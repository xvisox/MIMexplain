import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, cast

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import Tensor

MapSize = tuple[int, int]
AttentionMode = Literal["image", "text"]
PathLike = str | os.PathLike[str]
FloatArray = npt.NDArray[np.floating[Any]]
UInt8Array = npt.NDArray[np.uint8]


class ProjectionLayerLike(Protocol):
    weight: Tensor
    bias: Tensor | None

    def __call__(self, inputs: Tensor) -> Tensor: ...


class AttentionLayerLike(Protocol):
    q_proj: ProjectionLayerLike
    k_proj: ProjectionLayerLike
    v_proj: ProjectionLayerLike
    out_proj: ProjectionLayerLike


class EncoderLayerLike(Protocol):
    self_attn: AttentionLayerLike
    layer_norm1: torch.nn.Module
    layer_norm2: torch.nn.Module
    mlp: torch.nn.Module

    def __call__(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None, **kwargs: Any
    ) -> Any: ...


class EncoderStackLike(Protocol):
    layers: Sequence[EncoderLayerLike]


class VisionTransformerLike(Protocol):
    pre_layernorm: torch.nn.Module
    embeddings: torch.nn.Module
    encoder: EncoderStackLike
    post_layernorm: torch.nn.Module


class TextTransformerLike(Protocol):
    embeddings: torch.nn.Module
    encoder: EncoderStackLike
    final_layer_norm: torch.nn.Module
    eos_token_id: int | None


class AttentionOutputLike(Protocol):
    attentions: Sequence[Tensor] | None


class ContrastiveOutputLike(Protocol):
    logits_per_image: Tensor
    vision_model_output: AttentionOutputLike
    text_model_output: AttentionOutputLike


class ClipLikeModel(Protocol):
    vision_model: VisionTransformerLike
    text_model: TextTransformerLike
    visual_projection: torch.nn.Module
    text_projection: torch.nn.Module

    def zero_grad(self) -> None: ...

    def __call__(
        self,
        *,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> ContrastiveOutputLike: ...


class TextLayerIntermediates(NamedTuple):
    q_outs: list[Tensor]
    k_outs: list[Tensor]
    values: list[Tensor]


class ImageDenseEncoding(NamedTuple):
    outputs: Tensor
    v_final: Tensor
    last_input: Tensor
    values: Tensor
    q_out: Tensor
    k_out: Tensor
    attn_weights: Tensor
    att_output: Tensor
    map_size: MapSize


class TextDenseEncoding(NamedTuple):
    text_embedding: Tensor
    layer_intermediates: TextLayerIntermediates
    attns: list[Tensor]
    attn_outputs: list[Tensor]


def _unwrap_hidden_states(layer_output: Any) -> Tensor:
    if isinstance(layer_output, tuple):
        return cast(Tensor, layer_output[0])
    return cast(Tensor, layer_output)


def _run_vision_layer(layer: EncoderLayerLike, hidden_states: Tensor) -> Tensor:
    try:
        return _unwrap_hidden_states(
            layer(hidden_states, attention_mask=None, causal_attention_mask=None)
        )
    except TypeError:
        return _unwrap_hidden_states(layer(hidden_states, attention_mask=None))


def _build_text_attention_mask(
    input_ids: Tensor,
    attention_mask: Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    causal_mask = _make_causal_mask(
        input_ids.shape[0], input_ids.shape[1], dtype, device
    )
    if attention_mask is None:
        return causal_mask

    key_mask = attention_mask[:, None, :].to(dtype=dtype)
    padding_mask = (1.0 - key_mask) * torch.finfo(dtype).min
    return causal_mask + padding_mask


def _run_text_layer(
    layer: EncoderLayerLike, hidden_states: Tensor, attention_mask: Tensor
) -> Tensor:
    try:
        return _unwrap_hidden_states(
            layer(
                hidden_states, attention_mask=None, causal_attention_mask=attention_mask
            ),
        )
    except TypeError:
        return _unwrap_hidden_states(
            layer(hidden_states, attention_mask=attention_mask, is_causal=True)
        )


def _infer_patch_map_size(num_patches: int) -> MapSize:
    side_length = int(num_patches**0.5)
    if side_length * side_length != num_patches:
        raise ValueError(
            f"Expected a square patch grid, received {num_patches} patches."
        )
    return side_length, side_length


def _find_eos_positions(input_ids: Tensor, eos_token_id: int | None) -> Tensor:
    fallback_positions = input_ids.argmax(dim=-1)
    if eos_token_id is None:
        return fallback_positions

    eos_mask = input_ids.eq(eos_token_id)
    first_eos = eos_mask.int().argmax(dim=-1)
    has_eos = eos_mask.any(dim=-1)
    return torch.where(has_eos, first_eos, fallback_positions)


def _normalize_map(explanation: FloatArray) -> FloatArray:
    return (explanation - explanation.min()) / (
        explanation.max() - explanation.min() + 1e-8
    )


# The dense decomposition path below is the reusable attribution core for the
# repository's MetaCLIP-2 baseline.


def _attention_layer_single_head(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Single-head scaled dot product attention in LND format."""
    embed_dim = q.shape[2]
    q_t = (q * (float(embed_dim) ** -0.5)).transpose(0, 1)
    k_t = k.transpose(0, 1)
    v_t = v.transpose(0, 1)
    attn_weights = torch.bmm(q_t, k_t.transpose(1, 2))
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.bmm(attn_weights, v_t).transpose(0, 1)
    return attn_output, attn_weights.unsqueeze(1)


def _make_causal_mask(
    bsz: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Build an upper-triangular additive causal mask."""
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )
    return mask.unsqueeze(0).expand(bsz, -1, -1)


def clip_encode_image_dense(
    pixel_values: Tensor,
    clip_model: ClipLikeModel,
    interpolate_pos_encoding: bool = False,
) -> ImageDenseEncoding:
    """Decompose the vision encoder at its last layer and return dense Grad-ECLIP features."""
    model = clip_model.vision_model
    try:
        embeddings = model.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
    except TypeError:
        embeddings = model.embeddings(pixel_values)
    # Some MetaCLIP-2 checkpoints expose a legacy misspelled pre-layernorm attribute.
    pre_layernorm = getattr(model, "pre_layernorm", None)
    if pre_layernorm is None:
        pre_layernorm = getattr(model, "pre_layrnorm")
    hidden_states = cast(Tensor, pre_layernorm(embeddings))
    num_patches = hidden_states.shape[1] - 1
    map_size = _infer_patch_map_size(num_patches)

    for layer in model.encoder.layers[:-1]:
        hidden_states = _run_vision_layer(layer, hidden_states)

    last_input = hidden_states.requires_grad_(True)
    last_layer = model.encoder.layers[-1]
    attn_module = last_layer.self_attn
    x_normed = cast(Tensor, last_layer.layer_norm1(last_input))

    q_lnd = attn_module.q_proj(x_normed).permute(1, 0, 2)
    k_lnd = attn_module.k_proj(x_normed).permute(1, 0, 2)
    v_lnd = attn_module.v_proj(x_normed).permute(1, 0, 2)
    attn_output_lnd, attn_weights = _attention_layer_single_head(q_lnd, k_lnd, v_lnd)

    hidden_states = last_input + attn_module.out_proj(attn_output_lnd.permute(1, 0, 2))
    hidden_states = hidden_states + cast(
        Tensor, last_layer.mlp(last_layer.layer_norm2(hidden_states))
    )
    all_projected = cast(
        Tensor, clip_model.visual_projection(model.post_layernorm(hidden_states))
    )

    with torch.no_grad():
        qkv = torch._C._nn.linear(
            torch.stack((q_lnd, k_lnd, v_lnd)),
            attn_module.out_proj.weight,
            attn_module.out_proj.bias,
        )
        q_out = qkv[0]
        k_out = qkv[1]
        v_hs = last_input + qkv[2].permute(1, 0, 2)
        v_hs = v_hs + cast(Tensor, last_layer.mlp(last_layer.layer_norm2(v_hs)))
        v_final = cast(
            Tensor, clip_model.visual_projection(model.post_layernorm(v_hs))
        )[:, 1:]

    return ImageDenseEncoding(
        outputs=all_projected,
        v_final=v_final,
        last_input=last_input,
        values=v_lnd,
        q_out=q_out,
        k_out=k_out,
        attn_weights=attn_weights,
        att_output=attn_output_lnd,
        map_size=map_size,
    )


def clip_encode_text_dense(
    input_ids: Tensor,
    clip_model: ClipLikeModel,
    attention_mask: Tensor | None = None,
    n: int = 8,
) -> TextDenseEncoding:
    """Decompose the text encoder at its last ``n`` layers and return dense Grad-ECLIP features."""
    text_model = clip_model.text_model
    hidden_states = cast(Tensor, text_model.embeddings(input_ids=input_ids))
    effective_attention_mask = _build_text_attention_mask(
        input_ids,
        attention_mask,
        hidden_states.dtype,
        hidden_states.device,
    )

    frozen_prefix = max(len(text_model.encoder.layers) - n, 0)
    for layer in text_model.encoder.layers[:frozen_prefix]:
        hidden_states = _run_text_layer(layer, hidden_states, effective_attention_mask)

    attns: list[Tensor] = []
    attn_outputs: list[Tensor] = []
    values: list[Tensor] = []
    q_outs: list[Tensor] = []
    k_outs: list[Tensor] = []
    x_in = hidden_states

    for layer in text_model.encoder.layers[frozen_prefix:]:
        attn_module = layer.self_attn
        x_normed = cast(Tensor, layer.layer_norm1(x_in))
        q_lnd = attn_module.q_proj(x_normed).permute(1, 0, 2)
        k_lnd = attn_module.k_proj(x_normed).permute(1, 0, 2)
        v_lnd = attn_module.v_proj(x_normed).permute(1, 0, 2)

        attn_output_lnd, attn_w = _attention_layer_single_head(
            q_lnd,
            k_lnd,
            v_lnd,
            attn_mask=effective_attention_mask,
        )
        attn_outputs.append(attn_output_lnd)
        attns.append(attn_w)
        values.append(v_lnd)

        x = x_in + attn_module.out_proj(attn_output_lnd.permute(1, 0, 2))
        x_in = x + cast(Tensor, layer.mlp(layer.layer_norm2(x)))

        with torch.no_grad():
            qk = torch._C._nn.linear(
                torch.stack((q_lnd, k_lnd)),
                attn_module.out_proj.weight,
                attn_module.out_proj.bias,
            )
            q_outs.append(qk[0])
            k_outs.append(qk[1])

    x_out = cast(Tensor, text_model.final_layer_norm(x_in))
    eos_positions = _find_eos_positions(input_ids, text_model.eos_token_id)
    batch_indices = torch.arange(x_out.shape[0], device=x_out.device)
    text_embedding = cast(
        Tensor, clip_model.text_projection(x_out[batch_indices, eos_positions])
    )

    return TextDenseEncoding(
        text_embedding=text_embedding,
        layer_intermediates=TextLayerIntermediates(
            q_outs=q_outs, k_outs=k_outs, values=values
        ),
        attns=attns,
        attn_outputs=attn_outputs,
    )


def gradeclip_image(
    c: Tensor,
    q_out: Tensor,
    k_out: Tensor,
    v: Tensor,
    att_output: Tensor,
    map_size: MapSize,
    withksim: bool = True,
) -> Tensor:
    """Compute the Grad-ECLIP image explanation map."""
    grad_cls = torch.autograd.grad(
        c, att_output, retain_graph=True, create_graph=True
    )[0][:1, 0, :]
    if withksim:
        cls_to_patch_similarity = (
            F.normalize(q_out[:1, 0, :], dim=-1) * F.normalize(k_out[1:, 0, :], dim=-1)
        ).sum(-1)
        cls_to_patch_similarity = (
            cls_to_patch_similarity - cls_to_patch_similarity.min()
        ) / (cls_to_patch_similarity.max() - cls_to_patch_similarity.min() + 1e-8)
        explanation = F.relu(
            (grad_cls * v[1:, 0, :] * cls_to_patch_similarity[:, None]).sum(-1)
        )
    else:
        explanation = F.relu((grad_cls * v[1:, 0, :]).sum(-1))
    return explanation.reshape(*map_size)


def maskclip(
    txt_feats: Tensor, v_final: Tensor, k_out: Tensor, map_size: MapSize
) -> Tensor:
    """Compute the MaskCLIP explanation map."""
    sim = (
        F.normalize(v_final[0], dim=-1)
        @ F.normalize(txt_feats.reshape(1, -1), dim=-1).T
    ).squeeze(-1)
    key_similarity = (
        F.normalize(k_out[1:, 0, :], dim=-1)
        @ F.normalize(k_out[0, 0, :], dim=-1).unsqueeze(-1)
    ).squeeze(-1)
    return (sim * key_similarity).reshape(*map_size)


def gradcam(c: Tensor, layer_feat: Tensor, map_size: MapSize) -> Tensor:
    """Compute a Grad-CAM map from the input of the last encoder layer."""
    grad = torch.autograd.grad(c, layer_feat, retain_graph=True)[0].detach()
    return F.relu_((grad[0].mean(0) * layer_feat.detach()[0, 1:]).sum(-1)).reshape(
        *map_size
    )


def genericattention(
    pixel_values: Tensor,
    input_ids: Tensor,
    clip_model: ClipLikeModel,
    attention_mask: Tensor | None = None,
    start_layer: int = -1,
    flag: AttentionMode = "image",
) -> Tensor:
    """Compute generic attention-based relevance using Hugging Face attention outputs."""
    outputs = clip_model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
    )
    logits = outputs.logits_per_image
    batch_size = logits.shape[0]
    one_hot = torch.zeros_like(logits)
    one_hot[
        torch.arange(batch_size, device=logits.device),
        torch.arange(batch_size, device=logits.device),
    ] = 1
    one_hot_sum = (one_hot * logits).sum()
    clip_model.zero_grad()

    model_output = (
        outputs.vision_model_output if flag == "image" else outputs.text_model_output
    )
    if model_output.attentions is None:
        raise ValueError("The selected model did not return attention weights.")

    attentions = list(model_output.attentions)
    selected_start_layer = len(attentions) - 1 if start_layer == -1 else start_layer
    num_tokens = attentions[0].shape[-1]
    relevance = torch.eye(num_tokens, dtype=attentions[0].dtype, device=logits.device)
    relevance = relevance.unsqueeze(0).expand(batch_size, -1, -1).clone()

    for layer_index, attn in enumerate(attentions):
        if layer_index < selected_start_layer:
            continue
        grad = torch.autograd.grad(one_hot_sum, [attn], retain_graph=True)[0]
        relevance = relevance + torch.bmm(
            (grad.mean(1) * attn.mean(1)).clamp(min=0), relevance
        )

    if flag == "image":
        side_length = _infer_patch_map_size(relevance[:, 0, 1:].shape[-1])[0]
        return relevance[:, 0, 1:].reshape(batch_size, side_length, side_length)
    return relevance


def rolloutattention(
    all_layer_matrices: Sequence[Tensor],
    start_layer: int = 0,
    flag: AttentionMode = "image",
) -> Tensor:
    """Compute attention rollout across layers."""
    batch_size = all_layer_matrices[0].shape[0]
    num_tokens = all_layer_matrices[0].shape[-1]
    eye = (
        torch.eye(num_tokens, device=all_layer_matrices[0].device)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )
    matrices = [
        (matrix + eye) / (matrix + eye).sum(dim=-1, keepdim=True)
        for matrix in all_layer_matrices
    ]
    joint_attention = matrices[start_layer]
    for matrix in matrices[start_layer + 1 :]:
        joint_attention = matrix.bmm(joint_attention)
    if flag == "image":
        side_length = _infer_patch_map_size(joint_attention[:, 0, 1:].shape[-1])[0]
        return joint_attention[:, 0, 1:].reshape(batch_size, side_length, side_length)
    return joint_attention


def save_explanation(
    image: FloatArray,
    explanation: FloatArray,
    path: PathLike,
    tag: str,
    png: bool = True,
    npy: bool = False,
) -> None:
    """Overlay a heatmap on an image and persist it to disk."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_np = np.transpose(image, (1, 2, 0))
    img_np = _normalize_map(img_np)
    img_uint8 = (img_np * 255).astype(np.uint8)

    normalized_explanation = _normalize_map(explanation)
    resized_explanation = cv2.resize(
        normalized_explanation,
        (img_uint8.shape[1], img_uint8.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    color = cv2.applyColorMap(
        (resized_explanation * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = np.clip(img_uint8 * 0.5 + color * 0.5, 0, 255).astype(np.uint8)

    if npy:
        np.save(output_dir / f"explanation_{tag}.npy", resized_explanation)
    if png:
        cv2.imwrite(str(output_dir / f"explanation_{tag}.png"), overlay[:, :, ::-1])
