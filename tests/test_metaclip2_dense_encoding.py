from __future__ import annotations

import pytest
import torch
from transformers.models.metaclip_2.modeling_metaclip_2 import ALL_ATTENTION_FUNCTIONS

from explainer.gradeclip import clip_encode_image_dense
from explainer.io import load_image, pil_to_tensor
from explainer.metaclip2 import MetaClip2ExplanationPipeline

pytestmark = pytest.mark.slow


def test_clip_encode_image_dense_matches_official_last_layer(
    metaclip2_pipeline_cpu_small: MetaClip2ExplanationPipeline,
) -> None:
    image = pil_to_tensor(load_image("assets/examples/two_cats.jpg"))
    pixel_values = metaclip2_pipeline_cpu_small._prepare_image_inputs(image)["pixel_values"]
    encoding = clip_encode_image_dense(
        pixel_values,
        metaclip2_pipeline_cpu_small.model,
        interpolate_pos_encoding=metaclip2_pipeline_cpu_small.interpolate_pos_encoding,
    )

    vision_model = metaclip2_pipeline_cpu_small.model.vision_model
    try:
        embeddings = vision_model.embeddings(
            pixel_values,
            interpolate_pos_encoding=metaclip2_pipeline_cpu_small.interpolate_pos_encoding,
        )
    except TypeError:
        embeddings = vision_model.embeddings(pixel_values)

    pre_layernorm = getattr(vision_model, "pre_layernorm", None) or getattr(
        vision_model, "pre_layrnorm"
    )
    hidden_states = pre_layernorm(embeddings)
    for layer in vision_model.encoder.layers[:-1]:
        hidden_states = layer(hidden_states, attention_mask=None)

    last_layer = vision_model.encoder.layers[-1]
    attn_module = last_layer.self_attn
    x_normed = last_layer.layer_norm1(hidden_states)

    input_shape = x_normed.shape[:-1]
    hidden_shape = (*input_shape, -1, attn_module.head_dim)
    queries = attn_module.q_proj(x_normed).view(hidden_shape).transpose(1, 2)
    keys = attn_module.k_proj(x_normed).view(hidden_shape).transpose(1, 2)
    values = attn_module.v_proj(x_normed).view(hidden_shape).transpose(1, 2)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        attn_module.config._attn_implementation, None
    )
    attn_output, _ = attention_interface(
        attn_module,
        queries,
        keys,
        values,
        None,
        scaling=attn_module.scale,
        dropout=0.0,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    hidden_states = hidden_states + attn_module.out_proj(attn_output)
    hidden_states = hidden_states + last_layer.mlp(last_layer.layer_norm2(hidden_states))
    expected = metaclip2_pipeline_cpu_small.model.visual_projection(
        vision_model.post_layernorm(hidden_states)
    )

    torch.testing.assert_close(
        encoding.outputs.detach(),
        expected.detach(),
        atol=1e-4,
        rtol=1e-4,
    )
