from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoProcessor, MetaClip2Model

from explainer.constants import (
    DEFAULT_MODEL_NAME,
    DEFAULT_RESIZED_IMAGE_SIZE,
    DEFAULT_TEXT_LAYERS,
)
from explainer.examples import get_example
from explainer.gradeclip import (
    clip_encode_image_dense,
    clip_encode_text_dense,
    gradeclip_image,
)
from explainer.io import (
    ensure_chw_image,
    load_image,
    pil_to_tensor,
)


@dataclass(frozen=True)
class MetaClip2RunConfig:
    model_name: str = DEFAULT_MODEL_NAME
    device: str = "auto"
    text_layers: int = DEFAULT_TEXT_LAYERS
    with_key_similarity: bool = True
    resized_image_size: int = DEFAULT_RESIZED_IMAGE_SIZE


@dataclass(frozen=True)
class MetaClip2RunResult:
    image: Tensor
    explanation: Tensor
    map_size: tuple[int, int]
    similarity: Tensor
    image_embedding: Tensor
    text_embedding: Tensor


def choose_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


class MetaClip2ExplanationPipeline:
    def __init__(self, config: MetaClip2RunConfig | None = None) -> None:
        self.config = config or MetaClip2RunConfig()
        self.device = choose_device(self.config.device)

        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model: MetaClip2Model = cast(
            MetaClip2Model, AutoModel.from_pretrained(self.config.model_name)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        vision_config = self.model.config.vision_config
        native_image_size = int(vision_config.image_size)
        patch_size = int(vision_config.patch_size)
        if self.config.resized_image_size % patch_size != 0:
            raise ValueError(
                "The resized MetaCLIP-2 image size must be divisible by the model patch size."
            )
        self.interpolate_pos_encoding = (
            self.config.resized_image_size != native_image_size
        )

    def run_example(self, name: str) -> MetaClip2RunResult:
        example = get_example(name)
        image = pil_to_tensor(load_image(example.image_path))
        return self.run(image, example.caption)

    def run(
        self,
        image: Tensor,
        caption: str,
    ) -> MetaClip2RunResult:
        image_chw = ensure_chw_image(image)
        image_inputs = self._prepare_image_inputs(image_chw)
        aligned_image = self._decode_pixel_values(image_inputs["pixel_values"][0])
        text_inputs = self._prepare_text_inputs(caption)

        image_encoding = clip_encode_image_dense(
            image_inputs["pixel_values"],
            self.model,
            interpolate_pos_encoding=self.interpolate_pos_encoding,
        )
        text_encoding = clip_encode_text_dense(
            text_inputs["input_ids"],
            self.model,
            attention_mask=text_inputs.get("attention_mask"),
            n=self.config.text_layers,
        )

        image_embedding = F.normalize(image_encoding.outputs[:, 0], dim=-1)
        text_embedding = F.normalize(text_encoding.text_embedding, dim=-1)
        similarity = (image_embedding @ text_embedding.T)[0, 0]

        explanation = gradeclip_image(
            similarity,
            image_encoding.q_out,
            image_encoding.k_out,
            image_encoding.values,
            image_encoding.att_output,
            image_encoding.map_size,
            withksim=self.config.with_key_similarity,
        )
        return MetaClip2RunResult(
            image=aligned_image,
            explanation=explanation,
            map_size=image_encoding.map_size,
            similarity=similarity,
            image_embedding=image_embedding,
            text_embedding=text_embedding,
        )

    def _prepare_image_inputs(self, image: Tensor) -> dict[str, Tensor]:
        if isinstance(image, torch.Tensor):
            return {"pixel_values": self._pixel_values_from_tensor(image)}

        batch = cast(
            dict[str, Any],
            self.processor(
                images=image,
                return_tensors="pt",
                do_rescale=False,
                size={
                    "height": self.config.resized_image_size,
                    "width": self.config.resized_image_size,
                },
                crop_size={
                    "height": self.config.resized_image_size,
                    "width": self.config.resized_image_size,
                },
            ),
        )
        return move_batch_to_device(batch, self.device)

    def _pixel_values_from_tensor(self, image: Tensor) -> Tensor:
        image_chw = ensure_chw_image(image).to(self.device)
        size = self.config.resized_image_size
        if tuple(image_chw.shape[-2:]) != (size, size):
            image_chw = F.interpolate(
                image_chw.unsqueeze(0),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        image_processor = self.processor.image_processor
        mean = torch.tensor(
            image_processor.image_mean,
            dtype=image_chw.dtype,
            device=image_chw.device,
        )[:, None, None]
        std = torch.tensor(
            image_processor.image_std,
            dtype=image_chw.dtype,
            device=image_chw.device,
        )[:, None, None]
        return ((image_chw - mean) / std).unsqueeze(0)

    def _prepare_text_inputs(self, caption: str) -> dict[str, Tensor]:
        batch = cast(
            dict[str, Any],
            self.processor(text=[caption], padding=True, return_tensors="pt"),
        )
        return move_batch_to_device(batch, self.device)

    def _decode_pixel_values(self, pixel_values: Tensor) -> Tensor:
        image_processor = self.processor.image_processor
        mean = torch.tensor(
            image_processor.image_mean,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )[:, None, None]
        std = torch.tensor(
            image_processor.image_std,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )[:, None, None]
        return torch.clamp(pixel_values * std + mean, 0.0, 1.0)
