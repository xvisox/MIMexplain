from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


def load_image(image_source: str | Path) -> Image.Image:
    source = str(image_source)
    if source.startswith(("http://", "https://")):
        with urlopen(source) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")
    return Image.open(source).convert("RGB")


def pil_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def ensure_chw_image(image: Tensor) -> Tensor:
    if image.ndim != 3:
        raise ValueError("Image tensor must be 3-dimensional.")
    if image.shape[0] in (1, 3):
        return image
    if image.shape[-1] in (1, 3):
        return image.permute(2, 0, 1)
    raise ValueError("Image tensor must have 1 or 3 channels.")


def tensor_to_pil_image(image: Tensor) -> Image.Image:
    image_cpu = ensure_chw_image(image.detach().cpu())
    image_hwc = image_cpu.permute(1, 2, 0)

    image_np = image_hwc.numpy()
    if np.issubdtype(image_np.dtype, np.floating):
        scale = 255.0 if image_np.max(initial=0.0) <= 1.0 else 1.0
        image_np = np.clip(image_np * scale, 0.0, 255.0)
    else:
        image_np = np.clip(image_np, 0, 255)

    image_uint8 = image_np.astype(np.uint8)
    if image_uint8.shape[-1] == 1:
        image_uint8 = np.repeat(image_uint8, 3, axis=-1)
    return Image.fromarray(image_uint8, mode="RGB")


def _normalize_tensor(values: Tensor) -> Tensor:
    values = values.to(dtype=torch.float32)
    value_min = values.amin()
    value_max = values.amax()
    return (values - value_min) / (value_max - value_min + 1e-8)


def _normalize_image(image: Tensor) -> Tensor:
    image_chw = ensure_chw_image(image).to(dtype=torch.float32)
    if image_chw.shape[0] == 1:
        image_chw = image_chw.repeat(3, 1, 1)
    scale = 255.0 if image_chw.amax().item() > 1.0 else 1.0
    return torch.clamp(image_chw / scale, 0.0, 1.0)


def render_heatmap(explanation: Tensor, output_size: tuple[int, int]) -> Tensor:
    if explanation.ndim != 2:
        raise ValueError("Explanation tensor must be 2-dimensional.")

    heatmap = _normalize_tensor(explanation).unsqueeze(0).unsqueeze(0)
    heatmap = (
        F.interpolate(
            heatmap,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )

    red = torch.clamp(1.5 - torch.abs(4.0 * heatmap - 3.0), 0.0, 1.0)
    green = torch.clamp(1.5 - torch.abs(4.0 * heatmap - 2.0), 0.0, 1.0)
    blue = torch.clamp(1.5 - torch.abs(4.0 * heatmap - 1.0), 0.0, 1.0)
    return torch.stack((red, green, blue), dim=0)


def render_heatmap_overlay(
    image: Tensor, explanation: Tensor, alpha: float = 0.5
) -> Tensor:
    image_chw = _normalize_image(image)
    heatmap = render_heatmap(explanation, tuple(image_chw.shape[-2:]))
    return torch.clamp((1.0 - alpha) * image_chw + alpha * heatmap, 0.0, 1.0)


def save_heatmap_overlay(
    image: Tensor,
    explanation: Tensor,
    output_dir: Path,
    tag: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / f"explanation_{tag}.png"
    array_path = output_dir / f"explanation_{tag}.npy"
    tensor_to_pil_image(image).save(overlay_path)
    np.save(array_path, explanation.detach().cpu().numpy().astype(np.float32))
    return overlay_path, array_path
