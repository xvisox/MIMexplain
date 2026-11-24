from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from explainer.io import ensure_chw_image, load_image, pil_to_tensor


@dataclass(frozen=True)
class PerceptualMaskConfig:
    variance_kernel: int = 9
    gradient_kernel_size: int = 3
    structure_block_size: int = 5
    structure_kernel_size: int = 3
    use_saliency: bool = True
    budget_max_scale: float = 1.5
    variance_weight: float = 0.40
    gradient_weight: float = 0.35
    structure_weight: float = 0.25


def build_perceptual_allowance_map(
    image: Tensor | str | Path,
    *,
    output_shape: tuple[int, int] | None = None,
    config: PerceptualMaskConfig | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    effective_config = config or PerceptualMaskConfig()
    _validate_perceptual_config(effective_config)

    image_chw = _load_image_tensor(image)
    image_rgb_uint8 = _to_rgb_uint8(image_chw)
    image_bgr = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2BGR)

    gray8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray8.astype(np.float32) / 255.0

    mean = cv2.blur(gray, (effective_config.variance_kernel, effective_config.variance_kernel))
    mean2 = cv2.blur(
        gray * gray,
        (effective_config.variance_kernel, effective_config.variance_kernel),
    )
    variance = np.maximum(mean2 - mean * mean, 0.0)

    grad_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=effective_config.gradient_kernel_size,
    )
    grad_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=effective_config.gradient_kernel_size,
    )
    gradient = cv2.magnitude(grad_x, grad_y)

    structure = cv2.cornerMinEigenVal(
        gray8,
        blockSize=effective_config.structure_block_size,
        ksize=effective_config.structure_kernel_size,
    )

    variance_n = _norm01(variance)
    gradient_n = _norm01(gradient)
    structure_n = _norm01(structure)

    fused = (
        effective_config.variance_weight * variance_n
        + effective_config.gradient_weight * gradient_n
        + effective_config.structure_weight * structure_n
    )

    if effective_config.use_saliency:
        inverse_saliency = _compute_inverse_saliency(image_bgr)
        if inverse_saliency is not None:
            fused = fused * inverse_saliency

    fused = _norm01(fused)
    mask = torch.from_numpy(fused)
    desired_dtype = dtype or image_chw.dtype
    desired_device = device or image_chw.device
    mask = mask.to(device=desired_device, dtype=desired_dtype)

    if output_shape is not None and tuple(mask.shape) != output_shape:
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=output_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    return torch.clamp(mask, 0.0, 1.0)


def perceptual_budget_map(
    allowance_map: Tensor,
    *,
    max_epsilon: float,
    budget_max_scale: float,
) -> Tensor:
    scale = 1.0 + allowance_map * (budget_max_scale - 1.0)
    return max_epsilon * scale


def _load_image_tensor(image: Tensor | str | Path) -> Tensor:
    if isinstance(image, torch.Tensor):
        return ensure_chw_image(image.detach())
    return ensure_chw_image(pil_to_tensor(load_image(image)))


def _to_rgb_uint8(image: Tensor) -> np.ndarray:
    image_hwc = image.detach().cpu().permute(1, 2, 0).float().numpy()
    scaled = image_hwc * (255.0 if image_hwc.max(initial=0.0) <= 1.0 else 1.0)
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _validate_perceptual_config(config: PerceptualMaskConfig) -> None:
    for name, value in (
        ("variance_kernel", config.variance_kernel),
        ("gradient_kernel_size", config.gradient_kernel_size),
        ("structure_kernel_size", config.structure_kernel_size),
    ):
        if value % 2 == 0:
            raise ValueError(f"{name} must be odd, got {value}")
    if config.structure_block_size <= 1:
        raise ValueError("structure_block_size must be greater than 1")
    if config.budget_max_scale < 1.0:
        raise ValueError("budget_max_scale must be >= 1.0")


def _norm01(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = values.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < eps:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum + eps)


def _compute_inverse_saliency(image_bgr: np.ndarray) -> np.ndarray | None:
    if not hasattr(cv2, "saliency") or not hasattr(
        cv2.saliency, "StaticSaliencyFineGrained_create"
    ):
        return None
    detector = cv2.saliency.StaticSaliencyFineGrained_create()
    ok, saliency_map = detector.computeSaliency(image_bgr)
    if not ok or saliency_map is None:
        return None
    return 1.0 - _norm01(saliency_map.astype(np.float32))
