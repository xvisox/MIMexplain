from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from explainer.io import ensure_chw_image, load_image, pil_to_tensor


def prepare_target_explanation(
    *,
    target_explanation: Tensor | str | Path,
    explanation_shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(target_explanation, torch.Tensor):
        target_tensor = target_explanation
    else:
        target_tensor = pil_to_tensor(load_image(target_explanation))

    if target_tensor.ndim == 2:
        target_map = target_tensor
    elif target_tensor.ndim == 3:
        target_chw = ensure_chw_image(target_tensor)
        target_map = target_chw.float().mean(dim=0)
    else:
        raise ValueError("Target explanation must be a 2D map or an image tensor.")

    if not torch.isfinite(target_map).all():
        raise ValueError("Target explanation must contain only finite values.")

    normalized = target_map.to(device=device, dtype=dtype)
    if tuple(normalized.shape) != explanation_shape:
        normalized = F.interpolate(
            normalized.unsqueeze(0).unsqueeze(0),
            size=explanation_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    return normalize_target_explanation(normalized)


def normalize_target_explanation(target_explanation: Tensor) -> Tensor:
    min_value = float(target_explanation.amin().item())
    max_value = float(target_explanation.amax().item())
    if 0.0 <= min_value and max_value <= 1.0:
        return target_explanation
    if max_value - min_value < 1e-12:
        return torch.clamp(target_explanation, 0.0, 1.0)
    return (target_explanation - min_value) / (max_value - min_value)
