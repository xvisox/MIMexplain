from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from explainer.io import (
    load_image,
    pil_to_tensor,
    render_heatmap,
    render_heatmap_overlay,
    save_heatmap_overlay,
    tensor_to_pil_image,
)


def test_load_image_returns_rgb_pil(example_image_path: Path) -> None:
    img = load_image(example_image_path)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_pil_tensor_roundtrip_preserves_size(example_image_path: Path) -> None:
    original = load_image(example_image_path)
    tensor = pil_to_tensor(original)
    restored = tensor_to_pil_image(tensor)

    assert tuple(tensor.shape) == (3, original.height, original.width)
    assert restored.mode == "RGB"
    assert restored.size == original.size


def test_render_and_save_heatmap_overlay(tmp_path: Path) -> None:
    image = torch.full((3, 6, 8), 0.25, dtype=torch.float32)
    explanation = torch.tensor([[0.1, 0.3], [0.7, 1.0]], dtype=torch.float32)

    overlay = render_heatmap_overlay(image, explanation)
    overlay_path, array_path = save_heatmap_overlay(
        overlay,
        explanation,
        tmp_path,
        "io-test",
    )

    assert tuple(overlay.shape) == tuple(image.shape)
    assert overlay_path.exists()
    assert array_path.exists()
    np.testing.assert_allclose(np.load(array_path), explanation.numpy())


def test_render_heatmap_overlay_is_differentiable() -> None:
    image = torch.rand((3, 6, 8), dtype=torch.float32, requires_grad=True)
    explanation = torch.rand((2, 2), dtype=torch.float32, requires_grad=True)

    heatmap = render_heatmap(explanation, image.shape[-2:])
    overlay = render_heatmap_overlay(image, explanation)
    loss = heatmap.sum() + overlay.sum()
    loss.backward()

    assert heatmap.requires_grad
    assert overlay.requires_grad
    assert image.grad is not None
    assert explanation.grad is not None
