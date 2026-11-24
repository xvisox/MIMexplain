from __future__ import annotations

import numpy as np
import pytest
import torch

from explainer.io import load_image
from explainer.metaclip2 import MetaClip2ExplanationPipeline

pytestmark = pytest.mark.slow


def test_winter_bench_smoke(
    metaclip2_pipeline_cpu_small: MetaClip2ExplanationPipeline,
) -> None:
    result = metaclip2_pipeline_cpu_small.run_example("winter-bench")

    assert result.map_size == (14, 14)
    assert result.image.ndim == 3
    assert result.image.shape[0] == 3
    assert tuple(result.image.shape[-2:]) == (224, 224)
    assert tuple(result.explanation.shape) == result.map_size
    assert torch.isfinite(result.explanation).all()
    assert torch.all(result.explanation >= 0)

    assert result.similarity.ndim == 0
    assert torch.isfinite(result.similarity)
    assert result.image_embedding.shape[0] == 1
    assert result.text_embedding.shape[0] == 1
    assert result.image_embedding.shape[-1] == result.text_embedding.shape[-1]
    torch.testing.assert_close(
        result.image_embedding.norm(dim=-1),
        torch.ones(1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_run_accepts_image_tensor(
    metaclip2_pipeline_cpu_small: MetaClip2ExplanationPipeline,
    winter_bench_example,
) -> None:
    image = (
        torch.from_numpy(
            np.asarray(load_image(winter_bench_example.image_path), dtype=np.float32)
        ).permute(2, 0, 1)
        / 255.0
    )

    result = metaclip2_pipeline_cpu_small.run(
        image,
        winter_bench_example.caption,
    )

    assert tuple(result.image.shape[-2:]) == (224, 224)
    assert result.image.shape != image.shape
    assert tuple(result.explanation.shape) == result.map_size


def test_prepare_image_inputs_matches_pil_preprocessing(
    metaclip2_pipeline_cpu_small: MetaClip2ExplanationPipeline,
    winter_bench_example,
) -> None:
    pil_image = load_image(winter_bench_example.image_path)
    tensor_image = (
        torch.from_numpy(np.asarray(pil_image, dtype=np.float32)).permute(2, 0, 1)
        / 255.0
    )

    actual = metaclip2_pipeline_cpu_small._prepare_image_inputs(tensor_image)
    image_size = metaclip2_pipeline_cpu_small.config.resized_image_size
    expected = metaclip2_pipeline_cpu_small.processor(
        images=pil_image,
        return_tensors="pt",
        size={"height": image_size, "width": image_size},
        crop_size={"height": image_size, "width": image_size},
    )

    torch.testing.assert_close(
        actual["pixel_values"].cpu(),
        expected["pixel_values"],
        rtol=0.0,
        atol=0.02,
    )


def test_prepare_image_inputs_preserves_input_gradients(
    metaclip2_pipeline_cpu_small: MetaClip2ExplanationPipeline,
) -> None:
    image = torch.rand((3, 32, 32), dtype=torch.float32, requires_grad=True)

    pixel_values = metaclip2_pipeline_cpu_small._prepare_image_inputs(image)[
        "pixel_values"
    ]
    pixel_values.sum().backward()

    assert image.grad is not None


def test_winter_bench_reproducible(
    metaclip2_pipeline_cpu_small: MetaClip2ExplanationPipeline,
) -> None:
    torch.manual_seed(0)
    first = metaclip2_pipeline_cpu_small.run_example("winter-bench")
    torch.manual_seed(0)
    second = metaclip2_pipeline_cpu_small.run_example("winter-bench")

    first_arr = first.explanation.detach().cpu().numpy()
    second_arr = second.explanation.detach().cpu().numpy()
    np.testing.assert_allclose(first_arr, second_arr, atol=1e-5)
