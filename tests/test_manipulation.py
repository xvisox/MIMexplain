from __future__ import annotations

import pytest
import torch

from explainer.examples import get_example
from explainer.manipulation import (
    ManipulationConfig,
    MetaClip2ManipulationEngine,
    PerceptualMaskConfig,
    build_perceptual_allowance_map,
    perceptual_budget_map,
)
from explainer.manipulation.engine import _selection_score, _set_progress_postfix
from explainer.metaclip2 import MetaClip2ExplanationPipeline


class _FakeProgressBar:
    def __init__(self) -> None:
        self.postfix: dict[str, str] | None = None

    def set_postfix(self, values: dict[str, str]) -> None:
        self.postfix = values


def _ring_target(
    size: int = 14,
    outer_radius: float = 4.5,
    inner_radius: float = 2.5,
) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (size - 1) / 2.0
    distance = torch.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    return ((distance <= outer_radius) & (distance >= inner_radius)).float()


def test_ring_target_has_expected_shape_and_signal() -> None:
    mask = _ring_target()
    assert mask.shape == (14, 14)
    assert float(mask.min()) == 0.0
    assert float(mask.max()) == 1.0
    assert float(mask.sum()) > 0.0
    assert float(mask.std()) > 0.0


def test_build_perceptual_allowance_map_returns_unit_interval_tensor() -> None:
    image = torch.zeros((3, 16, 16), dtype=torch.float32)
    image[:, 4:12, 4:12] = 1.0
    image[:, ::2, ::2] = 0.5

    allowance = build_perceptual_allowance_map(
        image,
        config=PerceptualMaskConfig(use_saliency=False),
    )

    assert allowance.shape == (16, 16)
    assert float(allowance.min()) >= 0.0
    assert float(allowance.max()) <= 1.0
    assert float(allowance.std()) > 0.0


def test_perceptual_budget_map_preserves_base_budget_at_zero_allowance() -> None:
    allowance = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)

    budget = perceptual_budget_map(
        allowance,
        max_epsilon=30 / 255,
        budget_max_scale=1.5,
    )

    torch.testing.assert_close(
        budget,
        torch.tensor([[30 / 255, 37.5 / 255, 45 / 255]], dtype=torch.float32),
    )


def test_set_progress_postfix_reports_loss_and_linf() -> None:
    progress = _FakeProgressBar()

    _set_progress_postfix(
        progress,
        loss_value=1.23456,
        input_image=torch.tensor([[[0.0, 0.5]]], dtype=torch.float32),
        original_image=torch.tensor([[[0.0, 0.25]]], dtype=torch.float32),
    )

    assert progress.postfix == {"loss": "1.2346", "linf": "0.2500"}


def test_selection_score_defaults_to_last_iterate() -> None:
    # The default returns the converged final iterate: the score must strictly
    # decrease in step so the last step always wins, regardless of loss values.
    config = ManipulationConfig()
    assert config.selection_metric == "last"
    early = _selection_score(step=0, mask_loss=0.79, total_loss=0.79, config=config)
    late = _selection_score(step=1999, mask_loss=0.02, total_loss=3.8, config=config)
    assert late < early


def test_selection_score_mask_prefers_lowest_mask_error() -> None:
    # The unperturbed first iterate has zero perturbation penalty, so its TOTAL
    # loss is the global minimum whenever the perturbation weights are heavy.
    # The "mask" criterion ignores that and tracks the manipulation objective.
    config = ManipulationConfig(selection_metric="mask")
    baseline = _selection_score(step=0, mask_loss=0.79, total_loss=0.79, config=config)
    perturbed = _selection_score(
        step=800, mask_loss=0.02, total_loss=3.8, config=config
    )
    assert perturbed < baseline


def test_selection_score_total_reproduces_legacy_behavior() -> None:
    config = ManipulationConfig(selection_metric="total")
    baseline = _selection_score(step=0, mask_loss=0.79, total_loss=0.79, config=config)
    perturbed = _selection_score(
        step=800, mask_loss=0.02, total_loss=3.8, config=config
    )
    # Legacy selection prefers the unperturbed baseline (the bug under study).
    assert baseline < perturbed


def test_selection_score_rejects_unknown_metric() -> None:
    config = ManipulationConfig(selection_metric="bogus")
    with pytest.raises(ValueError, match="selection_metric"):
        _selection_score(step=0, mask_loss=0.1, total_loss=0.2, config=config)


@pytest.mark.slow
def test_run_experiment_smoke(
    metaclip2_pipeline_cpu_small: MetaClip2ExplanationPipeline,
) -> None:
    example = get_example("winter-bench")
    config = ManipulationConfig(
        steps=2,
        log_every=1,
        w_mask_l1=1.0,
        w_perturb_l2=5.0,
        w_pred_cos=0.5,
        perceptual_mask=PerceptualMaskConfig(use_saliency=False),
    )
    engine = MetaClip2ManipulationEngine(metaclip2_pipeline_cpu_small)
    target_explanation = _ring_target()

    result = engine.run(
        image=example.image_path,
        caption=example.caption,
        target_explanation=target_explanation,
        config=config,
        progress=False,
    )

    expected_keys = {
        "mask_l1",
        "mask_iou_at_mean",
        "pert_l2_rms",
        "pert_linf",
        "cos_preserve",
        "sim_delta",
        "sim_before",
        "sim_after",
        "best_loss",
    }
    assert expected_keys.issubset(result.metrics.keys())
    assert result.trace
    assert result.target_explanation.shape == result.best_explanation.shape
    assert result.aligned_best_image.shape[-2:] == (224, 224)
    assert result.perceptual_allowance is not None
    assert result.perturbation_budget is not None
    assert result.perceptual_allowance.shape == result.perturbation_budget.shape
