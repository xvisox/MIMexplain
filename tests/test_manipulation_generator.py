from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from explainer.manipulation import ManipulationConfig, build_uniform_budget
from explainer.manipulation.generator import (
    GeneratorConfig,
    PerturbationGenerator,
    apply_generator_delta,
    count_parameters,
    supervised_delta_loss,
)
from scripts.run_adversarial_generator import (
    cap_corpus,
    delta_from_result,
    discover_hubert_corpus_entries,
    exclude_paths_by_name,
    filter_corpus_manifest,
    iou_at_mean,
    should_checkpoint,
    validate_resolution,
)


# --------------------------------------------------------------------------- #
# apply_generator_delta (mirrors apply_universal_delta; no model needed)
# --------------------------------------------------------------------------- #
def test_apply_generator_delta_respects_budget_and_shape() -> None:
    image = torch.rand(3, 8, 8)
    budget = build_uniform_budget(
        (8, 8), max_epsilon=0.1, device=torch.device("cpu"), dtype=torch.float32
    )
    raw = torch.full((3, 8, 8), 50.0)  # saturates sigmoid; bound must still hold

    perturbed = apply_generator_delta(raw, budget, image)

    assert perturbed.shape == image.shape
    assert float((perturbed - image).abs().amax()) <= 0.1 + 1e-6
    assert float(perturbed.min()) >= 0.0
    assert float(perturbed.max()) <= 1.0


def test_apply_generator_delta_zero_delta_is_identity() -> None:
    image = torch.rand(3, 6, 6)
    budget = build_uniform_budget(
        (6, 6), max_epsilon=0.2, device=torch.device("cpu"), dtype=torch.float32
    )
    perturbed = apply_generator_delta(torch.zeros(3, 6, 6), budget, image)
    torch.testing.assert_close(perturbed, image)


# --------------------------------------------------------------------------- #
# PerturbationGenerator
# --------------------------------------------------------------------------- #
def test_generator_preserves_spatial_shape_and_is_finite() -> None:
    generator = PerturbationGenerator(GeneratorConfig())
    image = torch.rand(2, 3, 64, 64)

    raw_delta = generator(image)

    assert raw_delta.shape == image.shape
    assert torch.isfinite(raw_delta).all()


def test_generator_is_much_smaller_than_clip() -> None:
    generator = PerturbationGenerator(GeneratorConfig())

    params = count_parameters(generator)

    # "significantly smaller than CLIP (~150M)" but a real conv net, not trivial.
    assert 1_000_000 < params < 5_000_000


# --------------------------------------------------------------------------- #
# supervised_delta_loss
# --------------------------------------------------------------------------- #
def test_supervised_delta_loss_is_zero_at_exact_match() -> None:
    target = torch.rand(3, 8, 8)
    loss = supervised_delta_loss(target.clone(), target)
    assert math.isclose(float(loss), 0.0, abs_tol=1e-7)


def test_supervised_delta_loss_combines_l1_and_l2() -> None:
    pred = torch.zeros(3, 4, 4)
    target = torch.full((3, 4, 4), 2.0)
    # L1 mean = 2.0, L2 mean = 4.0; weighted sum with explicit weights.
    loss = supervised_delta_loss(pred, target, w_l1=1.0, w_l2=0.5)
    assert math.isclose(float(loss), 1.0 * 2.0 + 0.5 * 4.0, rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# GeneratorConfig.to_manipulation_config
# --------------------------------------------------------------------------- #
def test_to_manipulation_config_maps_shared_weights() -> None:
    config = GeneratorConfig()
    mcfg = config.to_manipulation_config()

    assert isinstance(mcfg, ManipulationConfig)
    assert mcfg.w_mask_l1 == config.w_mask_l1
    assert mcfg.w_mask_l2 == config.w_mask_l2
    assert mcfg.w_mask_focal_fg == config.w_mask_focal_fg
    assert mcfg.w_perturb_l1 == config.w_perturb_l1
    assert mcfg.w_perturb_l2 == config.w_perturb_l2
    assert mcfg.w_perturb_linf == config.w_perturb_linf
    assert mcfg.w_pred_cos == config.w_pred_cos
    assert mcfg.w_dist_kl == config.w_dist_kl
    assert mcfg.dist_kl_temperature == config.dist_kl_temperature
    assert mcfg.max_epsilon == config.max_epsilon


def test_to_manipulation_config_applies_rebalancing_scales() -> None:
    config = GeneratorConfig(w_mask_scale=4.0, w_perturb_scale=0.1, w_pred_scale=8.0)
    mcfg = config.to_manipulation_config()

    # Mask weights scaled up, perturbation penalties scaled down.
    assert mcfg.w_mask_l1 == config.w_mask_l1 * 4.0
    assert mcfg.w_mask_l2 == config.w_mask_l2 * 4.0
    assert mcfg.w_mask_focal_fg == config.w_mask_focal_fg * 4.0
    assert mcfg.w_perturb_l1 == config.w_perturb_l1 * 0.1
    assert mcfg.w_perturb_l2 == config.w_perturb_l2 * 0.1
    assert mcfg.w_perturb_linf == config.w_perturb_linf * 0.1
    # Cosine prediction-preservation scaled up; KL term left to its own knob.
    assert mcfg.w_pred_cos == config.w_pred_cos * 8.0
    assert mcfg.w_dist_kl == config.w_dist_kl


# --------------------------------------------------------------------------- #
# Script pure helpers (scripts.run_adversarial_generator)
# --------------------------------------------------------------------------- #
def test_delta_from_result_is_difference_within_box() -> None:
    original = torch.rand(3, 8, 8)
    # A bounded perturbation engine output stays within +-max_epsilon then clamps.
    delta_true = (torch.rand(3, 8, 8) - 0.5) * 2 * 0.1  # within +-0.1
    best_image = torch.clamp(original + delta_true, 0.0, 1.0)

    recovered = delta_from_result(best_image, original)

    assert recovered.shape == original.shape
    # clamping only shrinks magnitude, so the box bound still holds.
    assert float(recovered.abs().amax()) <= 0.1 + 1e-6


def test_filter_corpus_manifest_keeps_high_iou() -> None:
    rows = [
        {"img_id": "a", "mask_iou_at_mean": 0.95},
        {"img_id": "b", "mask_iou_at_mean": 0.30},
        {"img_id": "c", "mask_iou_at_mean": 0.80},
    ]

    kept = filter_corpus_manifest(rows, min_iou=0.8)

    assert [r["img_id"] for r in kept] == ["a", "c"]


def test_validate_resolution_rejects_small_and_non_multiple() -> None:
    validate_resolution(512)  # ok
    validate_resolution(336)  # ok (21x21, glyph fits)
    with pytest.raises(ValueError):
        validate_resolution(224)  # 14x14, glyph does not fit
    with pytest.raises(ValueError):
        validate_resolution(500)  # not a multiple of the patch size (16)


def test_cap_corpus_truncates_only_when_positive() -> None:
    items = [1, 2, 3, 4, 5]
    assert cap_corpus(items, 2) == [1, 2]
    assert cap_corpus(items, 0) == items  # 0 == no cap
    assert cap_corpus(items, 99) == items  # cap above length is a no-op
    assert cap_corpus(items, -1) == items  # negative treated as no cap


def test_exclude_paths_by_name_drops_training_images() -> None:
    pool = [Path("/data/a.jpg"), Path("/data/b.jpg"), Path("/other/c.jpg")]
    # Matches on filename, so a different directory prefix still excludes.
    kept = exclude_paths_by_name(pool, {"b.jpg"})
    assert [p.name for p in kept] == ["a.jpg", "c.jpg"]
    # Empty exclude set is a no-op (keeps order).
    assert exclude_paths_by_name(pool, set()) == pool


def test_iou_at_mean_perfect_and_disjoint() -> None:
    # Target is the right half; an explanation hot exactly on the right half
    # (above its own mean there, below elsewhere) gives IoU == 1.
    target = torch.zeros(4, 4)
    target[:, 2:] = 1.0
    explanation = torch.zeros(4, 4)
    explanation[:, 2:] = 1.0
    assert math.isclose(iou_at_mean(explanation, target), 1.0, rel_tol=1e-6)

    # Hot on the disjoint (left) half -> zero overlap with the target.
    disjoint = torch.zeros(4, 4)
    disjoint[:, :2] = 1.0
    assert math.isclose(iou_at_mean(disjoint, target), 0.0, abs_tol=1e-7)


def test_should_checkpoint_fires_on_interval_and_disables_at_zero() -> None:
    assert should_checkpoint(49, 50)  # completed step 50 (1-indexed)
    assert should_checkpoint(99, 50)
    assert not should_checkpoint(48, 50)
    assert not should_checkpoint(50, 50)
    # 0 (or negative) disables intermediate checkpointing entirely.
    assert not should_checkpoint(49, 0)
    assert not should_checkpoint(0, 0)


def _write_hubert_dir(
    root: Path, img_id: str, *, iou: float, with_delta: bool = True
) -> None:
    import json

    import numpy as np

    d = root / img_id
    d.mkdir(parents=True)
    (d / "complete.json").write_text(
        json.dumps(
            {
                "img_id": img_id,
                "source_image": f"/data/{img_id}.jpg",
                "prompt": f"Photo of {img_id}",
                "canonical_size": 512,
                "outputs": {"delta": "delta.npy"},
            }
        )
    )
    (d / "final_metrics.json").write_text(json.dumps({"mask_iou_at_mean": iou}))
    if with_delta:
        np.save(d / "delta.npy", np.zeros((3, 8, 8), dtype=np.float32))


def test_discover_hubert_corpus_entries_filters_and_reads(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_hubert_dir(tmp_path, "good", iou=0.9)
    _write_hubert_dir(tmp_path, "weak", iou=0.3)
    _write_hubert_dir(tmp_path, "no_delta", iou=0.95, with_delta=False)

    entries = discover_hubert_corpus_entries(tmp_path, min_iou=0.5)

    assert [e["img_id"] for e in entries] == ["good"]
    assert entries[0]["path"] == "/data/good.jpg"
    assert entries[0]["caption"] == "Photo of good"
    assert entries[0]["canonical_size"] == 512
    assert entries[0]["mask_iou_at_mean"] == 0.9


# --------------------------------------------------------------------------- #
# Slow plumbing smoke (real CPU model; box target at 224 since the glyph needs
# a >=19 grid that 224->14x14 cannot fit). Checks the second-order finetune +
# eval pipeline end to end, NOT the experiment outcome.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_finetune_and_eval_smoke(metaclip2_pipeline_cpu_small) -> None:  # type: ignore[no-untyped-def]
    from scripts.run_adversarial_generator import (
        evaluate_generator,
        finetune_generator,
    )

    pipeline = metaclip2_pipeline_cpu_small
    image = torch.rand(3, 224, 224)
    sample = pipeline.run(image.to(pipeline.device), "a photo")
    h, w = (int(d) for d in sample.explanation.shape)
    target = torch.zeros(h, w)
    target[h // 2 - 1 : h // 2 + 1, w // 2 - 1 : w // 2 + 1] = 1.0  # centered box

    images = [torch.rand(3, 224, 224), torch.rand(3, 224, 224)]
    captions = ["a photo", "a photo"]
    budget = build_uniform_budget(
        (224, 224), max_epsilon=30 / 255, device=pipeline.device, dtype=torch.float32
    )
    config = GeneratorConfig(
        base_channels=8, num_residual_blocks=1, finetune_steps=2, micro_batch=1
    )
    generator = PerturbationGenerator(config)

    trace = finetune_generator(
        generator, pipeline, images, captions, target, budget, config, progress=False
    )
    assert trace and "mask" in trace[0]

    metrics = evaluate_generator(
        generator,
        pipeline,
        images,
        captions,
        target,
        budget,
        config.to_manipulation_config(),
    )
    assert len(metrics) == 2
    assert "mask_iou_at_mean" in metrics[0]
