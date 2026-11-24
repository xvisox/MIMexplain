from __future__ import annotations

from pathlib import Path

import pytest
import torch

from explainer.examples import get_example
from explainer.manipulation import (
    ManipulationConfig,
    UniversalManipulationEngine,
    apply_universal_delta,
    average_metrics,
    build_uniform_budget,
)
from explainer.manipulation.universal import UniversalManipulationResult
from explainer.metaclip2 import MetaClip2ExplanationPipeline
from scripts.run_universal_mask_transfer import (
    _merge_rows,
    aggregate_records,
    build_per_image_rows,
    build_trace_rows,
    make_split,
    parse_k_values,
)


# --------------------------------------------------------------------------- #
# apply_universal_delta / build_uniform_budget (no model needed)
# --------------------------------------------------------------------------- #
def test_apply_universal_delta_respects_budget_and_shape() -> None:
    image = torch.rand(3, 8, 8)
    budget = build_uniform_budget(
        (8, 8), max_epsilon=0.1, device=torch.device("cpu"), dtype=torch.float32
    )
    # Large raw delta saturates the sigmoid; the bound must still hold.
    delta = torch.full((3, 8, 8), 50.0)
    perturbed = apply_universal_delta(delta, budget, image)

    assert perturbed.shape == image.shape
    assert float((perturbed - image).abs().amax()) <= 0.1 + 1e-6
    assert float(perturbed.min()) >= 0.0
    assert float(perturbed.max()) <= 1.0


def test_apply_universal_delta_zero_delta_is_identity() -> None:
    image = torch.rand(3, 6, 6)
    budget = build_uniform_budget(
        (6, 6), max_epsilon=0.2, device=torch.device("cpu"), dtype=torch.float32
    )
    perturbed = apply_universal_delta(torch.zeros(3, 6, 6), budget, image)
    # 2*sigmoid(0)-1 == 0, so a zero raw delta leaves the image unchanged.
    torch.testing.assert_close(perturbed, image)


def test_build_uniform_budget_is_constant_max_epsilon() -> None:
    budget = build_uniform_budget(
        (4, 5), max_epsilon=0.07, device=torch.device("cpu"), dtype=torch.float32
    )
    assert budget.shape == (4, 5)
    assert torch.allclose(budget, torch.full((4, 5), 0.07))


# --------------------------------------------------------------------------- #
# parse_k_values
# --------------------------------------------------------------------------- #
def test_parse_k_values_sorts_and_dedups() -> None:
    assert parse_k_values("8, 2, 2, 1, 4") == [1, 2, 4, 8]


def test_parse_k_values_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_k_values("1,0,2")


def test_parse_k_values_rejects_non_integer() -> None:
    with pytest.raises(ValueError, match="Invalid K value"):
        parse_k_values("1,x,2")


# --------------------------------------------------------------------------- #
# make_split: deterministic, disjoint, nested across K
# --------------------------------------------------------------------------- #
def _pool(n: int) -> list[Path]:
    return [Path(f"img_{i}.jpg") for i in range(n)]


def test_make_split_is_deterministic_for_same_seed() -> None:
    pool = _pool(20)
    test_a, train_a = make_split(pool, seed_key="0:0", test_size=5)
    test_b, train_b = make_split(pool, seed_key="0:0", test_size=5)
    assert test_a == test_b
    assert train_a == train_b


def test_make_split_disjoint_and_sized() -> None:
    pool = _pool(20)
    test, train = make_split(pool, seed_key="0:1", test_size=5)
    assert len(test) == 5
    assert len(train) == 15
    assert set(test).isdisjoint(set(train))
    assert set(test) | set(train) == set(pool)


def test_make_split_train_sets_are_nested_across_k() -> None:
    pool = _pool(20)
    _, train = make_split(pool, seed_key="0:0", test_size=5)
    # The first-K slices are prefixes, so smaller K is a subset of larger K.
    assert train[:2] == train[:4][:2]
    assert train[:4] == train[:8][:4]


def test_make_split_different_seeds_differ() -> None:
    pool = _pool(50)
    test_a, _ = make_split(pool, seed_key="0:0", test_size=10)
    test_b, _ = make_split(pool, seed_key="0:1", test_size=10)
    assert test_a != test_b


def test_make_split_rejects_too_few_images() -> None:
    with pytest.raises(ValueError, match="more than test_size"):
        make_split(_pool(5), seed_key="0:0", test_size=5)


# --------------------------------------------------------------------------- #
# aggregate_records: mean/std per (k, split, metric)
# --------------------------------------------------------------------------- #
def test_aggregate_records_computes_mean_and_std() -> None:
    records = [
        {"seed": 0, "k": 4, "split": "transfer", "mask_iou_at_mean": 0.2},
        {"seed": 1, "k": 4, "split": "transfer", "mask_iou_at_mean": 0.4},
        {"seed": 2, "k": 4, "split": "transfer", "mask_iou_at_mean": 0.6},
    ]
    agg = aggregate_records(records)
    assert len(agg) == 1
    row = agg[0]
    assert row["k"] == 4
    assert row["split"] == "transfer"
    assert row["metric"] == "mask_iou_at_mean"
    assert row["mean"] == pytest.approx(0.4)
    assert row["std"] == pytest.approx(0.16329, abs=1e-4)
    assert row["n"] == 3


def test_aggregate_records_accepts_non_numeric_k() -> None:
    # The generator eval (run_adversarial_generator.cmd_eval) tags rows with
    # k="G" instead of an integer K so the generator point is distinct on the
    # transfer-vs-K overlay; aggregation must group by it, not choke on int("G").
    records = [
        {"seed": 0, "k": "G", "split": "transfer", "mask_iou_at_mean": 0.05},
        {"seed": 1, "k": "G", "split": "transfer", "mask_iou_at_mean": 0.07},
    ]
    agg = aggregate_records(records)
    assert len(agg) == 1
    assert agg[0]["k"] == "G"
    assert agg[0]["mean"] == pytest.approx(0.06)
    assert agg[0]["n"] == 2


def test_aggregate_records_separates_groups() -> None:
    records = [
        {"seed": 0, "k": 1, "split": "train", "mask_l1": 0.1},
        {"seed": 0, "k": 1, "split": "transfer", "mask_l1": 0.5},
        {"seed": 0, "k": 2, "split": "train", "mask_l1": 0.2},
    ]
    agg = aggregate_records(records)
    keys = {(r["k"], r["split"], r["metric"]) for r in agg}
    assert keys == {
        (1, "train", "mask_l1"),
        (1, "transfer", "mask_l1"),
        (2, "train", "mask_l1"),
    }


# --------------------------------------------------------------------------- #
# build_per_image_rows / average_metrics
# --------------------------------------------------------------------------- #
def test_build_per_image_rows_attaches_id_and_metadata() -> None:
    paths = [Path("a.jpg"), Path("b.jpg")]
    metrics = [{"mask_iou_at_mean": 0.3}, {"mask_iou_at_mean": 0.7}]
    rows = build_per_image_rows(
        seed=2, k=4, split="transfer", paths=paths, metrics_list=metrics
    )
    assert rows == [
        {
            "seed": 2,
            "k": 4,
            "split": "transfer",
            "img_id": "a",
            "mask_iou_at_mean": 0.3,
        },
        {
            "seed": 2,
            "k": 4,
            "split": "transfer",
            "img_id": "b",
            "mask_iou_at_mean": 0.7,
        },
    ]


def test_build_per_image_rows_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        build_per_image_rows(
            seed=0, k=1, split="train", paths=[Path("a.jpg")], metrics_list=[]
        )


def test_average_metrics_matches_manual_mean() -> None:
    per_image = [
        {"mask_iou_at_mean": 0.2, "pert_linf": 0.10},
        {"mask_iou_at_mean": 0.6, "pert_linf": 0.20},
    ]
    averaged = average_metrics(per_image)
    assert averaged["mask_iou_at_mean"] == pytest.approx(0.4)
    assert averaged["pert_linf"] == pytest.approx(0.15)


def test_average_metrics_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        average_metrics([])


# --------------------------------------------------------------------------- #
# _merge_rows: split-invocation persistence (K=1,2 then K=4,8 into one dir)
# --------------------------------------------------------------------------- #
def test_merge_rows_preserves_prior_run_and_sorts() -> None:
    first_run = [
        {"seed": 0, "k": 1, "split": "train", "mask_l1": 0.1},
        {"seed": 0, "k": 2, "split": "train", "mask_l1": 0.2},
    ]
    second_run = [
        {"seed": 0, "k": 4, "split": "train", "mask_l1": 0.4},
        {"seed": 0, "k": 8, "split": "train", "mask_l1": 0.8},
    ]
    merged = _merge_rows(first_run, second_run, key_columns=("seed", "k", "split"))
    # The K=1,2 rows from the first run survive, K=4,8 added, sorted by key.
    assert [r["k"] for r in merged] == [1, 2, 4, 8]


def test_merge_rows_new_overwrites_matching_key() -> None:
    existing = [{"seed": 0, "k": 2, "split": "train", "mask_l1": 0.2}]
    new = [{"seed": 0, "k": 2, "split": "train", "mask_l1": 0.9}]
    merged = _merge_rows(existing, new, key_columns=("seed", "k", "split"))
    # Re-running a (seed, K, split) replaces it with the fresh value.
    assert merged == [{"seed": 0, "k": 2, "split": "train", "mask_l1": 0.9}]


def test_build_trace_rows_tags_seed_and_k() -> None:
    trace = [
        {"step": 0.0, "loss": 1.2, "mask": 0.8},
        {"step": 10.0, "loss": 0.5, "mask": 0.3},
    ]
    rows = build_trace_rows(seed=1, k=8, trace=trace)
    assert rows == [
        {"seed": 1, "k": 8, "step": 0.0, "loss": 1.2, "mask": 0.8},
        {"seed": 1, "k": 8, "step": 10.0, "loss": 0.5, "mask": 0.3},
    ]


def test_build_trace_rows_empty_trace() -> None:
    assert build_trace_rows(seed=0, k=1, trace=[]) == []


# --------------------------------------------------------------------------- #
# slow end-to-end smoke on the tiny CPU pipeline
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_universal_train_and_evaluate_smoke(
    metaclip2_pipeline_cpu_small: MetaClip2ExplanationPipeline,
) -> None:
    example = get_example("winter-bench")
    # Three images with batch_size=2 exercises micro-batch accumulation (2 then 1):
    # every step is a full pass over all three before one optimizer step.
    images = [torch.rand(3, 224, 224) for _ in range(3)]
    target = torch.zeros(14, 14)
    target[5:9, 5:9] = 1.0

    config = ManipulationConfig(
        steps=2,
        log_every=1,
        w_mask_l1=1.0,
        w_perturb_l2=5.0,
        max_epsilon=20 / 255,
    )
    engine = UniversalManipulationEngine(metaclip2_pipeline_cpu_small)
    result = engine.train(
        images,
        [example.caption] * len(images),
        target,
        config,
        batch_size=2,
        progress=False,
    )

    assert isinstance(result, UniversalManipulationResult)
    assert result.delta.shape == images[0].shape
    assert result.perturbation_budget.shape == (224, 224)
    assert "mask_iou_at_mean" in result.train_metrics
    assert result.trace
    # Per-image train metrics: one dict per training image (all K covered),
    # averaging to the mean.
    assert len(result.train_per_image) == 3
    assert average_metrics(result.train_per_image)["mask_l1"] == pytest.approx(
        result.train_metrics["mask_l1"]
    )

    test_images = [torch.rand(3, 224, 224), torch.rand(3, 224, 224)]
    per_image = engine.evaluate_per_image(
        result.delta,
        result.perturbation_budget,
        test_images,
        [example.caption, example.caption],
        result.target_explanation,
        config,
    )
    assert len(per_image) == len(test_images)
    assert "mask_iou_at_mean" in per_image[0]
    # evaluate() must equal the mean of the per-image metrics.
    averaged = engine.evaluate(
        result.delta,
        result.perturbation_budget,
        test_images,
        [example.caption, example.caption],
        result.target_explanation,
        config,
    )
    assert averaged["mask_l1"] == pytest.approx(
        average_metrics(per_image)["mask_l1"], abs=1e-5
    )


@pytest.mark.slow
def test_universal_train_with_distribution_anchor_records_kl(
    metaclip2_pipeline_cpu_small: MetaClip2ExplanationPipeline,
) -> None:
    from explainer.manipulation import DistributionAnchor

    pipeline = metaclip2_pipeline_cpu_small
    example = get_example("winter-bench")
    image = torch.rand(3, 224, 224)
    target = torch.zeros(14, 14)
    target[5:9, 5:9] = 1.0

    # Build a tiny anchor sized to the model's embedding dimension.
    embed_dim = pipeline.run(image, example.caption).image_embedding.shape[-1]
    anchor = DistributionAnchor(
        text_embeddings=torch.nn.functional.normalize(torch.rand(4, embed_dim), dim=-1),
        logit_scale=torch.tensor(100.0),
        labels=("a", "b", "c", "d"),
    )
    config = ManipulationConfig(steps=2, log_every=1, w_mask_l1=1.0, w_dist_kl=1.0)
    engine = UniversalManipulationEngine(pipeline)

    result = engine.train(
        [image],
        [example.caption],
        target,
        config,
        batch_size=1,
        progress=False,
        distribution_anchor=anchor,
    )
    # The KL preservation term must surface in metrics and the trace.
    assert "dist_kl" in result.train_metrics
    assert "dist_kl_T1" in result.train_metrics
    assert "dist" in result.trace[0]
