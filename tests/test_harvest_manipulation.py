from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from scripts import harvest_manipulation_results as harvest


def _write_image_dir(
    root: Path,
    img_id: str,
    *,
    label: str,
    baseline: dict[str, object],
    final_row: dict[str, object],
    complete: bool = True,
) -> Path:
    image_dir = root / img_id
    image_dir.mkdir(parents=True)
    (image_dir / "baseline_metrics.json").write_text(json.dumps(baseline))
    # Two rows: a step-0 row and the converged final row (the harvester must use
    # the LAST row, not the first).
    step0 = {**final_row, "step": 0, "mask_iou_at_mean": 0.05}
    with (image_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_row))
        writer.writeheader()
        writer.writerow({k: step0.get(k, "") for k in final_row})
        writer.writerow(final_row)
    if complete:
        (image_dir / "complete.json").write_text(
            json.dumps(
                {"img_id": img_id, "label": label, "prompt": f"Photo of {label}"}
            )
        )
    return image_dir


def _baseline(**overrides: object) -> dict[str, object]:
    base = {
        "mask_iou_at_mean": 0.05,
        "preserved_rank": 1,
        "preserved_prob": 0.90,
        "pert_ssim": 1.0,
        "top_k": [{"rank": 1, "label": "cat", "probability": 0.9}],
    }
    base.update(overrides)
    return base


def _final_row(**overrides: object) -> dict[str, object]:
    row = {
        "step": 2000,
        "mask_iou_at_mean": 0.94,
        "preserved_rank": 1,
        "preserved_prob": 0.88,
        "pert_ssim": 0.98,
        "pert_psnr": 41.0,
        "pert_linf": 0.078,
        "pert_l2_rms": 0.006,
        "cos_preserve": 0.984,
        "anchor_tvd": 0.02,
        "anchor_kl_T1": 0.0013,
        "top_1_label": "cat",
    }
    row.update(overrides)
    return row


def test_load_image_record_combines_baseline_final_and_metadata(tmp_path: Path) -> None:
    image_dir = _write_image_dir(
        tmp_path, "123", label="cat", baseline=_baseline(), final_row=_final_row()
    )

    record = harvest.load_image_record(image_dir)

    assert record is not None
    assert record["img_id"] == "123"
    assert record["label"] == "cat"
    assert record["baseline_mask_iou"] == 0.05
    assert record["final_mask_iou"] == 0.94  # last csv row, not the step-0 row
    assert record["baseline_preserved_rank"] == 1
    assert record["final_preserved_rank"] == 1
    assert record["final_pert_ssim"] == 0.98
    assert record["final_anchor_tvd"] == 0.02
    assert record["top1_preserved"] is True


def test_load_image_record_flags_top1_class_flip(tmp_path: Path) -> None:
    image_dir = _write_image_dir(
        tmp_path,
        "9",
        label="cat",
        baseline=_baseline(top_k=[{"rank": 1, "label": "cat", "probability": 0.9}]),
        final_row=_final_row(top_1_label="dog", preserved_rank=4),
    )

    record = harvest.load_image_record(image_dir)

    assert record is not None
    assert record["top1_preserved"] is False
    assert record["final_preserved_rank"] == 4


def test_load_image_record_returns_none_without_complete_marker(tmp_path: Path) -> None:
    image_dir = _write_image_dir(
        tmp_path,
        "x",
        label="cat",
        baseline=_baseline(),
        final_row=_final_row(),
        complete=False,
    )

    assert harvest.load_image_record(image_dir) is None


def test_harvest_records_collects_only_completed_dirs(tmp_path: Path) -> None:
    _write_image_dir(
        tmp_path, "a", label="cat", baseline=_baseline(), final_row=_final_row()
    )
    _write_image_dir(
        tmp_path, "b", label="dog", baseline=_baseline(), final_row=_final_row()
    )
    _write_image_dir(
        tmp_path,
        "c",
        label="fox",
        baseline=_baseline(),
        final_row=_final_row(),
        complete=False,
    )

    records = harvest.harvest_records(tmp_path)

    assert sorted(r["img_id"] for r in records) == ["a", "b"]


def test_summarize_reports_median_iqr_and_mean() -> None:
    records = [{"final_mask_iou": v} for v in (0.1, 0.2, 0.3, 0.4)]

    summary = harvest.summarize(records, ["final_mask_iou"])

    stats = summary["final_mask_iou"]
    assert stats["n"] == 4
    assert math.isclose(stats["mean"], 0.25)
    assert math.isclose(stats["median"], 0.25)
    # IQR = Q3 - Q1 (linear interpolation): Q1=0.175, Q3=0.325
    assert math.isclose(stats["iqr"], 0.15, rel_tol=1e-6)


def test_summarize_skips_non_finite_values() -> None:
    records = [
        {"final_pert_psnr": 40.0},
        {"final_pert_psnr": float("inf")},
        {"final_pert_psnr": 42.0},
    ]

    stats = harvest.summarize(records, ["final_pert_psnr"])["final_pert_psnr"]

    assert stats["n"] == 2
    assert math.isclose(stats["mean"], 41.0)


def test_prediction_preservation_rates() -> None:
    records = [
        {"top1_preserved": True, "final_preserved_rank": 1},
        {"top1_preserved": True, "final_preserved_rank": 1},
        {"top1_preserved": False, "final_preserved_rank": 3},
        {"top1_preserved": True, "final_preserved_rank": 2},
    ]

    rates = harvest.prediction_preservation(records)

    assert math.isclose(rates["top1_preserved_rate"], 0.75)
    assert math.isclose(rates["rank1_preserved_rate"], 0.5)
    assert rates["n"] == 4
