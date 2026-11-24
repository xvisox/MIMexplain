#!/usr/bin/env python3
"""Aggregate the per-image ``run_hubert_2plus2eq5_manipulation`` outputs.

The batch driver writes one directory per image, each holding:

- ``complete.json``       — run metadata (img_id, label, prompt)
- ``baseline_metrics.json`` — the step-0 (unperturbed) snapshot, including the
  Q2 fields ``preserved_rank`` / ``preserved_prob`` and the top-k ranking
- ``metrics.csv``         — the per-step optimization trace; the LAST row is the
  converged result and is the only place ``preserved_rank`` / ``anchor_tvd`` /
  ``anchor_kl_T1`` are recorded for the manipulated image
- ``final_metrics.json``  — the engine's converged metrics (mask/SSIM/PSNR/...)

``final_metrics.json`` does **not** carry the prediction-preservation fields, so
the Q2 story (does the prediction survive the manipulation?) can only be told by
pairing ``baseline_metrics.json`` (before) with the last ``metrics.csv`` row
(after). This module reads all three per image into one tidy record, then
reduces the 1000-image set to report-ready distributions, a prediction-drift
scatter, and (reusing the universal-sweep summary) the Q3 transfer-vs-K curve.

The pure data-reduction helpers (``load_image_record`` / ``harvest_records`` /
``summarize`` / ``prediction_preservation``) are unit-tested without the model;
plotting/IO is a thin shell around them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

# Metrics summarized for the main results table (key -> human label).
Q1_KEYS: dict[str, str] = {
    "final_mask_iou": "Explanation IoU @ mean (target glyph)",
    "final_pert_ssim": "SSIM (image fidelity)",
    "final_pert_psnr": "PSNR [dB]",
    "final_pert_linf": "L-inf perturbation",
}
Q2_KEYS: dict[str, str] = {
    "final_cos_preserve": "Embedding cosine (before vs after)",
    "final_anchor_tvd": "Prediction TVD (anchor vocab)",
    "final_anchor_kl_T1": "Prediction KL (anchor vocab)",
    "final_preserved_prob": "Preserved-prompt probability",
}


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without the model)
# --------------------------------------------------------------------------- #
def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def read_last_csv_row(path: Path) -> dict[str, str]:
    """Return the final data row of a CSV as a dict (empty if no data rows)."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def load_image_record(image_dir: Path) -> dict[str, object] | None:
    """Combine baseline + converged metrics + metadata for one image.

    Returns ``None`` for directories without a ``complete.json`` marker (a run
    that did not finish), so partial/locked runs are skipped, not half-counted.
    """
    complete_path = image_dir / "complete.json"
    baseline_path = image_dir / "baseline_metrics.json"
    metrics_path = image_dir / "metrics.csv"
    if not (
        complete_path.exists() and baseline_path.exists() and metrics_path.exists()
    ):
        return None

    meta = json.loads(complete_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    final = read_last_csv_row(metrics_path)
    if not final:
        return None

    baseline_top = baseline.get("top_k") or [{}]
    baseline_top1 = str(baseline_top[0].get("label", ""))
    final_top1 = str(final.get("top_1_label", ""))

    return {
        "img_id": str(meta.get("img_id", image_dir.name)),
        "label": meta.get("label"),
        "prompt": meta.get("prompt"),
        "baseline_mask_iou": _to_float(baseline.get("mask_iou_at_mean")),
        "final_mask_iou": _to_float(final.get("mask_iou_at_mean")),
        "baseline_preserved_rank": int(_to_float(baseline.get("preserved_rank"))),
        "final_preserved_rank": int(_to_float(final.get("preserved_rank"))),
        "baseline_preserved_prob": _to_float(baseline.get("preserved_prob")),
        "final_preserved_prob": _to_float(final.get("preserved_prob")),
        "final_pert_ssim": _to_float(final.get("pert_ssim")),
        "final_pert_psnr": _to_float(final.get("pert_psnr")),
        "final_pert_linf": _to_float(final.get("pert_linf")),
        "final_pert_l2_rms": _to_float(final.get("pert_l2_rms")),
        "final_cos_preserve": _to_float(final.get("cos_preserve")),
        "final_anchor_tvd": _to_float(final.get("anchor_tvd")),
        "final_anchor_kl_T1": _to_float(final.get("anchor_kl_T1")),
        "baseline_top1_label": baseline_top1,
        "final_top1_label": final_top1,
        "top1_preserved": bool(baseline_top1) and final_top1 == baseline_top1,
    }


def harvest_records(root: Path) -> list[dict[str, object]]:
    """Load every completed per-image record under ``root`` (sorted by id)."""
    records: list[dict[str, object]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        record = load_image_record(child)
        if record is not None:
            records.append(record)
    return records


def summarize(
    records: Sequence[dict[str, object]], keys: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Mean / median / IQR / min / max per key over finite values only."""
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [
            float(r[key])
            for r in records
            if key in r and math.isfinite(_to_float(r[key]))
        ]
        if not values:
            summary[key] = {"n": 0}
            continue
        quartiles = (
            statistics.quantiles(values, n=4, method="inclusive")
            if len(values) > 1
            else [values[0], values[0], values[0]]
        )
        summary[key] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "q1": quartiles[0],
            "q3": quartiles[2],
            "iqr": quartiles[2] - quartiles[0],
            "min": min(values),
            "max": max(values),
        }
    return summary


def prediction_preservation(
    records: Sequence[dict[str, object]],
) -> dict[str, float]:
    """Fraction of images whose prediction survives the explanation attack."""
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "top1_preserved_rate": math.nan,
            "rank1_preserved_rate": math.nan,
        }
    top1 = sum(1 for r in records if r.get("top1_preserved"))
    rank1 = sum(
        1 for r in records if int(_to_float(r.get("final_preserved_rank"))) == 1
    )
    return {
        "n": float(n),
        "top1_preserved_rate": top1 / n,
        "rank1_preserved_rate": rank1 / n,
    }


def load_universal_summary(path: Path) -> list[dict[str, object]]:
    """Load the universal-sweep ``results_summary.json`` (long-form rows)."""
    return list(json.loads(path.read_text()))


def universal_series(
    summary: Sequence[dict[str, object]], split: str, metric: str
) -> tuple[list[int], list[float], list[float]]:
    """(k, mean, std) for one split/metric, sorted by k."""
    rows = [r for r in summary if r.get("split") == split and r.get("metric") == metric]
    rows.sort(key=lambda r: int(_to_float(r.get("k"))))
    return (
        [int(_to_float(r["k"])) for r in rows],
        [float(_to_float(r["mean"])) for r in rows],
        [float(_to_float(r["std"])) for r in rows],
    )


# --------------------------------------------------------------------------- #
# Plotting / IO (thin shell; not unit-tested)
# --------------------------------------------------------------------------- #
def _write_records_csv(records: Sequence[dict[str, object]], path: Path) -> None:
    if not records:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _plot_q1_distributions(records, summary, baseline_iou, out_path, *, dpi=300) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    final_iou = [r["final_mask_iou"] for r in records]
    ssim = [r["final_pert_ssim"] for r in records]
    baseline_mean = statistics.fmean(baseline_iou)
    median_iou = statistics.median(final_iou)
    median_ssim = statistics.median(ssim)

    fig, (ax_iou, ax_ssim) = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(final_iou, bins=40, kde=True, color="#1f77b4", edgecolor="white", ax=ax_iou)
    ax_iou.axvline(
        baseline_mean,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"unperturbed baseline ({baseline_mean:.3f})",
    )
    ax_iou.axvline(
        median_iou,
        color="#111111",
        linestyle=":",
        linewidth=2,
        label=f"median ({median_iou:.3f})",
    )
    ax_iou.set_xlabel("Explanation IoU @ mean with target glyph")
    ax_iou.set_ylabel("images")
    ax_iou.set_title("Q1: targeted manipulation success")
    ax_iou.legend()

    sns.histplot(ssim, bins=40, kde=True, color="#2ca02c", edgecolor="white", ax=ax_ssim)
    ax_ssim.axvline(
        median_ssim,
        color="#111111",
        linestyle=":",
        linewidth=2,
        label=f"median ({median_ssim:.3f})",
    )
    ax_ssim.set_xlabel("SSIM (manipulated vs original image)")
    ax_ssim.set_ylabel("images")
    ax_ssim.set_title("Imperceptibility")
    ax_ssim.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_q2_scatter(records, out_path, *, dpi=300) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    iou = [r["final_mask_iou"] for r in records]
    drift = [r["final_anchor_tvd"] for r in records]
    groups = [
        "top-1 preserved" if r["top1_preserved"] else "top-1 flipped"
        for r in records
    ]
    counts = {
        "top-1 preserved": groups.count("top-1 preserved"),
        "top-1 flipped": groups.count("top-1 flipped"),
    }
    hue_order = ["top-1 preserved", "top-1 flipped"]
    palette = {"top-1 preserved": "#1f77b4", "top-1 flipped": "#d62728"}

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.scatterplot(
        x=iou,
        y=drift,
        hue=groups,
        hue_order=hue_order,
        palette=palette,
        s=22,
        alpha=0.55,
        edgecolor="none",
        ax=ax,
    )
    ax.set_xlabel("Manipulation success (explanation IoU with target)")
    ax.set_ylabel("Prediction drift (total variation, anchor vocab)")
    ax.set_title("Q2: explanation moves, prediction does not")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=[f"{label} ({counts.get(label, 0)})" for label in labels],
        title=None,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_q3_transfer(summary, baseline_iou, out_path, *, dpi=300) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 6))
    for split, color in (("train", "#1f77b4"), ("transfer", "#d62728")):
        ks, means, stds = universal_series(summary, split, "mask_iou_at_mean")
        if ks:
            ax.errorbar(
                ks,
                means,
                yerr=stds,
                marker="o",
                capsize=4,
                linewidth=2,
                color=color,
                label=f"{split} (in-sample fit)"
                if split == "train"
                else f"{split} (held-out)",
            )
    if baseline_iou:
        ax.axhline(
            statistics.fmean(baseline_iou),
            color="grey",
            linestyle=":",
            label=f"unperturbed chance ({statistics.fmean(baseline_iou):.3f})",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K (images sharing one universal perturbation)")
    ax.set_ylabel("Explanation IoU @ mean with target")
    ax.set_title("Q3: no single mask rules them all")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("mimuw_outputs/manipulation1"),
        help="Per-image output directory from run_hubert_2plus2eq5_manipulation.",
    )
    parser.add_argument(
        "--universal-summary",
        type=Path,
        default=Path("mimuw_outputs/universal_mask_transfer/results_summary.json"),
        help="results_summary.json from run_universal_mask_transfer (Q3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mimuw_outputs/harvest"),
        help="Where to write the aggregated CSV, summary JSON, and figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution (dots per inch) for the saved figures. Default: 300.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = harvest_records(args.input_dir)
    print(f"Loaded {len(records)} completed per-image records from {args.input_dir}")
    _write_records_csv(records, args.output_dir / "per_image.csv")

    summary = summarize(records, [*Q1_KEYS, *Q2_KEYS, "baseline_mask_iou"])
    preservation = prediction_preservation(records)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {"n": len(records), "summary": summary, "preservation": preservation},
            indent=2,
        )
    )

    print("\n=== Q1: manipulation success / imperceptibility (median [IQR]) ===")
    for key, label in Q1_KEYS.items():
        s = summary.get(key, {})
        if s.get("n"):
            print(
                f"  {label:40s} {s['median']:.4f}  [{s['q1']:.4f}, {s['q3']:.4f}]  (n={s['n']})"
            )
    print("\n=== Q2: prediction preservation ===")
    for key, label in Q2_KEYS.items():
        s = summary.get(key, {})
        if s.get("n"):
            print(f"  {label:40s} {s['median']:.4f}  [{s['q1']:.4f}, {s['q3']:.4f}]")
    print(
        f"  top-1 class preserved : {preservation['top1_preserved_rate'] * 100:.1f}% of {int(preservation['n'])} images"
    )
    print(
        f"  preserved-prompt rank==1 : {preservation['rank1_preserved_rate'] * 100:.1f}%"
    )

    baseline_iou = [
        float(r["baseline_mask_iou"])
        for r in records
        if math.isfinite(_to_float(r["baseline_mask_iou"]))
    ]
    if records:
        _plot_q1_distributions(
            records,
            summary,
            baseline_iou,
            args.output_dir / "q1_manipulation.png",
            dpi=args.dpi,
        )
        _plot_q2_scatter(
            records, args.output_dir / "q2_prediction_drift.png", dpi=args.dpi
        )

    if args.universal_summary.exists():
        universal = load_universal_summary(args.universal_summary)
        _plot_q3_transfer(
            universal,
            baseline_iou,
            args.output_dir / "q3_transfer_vs_k.png",
            dpi=args.dpi,
        )
        print(f"\nQ3 transfer curve written from {args.universal_summary}")
    else:
        print(
            f"\nQ3 summary not found at {args.universal_summary}; skipping transfer plot."
        )

    print(f"\nDone. Artifacts under {args.output_dir}")


if __name__ == "__main__":
    main()
