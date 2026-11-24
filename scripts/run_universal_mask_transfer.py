#!/usr/bin/env python3
"""Universal / transferable explanation-mask experiment.

Hypothesis: is there a *single* perturbation that forces an arbitrary image's
Grad-ECLIP explanation to a fixed target (the "2+2=5" glyph)?

For each random seed we carve a fixed held-out test set, then for each ``K`` we
train **one shared** ``delta`` on the first ``K`` training images and apply it,
frozen, to both the training images (in-sample fit) and the held-out test images
(transfer). Sweeping ``K`` and averaging over seeds produces metric-vs-K curves
that show whether transfer improves with more training images.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402

from explainer.io import load_image, render_heatmap_overlay  # noqa: E402
from explainer.manipulation import (  # noqa: E402
    DistributionAnchor,
    ManipulationConfig,
    UniversalManipulationEngine,
    apply_universal_delta,
    average_metrics,
)
from explainer.metaclip2 import (  # noqa: E402
    MetaClip2ExplanationPipeline,
    MetaClip2RunConfig,
)
from scripts.run_hubert_2plus2eq5_manipulation import (  # noqa: E402
    DEFAULT_LABELS_CSV,
    PRESERVED_PROMPT,
    build_centered_target_mask,
    build_distribution_anchor,
    discover_image_paths,
    fetch_imagenet_classes,
    load_image_labels,
    order_image_queue,
    resolve_image_prompt,
)

DEFAULT_INPUT_DIR = Path("data/pexels-1k/images")
DEFAULT_OUTPUT_DIR = Path("outputs/universal_mask_transfer")
DEFAULT_K_VALUES = "1,2,4,8,16,32,64"
# Loss weights proven by the per-image "manipulation3" ablation;
# kept identical to run_hubert_2plus2eq5_manipulation.build_manipulation_config.
DEFAULT_WEIGHTS: dict[str, float] = {
    "w_mask_l1": 1.5,
    "w_mask_l2": 3.0,
    "w_mask_focal_fg": 3.0,
    "w_perturb_l1": 50.0,
    "w_perturb_l2": 10.0,
    "w_perturb_linf": 50.0,
    "w_pred_cos": 1.0,
    "w_dist_kl": 1.0,
}
PLOT_METRICS: tuple[tuple[str, str, str], ...] = (
    ("mask_iou_at_mean", "transfer_iou_vs_k.png", "Explanation IoU @ mean"),
    ("mask_l1", "mask_l1_vs_k.png", "Explanation L1 error"),
)
SPLITS = ("train", "transfer")


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without the model)
# --------------------------------------------------------------------------- #
def parse_k_values(text: str) -> list[int]:
    """Parse a comma-separated ``K`` list into sorted, unique, positive ints."""
    values: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError as error:
            raise ValueError(
                f"Invalid K value {chunk!r}; expected an integer."
            ) from error
        if value <= 0:
            raise ValueError(f"K values must be positive; got {value}.")
        values.append(value)
    if not values:
        raise ValueError("No K values parsed.")
    return sorted(set(values))


def make_split(
    paths: Sequence[Path], *, seed_key: str, test_size: int
) -> tuple[list[Path], list[Path]]:
    """Deterministically split a pool into (fixed test set, train pool).

    The shuffle is seeded by ``seed_key`` (same SHA256 scheme as the batch
    driver), so the split is reproducible. The first ``test_size`` images form the
    held-out test set; the remainder is the train pool, from which ``K`` images
    are taken in order — making the ``K`` training sets nested.
    """
    if test_size <= 0:
        raise ValueError("test_size must be positive.")
    ordered = order_image_queue(list(paths), seed_key)
    if len(ordered) <= test_size:
        raise ValueError(
            f"Need more than test_size={test_size} images; got {len(ordered)}."
        )
    return ordered[:test_size], ordered[test_size:]


def aggregate_records(records: Sequence[dict]) -> list[dict]:
    """Reduce long-form per-seed records to mean/std per (k, split, metric).

    ``k`` is kept as-is (not cast to int): the universal sweep passes integer K
    values, but ``run_adversarial_generator.cmd_eval`` passes the label ``"G"``
    for the amortized generator. It is only ever a grouping/sort key, never used
    arithmetically, so both numeric and string ``k`` group correctly.
    """
    grouped: dict[tuple[object, str, str], list[float]] = {}
    for record in records:
        k = record["k"]
        split = str(record["split"])
        for key, value in record.items():
            if key in ("k", "split", "seed"):
                continue
            grouped.setdefault((k, split, key), []).append(float(value))
    rows: list[dict] = []
    for (k, split, metric), values in sorted(grouped.items()):
        rows.append(
            {
                "k": k,
                "split": split,
                "metric": metric,
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        )
    return rows


def build_per_image_rows(
    *,
    seed: int,
    k: int,
    split: str,
    paths: Sequence[Path],
    metrics_list: Sequence[dict[str, float]],
) -> list[dict]:
    """One row per image: (seed, k, split, img_id) + every metric.

    This is the full distribution behind the averaged ``results.csv`` rows, so
    the report can compute medians, IQR, per-image error bars, and significance
    tests over the meaningful ``n`` (images), not just the 3 seeds.
    """
    if len(paths) != len(metrics_list):
        raise ValueError("paths and metrics_list must have the same length.")
    rows: list[dict] = []
    for path, metrics in zip(paths, metrics_list, strict=True):
        rows.append(
            {"seed": seed, "k": k, "split": split, "img_id": path.stem, **metrics}
        )
    return rows


def build_trace_rows(
    *, seed: int, k: int, trace: Sequence[dict[str, float]]
) -> list[dict]:
    """Tag a run's optimization trace with (seed, k) for long-form persistence.

    Each trace entry already carries ``step`` plus the loss components, so the
    report can plot convergence curves per run.
    """
    return [{"seed": seed, "k": k, **row} for row in trace]


def build_captioned_pool(
    paths: Sequence[Path],
    label_lookup: dict[str, str],
    *,
    template: str,
    default_prompt: str,
) -> tuple[list[Path], dict[str, str]]:
    """Keep only images with a resolvable caption; return (paths, path->caption)."""
    kept: list[Path] = []
    captions: dict[str, str] = {}
    for path in paths:
        caption = resolve_image_prompt(
            path, label_lookup, template=template, default_prompt=default_prompt
        )
        if caption is None:
            continue
        kept.append(path)
        captions[str(path)] = caption
    return kept, captions


# --------------------------------------------------------------------------- #
# Model-dependent helpers
# --------------------------------------------------------------------------- #
def load_canonical_image(path: Path, size: int) -> torch.Tensor:
    """Load an image and resize to a square ``size`` so a shared delta applies."""
    pil = load_image(path).resize((size, size))
    array = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CanonicalImageCache:
    """Lazily load + cache canonical CPU tensors keyed by path."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._cache: dict[str, torch.Tensor] = {}

    def get(self, path: Path) -> torch.Tensor:
        key = str(path)
        if key not in self._cache:
            self._cache[key] = load_canonical_image(path, self._size)
        return self._cache[key]

    def many(self, paths: Sequence[Path]) -> list[torch.Tensor]:
        return [self.get(path) for path in paths]


def build_config(args: argparse.Namespace) -> ManipulationConfig:
    return ManipulationConfig(
        steps=args.steps,
        lr=args.lr,
        max_epsilon=args.max_epsilon,
        log_every=args.eval_every,
        seed=args.seed_base,
        w_mask_l1=args.w_mask_l1,
        w_mask_l2=args.w_mask_l2,
        w_mask_focal_fg=args.w_mask_focal_fg,
        w_perturb_l1=args.w_perturb_l1,
        w_perturb_l2=args.w_perturb_l2,
        w_perturb_linf=args.w_perturb_linf,
        w_pred_cos=args.w_pred_cos,
        w_dist_kl=args.w_dist_kl,
        dist_kl_temperature=args.dist_kl_temperature,
    )


# --------------------------------------------------------------------------- #
# Plotting / IO
# --------------------------------------------------------------------------- #
def _series(agg: Sequence[dict], split: str, metric: str) -> tuple[list, list, list]:
    rows = [r for r in agg if r["split"] == split and r["metric"] == metric]
    rows.sort(key=lambda r: r["k"])
    return (
        [r["k"] for r in rows],
        [r["mean"] for r in rows],
        [r["std"] for r in rows],
    )


def plot_metric_vs_k(
    agg: Sequence[dict], metric: str, *, ylabel: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for split in SPLITS:
        ks, means, stds = _series(agg, split, metric)
        if not ks:
            continue
        ax.errorbar(
            ks, means, yerr=stds, marker="o", capsize=4, label=split, linewidth=2
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K (training images for the shared mask)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs K (mean ± std over seeds)")
    ax.legend(title="split")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_perturbation_vs_k(agg: Sequence[dict], out_path: Path) -> None:
    fig, ax_linf = plt.subplots(figsize=(9, 6))
    ks, linf_means, linf_stds = _series(agg, "transfer", "pert_linf")
    ax_linf.errorbar(
        ks,
        linf_means,
        yerr=linf_stds,
        marker="o",
        capsize=4,
        color="#d62728",
        label="pert_linf",
        linewidth=2,
    )
    ax_linf.set_xscale("log", base=2)
    ax_linf.set_xlabel("K (training images for the shared mask)")
    ax_linf.set_ylabel("L-inf perturbation", color="#d62728")
    ax_ssim = ax_linf.twinx()
    ks2, ssim_means, ssim_stds = _series(agg, "transfer", "pert_ssim")
    ax_ssim.errorbar(
        ks2,
        ssim_means,
        yerr=ssim_stds,
        marker="s",
        capsize=4,
        color="#1f77b4",
        label="pert_ssim",
        linewidth=2,
    )
    ax_ssim.set_ylabel("SSIM (image fidelity)", color="#1f77b4")
    ax_linf.set_title("Transfer-set perturbation cost vs K")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_long_csv(
    rows: Sequence[dict], path: Path, *, id_columns: Sequence[str]
) -> None:
    metric_keys = sorted({k for r in rows for k in r if k not in id_columns})
    header = [*id_columns, *metric_keys]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row.get(col, "")) for col in header))
    path.write_text("\n".join(lines) + "\n")


def _load_rows(path: Path) -> list[dict]:
    """Load a previously written long-form JSON row list, or [] if absent."""
    if not path.exists():
        return []
    return list(json.loads(path.read_text()))


def _merge_rows(
    existing: Sequence[dict], new: Sequence[dict], *, key_columns: Sequence[str]
) -> list[dict]:
    """Merge row lists on ``key_columns``; ``new`` wins on a key collision.

    Lets a sweep be split across invocations into one output dir (e.g. K=1,2
    first, then K=4,8) without the second run clobbering the first: rows for
    untouched keys are preserved, and re-running a key overwrites it with fresh
    numbers. Sorted by key for stable, diff-friendly output.
    """
    merged: dict[tuple, dict] = {
        tuple(row[col] for col in key_columns): row for row in existing
    }
    for row in new:
        merged[tuple(row[col] for col in key_columns)] = row
    return [merged[key] for key in sorted(merged)]


def write_results(
    records: Sequence[dict],
    per_image_rows: Sequence[dict],
    trace_rows: Sequence[dict],
    output_dir: Path,
) -> None:
    # Merge with anything already on disk so a sweep split across invocations
    # (or resumed after an interruption) accumulates instead of overwriting.
    records = _merge_rows(
        _load_rows(output_dir / "results.json"),
        records,
        key_columns=("seed", "k", "split"),
    )
    per_image_rows = _merge_rows(
        _load_rows(output_dir / "per_image_results.json"),
        per_image_rows,
        key_columns=("seed", "k", "split", "img_id"),
    )
    trace_rows = _merge_rows(
        _load_rows(output_dir / "traces.json"),
        trace_rows,
        key_columns=("seed", "k", "step"),
    )

    # Averaged-over-eval-set rows (one per seed/K/split).
    _write_long_csv(
        records, output_dir / "results.csv", id_columns=("seed", "k", "split")
    )
    (output_dir / "results.json").write_text(json.dumps(list(records), indent=2))
    agg = aggregate_records(records)
    (output_dir / "results_summary.json").write_text(json.dumps(agg, indent=2))
    # Full per-image distribution behind those averages.
    _write_long_csv(
        per_image_rows,
        output_dir / "per_image_results.csv",
        id_columns=("seed", "k", "split", "img_id"),
    )
    (output_dir / "per_image_results.json").write_text(
        json.dumps(list(per_image_rows), indent=2)
    )
    # Per-(seed, K) optimization traces (loss curves over training steps).
    _write_long_csv(
        trace_rows, output_dir / "traces.csv", id_columns=("seed", "k", "step")
    )
    (output_dir / "traces.json").write_text(json.dumps(list(trace_rows), indent=2))


def save_qualitative_example(
    *,
    engine: UniversalManipulationEngine,
    delta: torch.Tensor,
    budget: torch.Tensor,
    image: torch.Tensor,
    caption: str,
    target: torch.Tensor,
    output_dir: Path,
    tag: str,
) -> None:
    device = engine.pipeline.device
    perturbed = apply_universal_delta(
        delta.to(device), budget.to(device), image.to(device)
    )
    run = engine.pipeline.run(perturbed, caption)
    overlay = render_heatmap_overlay(
        perturbed.detach().cpu(), run.explanation.detach().cpu(), alpha=0.45
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("source")
    axes[1].imshow(overlay.permute(1, 2, 0).numpy())
    axes[1].set_title("shared-mask explanation")
    axes[2].imshow(target.detach().cpu().numpy(), cmap="gray")
    axes[2].set_title("2+2=5 target")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"example_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    np.save(
        output_dir / f"delta_{tag}.npy",
        delta.detach().cpu().numpy().astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--resized-image-size", type=int, default=512)

    parser.add_argument("--k-values", default=DEFAULT_K_VALUES)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Optimizer updates; each is a full pass over all K training images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Gradient-accumulation micro-batch: image graphs summed per backward "
            "within each full pass. Memory knob only (<= this many second-order "
            "graphs alive at once); every step still covers all K images."
        ),
    )
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100)

    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--max-epsilon", type=float, default=30 / 255)
    for name, value in DEFAULT_WEIGHTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value)
    parser.add_argument("--dist-kl-temperature", type=float, default=2.0)
    parser.add_argument(
        "--anchor-stride",
        type=int,
        default=5,
        help="ImageNet class stride for the distribution anchor (hubert uses 5).",
    )

    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--id-column", default="img_id")
    parser.add_argument("--label-field", default="title")
    parser.add_argument("--label-template", default="Photo of {label}")

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build the split + baseline metrics on a couple images, no training.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sns.set_theme(style="whitegrid", context="talk")

    k_values = parse_k_values(args.k_values)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    label_lookup: dict[str, str] = {}
    if args.labels_csv and Path(args.labels_csv).exists():
        label_lookup = load_image_labels(
            Path(args.labels_csv),
            id_column=args.id_column,
            label_field=args.label_field,
        )

    all_paths = discover_image_paths(args.input_dir)
    pool, captions = build_captioned_pool(
        all_paths,
        label_lookup,
        template=args.label_template,
        default_prompt=PRESERVED_PROMPT,
    )
    print(f"Captioned image pool: {len(pool)} / {len(all_paths)} images")

    run_config = MetaClip2RunConfig(
        device=args.device,
        resized_image_size=args.resized_image_size,
        **({"model_name": args.model_name} if args.model_name else {}),
    )
    pipeline = MetaClip2ExplanationPipeline(run_config)
    engine = UniversalManipulationEngine(pipeline)
    cache = CanonicalImageCache(args.resized_image_size)
    config = build_config(args)

    # Determine the explanation-grid shape once, then build the glyph target.
    sample_run = pipeline.run(
        cache.get(pool[0]).to(pipeline.device), captions[str(pool[0])]
    )
    grid_shape = tuple(int(dim) for dim in sample_run.explanation.shape)
    target_mask = build_centered_target_mask(*grid_shape)
    print(f"Explanation grid: {grid_shape}; target glyph built.")

    # Distribution anchor for the KL preservation term (parity with hubert).
    distribution_anchor: DistributionAnchor | None = None
    if config.w_dist_kl > 0:
        imagenet_classes = fetch_imagenet_classes()
        distribution_anchor = build_distribution_anchor(
            pipeline, imagenet_classes, args.anchor_stride
        )
        (output_dir / "anchor_labels.json").write_text(
            json.dumps(list(distribution_anchor.labels), indent=2)
        )
        print(f"Distribution anchor: {len(distribution_anchor.labels)} labels")

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "args": {k: str(v) for k, v in vars(args).items()},
                "manipulation_config": asdict(config),
                "k_values": k_values,
                "grid_shape": list(grid_shape),
                "pool_size": len(pool),
            },
            indent=2,
        )
    )

    if args.validate_only:
        _run_validation(
            engine,
            cache,
            pool,
            captions,
            target_mask,
            config,
            output_dir,
            args,
            distribution_anchor,
        )
        return

    records: list[dict] = []
    per_image_rows: list[dict] = []
    trace_rows: list[dict] = []
    for seed_index in range(args.num_seeds):
        seed_key = f"{args.seed_base}:{seed_index}"
        test_paths, train_pool = make_split(
            pool, seed_key=seed_key, test_size=args.test_size
        )
        max_k = max(k_values)
        if len(train_pool) < max_k:
            raise ValueError(
                f"train pool {len(train_pool)} smaller than max K {max_k}; "
                "reduce --k-values or --test-size."
            )
        test_tensors = cache.many(test_paths)
        test_caps = [captions[str(p)] for p in test_paths]
        target_device = target_mask.to(pipeline.device)

        for k in k_values:
            train_paths = train_pool[:k]
            train_tensors = cache.many(train_paths)
            train_caps = [captions[str(p)] for p in train_paths]
            print(f"[seed {seed_index}] K={k}: training shared mask...")

            result = engine.train(
                train_tensors,
                train_caps,
                target_device,
                config,
                batch_size=args.batch_size,
                progress=True,
                distribution_anchor=distribution_anchor,
            )
            transfer_per_image = engine.evaluate_per_image(
                result.delta,
                result.perturbation_budget,
                test_tensors,
                test_caps,
                result.target_explanation,
                config,
                distribution_anchor=distribution_anchor,
            )
            transfer_metrics = average_metrics(transfer_per_image)
            records.append(
                {"seed": seed_index, "k": k, "split": "train", **result.train_metrics}
            )
            records.append(
                {"seed": seed_index, "k": k, "split": "transfer", **transfer_metrics}
            )
            per_image_rows.extend(
                build_per_image_rows(
                    seed=seed_index,
                    k=k,
                    split="train",
                    paths=train_paths,
                    metrics_list=result.train_per_image,
                )
            )
            per_image_rows.extend(
                build_per_image_rows(
                    seed=seed_index,
                    k=k,
                    split="transfer",
                    paths=test_paths,
                    metrics_list=transfer_per_image,
                )
            )
            trace_rows.extend(
                build_trace_rows(seed=seed_index, k=k, trace=result.trace)
            )
            print(
                f"[seed {seed_index}] K={k}: "
                f"train IoU={result.train_metrics['mask_iou_at_mean']:.3f} "
                f"transfer IoU={transfer_metrics['mask_iou_at_mean']:.3f} "
                f"transfer linf={transfer_metrics['pert_linf']:.3f}"
            )

            save_qualitative_example(
                engine=engine,
                delta=result.delta,
                budget=result.perturbation_budget,
                image=test_tensors[0],
                caption=test_caps[0],
                target=result.target_explanation,
                output_dir=output_dir,
                tag=f"seed{seed_index}_K{k}",
            )

            # Persist after every (seed, K) so an interrupted sweep keeps all
            # finished results. write_results rewrites the full accumulated set
            # each call, so the files always reflect everything completed so far.
            write_results(records, per_image_rows, trace_rows, output_dir)
            print(f"[seed {seed_index}] K={k}: results written to {output_dir}")

    # Aggregate over everything on disk (this run merged with any prior runs in
    # the same output dir) so plots span the full K range, not just this call.
    agg = aggregate_records(_load_rows(output_dir / "results.json"))
    for metric, filename, ylabel in PLOT_METRICS:
        plot_metric_vs_k(agg, metric, ylabel=ylabel, out_path=output_dir / filename)
    plot_perturbation_vs_k(agg, output_dir / "perturbation_vs_k.png")
    print(f"Done. Results + plots under {output_dir}")


def _run_validation(
    engine: UniversalManipulationEngine,
    cache: CanonicalImageCache,
    pool: Sequence[Path],
    captions: dict[str, str],
    target_mask: torch.Tensor,
    config: ManipulationConfig,
    output_dir: Path,
    args: argparse.Namespace,
    distribution_anchor: DistributionAnchor | None = None,
) -> None:
    test_paths, train_pool = make_split(
        pool,
        seed_key=f"{args.seed_base}:0",
        test_size=min(args.test_size, len(pool) - 1),
    )
    sample = train_pool[: min(2, len(train_pool))]
    delta = torch.zeros(3, args.resized_image_size, args.resized_image_size)
    budget = torch.full(
        (args.resized_image_size, args.resized_image_size),
        fill_value=args.max_epsilon,
    )
    metrics = engine.evaluate(
        delta,
        budget,
        cache.many(sample),
        [captions[str(p)] for p in sample],
        target_mask.to(engine.pipeline.device),
        config,
        distribution_anchor=distribution_anchor,
    )
    (output_dir / "validation.json").write_text(
        json.dumps(
            {
                "test_size": len(test_paths),
                "train_pool": len(train_pool),
                "baseline_metrics_zero_delta": metrics,
            },
            indent=2,
        )
    )
    print(f"Validation complete (zero-delta baseline): {metrics}")


if __name__ == "__main__":
    main()
