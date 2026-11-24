#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image, ImageDraw

from explainer.io import load_image, pil_to_tensor, render_heatmap_overlay
from explainer.manipulation import PerceptualMaskConfig, build_perceptual_allowance_map

DEFAULT_OUTPUT_DIR = Path("mimuw_outputs/manipulation1")
DEFAULT_ARTIFACT_FILENAMES = {
    "figure_path": "visualization_summary.png",
    "loss_figure_path": "visualization_losses.png",
    "explanation_gif_path": "visualization_explanation.gif",
    "image_gif_path": "visualization_images.gif",
    "perceptual_allowance_path": "perceptual_allowance.npy",
    "perceptual_allowance_overlay_path": "perceptual_allowance_overlay.png",
    "final_overlay_path": "final_overlay.png",
    "top5_figure_path": "visualization_top5_evolution.png",
    "dumbbell_figure_path": "visualization_distribution_dumbbell.png",
    "scatter_figure_path": "visualization_distribution_scatter.png",
}
EXPECTED_STEPS = [0, 200, 400, 600, 800, 1000]
GIF_TOTAL_DURATION_MS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single summary visualization for the Hubert 2+2=5 manipulation "
            "artifacts using reconstructed full-size checkpoint images."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--figure-path", type=Path)
    parser.add_argument("--loss-figure-path", type=Path)
    parser.add_argument("--explanation-gif-path", type=Path)
    parser.add_argument("--image-gif-path", type=Path)
    parser.add_argument("--perceptual-allowance-path", type=Path)
    parser.add_argument("--perceptual-allowance-overlay-path", type=Path)
    parser.add_argument("--final-overlay-path", type=Path)
    parser.add_argument("--top5-figure-path", type=Path)
    parser.add_argument("--dumbbell-figure-path", type=Path)
    parser.add_argument("--scatter-figure-path", type=Path)

    args = parser.parse_args()
    for arg_name, filename in DEFAULT_ARTIFACT_FILENAMES.items():
        if getattr(args, arg_name) is None:
            setattr(args, arg_name, args.output_dir / filename)
    return args


def load_metrics_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_source_image(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing source image: {path}")
    return pil_to_tensor(load_image(path))


def load_tensor_array(path_str: str) -> torch.Tensor:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return torch.from_numpy(np.load(path)).float()


def reconstruct_image(original_image: torch.Tensor, delta_path: str) -> torch.Tensor:
    delta = load_tensor_array(delta_path)
    if delta.shape != original_image.shape:
        raise ValueError(
            f"Delta shape {tuple(delta.shape)} does not match image shape "
            f"{tuple(original_image.shape)}."
        )
    return torch.clamp(original_image + delta, 0.0, 1.0)


def build_checkpoint_records(
    rows: list[dict[str, str]],
    original_image: torch.Tensor,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in rows:
        step = int(float(row["step"]))
        explanation_path = row.get("explanation_path", "").replace("outputs/", "mimuw_outputs/")
        if not explanation_path:
            raise ValueError(f"Missing explanation_path for step {step}.")
        explanation = load_tensor_array(explanation_path)
        delta_path = row.get("delta_path", "").strip().replace("outputs/", "mimuw_outputs/")
        if step > 0 and not delta_path:
            raise ValueError(f"Missing delta_path for non-baseline step {step}.")
        image = (
            original_image.clone()
            if not delta_path
            else reconstruct_image(original_image, delta_path)
        )
        overlay = render_heatmap_overlay(image, explanation, alpha=0.35)
        records.append(
            {
                "step": step,
                "loss": float(row["loss"]),
                "mask_loss": float(row["mask_loss"]),
                "perturb_loss": float(row["perturb_loss"]),
                "pred_loss": float(row["pred_loss"]),
                "mask_l1": float(row["mask_l1"]),
                "mask_iou_at_mean": float(row["mask_iou_at_mean"]),
                "pert_linf": float(row["pert_linf"]),
                "cos_preserve": float(row["cos_preserve"]),
                "sim_delta": float(row["sim_delta"]),
                "preserved_rank": float(row["preserved_rank"]),
                "preserved_prob": float(row["preserved_prob"]),
                "top_1_label": row.get("top_1_label", ""),
                "image": image,
                "explanation": explanation,
                "overlay": overlay,
            }
        )
    return sorted(records, key=lambda record: int(record["step"]))


def report_missing_steps(records: list[dict[str, object]]) -> list[int]:
    present = {int(record["step"]) for record in records}
    return [step for step in EXPECTED_STEPS if step not in present]


def records_for_figures(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [record for record in records if int(record["step"]) != 0]


def render_summary_figure(
    *,
    target_mask: torch.Tensor,
    records: list[dict[str, object]],
    output_path: Path,
) -> Path:
    if not records:
        raise ValueError("No non-baseline checkpoints available for summary figure.")
    steps = [int(record["step"]) for record in records]

    sns.set_theme(style="whitegrid", context="talk")

    fig = plt.figure(figsize=(24, 8))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0])

    ax_target = fig.add_subplot(grid[0, 0])
    ax_target.imshow(target_mask.numpy(), cmap="gray")
    ax_target.set_title("Target Mask")
    ax_target.axis("off")

    metric_axes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]
    metric_specs = [
        ("mask_l1", "Mask L1"),
        ("mask_iou_at_mean", "Mask IoU @ Mean"),
        ("pert_linf", "Perturbation Linf"),
        ("cos_preserve", "Cosine Preserve"),
        ("sim_delta", "Similarity Delta"),
        ("preserved_prob", "Preserved Prompt Prob"),
    ]
    for axis, (key, title) in zip(metric_axes, metric_specs):
        sns.lineplot(
            x=steps,
            y=[float(record[key]) for record in records],
            marker="o",
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel("")

    ax_rank = fig.add_subplot(grid[1, 3])
    sns.lineplot(
        x=steps,
        y=[float(record["preserved_rank"]) for record in records],
        marker="o",
        ax=ax_rank,
    )
    ax_rank.set_title("Preserved Prompt Rank")
    ax_rank.set_xlabel("Step")
    ax_rank.invert_yaxis()
    ax_rank.set_ylabel("Rank")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_loss_figure(
    *,
    records: list[dict[str, object]],
    output_path: Path,
) -> Path:
    if not records:
        raise ValueError("No non-baseline checkpoints available for loss figure.")
    steps = [int(record["step"]) for record in records]
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    loss_specs = [
        ("loss", "Total Loss"),
        ("mask_loss", "Mask Loss"),
        ("perturb_loss", "Perturbation Loss"),
        ("pred_loss", "Prediction Loss"),
    ]
    for key, label in loss_specs:
        sns.lineplot(
            x=steps,
            y=[float(record[key]) for record in records],
            marker="o",
            label=label,
            ax=ax,
        )
    ax.set_title("Optimization Losses")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_perceptual_allowance_artifacts(
    *,
    original_image: torch.Tensor,
    output_npy_path: Path,
    output_overlay_path: Path,
) -> tuple[Path, Path]:
    allowance = build_perceptual_allowance_map(
        original_image,
        output_shape=tuple(original_image.shape[-2:]),
        config=PerceptualMaskConfig(budget_max_scale=2.5),
        device=original_image.device,
        dtype=original_image.dtype,
    )
    output_npy_path.parent.mkdir(parents=True, exist_ok=True)
    output_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy_path, allowance.detach().cpu().numpy().astype(np.float32))
    overlay = render_heatmap_overlay(
        original_image.detach().cpu(),
        allowance.detach().cpu(),
        alpha=0.45,
    )
    Image.fromarray(_tensor_to_uint8_image(overlay)).save(output_overlay_path)
    return output_npy_path, output_overlay_path


def _tensor_to_uint8_image(image: torch.Tensor) -> np.ndarray:
    image_hwc = image.detach().cpu().permute(1, 2, 0).numpy()
    image_hwc = np.clip(image_hwc, 0.0, 1.0)
    return (image_hwc * 255).astype(np.uint8)


def _annotate_frame(frame: np.ndarray, step: int) -> Image.Image:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    text = f"step {step}"
    x = 18
    y = 18
    for dx, dy in [
        (-2, 0),
        (2, 0),
        (0, -2),
        (0, 2),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]:
        draw.text((x + dx, y + dy), text, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255))
    return image


def _build_image_comparison_frame(
    original_image: torch.Tensor,
    current_image: torch.Tensor,
) -> np.ndarray:
    delta = (current_image - original_image).abs().mean(dim=0)
    normalized = delta / delta.max().clamp_min(1e-8)
    base = original_image.detach().cpu().permute(1, 2, 0).numpy()
    base = np.clip(base, 0.0, 1.0)
    heatmap = plt.get_cmap("turbo")(normalized.detach().cpu().numpy())[..., :3]
    heatmap_overlay = np.clip((0.7 * base) + (0.8 * heatmap), 0.0, 1.0)

    modified_uint8 = _tensor_to_uint8_image(current_image)
    heatmap_uint8 = (heatmap_overlay * 255).astype(np.uint8)
    separator = np.full(
        (modified_uint8.shape[0], 12, modified_uint8.shape[2]),
        255,
        dtype=np.uint8,
    )
    return np.concatenate([modified_uint8, separator, heatmap_uint8], axis=1)


def write_gif(
    *,
    records: list[dict[str, object]],
    image_key: str,
    output_path: Path,
    original_image: torch.Tensor | None = None,
    final_frame_path: Path | None = None,
) -> Path:
    if image_key == "image":
        if original_image is None:
            raise ValueError("original_image is required for image comparison GIFs.")
        frames = [
            _annotate_frame(
                _build_image_comparison_frame(original_image, record["image"]),
                int(record["step"]),
            )
            for record in records
        ]
    else:
        frames = [
            _annotate_frame(_tensor_to_uint8_image(record[image_key]), int(record["step"]))
            for record in records
        ]
    duration_ms = max(20, round(GIF_TOTAL_DURATION_MS / max(len(frames), 1)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    if final_frame_path is not None:
        final_frame_path.parent.mkdir(parents=True, exist_ok=True)
        frames[-1].save(final_frame_path)
    return output_path


def _shorten_label(label: str, max_chars: int = 38) -> str:
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 1] + "…"


def load_anchor_artifacts(
    output_dir: Path,
    rows: list[dict[str, str]],
) -> tuple[list[str], np.ndarray, list[int], np.ndarray] | None:
    labels_path = output_dir / "anchor_labels.json"
    baseline_path = output_dir / "anchor_baseline_probs.npy"
    if not labels_path.exists() or not baseline_path.exists():
        return None
    anchor_labels: list[str] = json.loads(labels_path.read_text())
    baseline_probs = np.load(baseline_path).astype(np.float32)

    step_paths: list[tuple[int, Path]] = []
    for row in rows:
        anchor_path_str = row.get("anchor_probs_path", "").strip().replace(
            "outputs/", "mimuw_outputs/"
        )
        if not anchor_path_str:
            continue
        anchor_path = Path(anchor_path_str)
        if not anchor_path.exists():
            continue
        step_paths.append((int(float(row["step"])), anchor_path))
    if not step_paths:
        return None
    step_paths.sort(key=lambda pair: pair[0])
    steps = [step for step, _ in step_paths]
    probs_matrix = np.stack(
        [np.load(path).astype(np.float32) for _, path in step_paths],
        axis=0,
    )
    return anchor_labels, baseline_probs, steps, probs_matrix


def render_top5_evolution_figure(
    *,
    anchor_labels: list[str],
    baseline_probs: np.ndarray,
    steps: list[int],
    probs_matrix: np.ndarray,
    output_path: Path,
) -> Path:
    top5_indices = np.argsort(-baseline_probs)[:5]
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = sns.color_palette("tab10", n_colors=5)
    for color, idx in zip(palette, top5_indices):
        ax.plot(
            steps,
            probs_matrix[:, idx],
            marker="o",
            color=color,
            label=_shorten_label(anchor_labels[idx]),
        )
    ax.set_title("Top-5 baseline-prompt probabilities over time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.legend(loc="best", fontsize="small", frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_distribution_divergences(
    baseline_probs: np.ndarray, final_probs: np.ndarray
) -> dict[str, float]:
    eps = 1e-12
    p = baseline_probs.astype(np.float64) + eps
    q = final_probs.astype(np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pq = float(np.sum(p * (np.log(p) - np.log(q))))
    kl_qp = float(np.sum(q * (np.log(q) - np.log(p))))
    js = 0.5 * (
        float(np.sum(p * (np.log(p) - np.log(m))))
        + float(np.sum(q * (np.log(q) - np.log(m))))
    )
    tvd = 0.5 * float(np.abs(p - q).sum())
    return {
        "kl_baseline_to_final": kl_pq,
        "kl_final_to_baseline": kl_qp,
        "jensen_shannon": js,
        "total_variation": tvd,
    }


def render_distribution_dumbbell_figure(
    *,
    anchor_labels: list[str],
    baseline_probs: np.ndarray,
    final_probs: np.ndarray,
    output_path: Path,
    top_n: int = 15,
) -> Path:
    delta = final_probs - baseline_probs
    n_pick = int(min(top_n, delta.size))
    top_idx = np.argsort(-np.abs(delta))[:n_pick]
    top_idx = top_idx[np.argsort(-delta[top_idx])]
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, max(6.0, 0.45 * n_pick + 1.5)))
    y_pos = np.arange(n_pick)[::-1]
    for y, idx in zip(y_pos, top_idx):
        b = float(baseline_probs[idx])
        f = float(final_probs[idx])
        color = "#d62728" if f > b else "#1f77b4"
        ax.plot([b, f], [y, y], color=color, lw=2.5, alpha=0.85, zorder=1)
        ax.scatter(
            [b], [y], s=70, facecolor="white", edgecolor=color, lw=2, zorder=2
        )
        ax.scatter([f], [y], s=70, facecolor=color, edgecolor=color, zorder=3)
        ax.annotate(
            f"Δp={f - b:+.3f}",
            xy=(max(b, f), y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize="x-small",
            color=color,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_shorten_label(anchor_labels[idx]) for idx in top_idx])
    ax.set_xlabel("Probability")
    ax.set_title(f"Top-{n_pick} prompts by |Δ probability| (baseline → final)")
    legend_elements = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="white",
            markeredgecolor="#444", label="baseline", markersize=9, lw=0,
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#444",
            markeredgecolor="#444", label="final", markersize=9, lw=0,
        ),
        plt.Line2D([0], [0], color="#d62728", lw=2.5, label="prob ↑"),
        plt.Line2D([0], [0], color="#1f77b4", lw=2.5, label="prob ↓"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize="x-small")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_distribution_scatter_figure(
    *,
    anchor_labels: list[str],
    baseline_probs: np.ndarray,
    final_probs: np.ndarray,
    output_path: Path,
    annotate_top: int = 8,
) -> Path:
    eps = 1e-8
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        baseline_probs + eps,
        final_probs + eps,
        s=22,
        alpha=0.55,
        color="#1f77b4",
        edgecolor="none",
    )
    lo = float(min(baseline_probs.min(), final_probs.min()) + eps)
    hi = float(max(baseline_probs.max(), final_probs.max()) + eps)
    ax.plot([lo, hi], [lo, hi], color="#999", linestyle="--", lw=1, label="y = x")

    delta = final_probs - baseline_probs
    annotate_idx = np.argsort(-np.abs(delta))[:annotate_top]
    for idx in annotate_idx:
        ax.annotate(
            _shorten_label(anchor_labels[idx], max_chars=24),
            xy=(baseline_probs[idx] + eps, final_probs[idx] + eps),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize="xx-small",
            color="#444",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Baseline probability (log)")
    ax.set_ylabel("Final probability (log)")
    ax.set_title("Per-prompt baseline vs. final probability")

    divergences = compute_distribution_divergences(baseline_probs, final_probs)
    info_text = (
        f"KL(base || final) = {divergences['kl_baseline_to_final']:.4f}\n"
        f"KL(final || base) = {divergences['kl_final_to_baseline']:.4f}\n"
        f"JS divergence    = {divergences['jensen_shannon']:.4f}\n"
        f"Total variation  = {divergences['total_variation']:.4f}"
    )
    ax.text(
        0.03,
        0.97,
        info_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize="x-small",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    ax.legend(loc="lower right", fontsize="x-small")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()

    rows = load_metrics_rows(args.output_dir / "metrics.csv")
    if not rows:
        raise ValueError("metrics.csv is empty.")
    original_image = load_source_image(args.source_image)
    target_mask_path = args.output_dir / "target_mask.npy"
    if not target_mask_path.exists():
        raise FileNotFoundError(f"Missing target mask: {target_mask_path}")
    target_mask = torch.from_numpy(np.load(target_mask_path)).float()

    records = build_checkpoint_records(rows, original_image)
    figure_records = records_for_figures(records)
    missing_steps = report_missing_steps(records)
    allowance_npy_path, allowance_overlay_path = save_perceptual_allowance_artifacts(
        original_image=original_image,
        output_npy_path=args.perceptual_allowance_path,
        output_overlay_path=args.perceptual_allowance_overlay_path,
    )
    figure_path = render_summary_figure(
        target_mask=target_mask,
        records=figure_records,
        output_path=args.figure_path,
    )
    loss_figure_path = render_loss_figure(
        records=figure_records,
        output_path=args.loss_figure_path,
    )
    explanation_gif_path = write_gif(
        records=records,
        image_key="overlay",
        output_path=args.explanation_gif_path,
        final_frame_path=args.final_overlay_path,
    )
    image_gif_path = write_gif(
        records=records,
        image_key="image",
        output_path=args.image_gif_path,
        original_image=original_image,
    )

    print(f"summary_figure={figure_path}")
    print(f"loss_figure={loss_figure_path}")
    print(f"explanation_gif={explanation_gif_path}")
    print(f"final_overlay={args.final_overlay_path}")
    print(f"image_gif={image_gif_path}")
    print(f"perceptual_allowance={allowance_npy_path}")
    print(f"perceptual_allowance_overlay={allowance_overlay_path}")

    anchor_artifacts = load_anchor_artifacts(args.output_dir, rows)
    if anchor_artifacts is None:
        print("anchor_artifacts=missing")
    else:
        anchor_labels, baseline_probs, anchor_steps, probs_matrix = anchor_artifacts
        final_probs = probs_matrix[-1]
        top5_path = render_top5_evolution_figure(
            anchor_labels=anchor_labels,
            baseline_probs=baseline_probs,
            steps=anchor_steps,
            probs_matrix=probs_matrix,
            output_path=args.top5_figure_path,
        )
        dumbbell_path = render_distribution_dumbbell_figure(
            anchor_labels=anchor_labels,
            baseline_probs=baseline_probs,
            final_probs=final_probs,
            output_path=args.dumbbell_figure_path,
        )
        scatter_path = render_distribution_scatter_figure(
            anchor_labels=anchor_labels,
            baseline_probs=baseline_probs,
            final_probs=final_probs,
            output_path=args.scatter_figure_path,
        )
        divergences = compute_distribution_divergences(baseline_probs, final_probs)
        print(f"top5_evolution_figure={top5_path}")
        print(f"distribution_dumbbell_figure={dumbbell_path}")
        print(f"distribution_scatter_figure={scatter_path}")
        for name, value in divergences.items():
            print(f"distribution_{name}={value:.6f}")

    if missing_steps:
        print("missing_steps=" + ",".join(str(step) for step in missing_steps))


if __name__ == "__main__":
    main()
