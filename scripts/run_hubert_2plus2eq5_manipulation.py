#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F

from explainer.constants import DEFAULT_MODEL_NAME, DEFAULT_RESIZED_IMAGE_SIZE
from explainer.gradeclip import clip_encode_text_dense
from explainer.io import (
    load_image,
    pil_to_tensor,
    tensor_to_pil_image,
)
from explainer.manipulation import (
    DistributionAnchor,
    ManipulationConfig,
    ManipulationProgressSnapshot,
    ManipulationResult,
    MetaClip2ManipulationEngine,
    PerceptualMaskConfig,
)
from explainer.metaclip2 import (
    MetaClip2ExplanationPipeline,
    MetaClip2RunConfig,
    move_batch_to_device,
)

# Pexels dataset paths (repo-relative; resolved from the cwd, i.e. the repo root
# on both local and the synced remote checkout).
DEFAULT_INPUT_DIR = Path("data/pexels-1k/images")
DEFAULT_OUTPUT_DIR = Path("outputs/manipulation1")
# Fallback prompt used only when no dataset label is available for an image.
PRESERVED_PROMPT = "a photo of a person holding a creatine supplement package"
# Per-image labels source (Pexels labels.csv); matched by img_id == image stem.
DEFAULT_LABELS_CSV = Path("data/pexels-1k/labels.csv")
DEFAULT_ID_COLUMN = "img_id"
DEFAULT_LABEL_FIELD = "title"
DEFAULT_LABEL_TEMPLATE = "Photo of {label}"
DEFAULT_SAVE_EVERY = 100
DEFAULT_STEPS = 2000
DEFAULT_TOP_K = 10
DEFAULT_LOCK_TIMEOUT_HOURS = 1/6
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
IMAGENET_CLASSES_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)

CUSTOM_CLASSES = [
    "a photo of a person holding a pouch of creatine monohydrate",
    "a photo of a person holding a creatine supplement package",
    "a photo of a person holding sports nutrition supplements",
    "a photo of a person holding protein powder",
    "a photo of a person holding whey protein",
    "a photo of a person holding a pre-workout supplement",
    "a photo of a person holding BCAA powder",
    "a photo of a person holding amino acid supplement powder",
    "a photo of a person holding collagen powder",
    "a photo of a person holding an electrolyte powder",
    "a photo of a person holding a vitamin supplement",
    "a photo of a person holding a supplement pouch",
    "a photo of a person holding a supplement tub",
    "a photo of a person holding packaged nutrition products",
    "a photo of a person holding gym supplements",
    "a photo of a person holding powdered drink mix",
    "a photo of a person holding a recovery supplement",
    "a photo of a person holding a shaker bottle",
    "a photo of a person holding food packaging",
    "a photo of a person posing with fitness products",
]

GLYPHS: dict[str, list[str]] = {
    "2": ["111", "001", "111", "100", "111"],
    "+": ["010", "010", "111", "010", "010"],
    "=": ["000", "111", "000", "111", "000"],
    "5": ["111", "100", "111", "001", "111"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Hubert 2+2=5 Grad-ECLIP manipulation experiment for a "
            "flat folder of images."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Flat directory of images to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where experiment artifacts will be written.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device for MetaCLIP-2.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of optimization steps.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help="Save cadence for checkpoint snapshots.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="How many labels to store for each checkpoint ranking.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Write baseline metrics and exit without optimization.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="MetaCLIP-2 checkpoint name.",
    )
    parser.add_argument(
        "--lock-timeout-hours",
        type=float,
        default=DEFAULT_LOCK_TIMEOUT_HOURS,
        help="Reclaim per-image locks older than this many hours.",
    )
    parser.add_argument(
        "--queue-seed",
        help=(
            "Seed for image queue permutation. Defaults to this worker's "
            "hostname and process id."
        ),
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help=(
            "CSV of dataset labels (e.g. Pexels labels.csv) used as the per-image "
            "prompt and added to the prediction vocabulary. Rows are matched by "
            f"'{DEFAULT_ID_COLUMN}' == image filename stem. If the file is absent "
            "the run falls back to PRESERVED_PROMPT for every image."
        ),
    )
    parser.add_argument(
        "--id-column",
        default=DEFAULT_ID_COLUMN,
        help="Column in --labels-csv holding the image id (image filename stem).",
    )
    parser.add_argument(
        "--label-field",
        default=DEFAULT_LABEL_FIELD,
        help="Column in --labels-csv used as the image label (e.g. title, tags).",
    )
    parser.add_argument(
        "--label-template",
        default=DEFAULT_LABEL_TEMPLATE,
        help=(
            "Template applied to the dataset label to build the prompt. Must "
            "contain '{label}'. Use '{label}' for the raw value."
        ),
    )
    parser.add_argument(
        "--save-delta",
        action="store_true",
        help=(
            "Also save the final perturbation as 'delta.npy' per image, for the "
            "amortized-generator corpus. Forces a canonical square input "
            "(--canonical-size) and a uniform +-max_epsilon budget (no perceptual "
            "mask) so every saved delta has the fixed shape and box the generator "
            "consumes. Use a fresh --output-dir; this is not the report config."
        ),
    )
    parser.add_argument(
        "--canonical-size",
        type=int,
        default=DEFAULT_RESIZED_IMAGE_SIZE,
        help=(
            "Square side the image is resized to when --save-delta is set, so the "
            "saved delta matches the generator's input resolution (default 512)."
        ),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class ImageLock:
    path: Path
    acquired: bool

    def release(self) -> None:
        if self.acquired:
            shutil.rmtree(self.path, ignore_errors=True)


def discover_image_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def order_image_queue(paths: list[Path], seed: str) -> list[Path]:
    ordered = list(paths)
    seed_int = int.from_bytes(
        hashlib.sha256(seed.encode("utf-8")).digest()[:8],
        byteorder="big",
    )
    random.Random(seed_int).shuffle(ordered)
    return ordered


def image_output_dir(output_dir: Path, image_path: Path) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)
    safe_stem = safe_stem.strip("._") or "image"
    return output_dir / safe_stem


def should_skip_image(output_dir: Path) -> bool:
    return (output_dir / "complete.json").exists()


def lock_is_stale(lock_dir: Path, *, now: float, timeout_seconds: float) -> bool:
    if timeout_seconds <= 0:
        return False
    metadata_path = lock_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text())
        created_at = float(metadata["created_at"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        created_at = lock_dir.stat().st_mtime
    return now - created_at > timeout_seconds


def acquire_image_lock(
    *,
    output_dir: Path,
    image_path: Path,
    lock_timeout_seconds: float,
    worker_id: str,
    now: float | None = None,
) -> ImageLock:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = output_dir / ".lock"
    current_time = time.time() if now is None else now

    for _ in range(2):
        try:
            lock_dir.mkdir()
        except FileExistsError:
            if lock_is_stale(
                lock_dir,
                now=current_time,
                timeout_seconds=lock_timeout_seconds,
            ):
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            return ImageLock(path=lock_dir, acquired=False)
        metadata = {
            "worker_id": worker_id,
            "source_image": str(image_path),
            "created_at": current_time,
        }
        (lock_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return ImageLock(path=lock_dir, acquired=True)

    return ImageLock(path=lock_dir, acquired=False)


def fetch_imagenet_classes() -> list[str]:
    with urlopen(IMAGENET_CLASSES_URL, timeout=30) as response:
        return response.read().decode("utf-8").strip().splitlines()


def load_image_labels(
    csv_path: Path,
    *,
    id_column: str,
    label_field: str,
) -> dict[str, str]:
    """Map image id -> dataset label from a labels CSV (e.g. Pexels labels.csv)."""
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for required in (id_column, label_field):
            if required not in fieldnames:
                raise ValueError(
                    f"--labels-csv {csv_path} is missing column '{required}'; "
                    f"available columns: {fieldnames}"
                )
        lookup: dict[str, str] = {}
        for row in reader:
            image_id = (row.get(id_column) or "").strip()
            label = (row.get(label_field) or "").strip()
            if image_id and label:
                lookup[image_id] = label
    return lookup


def resolve_image_prompt(
    image_path: Path,
    label_lookup: dict[str, str],
    *,
    template: str,
    default_prompt: str,
) -> str | None:
    """Per-image prompt from the dataset label.

    Returns ``default_prompt`` when no label source is configured (backward
    compatible), the templated dataset label when one is found, or ``None`` when
    a label source exists but this image has no matching row (caller skips it).
    """
    if not label_lookup:
        return default_prompt
    label = label_lookup.get(image_path.stem)
    if not label:
        return None
    return template.format(label=label)


def build_prediction_vocabulary(
    imagenet_classes: list[str],
    image_label: str | None = None,
) -> list[str]:
    vocabulary = CUSTOM_CLASSES + [f"Photo of {label}" for label in imagenet_classes]
    if image_label and image_label not in vocabulary:
        vocabulary = vocabulary + [image_label]
    return vocabulary


def build_distribution_anchor(
    pipeline: MetaClip2ExplanationPipeline,
    imagenet_classes: list[str],
    stride: int,
) -> DistributionAnchor:
    if stride <= 0:
        raise ValueError("--anchor-stride must be positive.")
    anchor_labels = [f"Photo of {label}" for label in imagenet_classes[::stride]]
    batch = move_batch_to_device(
        pipeline.processor(text=anchor_labels, padding=True, return_tensors="pt"),
        pipeline.device,
    )
    with torch.no_grad():
        encoding = clip_encode_text_dense(
            batch["input_ids"],
            pipeline.model,
            attention_mask=batch.get("attention_mask"),
            n=pipeline.config.text_layers,
        )
        text_embeddings = F.normalize(encoding.text_embedding, dim=-1).detach()
        logit_scale = pipeline.model.logit_scale.exp().detach()
    return DistributionAnchor(
        text_embeddings=text_embeddings,
        logit_scale=logit_scale,
        labels=tuple(anchor_labels),
    )


def build_centered_target_mask(height: int, width: int) -> torch.Tensor:
    phrase = "2+2=5"
    glyph_rows = len(GLYPHS["2"])
    glyph_widths = [len(GLYPHS[ch][0]) for ch in phrase]
    total_width = sum(glyph_widths) + (len(phrase) - 1)
    if glyph_rows > height or total_width > width:
        raise ValueError(
            f"Target glyph layout {glyph_rows}x{total_width} does not fit "
            f"explanation map {height}x{width}."
        )

    mask = torch.zeros((height, width), dtype=torch.float32)
    top = (height - glyph_rows) // 2
    left = (width - total_width) // 2
    cursor = left
    for char in phrase:
        glyph = GLYPHS[char]
        for row_index, row in enumerate(glyph):
            for col_index, value in enumerate(row):
                if value == "1":
                    mask[top + row_index, cursor + col_index] = 1.0
        cursor += len(glyph[0]) + 1
    return mask


def compute_mask_iou_at_mean(
    explanation: torch.Tensor,
    target_explanation: torch.Tensor,
) -> float:
    adv_above_mean = (explanation > explanation.mean()).float()
    target_binary = (target_explanation > 0.5).float()
    intersection = (adv_above_mean * target_binary).sum()
    union = torch.clamp(adv_above_mean + target_binary, max=1.0).sum()
    return float((intersection / union).item()) if union.item() > 0 else 0.0


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean()
    if mse.item() < 1e-12:
        return float("inf")
    return float((20.0 * torch.log10(1.0 / mse.sqrt())).item())


def compute_ssim(
    a: torch.Tensor,
    b: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    coords = torch.arange(window_size, dtype=a.dtype, device=a.device)
    coords = coords - (window_size - 1) / 2.0
    gauss = torch.exp(-(coords**2) / (2.0 * sigma**2))
    gauss = gauss / gauss.sum()
    window_2d = gauss[:, None] * gauss[None, :]

    channels = a.shape[0]
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()

    a4 = a.unsqueeze(0)
    b4 = b.unsqueeze(0)
    pad = window_size // 2

    def conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, window, groups=channels, padding=pad)

    mu_a = conv(a4)
    mu_b = conv(b4)
    mu_aa = mu_a * mu_a
    mu_bb = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_aa = conv(a4 * a4) - mu_aa
    sigma_bb = conv(b4 * b4) - mu_bb
    sigma_ab = conv(a4 * b4) - mu_ab

    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_aa + mu_bb + c1) * (sigma_aa + sigma_bb + c2)
    return float((num / den).mean().item())


def score_vocabulary(
    pipeline: MetaClip2ExplanationPipeline,
    image_tensor: torch.Tensor,
    labels: list[str],
    batch_size: int = 64,
) -> torch.Tensor:
    image_pil = tensor_to_pil_image(image_tensor.detach().cpu())
    logits_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            batch_labels = labels[start : start + batch_size]
            inputs = pipeline.processor(
                images=image_pil,
                text=batch_labels,
                return_tensors="pt",
                padding=True,
            )
            batch = {key: value.to(pipeline.device) for key, value in inputs.items()}
            outputs = pipeline.model(
                **batch,
                interpolate_pos_encoding=pipeline.interpolate_pos_encoding,
            )
            logits_chunks.append(outputs.logits_per_image.squeeze(0).detach().cpu())
    return torch.cat(logits_chunks)


def build_snapshot_row(
    snapshot: ManipulationProgressSnapshot,
    labels: list[str],
    logits: torch.Tensor,
    top_k: int,
    prompt: str,
) -> tuple[dict[str, object], dict[str, object]]:
    probs = logits.softmax(dim=-1)
    preserved_index = labels.index(prompt)
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(labels)))

    diff = (snapshot.current_image - snapshot.original_image).detach()
    metrics: dict[str, object] = {
        "step": snapshot.step,
        "loss": snapshot.loss,
        "mask_loss": snapshot.mask_loss,
        "perturb_loss": snapshot.perturb_loss,
        "pred_loss": snapshot.pred_loss,
        "dist_loss": snapshot.dist_loss,
        "mask_l1": float(
            (snapshot.current_explanation - snapshot.target_explanation)
            .abs()
            .mean()
            .item()
        ),
        "mask_iou_at_mean": compute_mask_iou_at_mean(
            snapshot.current_explanation,
            snapshot.target_explanation,
        ),
        "pert_l2_rms": float(diff.pow(2).mean().sqrt().item()),
        "pert_linf": float(diff.abs().amax().item()),
        "pert_psnr": compute_psnr(snapshot.current_image, snapshot.original_image),
        "pert_ssim": compute_ssim(snapshot.current_image, snapshot.original_image),
        "cos_preserve": float(
            F.cosine_similarity(
                snapshot.image_embedding,
                snapshot.original_embedding,
                dim=-1,
            )
            .mean()
            .item()
        ),
        "sim_before": float(snapshot.original_similarity.item()),
        "sim_after": float(snapshot.similarity.item()),
        "sim_delta": float(
            (snapshot.similarity - snapshot.original_similarity).abs().item()
        ),
        "preserved_rank": int((logits > logits[preserved_index]).sum().item()) + 1,
        "preserved_prob": float(probs[preserved_index].item()),
    }
    if (
        snapshot.prediction_probs is not None
        and snapshot.baseline_prediction_probs is not None
    ):
        anchor_probs = snapshot.prediction_probs.detach().cpu()
        baseline_anchor = snapshot.baseline_prediction_probs.detach().cpu()
        eps = 1e-12
        kl_t1 = float(
            (
                baseline_anchor
                * (
                    (baseline_anchor + eps).log()
                    - (anchor_probs + eps).log()
                )
            )
            .sum()
            .item()
        )
        tvd = float(0.5 * (anchor_probs - baseline_anchor).abs().sum().item())
        metrics["anchor_kl_T1"] = kl_t1
        metrics["anchor_tvd"] = tvd
    ranking = [
        {
            "rank": rank,
            "label": labels[index],
            "probability": float(prob),
        }
        for rank, (index, prob) in enumerate(
            zip(top_indices.tolist(), top_probs.tolist()),
            start=1,
        )
    ]
    metrics_with_ranking = dict(metrics)
    metrics_with_ranking["top_k"] = ranking

    csv_row: dict[str, object] = dict(metrics)
    for entry in ranking:
        csv_row[f"top_{entry['rank']}_label"] = entry["label"]
        csv_row[f"top_{entry['rank']}_prob"] = entry["probability"]
    return metrics_with_ranking, csv_row


def save_snapshot(
    *,
    snapshot: ManipulationProgressSnapshot,
    pipeline: MetaClip2ExplanationPipeline,
    labels: list[str],
    output_dir: Path,
    top_k: int,
    csv_rows: list[dict[str, object]],
    prompt: str,
) -> None:
    logits = score_vocabulary(pipeline, snapshot.current_image, labels)
    metrics_json, csv_row = build_snapshot_row(snapshot, labels, logits, top_k, prompt)
    csv_rows.append(csv_row)

    if snapshot.step == 0:
        (output_dir / "baseline_metrics.json").write_text(
            json.dumps(metrics_json, indent=2)
        )


def write_metrics_csv(output_dir: Path, rows: list[dict[str, object]]) -> Path | None:
    if not rows:
        return None
    csv_path = output_dir / "metrics.csv"
    fieldnames = union_fieldnames(rows)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def union_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for fieldname in row:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
    return fieldnames


def create_baseline_snapshot(
    baseline: object,
    target_mask: torch.Tensor,
    anchor: DistributionAnchor | None = None,
) -> ManipulationProgressSnapshot:
    baseline_run = baseline
    baseline_mask_loss = float(
        (baseline_run.explanation - target_mask.to(baseline_run.explanation.device))
        .abs()
        .mean()
        .item()
    )
    baseline_probs: torch.Tensor | None = None
    if anchor is not None:
        with torch.no_grad():
            logits = (
                anchor.logit_scale
                * (baseline_run.image_embedding @ anchor.text_embeddings.T)
            ).squeeze(0)
            baseline_probs = F.softmax(logits, dim=-1).detach().cpu()
    return ManipulationProgressSnapshot(
        step=0,
        loss=baseline_mask_loss,
        mask_loss=baseline_mask_loss,
        perturb_loss=0.0,
        pred_loss=0.0,
        original_image=baseline_run.image.detach().clone(),
        current_image=baseline_run.image.detach().clone(),
        current_explanation=baseline_run.explanation.detach().clone(),
        target_explanation=target_mask.detach().to(baseline_run.explanation.device).clone(),
        image_embedding=baseline_run.image_embedding.detach().clone(),
        original_embedding=baseline_run.image_embedding.detach().clone(),
        similarity=baseline_run.similarity.detach().clone(),
        original_similarity=baseline_run.similarity.detach().clone(),
        dist_loss=0.0,
        prediction_probs=baseline_probs,
        baseline_prediction_probs=baseline_probs,
    )


def save_final_artifacts(
    *,
    result: ManipulationResult,
    output_dir: Path,
) -> None:
    np.save(
        output_dir / "final_explanation.npy",
        result.best_explanation.detach().cpu().numpy().astype(np.float32),
    )
    (output_dir / "final_metrics.json").write_text(
        json.dumps(result.metrics, indent=2)
    )


def write_complete_marker(
    *,
    output_dir: Path,
    image_path: Path,
    model_name: str,
    steps: int,
    validate_only: bool,
    prompt: str,
    img_id: str,
    label: str | None,
    save_delta: bool = False,
    canonical_size: int | None = None,
) -> Path:
    marker_path = output_dir / "complete.json"
    if validate_only:
        outputs: dict[str, str] = {
            "baseline_metrics": "baseline_metrics.json",
            "metrics_csv": "metrics.csv",
        }
    else:
        outputs = {
            "final_explanation": "final_explanation.npy",
            "final_metrics": "final_metrics.json",
            "metrics_csv": "metrics.csv",
        }
        if save_delta:
            outputs["delta"] = "delta.npy"
    marker = {
        "source_image": str(image_path),
        "img_id": img_id,
        "label": label,
        "prompt": prompt,
        "model_name": model_name,
        "steps": steps,
        "validate_only": validate_only,
        "canonical_size": canonical_size,
        "outputs": outputs,
    }
    marker_path.write_text(json.dumps(marker, indent=2))
    return marker_path


def write_batch_summary(
    output_dir: Path,
    rows: list[dict[str, object]],
) -> Path:
    summary_path = output_dir / "summary.csv"
    fieldnames = union_fieldnames(rows) or ["image", "output_dir", "status"]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def load_canonical_square_tensor(image_path: Path, size: int) -> torch.Tensor:
    """Load an image resized to a ``size``x``size`` square in [0, 1] CHW.

    Used for ``--save-delta`` so every saved perturbation has the fixed shape the
    amortized generator consumes. Mirrors ``run_universal_mask_transfer``'s
    ``load_canonical_image`` but is defined locally to avoid a circular import
    (that module imports helpers from here).
    """
    pil = load_image(image_path).resize((size, size))
    array = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def build_manipulation_config(
    *,
    steps: int,
    save_every: int,
    perceptual_mask_config: PerceptualMaskConfig | None,
) -> ManipulationConfig:
    return ManipulationConfig(
        steps=steps,
        lr=2e-2,
        log_every=save_every,
        w_mask_l1=1.5,
        w_mask_l2=3.0,
        w_mask_focal_fg=3.0,
        w_perturb_l1=50.0,
        w_perturb_l2=10.0,
        w_perturb_linf=50.0,
        w_pred_cos=1.0,
        w_dist_kl=1.0,
        dist_kl_temperature=2.0,
        perceptual_mask=perceptual_mask_config,
    )


def run_one_image(
    *,
    image_path: Path,
    output_dir: Path,
    pipeline: MetaClip2ExplanationPipeline,
    engine: MetaClip2ManipulationEngine,
    labels: list[str],
    distribution_anchor: DistributionAnchor,
    prompt: str,
    img_id: str,
    label: str | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, object]] = []

    if args.save_delta:
        image_tensor = load_canonical_square_tensor(
            image_path, args.canonical_size
        ).to(pipeline.device)
    else:
        image_tensor = pil_to_tensor(load_image(image_path)).to(pipeline.device)
    baseline = pipeline.run(image_tensor, prompt)
    target_mask = build_centered_target_mask(*baseline.explanation.shape).to(
        baseline.explanation.device
    )

    if args.validate_only:
        save_snapshot(
            snapshot=create_baseline_snapshot(
                baseline, target_mask, distribution_anchor
            ),
            pipeline=pipeline,
            labels=labels,
            output_dir=output_dir,
            top_k=args.top_k,
            csv_rows=csv_rows,
            prompt=prompt,
        )
        csv_path = write_metrics_csv(output_dir, csv_rows)
        write_complete_marker(
            output_dir=output_dir,
            image_path=image_path,
            model_name=args.model_name,
            steps=args.steps,
            validate_only=True,
            prompt=prompt,
            img_id=img_id,
            label=label,
        )
        return {
            "image": str(image_path),
            "output_dir": str(output_dir),
            "status": "processed",
            "img_id": img_id,
            "label": label,
            "metrics_csv": str(csv_path) if csv_path is not None else "",
            "prompt": prompt,
            "validate_only": True,
        }

    # --save-delta uses a uniform +-max_epsilon budget (no perceptual scaling) so
    # the saved delta lives in the exact box the amortized generator can produce.
    perceptual_mask_config = (
        None if args.save_delta else PerceptualMaskConfig(budget_max_scale=2.5)
    )
    result = engine.run(
        image=image_tensor,
        caption=prompt,
        target_explanation=target_mask,
        config=build_manipulation_config(
            steps=args.steps,
            save_every=args.save_every,
            perceptual_mask_config=perceptual_mask_config,
        ),
        progress=True,
        progress_callback=lambda snapshot: save_snapshot(
            snapshot=snapshot,
            pipeline=pipeline,
            labels=labels,
            output_dir=output_dir,
            top_k=args.top_k,
            csv_rows=csv_rows,
            prompt=prompt,
        ),
        callback_every=args.save_every,
        distribution_anchor=distribution_anchor,
    )

    csv_path = write_metrics_csv(output_dir, csv_rows)
    save_final_artifacts(result=result, output_dir=output_dir)
    if args.save_delta:
        delta = (result.best_image.detach().cpu() - image_tensor.detach().cpu()).float()
        np.save(output_dir / "delta.npy", delta.numpy().astype(np.float32))
    write_complete_marker(
        output_dir=output_dir,
        image_path=image_path,
        model_name=args.model_name,
        steps=args.steps,
        validate_only=False,
        prompt=prompt,
        img_id=img_id,
        label=label,
        save_delta=args.save_delta,
        canonical_size=args.canonical_size if args.save_delta else None,
    )

    return {
        "image": str(image_path),
        "output_dir": str(output_dir),
        "status": "processed",
        "img_id": img_id,
        "label": label,
        "metrics_csv": str(csv_path) if csv_path is not None else "",
        "prompt": prompt,
        **result.metrics,
    }


def main() -> None:
    args = parse_args()

    if args.save_every <= 0:
        raise ValueError("--save-every must be positive.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.lock_timeout_hours < 0:
        raise ValueError("--lock-timeout-hours must be non-negative.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_id = default_worker_id()
    queue_seed = args.queue_seed or worker_id
    image_paths = order_image_queue(discover_image_paths(args.input_dir), queue_seed)
    lock_timeout_seconds = args.lock_timeout_hours * 3600.0

    pipeline = MetaClip2ExplanationPipeline(
        MetaClip2RunConfig(model_name=args.model_name, device=args.device)
    )
    engine = MetaClip2ManipulationEngine(pipeline)

    imagenet_classes = fetch_imagenet_classes()
    distribution_anchor = build_distribution_anchor(pipeline, imagenet_classes, 5)
    (output_dir / "anchor_labels.json").write_text(
        json.dumps(list(distribution_anchor.labels), indent=2)
    )

    label_lookup: dict[str, str] = {}
    if args.labels_csv and args.labels_csv.exists():
        label_lookup = load_image_labels(
            args.labels_csv,
            id_column=args.id_column,
            label_field=args.label_field,
        )
        print(f"labels_csv={args.labels_csv} ({len(label_lookup)} labels)")
    else:
        print(
            f"labels_csv={args.labels_csv} not found; "
            f"falling back to PRESERVED_PROMPT for every image"
        )

    summary_rows: list[dict[str, object]] = []
    processed = 0
    skipped = 0
    locked = 0
    failed = 0
    total = len(image_paths)
    print(f"input_dir={args.input_dir}")
    print(f"output_dir={output_dir}")
    print(f"worker_id={worker_id}")
    print(f"queue_seed={queue_seed}")
    print(f"total_images={total}")

    for index, image_path in enumerate(image_paths, start=1):
        img_id = image_path.stem
        label = label_lookup.get(img_id)
        prompt = resolve_image_prompt(
            image_path,
            label_lookup,
            template=args.label_template,
            default_prompt=PRESERVED_PROMPT,
        )
        if prompt is None:
            skipped += 1
            no_label_row: dict[str, object] = {
                "image": str(image_path),
                "output_dir": str(image_output_dir(output_dir, image_path)),
                "status": "no_label",
                "img_id": img_id,
                "label": label,
            }
            summary_rows.append(no_label_row)
            print(f"[{index}/{total}] no_label {image_path.name}")
            print(
                f"processed={processed} skipped={skipped} locked={locked} "
                f"failed={failed} total={total}"
            )
            continue

        per_image_output_dir = image_output_dir(output_dir, image_path)
        if should_skip_image(per_image_output_dir):
            skipped += 1
            summary_rows.append(
                {
                    "image": str(image_path),
                    "output_dir": str(per_image_output_dir),
                    "status": "skipped",
                    "img_id": img_id,
                    "label": label,
                }
            )
            print(f"[{index}/{total}] skipped {image_path.name}")
            print(
                f"processed={processed} skipped={skipped} locked={locked} "
                f"failed={failed} total={total}"
            )
            continue

        lock = acquire_image_lock(
            output_dir=per_image_output_dir,
            image_path=image_path,
            lock_timeout_seconds=lock_timeout_seconds,
            worker_id=worker_id,
        )
        if not lock.acquired:
            locked += 1
            summary_rows.append(
                {
                    "image": str(image_path),
                    "output_dir": str(per_image_output_dir),
                    "status": "locked",
                    "img_id": img_id,
                    "label": label,
                }
            )
            print(f"[{index}/{total}] locked {image_path.name}")
            print(
                f"processed={processed} skipped={skipped} locked={locked} "
                f"failed={failed} total={total}"
            )
            continue

        try:
            row: dict[str, object]
            if should_skip_image(per_image_output_dir):
                skipped += 1
                row = {
                    "image": str(image_path),
                    "output_dir": str(per_image_output_dir),
                    "status": "skipped",
                    "img_id": img_id,
                    "label": label,
                }
                print(f"[{index}/{total}] skipped {image_path.name}")
            else:
                print(f"[{index}/{total}] processing {image_path.name}")
                try:
                    row = run_one_image(
                        image_path=image_path,
                        output_dir=per_image_output_dir,
                        pipeline=pipeline,
                        engine=engine,
                        labels=build_prediction_vocabulary(
                            imagenet_classes, image_label=prompt
                        ),
                        distribution_anchor=distribution_anchor,
                        prompt=prompt,
                        img_id=img_id,
                        label=label,
                        args=args,
                    )
                except Exception as exc:
                    failed += 1
                    row = {
                        "image": str(image_path),
                        "output_dir": str(per_image_output_dir),
                        "status": "failed",
                        "img_id": img_id,
                        "label": label,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"[{index}/{total}] failed {image_path.name}: {exc}")
                else:
                    processed += 1
        finally:
            lock.release()

        summary_rows.append(row)
        print(
            f"processed={processed} skipped={skipped} locked={locked} "
            f"failed={failed} total={total}"
        )

    summary_path = write_batch_summary(output_dir, summary_rows)
    print(f"summary_csv={summary_path}")
    if label_lookup:
        print(
            f"prompt_source=dataset_label field='{args.label_field}' "
            f"template='{args.label_template}'"
        )
    else:
        print(f"prompt_source=fallback preserved_prompt={PRESERVED_PROMPT}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
