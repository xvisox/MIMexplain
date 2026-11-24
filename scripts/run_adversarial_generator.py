#!/usr/bin/env python3
"""Amortized adversarial perturbation generator (Q3 "smaller network").

Trains a small network ``G`` so that ``apply_generator_delta(G(x), budget, x)``
forces the Grad-ECLIP explanation of MetaCLIP-2 toward the fixed "2+2=5" glyph
while preserving the model's prediction. This asks whether an input-conditioned
generator can *amortize* the per-image attack and succeed on unseen images where
a single universal mask fails (held-out IoU stuck at chance ~0.05).

Four subcommands, separated by cost so the expensive ones run on the remote GPU:

- ``gen-corpus`` (model, slow, remote): run the per-image
  :class:`MetaClip2ManipulationEngine` on canonically-resized images with a
  *uniform* budget (so deltas live in the same +-max_epsilon box the generator
  can represent), saving ``(image, delta)`` pairs as the supervised corpus.
- ``pretrain`` (cheap, no MetaCLIP, local): regress ``G`` onto the corpus deltas.
- ``finetune`` (second-order, remote): optimize ``G`` against the real Grad-ECLIP
  objective, mirroring ``UniversalManipulationEngine.train``'s accumulation loop.
- ``eval`` (model, slow, remote): apply the frozen ``G`` to held-out images and
  emit metrics in the same shape as ``run_universal_mask_transfer`` so the
  generator's held-out IoU overlays directly on the universal-mask curve.

The pure helpers (``delta_from_result`` / ``filter_corpus_manifest`` /
``validate_resolution``) are unit-tested without the model; the training/eval
core is importable for a single slow smoke test.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from explainer.manipulation import (
    DistributionAnchor,
    GeneratorConfig,
    ManipulationConfig,
    MetaClip2ManipulationEngine,
    PerturbationGenerator,
    apply_generator_delta,
    average_metrics,
    build_uniform_budget,
    count_parameters,
    supervised_delta_loss,
)
from explainer.manipulation.engine import (
    _anchor_logits,
    _bounded_delta,
    _compute_metrics,
    _cosine,
    _distribution_penalty,
    _mask_error,
    _perturbation_penalty,
    _prediction_penalty,
)
from explainer.metaclip2 import MetaClip2ExplanationPipeline, MetaClip2RunConfig
from scripts.run_hubert_2plus2eq5_manipulation import (
    DEFAULT_LABELS_CSV,
    PRESERVED_PROMPT,
    build_centered_target_mask,
    build_distribution_anchor,
    discover_image_paths,
    fetch_imagenet_classes,
    load_image_labels,
)
from scripts.run_universal_mask_transfer import (
    CanonicalImageCache,
    aggregate_records,
    build_captioned_pool,
    make_split,
)

# MetaCLIP-2 b16 patch size; the glyph "2+2=5" needs a grid width >= 19.
PATCH_SIZE = 16
MIN_GLYPH_GRID = 19
DEFAULT_OUTPUT_DIR = Path("outputs/advgen")
DEFAULT_INPUT_DIR = Path("data/pexels-1k/images")


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without the model)
# --------------------------------------------------------------------------- #
def delta_from_result(best_image: Tensor, original_image: Tensor) -> Tensor:
    """Recover the applied perturbation from an engine result.

    ``best_image = clamp(original + bounded_delta, 0, 1)``, so the difference is
    the perturbation as applied; clamping only shrinks magnitude, so it still
    respects the engine's L-inf budget.
    """
    return best_image - original_image


def filter_corpus_manifest(
    rows: Sequence[dict[str, object]], min_iou: float
) -> list[dict[str, object]]:
    """Keep only corpus entries whose per-image attack reached ``min_iou``.

    Cheap supervised pre-training on low-quality deltas teaches the generator
    garbage, so weak per-image solutions are dropped before regression.
    """
    return [
        r for r in rows if float(cast(float, r.get("mask_iou_at_mean", 0.0))) >= min_iou
    ]


def discover_hubert_corpus_entries(
    hubert_dir: Path, min_iou: float
) -> list[dict[str, object]]:
    """Scan a ``run_hubert ... --save-delta`` output dir for usable corpus entries.

    Each completed per-image dir holds ``complete.json`` (source image, prompt,
    canonical size), ``final_metrics.json`` (the per-image attack quality), and
    ``delta.npy`` (the saved perturbation). Entries missing the delta or below
    ``min_iou`` are skipped, so the generator only trains on strong per-image
    solutions.
    """
    entries: list[dict[str, object]] = []
    for child in sorted(hubert_dir.iterdir()):
        if not child.is_dir():
            continue
        complete = child / "complete.json"
        delta = child / "delta.npy"
        metrics = child / "final_metrics.json"
        if not (complete.exists() and delta.exists() and metrics.exists()):
            continue
        meta = json.loads(complete.read_text())
        iou = float(json.loads(metrics.read_text()).get("mask_iou_at_mean", 0.0))
        if iou < min_iou:
            continue
        entries.append(
            {
                "img_id": str(meta.get("img_id", child.name)),
                "path": meta.get("source_image"),
                "caption": meta.get("prompt"),
                "canonical_size": meta.get("canonical_size"),
                "mask_iou_at_mean": iou,
                "delta_path": str(delta),
            }
        )
    return entries


def iou_at_mean(explanation: Tensor, target: Tensor) -> float:
    """IoU of (explanation > its mean) vs (target > 0.5).

    Mirrors ``mask_iou_at_mean`` in ``engine._compute_metrics`` exactly so the
    number printed during finetune matches the one reported at eval time.
    """
    adv_above_mean = (explanation > explanation.mean()).float()
    target_binary = (target > 0.5).float()
    intersection = (adv_above_mean * target_binary).sum()
    union = torch.clamp(adv_above_mean + target_binary, max=1.0).sum()
    return float((intersection / union).item()) if union.item() > 0 else 0.0


def cap_corpus(items: list, max_corpus: int) -> list:
    """Truncate a corpus list to its first ``max_corpus`` entries (0 == no cap).

    Lets a tuning/overfit run train on a handful of images (fast, diagnostic)
    instead of the whole filtered corpus, since each finetune step is a full
    pass over every image (step time scales with corpus size).
    """
    return items[:max_corpus] if max_corpus and max_corpus > 0 else items


def exclude_paths_by_name(paths: Sequence[Path], exclude_names: set[str]) -> list[Path]:
    """Drop paths whose filename is in ``exclude_names`` (held-out leakage guard).

    The transfer split must be disjoint from the images the generator trained on,
    or the held-out IoU is contaminated. Matching on filename is robust to
    absolute-vs-relative path differences between the corpus manifest and the
    eval pool.
    """
    return [p for p in paths if p.name not in exclude_names]


def should_checkpoint(step: int, checkpoint_every: int) -> bool:
    """Whether to persist a checkpoint after completing ``step`` (0-indexed).

    Fires every ``checkpoint_every`` steps so a long second-order finetune never
    loses more than that many steps of progress to a crash, OOM, or kill.
    ``checkpoint_every <= 0`` disables intermediate saving (final save only).
    """
    return checkpoint_every > 0 and (step + 1) % checkpoint_every == 0


def validate_resolution(size: int) -> None:
    """Raise unless ``size`` is a patch multiple large enough for the glyph."""
    if size % PATCH_SIZE != 0:
        raise ValueError(
            f"--resized-image-size must be a multiple of {PATCH_SIZE}; got {size}."
        )
    grid = size // PATCH_SIZE
    if grid < MIN_GLYPH_GRID:
        raise ValueError(
            f"--resized-image-size {size} gives a {grid}x{grid} grid; the '2+2=5' "
            f"glyph needs at least {MIN_GLYPH_GRID} (>= {MIN_GLYPH_GRID * PATCH_SIZE})."
        )


# --------------------------------------------------------------------------- #
# Baselines + training/eval core (importable for the slow smoke test)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Baseline:
    embedding: Tensor
    similarity: Tensor
    logits: Tensor | None


def _baseline(
    pipeline: MetaClip2ExplanationPipeline,
    image: Tensor,
    caption: str,
    distribution_anchor: DistributionAnchor | None,
) -> _Baseline:
    run = pipeline.run(image.to(pipeline.device), caption)
    embedding = run.image_embedding.detach()
    logits: Tensor | None = None
    if distribution_anchor is not None:
        with torch.no_grad():
            logits = _anchor_logits(embedding, distribution_anchor).detach()
    return _Baseline(
        embedding=embedding, similarity=run.similarity.detach(), logits=logits
    )


def finetune_generator(
    generator: PerturbationGenerator,
    pipeline: MetaClip2ExplanationPipeline,
    images: Sequence[Tensor],
    captions: Sequence[str],
    target: Tensor,
    budget: Tensor,
    config: GeneratorConfig,
    *,
    corpus_deltas: Sequence[Tensor] | None = None,
    distribution_anchor: DistributionAnchor | None = None,
    progress: bool = True,
    checkpoint_every: int = 0,
    on_checkpoint: Callable[[int, list[dict[str, float]]], None] | None = None,
) -> list[dict[str, float]]:
    """Second-order finetune of ``G`` (mirrors ``UniversalManipulationEngine.train``).

    The mask + perturbation + prediction + distribution terms are the engine's,
    so the objective is byte-for-byte the per-image one; ``corpus_deltas`` adds an
    anchor pulling ``G`` toward its pretrained solution. Images are processed one
    at a time (``pipeline.run`` rejects batches) with gradient accumulation over a
    micro-batch, so ``micro_batch`` is the second-order memory knob.
    """
    device = pipeline.device
    target = target.to(device)
    budget = budget.to(device)
    generator.to(device).train()
    mcfg = config.to_manipulation_config()
    optimizer = torch.optim.Adam(generator.parameters(), lr=config.finetune_lr)

    baselines = [
        _baseline(pipeline, images[i], captions[i], distribution_anchor)
        for i in range(len(images))
    ]
    torch.manual_seed(config.seed)

    count = len(images)
    micro = min(config.micro_batch, count)
    trace: list[dict[str, float]] = []
    iterator: range = range(config.finetune_steps)
    if progress:
        iterator = _maybe_tqdm(iterator, "finetune")

    for step in iterator:
        optimizer.zero_grad()
        step_losses = {
            "loss": 0.0,
            "mask": 0.0,
            "perturb": 0.0,
            "pred": 0.0,
            "dist": 0.0,
        }
        step_metrics = {"iou": 0.0, "cos_preserve": 0.0, "pert_linf": 0.0}
        for chunk_start in range(0, count, micro):
            chunk = range(chunk_start, min(chunk_start + micro, count))
            chunk_loss: Tensor | None = None
            for index in chunk:
                x = images[index].to(device)
                raw = generator(x.unsqueeze(0)).squeeze(0)
                perturbed = apply_generator_delta(raw, budget, x)
                run = pipeline.run(perturbed, captions[index])
                mask = _mask_error(run.explanation, target, mcfg)
                perturb = _perturbation_penalty(perturbed, x, mcfg)
                pred = _prediction_penalty(
                    run.image_embedding,
                    run.similarity,
                    baselines[index].embedding,
                    baselines[index].similarity,
                    mcfg,
                )
                dist = _distribution_penalty(
                    run.image_embedding,
                    distribution_anchor,
                    baselines[index].logits,
                    mcfg,
                )
                image_loss = mask + perturb + pred + dist
                if corpus_deltas is not None and config.w_supervised_anchor > 0:
                    bounded = _bounded_delta(raw, budget)
                    image_loss = (
                        image_loss
                        + config.w_supervised_anchor
                        * supervised_delta_loss(
                            bounded,
                            corpus_deltas[index].to(device),
                            w_l1=config.w_sup_l1,
                            w_l2=config.w_sup_l2,
                        )
                    )
                chunk_loss = (
                    image_loss if chunk_loss is None else chunk_loss + image_loss
                )
                step_losses["loss"] += float(image_loss.item()) / count
                step_losses["mask"] += float(mask.item()) / count
                step_losses["perturb"] += float(perturb.item()) / count
                step_losses["pred"] += float(pred.item()) / count
                step_losses["dist"] += float(dist.item()) / count
                with torch.no_grad():
                    step_metrics["iou"] += (
                        iou_at_mean(run.explanation.detach(), target) / count
                    )
                    step_metrics["cos_preserve"] += (
                        _cosine(
                            run.image_embedding.detach(), baselines[index].embedding
                        )
                        / count
                    )
                    step_metrics["pert_linf"] += (
                        float((perturbed - x).detach().abs().amax().item()) / count
                    )
            assert chunk_loss is not None
            (chunk_loss / count).backward()
        optimizer.step()
        # Losses + metrics are already synced to the CPU every step for the
        # bookkeeping above, so logging every step only adds a print + a small
        # dict append (no extra GPU sync). Saving the model is the one I/O-heavy
        # op, so it stays on the --checkpoint-every cadence below.
        trace.append({"step": float(step), **step_losses, **step_metrics})
        _console(
            f"[finetune] step {step + 1}/{config.finetune_steps} "
            f"loss={step_losses['loss']:.4f} mask={step_losses['mask']:.4f} "
            f"iou={step_metrics['iou']:.3f} "
            f"cos={step_metrics['cos_preserve']:.4f} "
            f"linf={step_metrics['pert_linf']:.4f}"
        )
        if on_checkpoint is not None and should_checkpoint(step, checkpoint_every):
            on_checkpoint(step, trace)
    return trace


def evaluate_generator(
    generator: PerturbationGenerator,
    pipeline: MetaClip2ExplanationPipeline,
    images: Sequence[Tensor],
    captions: Sequence[str],
    target: Tensor,
    budget: Tensor,
    config: ManipulationConfig,
    *,
    distribution_anchor: DistributionAnchor | None = None,
) -> list[dict[str, float]]:
    """Apply the frozen ``G`` per image and return per-image metrics.

    Metrics come from the shared :func:`_compute_metrics`, so they match the
    per-image and universal engines exactly (directly comparable held-out IoU).
    When a distribution anchor is supplied, ``top1_preserved`` (1.0 if the
    argmax over the ImageNet anchor vocabulary is unchanged by the perturbation)
    is added so prediction-preservation is comparable to the per-image baseline's
    top-1-preserved fraction, not just to ``cos_preserve``.
    """
    device = pipeline.device
    target = target.to(device)
    budget = budget.to(device)
    generator.to(device).eval()
    per_image: list[dict[str, float]] = []
    for index, image in enumerate(images):
        x = image.to(device)
        with torch.no_grad():
            raw = generator(x.unsqueeze(0)).squeeze(0)
            perturbed = apply_generator_delta(raw, budget, x).detach()
        base = _baseline(pipeline, x, captions[index], distribution_anchor)
        run = pipeline.run(perturbed, captions[index])
        metrics = _compute_metrics(
            best_image=perturbed.detach(),
            original_image=x,
            final_explanation=run.explanation.detach(),
            final_embedding=run.image_embedding.detach(),
            original_embedding=base.embedding,
            final_similarity=run.similarity.detach(),
            original_similarity=base.similarity,
            target_explanation=target,
            best_loss=0.0,
            config=config,
            distribution_anchor=distribution_anchor,
            original_logits=base.logits,
        )
        if distribution_anchor is not None and base.logits is not None:
            with torch.no_grad():
                final_logits = _anchor_logits(
                    run.image_embedding.detach(), distribution_anchor
                )
                metrics["top1_preserved"] = float(
                    int(final_logits.argmax()) == int(base.logits.argmax())
                )
        per_image.append(metrics)
    return per_image


def pretrain_generator(
    generator: PerturbationGenerator,
    images: Sequence[Tensor],
    deltas: Sequence[Tensor],
    budget: Tensor,
    config: GeneratorConfig,
    device: torch.device,
    *,
    progress: bool = True,
    checkpoint_every: int = 0,
    on_checkpoint: Callable[[int, list[dict[str, float]]], None] | None = None,
) -> list[dict[str, float]]:
    """Cheap supervised regression of ``G`` onto corpus deltas (no MetaCLIP)."""
    generator.to(device).train()
    budget = budget.to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=config.pretrain_lr)
    count = len(images)
    batch = max(1, min(config.pretrain_batch_size, count))
    trace: list[dict[str, float]] = []
    generator_seed = torch.Generator().manual_seed(config.seed)
    iterator: range = range(config.pretrain_epochs)
    if progress:
        iterator = _maybe_tqdm(iterator, "pretrain")

    for epoch in iterator:
        order = torch.randperm(count, generator=generator_seed).tolist()
        epoch_loss = 0.0
        pred_l1 = 0.0  # mean |bounded predicted delta| -- is G producing deltas?
        tgt_l1 = 0.0  # mean |corpus target delta|     -- the magnitude to match
        for start in range(0, count, batch):
            idx = order[start : start + batch]
            x = torch.stack([images[i] for i in idx]).to(device)
            target_delta = torch.stack([deltas[i] for i in idx]).to(device)
            bounded = _bounded_delta(generator(x), budget)
            loss = supervised_delta_loss(
                bounded, target_delta, w_l1=config.w_sup_l1, w_l2=config.w_sup_l2
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            weight = len(idx) / count
            epoch_loss += float(loss.item()) * weight
            pred_l1 += float(bounded.detach().abs().mean().item()) * weight
            tgt_l1 += float(target_delta.abs().mean().item()) * weight
        trace.append(
            {
                "epoch": float(epoch),
                "loss": epoch_loss,
                "pred_l1": pred_l1,
                "tgt_l1": tgt_l1,
            }
        )
        _console(
            f"[pretrain] epoch {epoch + 1}/{config.pretrain_epochs} "
            f"loss={epoch_loss:.5f} pred_l1={pred_l1:.4f} tgt_l1={tgt_l1:.4f}"
        )
        if on_checkpoint is not None and should_checkpoint(epoch, checkpoint_every):
            on_checkpoint(epoch, trace)
    return trace


def _maybe_tqdm(iterator: range, desc: str) -> range:
    try:
        import tqdm  # type: ignore[import-untyped]

        return tqdm.tqdm(iterator, desc=desc)
    except ImportError:
        return iterator


def _console(message: str) -> None:
    """Emit a console line without clobbering an active tqdm progress bar."""
    try:
        import tqdm  # type: ignore[import-untyped]

        tqdm.tqdm.write(message)
    except ImportError:
        print(message)


def _atomic_save_state(generator: PerturbationGenerator, path: Path) -> None:
    """Save ``generator`` to ``path`` via a temp file + rename (crash-safe).

    A kill mid-write would otherwise leave a truncated, unloadable checkpoint;
    ``os.replace`` is atomic on the same filesystem, so the visible file is
    always a complete previous checkpoint or the new one.
    """
    tmp = path.with_name(path.name + ".tmp")
    torch.save(generator.state_dict(), tmp)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _save_generator_checkpoint(
    generator: PerturbationGenerator,
    output_dir: Path,
    model_name: str,
    trace: Sequence[dict[str, float]],
    trace_name: str,
) -> None:
    _atomic_save_state(generator, output_dir / model_name)
    _atomic_write_text(output_dir / trace_name, json.dumps(list(trace), indent=2))


# --------------------------------------------------------------------------- #
# Shared setup
# --------------------------------------------------------------------------- #
def _build_pool(
    input_dir: Path, labels_csv: Path, template: str
) -> tuple[list[Path], dict[str, str]]:
    label_lookup: dict[str, str] = {}
    if labels_csv and Path(labels_csv).exists():
        label_lookup = load_image_labels(
            Path(labels_csv), id_column="img_id", label_field="title"
        )
    paths = discover_image_paths(input_dir)
    return build_captioned_pool(
        paths, label_lookup, template=template, default_prompt=PRESERVED_PROMPT
    )


def _maybe_anchor(
    pipeline: MetaClip2ExplanationPipeline, config: GeneratorConfig, stride: int
) -> DistributionAnchor | None:
    if config.w_dist_kl <= 0:
        return None
    return build_distribution_anchor(pipeline, fetch_imagenet_classes(), stride)


def _target_for(
    pipeline: MetaClip2ExplanationPipeline, image: Tensor, caption: str
) -> Tensor:
    sample = pipeline.run(image.to(pipeline.device), caption)
    grid = tuple(int(d) for d in sample.explanation.shape)
    return build_centered_target_mask(*grid).to(pipeline.device)


def _generator_config(args: argparse.Namespace) -> GeneratorConfig:
    return GeneratorConfig(
        base_channels=args.base_channels,
        num_residual_blocks=args.num_residual_blocks,
        max_epsilon=args.max_epsilon,
        pretrain_lr=args.pretrain_lr,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_batch_size=args.pretrain_batch_size,
        finetune_lr=args.finetune_lr,
        finetune_steps=args.finetune_steps,
        micro_batch=args.micro_batch,
        w_dist_kl=args.w_dist_kl,
        w_mask_scale=args.w_mask_scale,
        w_perturb_scale=args.w_perturb_scale,
        w_pred_scale=args.w_pred_scale,
        seed=args.seed,
    )


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_gen_corpus(args: argparse.Namespace) -> None:
    validate_resolution(args.resized_image_size)
    output_dir = args.output_dir
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    pool, captions = _build_pool(args.input_dir, args.labels_csv, args.label_template)
    _, train_pool = make_split(
        pool, seed_key=f"{args.seed}:0", test_size=args.test_size
    )
    pipeline = MetaClip2ExplanationPipeline(
        MetaClip2RunConfig(
            device=args.device, resized_image_size=args.resized_image_size
        )
    )
    engine = MetaClip2ManipulationEngine(pipeline)
    cache = CanonicalImageCache(args.resized_image_size)
    gen_config = _generator_config(args)
    distribution_anchor = _maybe_anchor(pipeline, gen_config, args.anchor_stride)

    target = build_centered_target_mask(
        args.resized_image_size // PATCH_SIZE, args.resized_image_size // PATCH_SIZE
    ).to(pipeline.device)
    # Uniform budget (no perceptual mask) so corpus deltas live in the +-eps box.
    config = dataclasses.replace(
        gen_config.to_manipulation_config(), steps=args.steps, log_every=args.steps
    )

    manifest: list[dict[str, object]] = []
    selected = train_pool[: args.num_corpus_images]
    for position, path in enumerate(selected, start=1):
        canonical = cache.get(path)
        caption = captions[str(path)]
        result = engine.run(
            canonical,
            caption,
            target,
            config,
            progress=False,
            distribution_anchor=distribution_anchor,
        )
        delta = delta_from_result(result.best_image.detach().cpu(), canonical)
        np.save(
            corpus_dir / f"{path.stem}_delta.npy",
            delta.numpy().astype(np.float32),
        )
        manifest.append(
            {
                "img_id": path.stem,
                "path": str(path),
                "caption": caption,
                "mask_iou_at_mean": result.metrics["mask_iou_at_mean"],
                "pert_linf": result.metrics["pert_linf"],
                "cos_preserve": result.metrics["cos_preserve"],
            }
        )
        print(
            f"[{position}/{len(selected)}] {path.stem} "
            f"iou={result.metrics['mask_iou_at_mean']:.3f} "
            f"linf={result.metrics['pert_linf']:.3f}"
        )
        (corpus_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    (corpus_dir / "config.json").write_text(
        json.dumps(
            {
                "resized_image_size": args.resized_image_size,
                "max_epsilon": args.max_epsilon,
                "seed": args.seed,
                "test_size": args.test_size,
                "num_corpus_images": len(selected),
                "steps": args.steps,
                "label_template": args.label_template,
            },
            indent=2,
        )
    )
    print(f"Corpus written to {corpus_dir} ({len(manifest)} deltas)")


def _load_corpus(
    corpus_dir: Path, min_iou: float
) -> tuple[list[Tensor], list[Tensor], list[str], dict[str, object]]:
    corpus_config = json.loads((corpus_dir / "config.json").read_text())
    manifest = filter_corpus_manifest(
        json.loads((corpus_dir / "manifest.json").read_text()), min_iou
    )
    cache = CanonicalImageCache(int(corpus_config["resized_image_size"]))
    images: list[Tensor] = []
    deltas: list[Tensor] = []
    captions: list[str] = []
    for row in manifest:
        images.append(cache.get(Path(str(row["path"]))))
        deltas.append(
            torch.from_numpy(np.load(corpus_dir / f"{row['img_id']}_delta.npy"))
        )
        captions.append(str(row["caption"]))
    return images, deltas, captions, corpus_config


def load_hubert_corpus(
    hubert_dir: Path, min_iou: float
) -> tuple[list[Tensor], list[Tensor], list[str], dict[str, object]]:
    """Load an ``(image, delta)`` corpus from a ``run_hubert --save-delta`` dir.

    The deltas were optimized on canonical square images at ``canonical_size`` with
    a uniform budget, so they drop straight into the generator's input space.
    """
    entries = discover_hubert_corpus_entries(hubert_dir, min_iou)
    if not entries:
        raise ValueError(
            f"No hubert deltas above --min-iou={min_iou} in {hubert_dir}. "
            "Did you run run_hubert_2plus2eq5_manipulation.py with --save-delta?"
        )
    sizes = {int(cast(int, e["canonical_size"])) for e in entries}
    if len(sizes) != 1:
        raise ValueError(f"Corpus mixes canonical sizes {sizes}; expected one.")
    size = sizes.pop()
    cache = CanonicalImageCache(size)
    images: list[Tensor] = []
    deltas: list[Tensor] = []
    captions: list[str] = []
    for entry in entries:
        images.append(cache.get(Path(str(entry["path"]))))
        deltas.append(torch.from_numpy(np.load(str(entry["delta_path"]))))
        captions.append(str(entry["caption"]))
    return images, deltas, captions, {"resized_image_size": size}


def _resolve_corpus(
    args: argparse.Namespace,
) -> tuple[list[Tensor], list[Tensor], list[str], dict[str, object]]:
    """Load the training corpus from a hubert output dir or the flat corpus dir."""
    if getattr(args, "hubert_dir", None):
        images, deltas, captions, cfg = load_hubert_corpus(
            args.hubert_dir, args.min_iou
        )
    else:
        images, deltas, captions, cfg = _load_corpus(args.corpus_dir, args.min_iou)
    max_corpus = getattr(args, "max_corpus", 0) or 0
    if max_corpus > 0:
        images = cap_corpus(images, max_corpus)
        deltas = cap_corpus(deltas, max_corpus)
        captions = cap_corpus(captions, max_corpus)
    return images, deltas, captions, cfg


def cmd_pretrain(args: argparse.Namespace) -> None:
    images, deltas, _captions, corpus_config = _resolve_corpus(args)
    if not images:
        raise ValueError(f"No corpus entries above --min-iou={args.min_iou}.")
    size = int(cast(int, corpus_config["resized_image_size"]))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)  # fail fast if not writable
    device = torch.device(args.device) if args.device != "auto" else _auto_device()
    gen_config = _generator_config(args)
    generator = PerturbationGenerator(gen_config)
    budget = build_uniform_budget(
        (size, size),
        max_epsilon=gen_config.max_epsilon,
        device=device,
        dtype=torch.float32,
    )
    print(
        f"Pretraining G ({count_parameters(generator):,} params) on {len(images)} deltas"
    )

    def _checkpoint(epoch: int, trace: list[dict[str, float]]) -> None:
        _save_generator_checkpoint(
            generator, output_dir, "g_pretrained.pt", trace, "pretrain_trace.json"
        )
        print(
            f"[checkpoint] epoch {epoch + 1}: saved g_pretrained.pt "
            f"(loss {trace[-1]['loss']:.5f})"
        )

    trace = pretrain_generator(
        generator,
        images,
        deltas,
        budget,
        gen_config,
        device,
        checkpoint_every=args.checkpoint_every,
        on_checkpoint=_checkpoint,
    )
    _save_generator_checkpoint(
        generator, output_dir, "g_pretrained.pt", trace, "pretrain_trace.json"
    )
    print(f"Saved g_pretrained.pt (final loss {trace[-1]['loss']:.5f})")


def cmd_finetune(args: argparse.Namespace) -> None:
    images, deltas, captions, corpus_config = _resolve_corpus(args)
    size = int(cast(int, corpus_config["resized_image_size"]))
    validate_resolution(size)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)  # fail fast before the slow load
    # Record corpus provenance so `eval` can reconstruct the exact training set
    # and keep the held-out (transfer) split disjoint from it (no leakage).
    (output_dir / "finetune_config.json").write_text(
        json.dumps(
            {
                "hubert_dir": str(args.hubert_dir) if args.hubert_dir else None,
                "min_iou": args.min_iou,
                "max_corpus": args.max_corpus,
                "resized_image_size": size,
                "num_train_images": len(images),
                "w_mask_scale": args.w_mask_scale,
                "w_perturb_scale": args.w_perturb_scale,
                "w_pred_scale": args.w_pred_scale,
                "finetune_lr": args.finetune_lr,
            },
            indent=2,
        )
    )
    pipeline = MetaClip2ExplanationPipeline(
        MetaClip2RunConfig(device=args.device, resized_image_size=size)
    )
    gen_config = _generator_config(args)
    generator = PerturbationGenerator(gen_config)
    if args.init and Path(args.init).exists():
        generator.load_state_dict(torch.load(args.init, map_location=pipeline.device))
        print(f"Initialized G from {args.init}")
    distribution_anchor = _maybe_anchor(pipeline, gen_config, args.anchor_stride)
    target = build_centered_target_mask(size // PATCH_SIZE, size // PATCH_SIZE)
    budget = build_uniform_budget(
        (size, size),
        max_epsilon=gen_config.max_epsilon,
        device=pipeline.device,
        dtype=torch.float32,
    )

    def _checkpoint(step: int, trace: list[dict[str, float]]) -> None:
        _save_generator_checkpoint(
            generator, output_dir, "g_finetuned.pt", trace, "finetune_trace.json"
        )
        print(
            f"[checkpoint] step {step + 1}: saved g_finetuned.pt "
            f"(loss {trace[-1]['loss']:.5f})"
        )

    trace = finetune_generator(
        generator,
        pipeline,
        images,
        captions,
        target,
        budget,
        gen_config,
        corpus_deltas=deltas,
        distribution_anchor=distribution_anchor,
        checkpoint_every=args.checkpoint_every,
        on_checkpoint=_checkpoint,
    )
    _save_generator_checkpoint(
        generator, output_dir, "g_finetuned.pt", trace, "finetune_trace.json"
    )
    print(f"Saved g_finetuned.pt (final loss {trace[-1]['loss']:.5f})")


def _resolve_eval_provenance(
    args: argparse.Namespace,
) -> tuple[Path | None, float, int]:
    """Resolve the training-corpus provenance for leakage-safe eval splits.

    Explicit CLI flags win; otherwise fall back to the ``finetune_config.json``
    written next to the generator so ``eval`` reconstructs the exact training set
    without the caller re-typing ``--hubert-dir/--min-iou/--max-corpus``.
    """
    hubert_dir = getattr(args, "hubert_dir", None)
    min_iou = getattr(args, "min_iou", 0.5)
    max_corpus = getattr(args, "max_corpus", 0)
    if hubert_dir is None:
        config_path = Path(args.generator).parent / "finetune_config.json"
        if config_path.exists():
            prov = json.loads(config_path.read_text())
            if prov.get("hubert_dir"):
                hubert_dir = Path(str(prov["hubert_dir"]))
                min_iou = float(prov.get("min_iou", min_iou))
                max_corpus = int(prov.get("max_corpus", max_corpus))
                print(f"Using training provenance from {config_path}")
    return hubert_dir, min_iou, max_corpus


def cmd_eval(args: argparse.Namespace) -> None:
    pipeline = MetaClip2ExplanationPipeline(
        MetaClip2RunConfig(
            device=args.device, resized_image_size=args.resized_image_size
        )
    )
    gen_config = _generator_config(args)
    generator = PerturbationGenerator(gen_config)
    generator.load_state_dict(torch.load(args.generator, map_location=pipeline.device))
    distribution_anchor = _maybe_anchor(pipeline, gen_config, args.anchor_stride)

    size = args.resized_image_size
    # Reconstruct the exact training set: its images are excluded from `transfer`
    # (clean held-out) and become the `train` split (true seen-image IoU).
    train_images: list[Tensor] = []
    train_captions: list[str] = []
    exclude_names: set[str] = set()
    hubert_dir, min_iou, max_corpus = _resolve_eval_provenance(args)
    if hubert_dir is not None:
        entries = cap_corpus(
            discover_hubert_corpus_entries(hubert_dir, min_iou), max_corpus
        )
        if not entries:
            raise ValueError(
                f"No hubert deltas above --min-iou={min_iou} in {hubert_dir}."
            )
        size = int(cast(int, entries[0]["canonical_size"]))
        exclude_names = {Path(str(e["path"])).name for e in entries}
        train_cache = CanonicalImageCache(size)
        seen = entries[: args.eval_train]
        train_images = [train_cache.get(Path(str(e["path"]))) for e in seen]
        train_captions = [str(e["caption"]) for e in seen]

    validate_resolution(size)
    cache = CanonicalImageCache(size)
    target = build_centered_target_mask(size // PATCH_SIZE, size // PATCH_SIZE)
    budget = build_uniform_budget(
        (size, size),
        max_epsilon=gen_config.max_epsilon,
        device=pipeline.device,
        dtype=torch.float32,
    )
    mcfg = gen_config.to_manipulation_config()

    pool, captions = _build_pool(args.input_dir, args.labels_csv, args.label_template)
    test_paths, train_pool = make_split(
        pool, seed_key=f"{args.seed}:0", test_size=args.test_size
    )
    transfer_paths = exclude_paths_by_name(test_paths, exclude_names)[: args.test_size]

    eval_sets: list[tuple[str, list[Tensor], list[str]]] = [
        (
            "transfer",
            cache.many(transfer_paths),
            [captions[str(p)] for p in transfer_paths],
        )
    ]
    if exclude_names:
        eval_sets.append(("train", train_images, train_captions))
    else:
        seen_paths = train_pool[: args.eval_train]
        eval_sets.append(
            ("train", cache.many(seen_paths), [captions[str(p)] for p in seen_paths])
        )

    records: list[dict[str, object]] = []
    for split, imgs, caps in eval_sets:
        if not imgs:
            continue
        per_image = evaluate_generator(
            generator,
            pipeline,
            imgs,
            caps,
            target,
            budget,
            mcfg,
            distribution_anchor=distribution_anchor,
        )
        avg = average_metrics(per_image)
        records.append({"seed": args.seed, "k": "G", "split": split, **avg})
        print(
            f"[{split}] mask_iou={avg['mask_iou_at_mean']:.3f} "
            f"cos={avg.get('cos_preserve', float('nan')):.3f} "
            f"top1={avg.get('top1_preserved', float('nan')):.3f} "
            f"ssim={avg.get('pert_ssim', float('nan')):.3f} (n={len(per_image)})"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(records, indent=2))
    (args.output_dir / "results_summary.json").write_text(
        json.dumps(aggregate_records(records), indent=2)
    )
    print(f"Eval written to {args.output_dir}")


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resized-image-size", type=int, default=512)
    parser.add_argument("--max-epsilon", type=float, default=30 / 255)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-residual-blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchor-stride", type=int, default=5)
    parser.add_argument(
        "--w-dist-kl",
        type=float,
        default=1.0,
        help=(
            "Weight of the prediction-distribution (KL) preservation term. Set 0 "
            "to skip the ImageNet anchor entirely (no network fetch; offline runs)."
        ),
    )
    parser.add_argument("--pretrain-lr", type=float, default=2e-4)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--pretrain-batch-size", type=int, default=8)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-steps", type=int, default=1500)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument(
        "--w-mask-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the mask (manipulation) loss weights. >1 makes the attack "
            "objective win the gradient competition against the perturbation "
            "penalty (default 1.0 == per-image manipulation3 balance)."
        ),
    )
    parser.add_argument(
        "--w-perturb-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the perturbation (imperceptibility) penalty weights. <1 "
            "stops the generator from collapsing delta -> 0 when amortizing at a "
            "small finetune LR (default 1.0 == per-image manipulation3 balance)."
        ),
    )
    parser.add_argument(
        "--w-pred-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the cosine prediction-preservation weight. >1 keeps the "
            "model's embedding (and top-1) from drifting as the mask term pushes "
            "the attack harder (default 1.0 == per-image manipulation3 balance)."
        ),
    )
    parser.add_argument(
        "--max-corpus",
        type=int,
        default=0,
        help=(
            "Use only the first N corpus images (0 == all). Each finetune step "
            "is a full pass over the corpus, so a small cap (e.g. 4-8) gives a "
            "fast overfit/sanity run before committing to the whole corpus."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help=(
            "Save the model + trace every N steps/epochs (atomic overwrite) so a "
            "crash/OOM/kill loses at most N steps. 0 disables (final save only)."
        ),
    )


def _add_dataset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--label-template", default="Photo of {label}")
    parser.add_argument("--test-size", type=int, default=50)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_corpus = sub.add_parser(
        "gen-corpus", help="Generate the supervised (image, delta) corpus."
    )
    _add_common(p_corpus)
    _add_dataset(p_corpus)
    p_corpus.add_argument("--num-corpus-images", type=int, default=200)
    p_corpus.add_argument("--steps", type=int, default=250)
    p_corpus.set_defaults(func=cmd_gen_corpus)

    p_pre = sub.add_parser(
        "pretrain", help="Supervised regression of G onto the corpus."
    )
    _add_common(p_pre)
    p_pre.add_argument("--corpus-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "corpus")
    p_pre.add_argument(
        "--hubert-dir",
        type=Path,
        default=None,
        help=(
            "Use a run_hubert ... --save-delta output dir as the corpus instead "
            "of --corpus-dir (reads per-image complete.json + delta.npy)."
        ),
    )
    p_pre.add_argument("--min-iou", type=float, default=0.5)
    p_pre.set_defaults(func=cmd_pretrain)

    p_fine = sub.add_parser("finetune", help="Second-order GAN-alike finetune of G.")
    _add_common(p_fine)
    p_fine.add_argument(
        "--corpus-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "corpus"
    )
    p_fine.add_argument(
        "--hubert-dir",
        type=Path,
        default=None,
        help="Use a run_hubert ... --save-delta output dir as the corpus.",
    )
    p_fine.add_argument("--min-iou", type=float, default=0.5)
    p_fine.add_argument(
        "--init", type=Path, default=DEFAULT_OUTPUT_DIR / "g_pretrained.pt"
    )
    p_fine.set_defaults(func=cmd_finetune)

    p_eval = sub.add_parser("eval", help="Apply frozen G to held-out images.")
    _add_common(p_eval)
    _add_dataset(p_eval)
    p_eval.add_argument(
        "--generator", type=Path, default=DEFAULT_OUTPUT_DIR / "g_finetuned.pt"
    )
    p_eval.add_argument(
        "--hubert-dir",
        type=Path,
        default=None,
        help=(
            "Training corpus dir, to exclude its images from the transfer split "
            "(no leakage) and report their seen-image IoU as the train split. "
            "Auto-read from finetune_config.json next to --generator if omitted."
        ),
    )
    p_eval.add_argument("--min-iou", type=float, default=0.5)
    p_eval.add_argument("--eval-train", type=int, default=50)
    p_eval.set_defaults(func=cmd_eval)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
