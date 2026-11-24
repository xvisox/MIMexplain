"""Train a single *shared* perturbation across many images.

Where :class:`~explainer.manipulation.engine.MetaClip2ManipulationEngine`
optimizes a fresh per-image ``delta``, this module optimizes **one** ``delta``
jointly over a set of images so it can be applied, frozen, to *unseen* images.
It is the core of the universal / transferable explanation-mask experiment: does
a single perturbation force an arbitrary image's Grad-ECLIP explanation to a
fixed target?

Every loss and metric is reused from ``engine`` so the per-image and universal
paths share one objective definition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from explainer.io import ensure_chw_image, load_image, pil_to_tensor
from explainer.manipulation.engine import (
    DistributionAnchor,
    ManipulationConfig,
    _anchor_logits,
    _bounded_delta,
    _compute_metrics,
    _distribution_penalty,
    _mask_error,
    _perturbation_penalty,
    _prediction_penalty,
)
from explainer.manipulation.targets import prepare_target_explanation
from explainer.metaclip2 import MetaClip2ExplanationPipeline

ImageInput = Tensor | str | Path


def build_uniform_budget(
    image_shape: tuple[int, int],
    *,
    max_epsilon: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Per-pixel L-inf budget shared by every image.

    A *shared* perturbation cannot use a per-image perceptual budget, so the
    universal path always uses a uniform ``max_epsilon`` cap. Shape matches the
    2D convention expected by :func:`_bounded_delta` (it ``unsqueeze(0)``-es the
    budget to broadcast over the channel dimension).
    """
    return torch.full(image_shape, fill_value=max_epsilon, device=device, dtype=dtype)


def _map_shape(explanation: Tensor) -> tuple[int, int]:
    return int(explanation.shape[0]), int(explanation.shape[1])


def _hw(image: Tensor) -> tuple[int, int]:
    return int(image.shape[-2]), int(image.shape[-1])


def average_metrics(per_image: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric key across a list of per-image metric dicts."""
    if not per_image:
        raise ValueError("Cannot average an empty metrics list.")
    totals: dict[str, float] = {}
    for metrics in per_image:
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / len(per_image) for key, value in totals.items()}


def apply_universal_delta(delta: Tensor, budget: Tensor, image: Tensor) -> Tensor:
    """Apply a shared, budget-bounded perturbation to one image.

    Mirrors the per-step construction in the per-image engine:
    ``clamp(image + bounded_delta, 0, 1)``. Because ``_bounded_delta`` maps the
    raw parameter through ``budget * (2*sigmoid(delta) - 1)``, the perturbation
    magnitude never exceeds ``budget`` (i.e. ``max_epsilon``) before clamping.
    """
    bounded = _bounded_delta(delta, budget)
    return torch.clamp(image + bounded, 0.0, 1.0)


@dataclass(frozen=True)
class UniversalManipulationResult:
    config: ManipulationConfig
    delta: Tensor = field(repr=False)
    perturbation_budget: Tensor = field(repr=False)
    target_explanation: Tensor = field(repr=False)
    train_metrics: dict[str, float]
    train_per_image: tuple[dict[str, float], ...] = field(repr=False)
    trace: tuple[dict[str, float], ...] = field(repr=False)


@dataclass(frozen=True)
class _Baseline:
    """Cached unperturbed forward pass for one training/eval image."""

    image: Tensor
    caption: str
    embedding: Tensor
    similarity: Tensor
    original_logits: Tensor | None = None


class UniversalManipulationEngine:
    """Optimize one shared perturbation across a list of (image, caption) pairs."""

    def __init__(self, pipeline: MetaClip2ExplanationPipeline) -> None:
        self.pipeline = pipeline

    def train(
        self,
        images: Sequence[ImageInput],
        captions: Sequence[str],
        target_explanation: Tensor | str | Path,
        config: ManipulationConfig | None = None,
        *,
        batch_size: int = 4,
        progress: bool = True,
        distribution_anchor: DistributionAnchor | None = None,
    ) -> UniversalManipulationResult:
        effective_config = config or ManipulationConfig()
        if len(images) == 0:
            raise ValueError("train() requires at least one image.")
        if len(images) != len(captions):
            raise ValueError("images and captions must have the same length.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        device = self.pipeline.device
        image_tensors = self._load_aligned_images(images, device)
        baselines = [
            self._baseline(image_tensors[i], captions[i], distribution_anchor)
            for i in range(len(image_tensors))
        ]

        first_baseline_run = self.pipeline.run(image_tensors[0], captions[0])
        prepared_target = prepare_target_explanation(
            target_explanation=target_explanation,
            explanation_shape=_map_shape(first_baseline_run.explanation),
            device=device,
            dtype=first_baseline_run.explanation.dtype,
        )

        budget = build_uniform_budget(
            _hw(image_tensors[0]),
            max_epsilon=effective_config.max_epsilon,
            device=device,
            dtype=image_tensors[0].dtype,
        )

        torch.manual_seed(effective_config.seed)

        delta = torch.nn.Parameter(torch.zeros_like(image_tensors[0]))
        optimizer = torch.optim.Adam([delta], lr=effective_config.lr)

        trace: list[dict[str, float]] = []
        count = len(image_tensors)
        # Every optimizer step is a full pass over all K images, then one update,
        # so the gradient is the exact mean over the whole training set (the
        # fairest objective for the K-sweep). The pass is split into
        # gradient-accumulation micro-batches of ``batch_size``: each micro-batch
        # sums its image losses and calls ``.backward()`` once (fewer backward
        # calls), accumulating into ``delta.grad``. Because the Grad-ECLIP map is
        # second-order (``create_graph=True``), at most ``batch_size`` image graphs
        # are alive per backward, so ``batch_size`` is the memory knob: keep it
        # small (default 4) to stay within GPU memory; larger holds more graphs.
        micro = min(batch_size, count)

        iterator: range = range(effective_config.steps)
        if progress:
            iterator = self._maybe_wrap_progress(iterator, "universal")

        for step in iterator:
            optimizer.zero_grad()
            step_losses = {
                "loss": 0.0,
                "mask": 0.0,
                "perturb": 0.0,
                "pred": 0.0,
                "dist": 0.0,
            }
            for chunk_start in range(0, count, micro):
                chunk = range(chunk_start, min(chunk_start + micro, count))
                chunk_loss: Tensor | None = None
                for index in chunk:
                    input_image = apply_universal_delta(
                        delta, budget, image_tensors[index]
                    )
                    run = self.pipeline.run(input_image, captions[index])
                    mask_loss = _mask_error(
                        run.explanation, prepared_target, effective_config
                    )
                    perturb_loss = _perturbation_penalty(
                        input_image, image_tensors[index], effective_config
                    )
                    pred_loss = _prediction_penalty(
                        run.image_embedding,
                        run.similarity,
                        baselines[index].embedding,
                        baselines[index].similarity,
                        effective_config,
                    )
                    dist_loss = _distribution_penalty(
                        run.image_embedding,
                        distribution_anchor,
                        baselines[index].original_logits,
                        effective_config,
                    )
                    image_loss = mask_loss + perturb_loss + pred_loss + dist_loss
                    chunk_loss = (
                        image_loss if chunk_loss is None else chunk_loss + image_loss
                    )
                    step_losses["loss"] += float(image_loss.item()) / count
                    step_losses["mask"] += float(mask_loss.item()) / count
                    step_losses["perturb"] += float(perturb_loss.item()) / count
                    step_losses["pred"] += float(pred_loss.item()) / count
                    step_losses["dist"] += float(dist_loss.item()) / count
                assert chunk_loss is not None
                # One backward per micro-batch; /count makes the accumulated
                # gradient the mean over all K images.
                (chunk_loss / count).backward()
            optimizer.step()

            if (
                step % effective_config.log_every == 0
                or step == effective_config.steps - 1
            ):
                trace.append({"step": float(step), **step_losses})

        train_per_image = self.evaluate_per_image(
            delta.detach(),
            budget,
            image_tensors,
            captions,
            prepared_target,
            effective_config,
            baselines=baselines,
            distribution_anchor=distribution_anchor,
        )

        return UniversalManipulationResult(
            config=effective_config,
            delta=delta.detach(),
            perturbation_budget=budget,
            target_explanation=prepared_target,
            train_metrics=average_metrics(train_per_image),
            train_per_image=tuple(train_per_image),
            trace=tuple(trace),
        )

    def evaluate_per_image(
        self,
        delta: Tensor,
        budget: Tensor,
        images: Sequence[ImageInput],
        captions: Sequence[str],
        target_explanation: Tensor | str | Path,
        config: ManipulationConfig | None = None,
        *,
        baselines: Sequence[_Baseline] | None = None,
        distribution_anchor: DistributionAnchor | None = None,
    ) -> list[dict[str, float]]:
        """Apply a frozen ``delta`` to each image, returning per-image metrics.

        One metrics dict per image, in input order, so callers can attach an id
        and persist the full distribution (the meaningful ``n`` for analysis).
        Metrics come from the shared :func:`_compute_metrics` so they match the
        per-image engine exactly. :meth:`evaluate` averages these. Passing a
        ``distribution_anchor`` adds the ``dist_kl`` / ``dist_kl_T1`` metrics.
        """
        effective_config = config or ManipulationConfig()
        if len(images) == 0:
            raise ValueError("evaluate() requires at least one image.")
        if len(images) != len(captions):
            raise ValueError("images and captions must have the same length.")

        device = self.pipeline.device
        frozen_delta = delta.detach().to(device)
        budget = budget.to(device)
        image_tensors = self._load_aligned_images(images, device)

        if baselines is None:
            baselines = [
                self._baseline(image_tensors[i], captions[i], distribution_anchor)
                for i in range(len(image_tensors))
            ]

        if isinstance(target_explanation, torch.Tensor):
            prepared_target = target_explanation.to(device)
        else:
            sample_run = self.pipeline.run(image_tensors[0], captions[0])
            prepared_target = prepare_target_explanation(
                target_explanation=target_explanation,
                explanation_shape=_map_shape(sample_run.explanation),
                device=device,
                dtype=sample_run.explanation.dtype,
            )

        per_image: list[dict[str, float]] = []
        for index, baseline in enumerate(baselines):
            perturbed = apply_universal_delta(
                frozen_delta, budget, image_tensors[index]
            )
            run = self.pipeline.run(perturbed, captions[index])
            per_image.append(
                _compute_metrics(
                    best_image=perturbed.detach(),
                    original_image=image_tensors[index],
                    final_explanation=run.explanation.detach(),
                    final_embedding=run.image_embedding.detach(),
                    original_embedding=baseline.embedding,
                    final_similarity=run.similarity.detach(),
                    original_similarity=baseline.similarity,
                    target_explanation=prepared_target,
                    best_loss=0.0,
                    config=effective_config,
                    distribution_anchor=distribution_anchor,
                    original_logits=baseline.original_logits,
                )
            )
        return per_image

    def evaluate(
        self,
        delta: Tensor,
        budget: Tensor,
        images: Sequence[ImageInput],
        captions: Sequence[str],
        target_explanation: Tensor | str | Path,
        config: ManipulationConfig | None = None,
        *,
        baselines: Sequence[_Baseline] | None = None,
        distribution_anchor: DistributionAnchor | None = None,
    ) -> dict[str, float]:
        """Apply a frozen ``delta`` to each image and average the metrics.

        This is the transfer measurement: ``images`` may be held-out examples the
        ``delta`` was never trained on. Thin mean over :meth:`evaluate_per_image`.
        """
        per_image = self.evaluate_per_image(
            delta,
            budget,
            images,
            captions,
            target_explanation,
            config,
            baselines=baselines,
            distribution_anchor=distribution_anchor,
        )
        return average_metrics(per_image)

    def _baseline(
        self,
        image: Tensor,
        caption: str,
        distribution_anchor: DistributionAnchor | None = None,
    ) -> _Baseline:
        run = self.pipeline.run(image, caption)
        embedding = run.image_embedding.detach()
        original_logits: Tensor | None = None
        if distribution_anchor is not None:
            with torch.no_grad():
                original_logits = _anchor_logits(
                    embedding, distribution_anchor
                ).detach()
        return _Baseline(
            image=image,
            caption=caption,
            embedding=embedding,
            similarity=run.similarity.detach(),
            original_logits=original_logits,
        )

    def _load_aligned_images(
        self, images: Sequence[ImageInput], device: torch.device
    ) -> list[Tensor]:
        tensors = [self._load_image_tensor(image).to(device) for image in images]
        reference_shape = tuple(tensors[0].shape)
        for tensor in tensors[1:]:
            if tuple(tensor.shape) != reference_shape:
                raise ValueError(
                    "A shared perturbation requires identically shaped images; "
                    f"got {tuple(tensor.shape)} vs {reference_shape}. Resize all "
                    "images to a common square size before calling the engine."
                )
        return tensors

    def _load_image_tensor(self, image: ImageInput) -> Tensor:
        if isinstance(image, torch.Tensor):
            return ensure_chw_image(image)
        return ensure_chw_image(pil_to_tensor(load_image(image)))

    @staticmethod
    def _maybe_wrap_progress(iterator: range, desc: str) -> range:
        try:
            import tqdm  # type: ignore[import-untyped]

            return tqdm.tqdm(iterator, desc=desc)
        except ImportError:
            return iterator
