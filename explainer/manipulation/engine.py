from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from explainer.io import ensure_chw_image, load_image, pil_to_tensor
from explainer.manipulation.perceptual import (
    PerceptualMaskConfig,
    build_perceptual_allowance_map,
    perceptual_budget_map,
)
from explainer.manipulation.targets import prepare_target_explanation
from explainer.metaclip2 import MetaClip2ExplanationPipeline


@dataclass(frozen=True)
class DistributionAnchor:
    text_embeddings: Tensor = field(repr=False)
    logit_scale: Tensor = field(repr=False)
    labels: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True)
class ManipulationConfig:
    steps: int = 250
    lr: float = 7e-3
    max_epsilon: float = 30 / 255
    delta_init_std: float = 2.0
    log_every: int = 50
    seed: int = 0

    w_mask_l1: float = 1.0
    w_mask_l2: float = 0.0
    w_mask_focal_fg: float = 0.0
    w_mask_focal_bg: float = 0.0
    w_perturb_l1: float = 2.0
    w_perturb_l2: float = 5.0
    w_perturb_linf: float = 0.0
    w_pred_cos: float = 0.0
    w_sim_delta: float = 0.0
    w_dist_kl: float = 0.0
    dist_kl_temperature: float = 2.0

    # Criterion used to pick the returned ("best") iterate.
    # "last": the final iterate after all steps. This is the default: the
    #   optimization has converged by then, the perturbation is hard-capped by the
    #   budget, and reporting the converged result avoids cherry-picking a single
    #   favorable step.
    # "mask": the lowest mask-alignment-error iterate. Picks the most on-target
    #   mask within budget, at the cost of selection bias toward a lucky step.
    # "total": legacy behaviour — the full penalized loss. With heavy perturbation
    #   weights this degenerately selects the unperturbed first iterate (delta=0),
    #   whose perturbation penalty is zero, discarding every real manipulation.
    selection_metric: str = "last"

    perceptual_mask: PerceptualMaskConfig | None = None


@dataclass(frozen=True)
class ManipulationResult:
    config: ManipulationConfig
    caption: str
    metrics: dict[str, float]
    best_image: Tensor = field(repr=False)
    aligned_best_image: Tensor = field(repr=False)
    best_explanation: Tensor = field(repr=False)
    target_explanation: Tensor = field(repr=False)
    trace: tuple[dict[str, float], ...] = field(repr=False)
    perceptual_allowance: Tensor | None = field(default=None, repr=False)
    perturbation_budget: Tensor | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ManipulationProgressSnapshot:
    step: int
    loss: float
    mask_loss: float
    perturb_loss: float
    pred_loss: float
    original_image: Tensor = field(repr=False)
    current_image: Tensor = field(repr=False)
    current_explanation: Tensor = field(repr=False)
    target_explanation: Tensor = field(repr=False)
    image_embedding: Tensor = field(repr=False)
    original_embedding: Tensor = field(repr=False)
    similarity: Tensor = field(repr=False)
    original_similarity: Tensor = field(repr=False)
    delta: Tensor | None = field(default=None, repr=False)
    dist_loss: float = 0.0
    prediction_probs: Tensor | None = field(default=None, repr=False)
    baseline_prediction_probs: Tensor | None = field(default=None, repr=False)


ManipulationProgressCallback = Callable[[ManipulationProgressSnapshot], None]


class MetaClip2ManipulationEngine:
    def __init__(self, pipeline: MetaClip2ExplanationPipeline) -> None:
        self.pipeline = pipeline

    def run(
        self,
        image: Tensor | str | Path,
        caption: str,
        target_explanation: Tensor | str | Path,
        config: ManipulationConfig | None = None,
        progress: bool = True,
        progress_callback: ManipulationProgressCallback | None = None,
        callback_every: int | None = None,
        distribution_anchor: DistributionAnchor | None = None,
    ) -> ManipulationResult:
        effective_config = config or ManipulationConfig()
        callback_interval = (
            effective_config.log_every if callback_every is None else callback_every
        )
        if callback_interval <= 0:
            raise ValueError("callback_every must be positive when provided.")

        torch.manual_seed(effective_config.seed)

        device = self.pipeline.device
        original_image_tensor = self._load_image_tensor(image).to(device)

        baseline = self.pipeline.run(original_image_tensor, caption)
        aligned_original_image = baseline.image.detach().clone()
        original_embedding = baseline.image_embedding.detach()
        original_similarity = baseline.similarity.detach()
        explanation_shape = tuple(int(dim) for dim in baseline.explanation.shape)

        original_logits: Tensor | None = None
        baseline_probs_cpu: Tensor | None = None
        if distribution_anchor is not None:
            with torch.no_grad():
                original_logits = _anchor_logits(
                    original_embedding, distribution_anchor
                ).detach()
                baseline_probs_cpu = F.softmax(original_logits, dim=-1).detach().cpu()

        prepared_target = prepare_target_explanation(
            target_explanation=target_explanation,
            explanation_shape=explanation_shape,
            device=device,
            dtype=baseline.explanation.dtype,
        )
        perceptual_allowance = self._build_perceptual_allowance(
            image=original_image_tensor,
            config=effective_config,
        )
        perturbation_budget = self._build_perturbation_budget(
            config=effective_config,
            perceptual_allowance=perceptual_allowance,
            device=device,
            dtype=original_image_tensor.dtype,
            image_shape=tuple(original_image_tensor.shape[-2:]),
        )

        delta = torch.nn.Parameter(torch.zeros_like(original_image_tensor))
        optimizer = torch.optim.Adam([delta], lr=effective_config.lr)

        best_score = float("inf")
        best_loss = float("inf")
        best_image: Tensor | None = None
        trace: list[dict[str, float]] = []

        baseline_mask_loss = float(
            _mask_error(baseline.explanation, prepared_target, effective_config).item()
        )
        if progress_callback is not None:
            progress_callback(
                ManipulationProgressSnapshot(
                    step=0,
                    loss=baseline_mask_loss,
                    mask_loss=baseline_mask_loss,
                    perturb_loss=0.0,
                    pred_loss=0.0,
                    delta=None,
                    original_image=aligned_original_image.clone(),
                    current_image=aligned_original_image.clone(),
                    current_explanation=baseline.explanation.detach().clone(),
                    target_explanation=prepared_target.detach().clone(),
                    image_embedding=baseline.image_embedding.detach().clone(),
                    original_embedding=original_embedding.detach().clone(),
                    similarity=baseline.similarity.detach().clone(),
                    original_similarity=original_similarity.detach().clone(),
                    dist_loss=0.0,
                    prediction_probs=baseline_probs_cpu,
                    baseline_prediction_probs=baseline_probs_cpu,
                )
            )

        iterator = range(effective_config.steps)
        if progress:
            iterator = self._maybe_wrap_progress(iterator, "manipulation")

        for step in iterator:
            computed_delta = _bounded_delta(delta, perturbation_budget)
            input_image = torch.clamp(original_image_tensor + computed_delta, 0.0, 1.0)

            run = self.pipeline.run(input_image, caption)

            mask_loss = _mask_error(run.explanation, prepared_target, effective_config)
            perturb_loss = _perturbation_penalty(
                input_image,
                original_image_tensor,
                effective_config,
            )
            pred_loss = _prediction_penalty(
                run.image_embedding,
                run.similarity,
                original_embedding,
                original_similarity,
                effective_config,
            )
            dist_loss = _distribution_penalty(
                run.image_embedding,
                distribution_anchor,
                original_logits,
                effective_config,
            )
            loss = mask_loss + perturb_loss + pred_loss + dist_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            _set_progress_postfix(
                iterator,
                loss_value=loss_value,
                input_image=input_image,
                original_image=original_image_tensor,
            )
            selection_score = _selection_score(
                step=step,
                mask_loss=float(mask_loss.item()),
                total_loss=loss_value,
                config=effective_config,
            )
            if selection_score < best_score:
                best_score = selection_score
                best_loss = loss_value
                best_image = input_image.detach().clone()

            if (
                step % effective_config.log_every == 0
                or step == effective_config.steps - 1
            ):
                trace.append(
                    _build_trace_row(
                        step=step,
                        loss=loss_value,
                        mask_loss=float(mask_loss.item()),
                        perturb_loss=float(perturb_loss.item()),
                        pred_loss=float(pred_loss.item()),
                        dist_loss=float(dist_loss.item()),
                        input_image=input_image,
                        original_image=original_image_tensor,
                        image_embedding=run.image_embedding,
                        original_embedding=original_embedding,
                        similarity=run.similarity,
                        original_similarity=original_similarity,
                    )
                )

            completed_step = step + 1
            if progress_callback is not None and (
                completed_step % callback_interval == 0
                or completed_step == effective_config.steps
            ):
                callback_delta = (
                    _bounded_delta(delta, perturbation_budget).detach().clone()
                )
                callback_image = torch.clamp(
                    original_image_tensor + callback_delta,
                    0.0,
                    1.0,
                )
                callback_run = self.pipeline.run(callback_image, caption)
                callback_mask_loss = _mask_error(
                    callback_run.explanation,
                    prepared_target,
                    effective_config,
                )
                callback_perturb_loss = _perturbation_penalty(
                    callback_image,
                    original_image_tensor,
                    effective_config,
                )
                callback_pred_loss = _prediction_penalty(
                    callback_run.image_embedding,
                    callback_run.similarity,
                    original_embedding,
                    original_similarity,
                    effective_config,
                )
                callback_dist_loss = _distribution_penalty(
                    callback_run.image_embedding,
                    distribution_anchor,
                    original_logits,
                    effective_config,
                )
                callback_prediction_probs: Tensor | None = None
                if distribution_anchor is not None:
                    with torch.no_grad():
                        cb_logits = _anchor_logits(
                            callback_run.image_embedding, distribution_anchor
                        )
                        callback_prediction_probs = (
                            F.softmax(cb_logits, dim=-1).detach().cpu()
                        )
                progress_callback(
                    ManipulationProgressSnapshot(
                        step=completed_step,
                        loss=float(
                            (
                                callback_mask_loss
                                + callback_perturb_loss
                                + callback_pred_loss
                                + callback_dist_loss
                            ).item()
                        ),
                        mask_loss=float(callback_mask_loss.item()),
                        perturb_loss=float(callback_perturb_loss.item()),
                        pred_loss=float(callback_pred_loss.item()),
                        delta=callback_delta,
                        original_image=aligned_original_image.clone(),
                        current_image=callback_run.image.detach().clone(),
                        current_explanation=callback_run.explanation.detach().clone(),
                        target_explanation=prepared_target.detach().clone(),
                        image_embedding=callback_run.image_embedding.detach().clone(),
                        original_embedding=original_embedding.detach().clone(),
                        similarity=callback_run.similarity.detach().clone(),
                        original_similarity=original_similarity.detach().clone(),
                        dist_loss=float(callback_dist_loss.item()),
                        prediction_probs=callback_prediction_probs,
                        baseline_prediction_probs=baseline_probs_cpu,
                    )
                )

        assert best_image is not None

        final = self.pipeline.run(best_image, caption)
        final_explanation = final.explanation.detach()
        metrics = _compute_metrics(
            best_image=best_image,
            original_image=original_image_tensor,
            final_explanation=final_explanation,
            final_embedding=final.image_embedding.detach(),
            original_embedding=original_embedding,
            final_similarity=final.similarity.detach(),
            original_similarity=original_similarity,
            target_explanation=prepared_target,
            best_loss=best_loss,
            config=effective_config,
            distribution_anchor=distribution_anchor,
            original_logits=original_logits,
        )

        return ManipulationResult(
            config=effective_config,
            caption=caption,
            metrics=metrics,
            best_image=best_image,
            aligned_best_image=final.image.detach(),
            best_explanation=final_explanation,
            target_explanation=prepared_target,
            trace=tuple(trace),
            perceptual_allowance=perceptual_allowance,
            perturbation_budget=perturbation_budget,
        )

    def _build_perceptual_allowance(
        self,
        *,
        image: Tensor,
        config: ManipulationConfig,
    ) -> Tensor | None:
        if config.perceptual_mask is None:
            return None
        return build_perceptual_allowance_map(
            image,
            output_shape=tuple(image.shape[-2:]),
            config=config.perceptual_mask,
            device=image.device,
            dtype=image.dtype,
        )

    def _build_perturbation_budget(
        self,
        *,
        config: ManipulationConfig,
        perceptual_allowance: Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
        image_shape: tuple[int, int],
    ) -> Tensor:
        if perceptual_allowance is None:
            return torch.full(
                image_shape,
                fill_value=config.max_epsilon,
                device=device,
                dtype=dtype,
            )
        assert config.perceptual_mask is not None
        return perceptual_budget_map(
            perceptual_allowance,
            max_epsilon=config.max_epsilon,
            budget_max_scale=config.perceptual_mask.budget_max_scale,
        ).to(device=device, dtype=dtype)

    def _load_image_tensor(self, image: Tensor | str | Path) -> Tensor:
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


def _selection_score(
    *, step: int, mask_loss: float, total_loss: float, config: ManipulationConfig
) -> float:
    """Score used to choose the returned iterate; lower is better.

    See ``ManipulationConfig.selection_metric`` for why "last" is the default.
    """
    if config.selection_metric == "last":
        # Strictly decreasing in step, so the final iterate always wins.
        return -float(step)
    if config.selection_metric == "mask":
        return mask_loss
    if config.selection_metric == "total":
        return total_loss
    raise ValueError(
        f"Unknown selection_metric {config.selection_metric!r}; "
        "expected 'last', 'mask', or 'total'."
    )


def _mask_error(
    explanation: Tensor, target: Tensor, config: ManipulationConfig
) -> Tensor:
    weight = torch.ones_like(target)
    if config.w_mask_focal_fg > 0:
        weight = weight + config.w_mask_focal_fg * target
    if config.w_mask_focal_bg > 0:
        weight = weight + config.w_mask_focal_bg * (1.0 - target)

    loss = torch.zeros((), device=explanation.device, dtype=explanation.dtype)
    if config.w_mask_l1 > 0:
        loss = loss + config.w_mask_l1 * (weight * (explanation - target).abs()).mean()
    if config.w_mask_l2 > 0:
        loss = loss + config.w_mask_l2 * (weight * (explanation - target).pow(2)).mean()
    return loss


def _perturbation_penalty(
    input_image: Tensor,
    original_image: Tensor,
    config: ManipulationConfig,
) -> Tensor:
    delta = input_image - original_image
    loss = torch.zeros((), device=input_image.device, dtype=input_image.dtype)

    if config.w_perturb_l2 > 0:
        loss = loss + config.w_perturb_l2 * delta.pow(2).mean()
    if config.w_perturb_linf > 0:
        loss = loss + config.w_perturb_linf * delta.abs().amax()
    if config.w_perturb_l1 > 0:
        loss = loss + config.w_perturb_l1 * delta.abs().mean()
    return loss


def _bounded_delta(delta: Tensor, perturbation_budget: Tensor) -> Tensor:
    return perturbation_budget.unsqueeze(0) * (2.0 * torch.sigmoid(delta) - 1.0)


def _prediction_penalty(
    image_embedding: Tensor,
    similarity: Tensor,
    original_embedding: Tensor,
    original_similarity: Tensor,
    config: ManipulationConfig,
) -> Tensor:
    loss = torch.zeros((), device=image_embedding.device, dtype=image_embedding.dtype)
    if config.w_pred_cos > 0:
        cos = F.cosine_similarity(image_embedding, original_embedding, dim=-1).mean()
        loss = loss + config.w_pred_cos * (1.0 - cos)
    if config.w_sim_delta > 0:
        loss = loss + config.w_sim_delta * (similarity - original_similarity).abs()
    return loss


def _anchor_logits(image_embedding: Tensor, anchor: DistributionAnchor) -> Tensor:
    return (anchor.logit_scale * (image_embedding @ anchor.text_embeddings.T)).squeeze(
        0
    )


def _distribution_penalty(
    image_embedding: Tensor,
    anchor: DistributionAnchor | None,
    original_logits: Tensor | None,
    config: ManipulationConfig,
) -> Tensor:
    if anchor is None or original_logits is None or config.w_dist_kl <= 0.0:
        return torch.zeros(
            (), device=image_embedding.device, dtype=image_embedding.dtype
        )
    temperature = float(config.dist_kl_temperature)
    current_logits = _anchor_logits(image_embedding, anchor)
    teacher_probs = F.softmax(original_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(current_logits / temperature, dim=-1)
    kl = F.kl_div(
        student_log_probs.unsqueeze(0),
        teacher_probs.unsqueeze(0),
        reduction="batchmean",
    )
    return config.w_dist_kl * (temperature**2) * kl


def _build_trace_row(
    *,
    step: int,
    loss: float,
    mask_loss: float,
    perturb_loss: float,
    pred_loss: float,
    dist_loss: float,
    input_image: Tensor,
    original_image: Tensor,
    image_embedding: Tensor,
    original_embedding: Tensor,
    similarity: Tensor,
    original_similarity: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        cos_pres = _cosine(image_embedding, original_embedding)
        sim_delta_val = float((similarity - original_similarity).abs().item())
        linf = float((input_image - original_image).abs().amax().item())
    return {
        "step": float(step),
        "loss": loss,
        "mask_loss": mask_loss,
        "perturb_loss": perturb_loss,
        "pred_loss": pred_loss,
        "dist_loss": dist_loss,
        "cos_preserve": cos_pres,
        "sim_delta": sim_delta_val,
        "linf": linf,
    }


def _set_progress_postfix(
    iterator: object,
    *,
    loss_value: float,
    input_image: Tensor,
    original_image: Tensor,
) -> None:
    if not hasattr(iterator, "set_postfix"):
        return
    linf = float((input_image - original_image).abs().amax().item())
    iterator.set_postfix(  # type: ignore[attr-defined]
        {
            "loss": f"{loss_value:.4f}",
            "linf": f"{linf:.4f}",
        }
    )


def _compute_metrics(
    *,
    best_image: Tensor,
    original_image: Tensor,
    final_explanation: Tensor,
    final_embedding: Tensor,
    original_embedding: Tensor,
    final_similarity: Tensor,
    original_similarity: Tensor,
    target_explanation: Tensor,
    best_loss: float,
    config: ManipulationConfig | None = None,
    distribution_anchor: DistributionAnchor | None = None,
    original_logits: Tensor | None = None,
) -> dict[str, float]:
    diff = (best_image - original_image).detach()
    adv_above_mean = (final_explanation > final_explanation.mean()).float()
    target_binary = (target_explanation > 0.5).float()
    intersection = (adv_above_mean * target_binary).sum()
    union = torch.clamp(adv_above_mean + target_binary, max=1.0).sum()
    iou = float((intersection / union).item()) if union.item() > 0 else 0.0

    metrics: dict[str, float] = {
        "mask_l1": float((final_explanation - target_explanation).abs().mean().item()),
        "mask_iou_at_mean": iou,
        "pert_l2_rms": float(diff.pow(2).mean().sqrt().item()),
        "pert_linf": float(diff.abs().amax().item()),
        "pert_psnr": _psnr(best_image.clamp(0.0, 1.0), original_image),
        "pert_ssim": _ssim(best_image.clamp(0.0, 1.0), original_image),
        "cos_preserve": _cosine(final_embedding, original_embedding),
        "sim_delta": float((final_similarity - original_similarity).abs().item()),
        "sim_before": float(original_similarity.item()),
        "sim_after": float(final_similarity.item()),
        "best_loss": best_loss,
    }

    if distribution_anchor is not None and original_logits is not None:
        with torch.no_grad():
            final_logits = _anchor_logits(final_embedding, distribution_anchor)
            t = float(config.dist_kl_temperature) if config is not None else 2.0
            teacher_probs_t = F.softmax(original_logits / t, dim=-1)
            student_log_t = F.log_softmax(final_logits / t, dim=-1)
            metrics["dist_kl"] = float(
                (t**2)
                * F.kl_div(
                    student_log_t.unsqueeze(0),
                    teacher_probs_t.unsqueeze(0),
                    reduction="batchmean",
                ).item()
            )
            teacher_probs_unit = F.softmax(original_logits, dim=-1)
            student_log_unit = F.log_softmax(final_logits, dim=-1)
            metrics["dist_kl_T1"] = float(
                F.kl_div(
                    student_log_unit.unsqueeze(0),
                    teacher_probs_unit.unsqueeze(0),
                    reduction="batchmean",
                ).item()
            )

    return metrics


def _cosine(a: Tensor, b: Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=-1).mean().item())


def _psnr(a: Tensor, b: Tensor) -> float:
    mse = (a - b).pow(2).mean()
    if mse.item() < 1e-12:
        return float("inf")
    return float((20.0 * torch.log10(1.0 / mse.sqrt())).item())


def _ssim(a: Tensor, b: Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
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

    def conv(x: Tensor) -> Tensor:
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
