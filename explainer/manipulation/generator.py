"""A small, input-conditioned perturbation generator (the Q3 "smaller network").

Where :class:`~explainer.manipulation.engine.MetaClip2ManipulationEngine`
optimizes a fresh per-image ``delta`` and
:class:`~explainer.manipulation.universal.UniversalManipulationEngine` optimizes
one *shared* ``delta``, this module learns a network ``G`` that *maps an image to
its perturbation*: ``delta = G(image)``. The hypothesis is that an amortized,
input-conditioned attacker can succeed on unseen images where a single universal
mask fails (held-out IoU stuck at chance), by producing a different effective
delta per image in one forward pass.

This is the AdvGAN generator without the discriminator (a documented future
ablation): ``G`` is trained against the *same* second-order Grad-ECLIP objective
used by the other two engines, so its loss terms and bounded-delta application
are byte-for-byte the per-image objective. Two-stage training (cheap supervised
pre-training to regress precomputed deltas, then second-order finetuning) lives
in ``scripts/run_adversarial_generator.py``; the reusable nets, the bounded
application, and the config mapping live here and are unit-tested without the
model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from explainer.manipulation.engine import ManipulationConfig, _bounded_delta


def apply_generator_delta(raw_delta: Tensor, budget: Tensor, image: Tensor) -> Tensor:
    """Apply a generator's raw output as a budget-bounded perturbation.

    Identical construction to
    :func:`~explainer.manipulation.universal.apply_universal_delta`: the raw
    network output is squashed through ``budget * (2*sigmoid(delta) - 1)`` so the
    L-inf magnitude never exceeds ``max_epsilon`` before clamping back to
    ``[0, 1]``. Reusing the same bound keeps the generator directly comparable to
    the per-image and universal engines.
    """
    bounded = _bounded_delta(raw_delta, budget)
    return torch.clamp(image + bounded, 0.0, 1.0)


def count_parameters(module: nn.Module) -> int:
    """Total number of parameters (trainable or not) in a module."""
    return sum(parameter.numel() for parameter in module.parameters())


def supervised_delta_loss(
    bounded_pred: Tensor,
    target_delta: Tensor,
    *,
    w_l1: float = 1.0,
    w_l2: float = 1.0,
) -> Tensor:
    """``w_l1 * L1 + w_l2 * L2`` between the bounded predicted delta and a target.

    The prediction is the *bounded* delta (what actually gets applied), so the
    network learns to reproduce the perturbation as it will be used, not its raw
    pre-sigmoid parameterization.
    """
    l1 = (bounded_pred - target_delta).abs().mean()
    l2 = (bounded_pred - target_delta).pow(2).mean()
    return w_l1 * l1 + w_l2 * l2


@dataclass(frozen=True)
class GeneratorConfig:
    """Architecture + training hyperparameters for the perturbation generator.

    The ``w_*`` / ``max_epsilon`` / ``dist_kl_temperature`` fields mirror the
    per-image "manipulation3" weights so :meth:`to_manipulation_config` can hand
    the engine helpers the exact same objective during second-order finetuning.
    """

    # Architecture
    base_channels: int = 32
    num_residual_blocks: int = 4
    max_epsilon: float = 30 / 255

    # Stage 1: supervised pre-training (no MetaCLIP, no second order)
    pretrain_lr: float = 2e-4
    pretrain_epochs: int = 30
    pretrain_batch_size: int = 8
    w_sup_l1: float = 1.0
    w_sup_l2: float = 1.0

    # Stage 2: second-order finetuning
    finetune_lr: float = 1e-4
    finetune_steps: int = 1500
    micro_batch: int = 2
    w_supervised_anchor: float = 0.1

    # Shared manipulation objective (manipulation3 weights)
    w_mask_l1: float = 1.5
    w_mask_l2: float = 3.0
    w_mask_focal_fg: float = 3.0
    w_perturb_l1: float = 50.0
    w_perturb_l2: float = 10.0
    w_perturb_linf: float = 50.0
    w_pred_cos: float = 1.0
    w_dist_kl: float = 1.0
    dist_kl_temperature: float = 2.0

    # Global rebalancing knobs (1.0 == the per-image "manipulation3" balance).
    # Amortizing across images at a small finetune LR, the per-image balance lets
    # the heavy perturbation penalty win and drive delta -> 0 (mask frozen at
    # chance IoU). Scale the perturbation penalty down and/or the mask term up so
    # the manipulation objective wins the gradient competition.
    w_mask_scale: float = 1.0
    w_perturb_scale: float = 1.0
    # Cutting w_perturb_scale frees delta to grow, which (unlike the per-image
    # attack, where the heavy perturb penalty kept delta -- and thus the
    # embedding drift -- small) lets the prediction drift. Scale the cosine
    # prediction-preservation term up to recover it independently of delta size.
    w_pred_scale: float = 1.0

    seed: int = 0

    def to_manipulation_config(self) -> ManipulationConfig:
        """Build the engine ``ManipulationConfig`` for the shared loss terms."""
        return ManipulationConfig(
            max_epsilon=self.max_epsilon,
            w_mask_l1=self.w_mask_l1 * self.w_mask_scale,
            w_mask_l2=self.w_mask_l2 * self.w_mask_scale,
            w_mask_focal_fg=self.w_mask_focal_fg * self.w_mask_scale,
            w_perturb_l1=self.w_perturb_l1 * self.w_perturb_scale,
            w_perturb_l2=self.w_perturb_l2 * self.w_perturb_scale,
            w_perturb_linf=self.w_perturb_linf * self.w_perturb_scale,
            w_pred_cos=self.w_pred_cos * self.w_pred_scale,
            w_dist_kl=self.w_dist_kl,
            dist_kl_temperature=self.dist_kl_temperature,
            seed=self.seed,
        )


class ResidualBlock(nn.Module):
    """Conv-IN-ReLU-Conv-IN with an identity skip (AdvGAN bottleneck block)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, padding_mode="reflect"
            ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, padding_mode="reflect"
            ),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class PerturbationGenerator(nn.Module):
    """Map an image ``[B,3,H,W]`` in ``[0,1]`` to a raw delta ``[B,3,H,W]``.

    Encoder -> residual bottleneck -> decoder, all from raw ``torch.nn`` (no
    torchvision). The output is the *raw* (pre-sigmoid) delta; the L-inf bound is
    applied externally by :func:`apply_generator_delta`, so the bounding math is
    identical to the per-image and universal engines.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        base = config.base_channels

        def conv(in_ch: int, out_ch: int, k: int, stride: int, pad: int) -> nn.Conv2d:
            return nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=k,
                stride=stride,
                padding=pad,
                padding_mode="reflect",
            )

        self.encoder = nn.Sequential(
            conv(3, base, 7, 1, 3),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True),
            conv(base, base * 2, 3, 2, 1),
            nn.InstanceNorm2d(base * 2),
            nn.ReLU(inplace=True),
            conv(base * 2, base * 4, 3, 2, 1),
            nn.InstanceNorm2d(base * 4),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(base * 4) for _ in range(config.num_residual_blocks)]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                base * 4, base * 2, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base * 2, base, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 3, kernel_size=7, padding=3, padding_mode="reflect"),
        )

    def forward(self, image: Tensor) -> Tensor:
        encoded = self.encoder(image)
        bottlenecked = self.bottleneck(encoded)
        return self.decoder(bottlenecked)
