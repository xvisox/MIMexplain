from __future__ import annotations

import numpy as np
import pytest
import torch

from explainer.gradeclip import (
    _build_text_attention_mask,
    _find_eos_positions,
    _infer_patch_map_size,
    _make_causal_mask,
    _normalize_map,
)


def test_infer_patch_map_size_square() -> None:
    assert _infer_patch_map_size(196) == (14, 14)
    assert _infer_patch_map_size(3844) == (62, 62)


def test_infer_patch_map_size_non_square_raises() -> None:
    with pytest.raises(ValueError, match="square patch grid"):
        _infer_patch_map_size(197)


def test_normalize_map_unit_interval() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = _normalize_map(arr)
    assert np.isclose(out.min(), 0.0)
    assert np.isclose(out.max(), 1.0)
    assert out[0, 0] < out[0, 1] < out[1, 0] < out[1, 1]


def test_normalize_map_constant_input_eps_guard() -> None:
    arr = np.full((3, 3), 5.0)
    out = _normalize_map(arr)
    assert np.isfinite(out).all()
    assert np.allclose(out, 0.0)


def test_explanation_loss_backpropagates_through_created_gradient_graph() -> None:
    image = torch.tensor([[0.2, -0.3], [0.4, -0.5]], requires_grad=True)
    features = image.square()
    score = features.pow(2).sum()
    explanation = torch.autograd.grad(
        score, features, retain_graph=True, create_graph=True
    )[0]

    target = torch.zeros_like(explanation)
    loss = (explanation - target).pow(2).mean()
    loss.backward()

    assert image.grad is not None
    assert image.grad.abs().sum().item() > 0


def test_explanation_loss_without_created_gradient_graph_cannot_backpropagate() -> None:
    image = torch.tensor([[0.2, -0.3], [0.4, -0.5]], requires_grad=True)
    features = image.square()
    score = features.pow(2).sum()
    explanation = torch.autograd.grad(
        score, features, retain_graph=True, create_graph=False
    )[0]

    target = torch.zeros_like(explanation)
    loss = (explanation - target).pow(2).mean()

    assert not loss.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward()


def test_make_causal_mask_shape_and_upper_tri_infs() -> None:
    mask = _make_causal_mask(2, 4, torch.float32, torch.device("cpu"))

    assert mask.shape == (2, 4, 4)
    assert torch.equal(mask[0], mask[1])

    expected_upper = torch.tensor(
        [
            [0.0, float("-inf"), float("-inf"), float("-inf")],
            [0.0, 0.0, float("-inf"), float("-inf")],
            [0.0, 0.0, 0.0, float("-inf")],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(mask[0], expected_upper)


def test_build_text_attention_mask_no_padding() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    built = _build_text_attention_mask(
        input_ids, None, torch.float32, torch.device("cpu")
    )
    causal = _make_causal_mask(1, 4, torch.float32, torch.device("cpu"))
    assert torch.equal(built, causal)


def test_build_text_attention_mask_with_padding() -> None:
    input_ids = torch.tensor([[1, 2, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    built = _build_text_attention_mask(
        input_ids, attention_mask, torch.float32, torch.device("cpu")
    )

    finite_min = torch.finfo(torch.float32).min
    blocked_threshold = finite_min / 2.0

    assert (built[0, :, 3] <= blocked_threshold).all()
    assert built[0, 0, 0].item() == 0.0
    assert built[0, 1, 1].item() == 0.0
    assert built[0, 2, 2].item() == 0.0
    assert built[0, 3, 3].item() <= blocked_threshold
    assert torch.isneginf(built[0, 0, 1])
    assert built[0, 1, 0].item() == 0.0
    assert built[0, 3, 2].item() == 0.0


def test_find_eos_positions_with_eos() -> None:
    input_ids = torch.tensor([[1, 2, 49407, 49407]])
    positions = _find_eos_positions(input_ids, eos_token_id=49407)
    assert positions.tolist() == [2]


def test_find_eos_positions_no_eos_fallback() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    positions = _find_eos_positions(input_ids, eos_token_id=49407)
    assert positions.tolist() == [3]


def test_find_eos_positions_none_token_id() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    positions = _find_eos_positions(input_ids, eos_token_id=None)
    assert positions.tolist() == [3]
