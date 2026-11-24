from __future__ import annotations

import pytest
import torch

from explainer.metaclip2 import choose_device, move_batch_to_device


def test_choose_device_explicit_cpu() -> None:
    assert choose_device("cpu") == torch.device("cpu")


def test_choose_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert choose_device("auto").type == "cuda"


def test_choose_device_auto_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert choose_device("auto").type == "cpu"


def test_move_batch_to_device_filters_non_tensors() -> None:
    device = torch.device("cpu")
    batch = {
        "pixel_values": torch.ones(1, 3),
        "caption": "not a tensor",
        "count": 42,
    }
    out = move_batch_to_device(batch, device)

    assert set(out.keys()) == {"pixel_values"}
    assert out["pixel_values"].device == device
