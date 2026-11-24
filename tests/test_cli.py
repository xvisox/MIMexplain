from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import torch

from explainer import cli


def test_main_saves_overlay_and_explanation(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    explanation = torch.tensor([[0.1, 0.3], [0.5, 0.7]], dtype=torch.float32)
    result = SimpleNamespace(
        image=torch.full((3, 4, 4), 0.5, dtype=torch.float32),
        explanation=explanation,
        map_size=(2, 2),
    )

    args = argparse.Namespace(
        example="fake",
        output_dir=tmp_path,
        device="cpu",
        text_layers=4,
        without_key_sim=False,
        resized_image_size=224,
        model_name="fake-model",
    )

    class FakePipeline:
        def __init__(self, config) -> None:
            self.config = config

        def run_example(self, name: str):
            assert name == "fake"
            return result

    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(cli, "MetaClip2ExplanationPipeline", FakePipeline)

    cli.main()

    overlay_path = tmp_path / "explanation_fake.png"
    array_path = tmp_path / "explanation_fake.npy"
    assert overlay_path.exists()
    np.testing.assert_allclose(np.load(array_path), explanation.numpy())

    captured = capsys.readouterr()
    assert "example=fake" in captured.out
    assert "map_size=2x2" in captured.out
    assert f"overlay={overlay_path}" in captured.out
    assert f"array={array_path}" in captured.out
