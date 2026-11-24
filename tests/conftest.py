from __future__ import annotations

from pathlib import Path

import pytest

from explainer.constants import EXAMPLES_DIR
from explainer.examples import ExampleSpec, get_example
from explainer.metaclip2 import MetaClip2ExplanationPipeline, MetaClip2RunConfig


@pytest.fixture
def example_image_path() -> Path:
    return EXAMPLES_DIR / "example_winter_bench.jpg"


@pytest.fixture
def winter_bench_example() -> ExampleSpec:
    return get_example("winter-bench")


@pytest.fixture(scope="session")
def metaclip2_pipeline_cpu_small() -> MetaClip2ExplanationPipeline:
    config = MetaClip2RunConfig(
        device="cpu",
        resized_image_size=224,
    )
    return MetaClip2ExplanationPipeline(config)
