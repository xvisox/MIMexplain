from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from explainer.constants import EXAMPLES_DIR


@dataclass(frozen=True)
class ExampleSpec:
    name: str
    image_path: Path
    caption: str
    description: str


EXAMPLES: dict[str, ExampleSpec] = {
    "winter-bench": ExampleSpec(
        name="winter-bench",
        image_path=EXAMPLES_DIR / "example_winter_bench.jpg",
        caption="bench",
        description="Neutral outdoor bench scene used for smoke tests.",
    ),
}


def get_example(name: str) -> ExampleSpec:
    try:
        return EXAMPLES[name]
    except KeyError as error:
        choices = ", ".join(sorted(EXAMPLES))
        raise KeyError(
            f"Unknown example '{name}'. Available examples: {choices}."
        ) from error


def list_example_names() -> tuple[str, ...]:
    return tuple(sorted(EXAMPLES))
