from __future__ import annotations

import pytest

from explainer.examples import get_example, list_example_names


def test_list_example_names_is_sorted_tuple() -> None:
    names = list_example_names()
    assert isinstance(names, tuple)
    assert list(names) == sorted(names)
    assert "winter-bench" in names


def test_get_example_returns_spec_with_existing_file() -> None:
    spec = get_example("winter-bench")
    assert spec.name == "winter-bench"
    assert spec.image_path.exists()


def test_get_example_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="not-a-real-example"):
        get_example("not-a-real-example")
