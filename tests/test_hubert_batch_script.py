from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import run_hubert_2plus2eq5_manipulation as script


def test_discover_image_paths_reads_supported_flat_files_only(tmp_path: Path) -> None:
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "a.JPG").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("skip")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.jpg").write_bytes(b"")

    paths = script.discover_image_paths(tmp_path)

    assert [path.name for path in paths] == ["a.JPG", "b.png"]


def test_order_image_queue_permutates_without_changing_members() -> None:
    paths = [Path(f"image_{index}.jpg") for index in range(10)]

    ordered = script.order_image_queue(paths, seed="worker-a")

    assert sorted(ordered) == sorted(paths)
    assert ordered != paths


def test_order_image_queue_is_stable_for_same_seed() -> None:
    paths = [Path(f"image_{index}.jpg") for index in range(10)]

    assert script.order_image_queue(paths, seed="worker-a") == script.order_image_queue(
        paths, seed="worker-a"
    )


def test_image_output_dir_sanitizes_filename_stem(tmp_path: Path) -> None:
    output_dir = script.image_output_dir(
        tmp_path,
        Path("/remote/folder/Hubert photo (1).jpg"),
    )

    assert output_dir == tmp_path / "Hubert_photo_1"


def test_should_skip_image_when_complete_marker_exists(tmp_path: Path) -> None:
    image_dir = tmp_path / "image"
    image_dir.mkdir()
    (image_dir / "complete.json").write_text("{}")

    assert script.should_skip_image(image_dir)


def test_should_recompute_image_when_directory_is_partial(tmp_path: Path) -> None:
    image_dir = tmp_path / "image"
    image_dir.mkdir()
    (image_dir / "metrics.csv").write_text("step,loss\n")

    assert not script.should_skip_image(image_dir)


def test_acquire_image_lock_creates_metadata_and_blocks_second_worker(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "image"

    first = script.acquire_image_lock(
        output_dir=image_dir,
        image_path=Path("/remote/images/a.jpg"),
        lock_timeout_seconds=3600,
        worker_id="worker-a",
    )
    second = script.acquire_image_lock(
        output_dir=image_dir,
        image_path=Path("/remote/images/a.jpg"),
        lock_timeout_seconds=3600,
        worker_id="worker-b",
    )

    assert first.acquired
    assert not second.acquired
    metadata = json.loads((image_dir / ".lock" / "metadata.json").read_text())
    assert metadata["worker_id"] == "worker-a"
    assert metadata["source_image"] == "/remote/images/a.jpg"

    first.release()
    assert not (image_dir / ".lock").exists()


def test_acquire_image_lock_reclaims_stale_lock(tmp_path: Path) -> None:
    image_dir = tmp_path / "image"
    lock_dir = image_dir / ".lock"
    lock_dir.mkdir(parents=True)
    (lock_dir / "metadata.json").write_text(json.dumps({"created_at": 1.0}))

    lock = script.acquire_image_lock(
        output_dir=image_dir,
        image_path=Path("/remote/images/a.jpg"),
        lock_timeout_seconds=1,
        worker_id="worker-b",
        now=10.0,
    )

    assert lock.acquired
    metadata = json.loads((lock_dir / "metadata.json").read_text())
    assert metadata["worker_id"] == "worker-b"


def test_write_batch_summary_uses_union_of_metric_fields(tmp_path: Path) -> None:
    rows = [
        {"image": "a.jpg", "status": "processed", "mask_l1": 0.1},
        {"image": "b.jpg", "status": "failed", "error": "boom"},
    ]

    path = script.write_batch_summary(tmp_path, rows)

    with path.open(newline="") as handle:
        written = list(csv.DictReader(handle))
    assert written == [
        {"image": "a.jpg", "status": "processed", "mask_l1": "0.1", "error": ""},
        {"image": "b.jpg", "status": "failed", "mask_l1": "", "error": "boom"},
    ]


def test_write_batch_summary_has_header_for_empty_batches(tmp_path: Path) -> None:
    path = script.write_batch_summary(tmp_path, [])

    assert path.read_text() == "image,output_dir,status\n"


def test_write_complete_marker_records_run_metadata(tmp_path: Path) -> None:
    marker = script.write_complete_marker(
        output_dir=tmp_path,
        image_path=Path("/remote/images/a.jpg"),
        model_name="facebook/metaclip-2-test",
        steps=10,
        validate_only=False,
        prompt="Photo of a cat",
        img_id="a",
        label="a cat",
    )

    data = json.loads(marker.read_text())
    assert data["source_image"] == "/remote/images/a.jpg"
    assert data["model_name"] == "facebook/metaclip-2-test"
    assert data["steps"] == 10
    assert data["validate_only"] is False
    assert data["outputs"] == {
        "final_explanation": "final_explanation.npy",
        "final_metrics": "final_metrics.json",
        "metrics_csv": "metrics.csv",
    }


def test_write_complete_marker_records_delta_output(tmp_path: Path) -> None:
    marker = script.write_complete_marker(
        output_dir=tmp_path,
        image_path=Path("/remote/images/a.jpg"),
        model_name="facebook/metaclip-2-test",
        steps=10,
        validate_only=False,
        prompt="Photo of a cat",
        img_id="a",
        label="a cat",
        save_delta=True,
        canonical_size=512,
    )

    data = json.loads(marker.read_text())
    assert data["canonical_size"] == 512
    assert data["outputs"]["delta"] == "delta.npy"


def test_write_complete_marker_records_validate_only_outputs(tmp_path: Path) -> None:
    marker = script.write_complete_marker(
        output_dir=tmp_path,
        image_path=Path("/remote/images/a.jpg"),
        model_name="facebook/metaclip-2-test",
        steps=10,
        validate_only=True,
        prompt="Photo of a cat",
        img_id="a",
        label="a cat",
    )

    data = json.loads(marker.read_text())
    assert data["outputs"] == {
        "baseline_metrics": "baseline_metrics.json",
        "metrics_csv": "metrics.csv",
    }
