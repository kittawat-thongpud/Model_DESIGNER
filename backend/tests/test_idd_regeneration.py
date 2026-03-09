from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.controllers import dataset_controller
from app.services import coco_converter


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _idd_payload(file_name: str) -> dict:
    return {
        "images": [{"id": 1, "file_name": file_name, "width": 100, "height": 50}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 5, 20, 10], "area": 200, "iscrowd": 0}],
        "categories": [{"id": 0, "name": "person"}],
    }


def test_is_already_converted_requires_expected_split_dirs(tmp_path: Path):
    dataset_path = tmp_path / "idd"
    _write_json(dataset_path / "annotations" / "idd_detection_train.json", _idd_payload("frontFar/a.jpg"))
    _write_json(dataset_path / "annotations" / "idd_detection_val.json", _idd_payload("frontFar/b.jpg"))
    (dataset_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    flat_label = dataset_path / "labels" / "frontFar" / "a.txt"
    flat_label.parent.mkdir(parents=True, exist_ok=True)
    flat_label.write_text("0 0.5 0.5 0.2 0.2\n")
    marker_path = dataset_path / "labels" / ".coco_yolo_conversion.json"
    marker_path.write_text(json.dumps({"annotations_fingerprint": coco_converter._annotations_fingerprint(dataset_path)}))

    assert coco_converter.is_already_converted(dataset_path) is False


def test_convert_coco_to_yolo_regenerates_split_labels_and_cleans_flat_dirs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(coco_converter, "DATASETS_DIR", tmp_path)
    dataset_path = tmp_path / "idd"
    _write_json(dataset_path / "annotations" / "idd_detection_train.json", _idd_payload("frontFar/a.jpg"))
    _write_json(dataset_path / "annotations" / "idd_detection_val.json", _idd_payload("sideLeft/b.jpg"))
    (dataset_path / "images" / "train" / "frontFar").mkdir(parents=True, exist_ok=True)
    (dataset_path / "images" / "val" / "sideLeft").mkdir(parents=True, exist_ok=True)
    (dataset_path / "images" / "train" / "frontFar" / "a.jpg").write_bytes(b"")
    (dataset_path / "images" / "val" / "sideLeft" / "b.jpg").write_bytes(b"")
    stale_dir = dataset_path / "labels" / "frontFar"
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "stale.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    result = coco_converter.convert_coco_to_yolo("idd", cls91to80=False)

    assert result["status"] == "success"
    assert (dataset_path / "labels" / "train" / "frontFar" / "a.txt").exists()
    assert (dataset_path / "labels" / "val" / "sideLeft" / "b.txt").exists()
    assert not stale_dir.exists()


def test_scan_dataset_meta_triggers_idd_label_self_heal(monkeypatch):
    calls: list[str] = []

    class StubPlugin:
        def is_available(self):
            return True

        def disk_size_bytes(self):
            return 123

        def scan_splits(self):
            return {
                "train": {"total": 4, "labeled": 3},
                "val": {"total": 2, "labeled": 1},
            }

    monkeypatch.setattr(dataset_controller.coco_converter, "auto_convert_if_needed", lambda name: calls.append(name) or None)
    monkeypatch.setattr(dataset_controller, "get_dataset_plugin", lambda name: StubPlugin() if name == "idd" else None)
    monkeypatch.setattr(dataset_controller, "_load_split_config", lambda name: {})
    monkeypatch.setattr(
        dataset_controller,
        "_compute_transfer_counts",
        lambda cfg, raw_train, raw_test, raw_val: {
            "train_count": raw_train,
            "val_count": raw_val,
            "test_count": raw_test,
        },
    )
    monkeypatch.setattr(dataset_controller, "_save_meta", lambda name, meta: None)

    meta = dataset_controller._scan_dataset_meta("idd")

    assert calls == ["idd"]
    assert meta["available"] is True
    assert meta["splits"]["train"]["effective"] == 3
    assert meta["splits"]["val"]["effective"] == 1
