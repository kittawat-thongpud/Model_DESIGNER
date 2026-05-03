from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.plugins.loader import discover_plugins, get_dataset_plugin
from app.plugins.datasets import bdd100k as bdd_mod
from app.plugins.datasets import cityscapes as city_mod
from app.plugins.datasets import kitti as kitti_mod
from app.services import dataset_yaml


def _write_image(path: Path, size: tuple[int, int] = (100, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 64, 32)).save(path)


def test_driving_plugins_are_discovered():
    discover_plugins()
    for name in ("kitti", "cityscapes", "bdd100k"):
        plugin = get_dataset_plugin(name)
        assert plugin is not None
        assert plugin.task_type == "detection"
        assert plugin.manual_download is True


def test_kitti_rebuilds_yolo_ready_layout(tmp_path: Path, monkeypatch):
    root = tmp_path / "kitti"
    monkeypatch.setattr(kitti_mod, "_ROOT", root)
    monkeypatch.setattr(kitti_mod, "_INDEX_DIR", root / "_index")

    _write_image(root / "training" / "image_2" / "000001.png")
    _write_image(root / "training" / "image_2" / "000002.png")
    _write_image(root / "testing" / "image_2" / "000003.png")
    label_dir = root / "training" / "label_2"
    label_dir.mkdir(parents=True)
    (label_dir / "000001.txt").write_text(
        "Car 0 0 0 10 5 40 30 0 0 0 0 0 0 0\n"
        "DontCare 0 0 0 1 1 2 2 0 0 0 0 0 0 0\n"
    )
    (label_dir / "000002.txt").write_text(
        "Pedestrian 0 0 0 20 10 30 40 0 0 0 0 0 0 0\n"
    )

    plugin = kitti_mod.KITTIPlugin()
    plugin.rebuild_index()

    assert plugin.is_available()
    splits = plugin.scan_splits()
    assert splits["train"]["total"] + splits["val"]["total"] == 2
    assert splits["train"]["labeled"] + splits["val"]["labeled"] == 2
    assert splits["test"]["total"] == 1
    assert (root / "train.txt").exists()
    assert (root / "val.txt").exists()
    assert list((root / "labels").rglob("*.txt"))

    raw = plugin.load_train() or plugin.load_val()
    _, target = plugin.wrap_for_training(raw)[0]
    assert torch.isfinite(target["boxes"]).all()
    assert int(target["labels"][0]) in range(plugin.num_classes)


def test_cityscapes_rebuilds_polygon_boxes(tmp_path: Path, monkeypatch):
    root = tmp_path / "cityscapes"
    monkeypatch.setattr(city_mod, "_ROOT", root)
    monkeypatch.setattr(city_mod, "_INDEX_DIR", root / "_index")

    train_stem = "aachen_000000_000019"
    val_stem = "bochum_000000_000019"
    _write_image(root / "leftImg8bit" / "train" / "aachen" / f"{train_stem}_leftImg8bit.png")
    _write_image(root / "leftImg8bit" / "val" / "bochum" / f"{val_stem}_leftImg8bit.png")
    ann = {
        "objects": [
            {"label": "car", "polygon": [[10, 5], [40, 5], [40, 30], [10, 30]]},
            {"label": "road", "polygon": [[0, 0], [1, 1]]},
            {"label": "cargroup", "polygon": [[0, 0], [50, 0], [50, 40], [0, 40]]},
        ]
    }
    for split, city, stem in (("train", "aachen", train_stem), ("val", "bochum", val_stem)):
        path = root / "gtFine" / split / city / f"{stem}_gtFine_polygons.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(ann))

    plugin = city_mod.CityscapesPlugin()
    plugin.rebuild_index()

    assert plugin.is_available()
    assert plugin.scan_splits()["train"]["labeled"] == 1
    assert plugin.scan_splits()["val"]["labeled"] == 1
    assert (root / "train.txt").exists()
    assert (root / "labels" / "train" / "aachen" / f"{train_stem}_leftImg8bit.txt").exists()

    _, target = plugin.wrap_for_training(plugin.load_train())[0]
    assert target["labels"].tolist() == [plugin.class_names.index("car")]
    assert torch.isfinite(target["boxes"]).all()


def test_bdd100k_rebuilds_detection_json(tmp_path: Path, monkeypatch):
    root = tmp_path / "bdd100k"
    monkeypatch.setattr(bdd_mod, "_ROOT", root)
    monkeypatch.setattr(bdd_mod, "_INDEX_DIR", root / "_index")

    _write_image(root / "images" / "100k" / "train" / "b1.jpg")
    _write_image(root / "images" / "100k" / "val" / "b2.jpg")
    _write_image(root / "images" / "100k" / "test" / "b3.jpg")
    labels = root / "labels" / "det_20"
    labels.mkdir(parents=True)
    (labels / "det_train.json").write_text(json.dumps([
        {"name": "b1.jpg", "labels": [{"category": "car", "box2d": {"x1": 10, "y1": 5, "x2": 50, "y2": 30}}]}
    ]))
    (labels / "det_val.json").write_text(json.dumps([
        {"name": "b2.jpg", "labels": [{"category": "pedestrian", "box2d": {"x1": 20, "y1": 10, "x2": 40, "y2": 40}}]}
    ]))

    plugin = bdd_mod.BDD100KPlugin()
    plugin.rebuild_index()

    assert plugin.is_available()
    splits = plugin.scan_splits()
    assert splits["train"] == {"total": 1, "labeled": 1}
    assert splits["val"] == {"total": 1, "labeled": 1}
    assert splits["test"] == {"total": 1, "labeled": 0}
    assert (root / "train.txt").exists()
    assert (root / "labels" / "train" / "b1.txt").exists()

    _, target = plugin.wrap_for_training(plugin.load_train())[0]
    assert torch.isfinite(target["boxes"]).all()
    assert int(target["labels"][0]) == plugin.class_names.index("car")


def test_generated_yaml_prefers_root_split_txt(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dataset_yaml, "DATASETS_DIR", tmp_path)
    root = tmp_path / "kitti"
    root.mkdir()
    (root / "train.txt").write_text("/tmp/a.png\n")
    (root / "val.txt").write_text("/tmp/b.png\n")

    out = tmp_path / "data.yaml"
    dataset_yaml.write_data_yaml("kitti", out)
    text = out.read_text()

    assert "train.txt" in text
    assert "val.txt" in text
    assert "nc: 8" in text
