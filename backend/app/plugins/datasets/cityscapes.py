"""Cityscapes detection dataset plugin.

Manual-only plugin for official Cityscapes leftImg8bit and gtFine archives.
Detection boxes are derived from gtFine polygon annotations for instance classes.
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

from app.config import DATASETS_DIR

from ..base import DatasetPlugin
from ..loader import register_dataset
from ._driving_common import (
    DrivingDetectionRawDataset,
    DrivingDetectionWrapper,
    has_images,
    image_files,
    image_size,
    link_or_copy,
    reset_dir,
    write_split_txt,
    write_yolo_label,
)


_ROOT = DATASETS_DIR / "cityscapes"
_INDEX_DIR = _ROOT / "_index"
_INDEX_VERSION = 1

_CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
_CLASS_MAP = {name: i for i, name in enumerate(_CLASSES)}


def _source_img_dir(split: str) -> Path | None:
    candidates = [
        _ROOT / "leftImg8bit" / split,
        _ROOT / "leftImg8bit_trainvaltest" / "leftImg8bit" / split,
        _ROOT / split,
    ]
    for candidate in candidates:
        if has_images(candidate):
            return candidate
    return None


def _source_ann_dir(split: str) -> Path | None:
    candidates = [
        _ROOT / "gtFine" / split,
        _ROOT / "gtFine_trainvaltest" / "gtFine" / split,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*_gtFine_polygons.json")):
            return candidate
    return None


def _ann_for_image(img: Path, img_root: Path, ann_root: Path | None) -> Path | None:
    if ann_root is None:
        return None
    rel = img.relative_to(img_root)
    stem = rel.stem.replace("_leftImg8bit", "")
    candidate = ann_root / rel.parent / f"{stem}_gtFine_polygons.json"
    if candidate.exists():
        return candidate
    matches = list(ann_root.rglob(f"{stem}_gtFine_polygons.json"))
    return matches[0] if matches else None


def _parse_polygons(path: Path | None, img_w: int, img_h: int) -> list[dict]:
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    anns: list[dict] = []
    for obj in data.get("objects", []) or []:
        label = str(obj.get("label", ""))
        if label.endswith("group"):
            continue
        if label not in _CLASS_MAP:
            continue
        polygon = obj.get("polygon") or []
        if not polygon:
            continue
        try:
            xs = [float(p[0]) for p in polygon]
            ys = [float(p[1]) for p in polygon]
        except (TypeError, ValueError, IndexError):
            continue
        x1 = max(0.0, min(float(img_w), min(xs)))
        y1 = max(0.0, min(float(img_h), min(ys)))
        x2 = max(0.0, min(float(img_w), max(xs)))
        y2 = max(0.0, min(float(img_h), max(ys)))
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        anns.append({"bbox": [x1, y1, w, h], "category_id": _CLASS_MAP[label]})
    return anns


class CityscapesPlugin(DatasetPlugin):
    def __init__(self):
        self._indices: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "cityscapes"

    @property
    def display_name(self) -> str:
        return "Cityscapes Detection"

    @property
    def task_type(self) -> str:
        return "detection"

    @property
    def input_shape(self) -> list[int]:
        return [3, 2048, 1024]

    @property
    def num_classes(self) -> int:
        return len(_CLASSES)

    @property
    def class_names(self) -> list[str]:
        return list(_CLASSES)

    @property
    def train_size(self) -> int:
        return len(self._get_index("train"))

    @property
    def val_size(self) -> int:
        return len(self._get_index("val"))

    @property
    def test_size(self) -> int:
        return len(self._get_index("test"))

    @property
    def normalization(self) -> tuple[tuple, tuple]:
        return ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    @property
    def data_dirs(self) -> list[str]:
        return ["cityscapes"]

    @property
    def manual_download(self) -> bool:
        return True

    @property
    def upload_instructions(self) -> str:
        return (
            "1. Register/login at https://www.cityscapes-dataset.com/downloads/\n"
            "2. Download leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip\n"
            "3. Upload both archives, or place the extracted folders in data/datasets/cityscapes/\n"
            "\nExpected source layout:\n"
            "  leftImg8bit/train|val|test/<city>/*_leftImg8bit.png\n"
            "  gtFine/train|val/<city>/*_gtFine_polygons.json"
        )

    def is_available(self) -> bool:
        return has_images(_ROOT / "images" / "train") and (_ROOT / "train.txt").exists()

    def _get_index(self, split: str) -> list[dict]:
        with self._lock:
            if split in self._indices:
                return self._indices[split]
        path = _INDEX_DIR / f"{split}_index.json"
        if path.exists():
            try:
                raw = json.loads(path.read_text())
                if raw.get("version") == _INDEX_VERSION:
                    index = raw.get("images", [])
                    with self._lock:
                        self._indices[split] = index
                    return index
            except Exception:
                pass
        return []

    def rebuild_index(self) -> None:
        with self._lock:
            self._indices.clear()
        if _INDEX_DIR.exists():
            shutil.rmtree(_INDEX_DIR)
        _INDEX_DIR.mkdir(parents=True, exist_ok=True)

        for split in ("train", "val", "test"):
            reset_dir(_ROOT / "images" / split)
            reset_dir(_ROOT / "labels" / split)

        for split in ("train", "val", "test"):
            src_dir = _source_img_dir(split)
            ann_dir = _source_ann_dir(split) if split != "test" else None
            index: list[dict] = []
            paths: list[Path] = []
            if src_dir is not None:
                for src in image_files(src_dir):
                    rel = src.relative_to(src_dir)
                    dst = _ROOT / "images" / split / rel
                    link_or_copy(src, dst)
                    size = image_size(dst)
                    if size is None:
                        continue
                    img_w, img_h = size
                    anns = _parse_polygons(_ann_for_image(src, src_dir, ann_dir), img_w, img_h)
                    if split != "test":
                        write_yolo_label(_ROOT / "labels" / split / rel.with_suffix(".txt"), anns, img_w, img_h)
                    paths.append(dst)
                    index.append({"file": str(rel), "w": img_w, "h": img_h, "anns": anns})
            index.sort(key=lambda e: e["file"])
            (_INDEX_DIR / f"{split}_index.json").write_text(
                json.dumps({"version": _INDEX_VERSION, "images": index}, separators=(",", ":"))
            )
            write_split_txt(_ROOT, split, sorted(paths))
            with self._lock:
                self._indices[split] = index

    def scan_splits(self) -> dict[str, dict]:
        return {
            split: {
                "total": len(index := self._get_index(split)),
                "labeled": sum(1 for e in index if e.get("anns")),
            }
            for split in ("train", "val", "test")
        }

    def download(self, state: dict) -> None:
        raise NotImplementedError(
            "Cityscapes requires manual download from the official Cityscapes website. "
            "Please upload the official leftImg8bit and gtFine archives."
        )

    def clear_data(self) -> list[str]:
        deleted = []
        if _ROOT.exists():
            shutil.rmtree(_ROOT)
            deleted.append(str(_ROOT))
        with self._lock:
            self._indices.clear()
        return deleted

    def _load_split(self, split: str, transform=None):
        index = self._get_index(split)
        img_dir = _ROOT / "images" / split
        if not index or not has_images(img_dir):
            return None
        return DrivingDetectionRawDataset(img_dir, index, transform=transform)

    def load_train(self, transform=None):
        return self._load_split("train", transform)

    def load_val(self, transform=None):
        return self._load_split("val", transform)

    def load_test(self, transform=None):
        return self._load_split("test", transform)

    def wrap_for_training(self, dataset) -> DrivingDetectionWrapper:
        return DrivingDetectionWrapper(dataset)


register_dataset(CityscapesPlugin())
