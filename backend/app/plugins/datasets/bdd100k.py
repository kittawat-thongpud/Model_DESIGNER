"""BDD100K detection dataset plugin.

Manual-only plugin for official BDD100K 100k images and detection labels.
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


_ROOT = DATASETS_DIR / "bdd100k"
_INDEX_DIR = _ROOT / "_index"
_INDEX_VERSION = 1

_FALLBACK_CLASSES = [
    "traffic light",
    "traffic sign",
    "car",
    "pedestrian",
    "bus",
    "truck",
    "rider",
    "bicycle",
    "motorcycle",
    "train",
]


def _source_img_dir(split: str) -> Path | None:
    candidates = [
        _ROOT / "images" / "100k" / split,
        _ROOT / "bdd100k" / "images" / "100k" / split,
        _ROOT / "100k" / split,
        _ROOT / split,
    ]
    for candidate in candidates:
        if has_images(candidate):
            return candidate
    return None


def _find_label_file(split: str) -> Path | None:
    candidates = [
        _ROOT / "labels" / "det_20" / f"det_{split}.json",
        _ROOT / "labels" / "det_20" / f"bdd100k_labels_images_det_{split}.json",
        _ROOT / "labels" / f"det_{split}.json",
        _ROOT / "labels" / f"bdd100k_labels_images_{split}.json",
        _ROOT / "bdd100k" / "labels" / "det_20" / f"det_{split}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    labels_root = _ROOT / "labels"
    if labels_root.exists():
        for path in labels_root.rglob("*.json"):
            stem = path.stem.lower()
            if split in stem and ("det" in stem or "label" in stem):
                return path
    return None


def _load_json(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _discover_classes() -> list[str]:
    seen: set[str] = set()
    for split in ("train", "val"):
        for item in _load_json(_find_label_file(split)):
            for label in item.get("labels", []) or []:
                category = label.get("category")
                if category:
                    seen.add(str(category))
    classes = [c for c in _FALLBACK_CLASSES if c in seen]
    classes.extend(sorted(seen - set(classes)))
    return classes or list(_FALLBACK_CLASSES)


def _frame_map(split: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in _load_json(_find_label_file(split)):
        name = item.get("name") or item.get("url")
        if not name:
            continue
        result[Path(str(name)).name] = item
    return result


def _parse_anns(item: dict | None, class_to_idx: dict[str, int], img_w: int, img_h: int) -> list[dict]:
    if not item:
        return []
    anns: list[dict] = []
    for label in item.get("labels", []) or []:
        box = label.get("box2d")
        category = label.get("category")
        if not box or category not in class_to_idx:
            continue
        try:
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        x1 = max(0.0, min(float(img_w), x1))
        y1 = max(0.0, min(float(img_h), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        anns.append({"bbox": [x1, y1, w, h], "category_id": class_to_idx[category]})
    return anns


class BDD100KPlugin(DatasetPlugin):
    def __init__(self):
        self._indices: dict[str, list[dict]] = {}
        self._classes: list[str] | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "bdd100k"

    @property
    def display_name(self) -> str:
        return "BDD100K Detection"

    @property
    def task_type(self) -> str:
        return "detection"

    @property
    def input_shape(self) -> list[int]:
        return [3, 1280, 720]

    def _class_names(self) -> list[str]:
        if self._classes is None:
            cats_path = _INDEX_DIR / "categories.json"
            if cats_path.exists():
                try:
                    self._classes = json.loads(cats_path.read_text())
                except Exception:
                    self._classes = None
            if self._classes is None:
                self._classes = _discover_classes()
        return self._classes

    @property
    def num_classes(self) -> int:
        return len(self._class_names())

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names())

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
        return ["bdd100k"]

    @property
    def manual_download(self) -> bool:
        return True

    @property
    def upload_instructions(self) -> str:
        return (
            "1. Download BDD100K 100k images and detection labels from the official BDD100K portal\n"
            "2. Upload/extract the archives into this dataset\n"
            "\nExpected source layout:\n"
            "  images/100k/train/*.jpg + images/100k/val/*.jpg + images/100k/test/*.jpg\n"
            "  labels/det_20/det_train.json + det_val.json"
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
            self._classes = None
        if _INDEX_DIR.exists():
            shutil.rmtree(_INDEX_DIR)
        _INDEX_DIR.mkdir(parents=True, exist_ok=True)

        classes = _discover_classes()
        class_to_idx = {name: i for i, name in enumerate(classes)}
        (_INDEX_DIR / "categories.json").write_text(json.dumps(classes, indent=1))
        self._classes = classes

        for split in ("train", "val", "test"):
            reset_dir(_ROOT / "images" / split)
            reset_dir(_ROOT / "labels" / split)

        for split in ("train", "val", "test"):
            src_dir = _source_img_dir(split)
            frames = _frame_map(split) if split != "test" else {}
            index: list[dict] = []
            paths: list[Path] = []
            if src_dir is not None:
                for src in image_files(src_dir):
                    dst = _ROOT / "images" / split / src.name
                    link_or_copy(src, dst)
                    size = image_size(dst)
                    if size is None:
                        continue
                    img_w, img_h = size
                    anns = _parse_anns(frames.get(src.name), class_to_idx, img_w, img_h)
                    if split != "test":
                        write_yolo_label(_ROOT / "labels" / split / f"{src.stem}.txt", anns, img_w, img_h)
                    paths.append(dst)
                    index.append({"file": dst.name, "w": img_w, "h": img_h, "anns": anns})
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
            "BDD100K requires manual download from the official BDD100K portal. "
            "Please upload the official 100k image and detection label archives."
        )

    def clear_data(self) -> list[str]:
        deleted = []
        if _ROOT.exists():
            shutil.rmtree(_ROOT)
            deleted.append(str(_ROOT))
        with self._lock:
            self._indices.clear()
            self._classes = None
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


register_dataset(BDD100KPlugin())
