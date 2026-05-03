"""KITTI Object Detection dataset plugin.

Manual-only plugin for the official KITTI object detection archives:
``data_object_image_2.zip`` and ``data_object_label_2.zip``.
"""
from __future__ import annotations

import hashlib
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


_ROOT = DATASETS_DIR / "kitti"
_INDEX_DIR = _ROOT / "_index"
_INDEX_VERSION = 1

_CLASSES = [
    "car",
    "van",
    "truck",
    "pedestrian",
    "person_sitting",
    "cyclist",
    "tram",
    "misc",
]
_CLASS_MAP = {name: i for i, name in enumerate(_CLASSES)}
_KITTI_NAME_MAP = {
    "car": "car",
    "van": "van",
    "truck": "truck",
    "pedestrian": "pedestrian",
    "person_sitting": "person_sitting",
    "cyclist": "cyclist",
    "tram": "tram",
    "misc": "misc",
}


def _find_img_dir(split: str) -> Path | None:
    if split in {"train", "val"}:
        candidates = [
            _ROOT / "training" / "image_2",
            _ROOT / "image_2",
        ]
    else:
        candidates = [
            _ROOT / "testing" / "image_2",
            _ROOT / "test" / "image_2",
        ]
    for candidate in candidates:
        if has_images(candidate):
            return candidate
    return None


def _find_label_dir() -> Path | None:
    for candidate in (_ROOT / "training" / "label_2", _ROOT / "label_2", _ROOT / "labels_raw"):
        if candidate.exists() and any(candidate.glob("*.txt")):
            return candidate
    return None


def _read_existing_split_stems(split: str) -> set[str] | None:
    txt = _ROOT / f"{split}.txt"
    if not txt.exists():
        return None
    stems = {
        Path(line.strip()).stem
        for line in txt.read_text().splitlines()
        if line.strip()
    }
    return stems or None


def _stable_is_val(stem: str) -> bool:
    bucket = int(hashlib.md5(stem.encode()).hexdigest()[:8], 16) % 100
    return bucket < 20


def _parse_label_file(path: Path, img_w: int, img_h: int) -> list[dict]:
    anns: list[dict] = []
    if not path.exists():
        return anns
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        raw_name = parts[0].lower()
        if raw_name == "dontcare":
            continue
        cls_name = _KITTI_NAME_MAP.get(raw_name)
        if cls_name is None:
            continue
        try:
            x1, y1, x2, y2 = (float(parts[i]) for i in range(4, 8))
        except ValueError:
            continue
        x1 = max(0.0, min(float(img_w), x1))
        y1 = max(0.0, min(float(img_h), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        anns.append({"bbox": [x1, y1, w, h], "category_id": _CLASS_MAP[cls_name]})
    return anns


class KITTIPlugin(DatasetPlugin):
    def __init__(self):
        self._indices: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "kitti"

    @property
    def display_name(self) -> str:
        return "KITTI Object Detection"

    @property
    def task_type(self) -> str:
        return "detection"

    @property
    def input_shape(self) -> list[int]:
        return [3, 1242, 375]

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
        return ["kitti"]

    @property
    def manual_download(self) -> bool:
        return True

    @property
    def upload_instructions(self) -> str:
        return (
            "1. Go to https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d\n"
            "2. Download the official object detection image and label archives\n"
            "3. Upload data_object_image_2.zip and data_object_label_2.zip, or place both extracted folders in data/datasets/kitti/\n"
            "\nExpected source layout:\n"
            "  training/image_2/*.png\n"
            "  training/label_2/*.txt\n"
            "  testing/image_2/*.png (optional)"
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

        src_train = _find_img_dir("train")
        label_dir = _find_label_dir()
        src_test = _find_img_dir("test")

        for split in ("train", "val", "test"):
            reset_dir(_ROOT / "images" / split)
            reset_dir(_ROOT / "labels" / split)

        train_stems = _read_existing_split_stems("train")
        val_stems = _read_existing_split_stems("val")
        have_user_split = train_stems is not None or val_stems is not None

        split_indices: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        split_paths: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

        if src_train is not None:
            for src in image_files(src_train):
                stem = src.stem
                if have_user_split:
                    split = "val" if val_stems and stem in val_stems else "train"
                else:
                    split = "val" if _stable_is_val(stem) else "train"
                dst = _ROOT / "images" / split / src.name
                link_or_copy(src, dst)
                size = image_size(dst)
                if size is None:
                    continue
                img_w, img_h = size
                anns = _parse_label_file(label_dir / f"{stem}.txt", img_w, img_h) if label_dir else []
                label_path = _ROOT / "labels" / split / f"{stem}.txt"
                write_yolo_label(label_path, anns, img_w, img_h)
                split_paths[split].append(dst)
                split_indices[split].append({
                    "file": dst.name,
                    "w": img_w,
                    "h": img_h,
                    "anns": anns,
                })

        if src_test is not None:
            for src in image_files(src_test):
                dst = _ROOT / "images" / "test" / src.name
                link_or_copy(src, dst)
                size = image_size(dst)
                if size is None:
                    continue
                img_w, img_h = size
                split_paths["test"].append(dst)
                split_indices["test"].append({"file": dst.name, "w": img_w, "h": img_h, "anns": []})

        for split, index in split_indices.items():
            index.sort(key=lambda e: e["file"])
            (_INDEX_DIR / f"{split}_index.json").write_text(
                json.dumps({"version": _INDEX_VERSION, "images": index}, separators=(",", ":"))
            )
            write_split_txt(_ROOT, split, sorted(split_paths[split]))
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
            "KITTI requires manual download from the official KITTI website. "
            "Please upload the official object detection archives."
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


register_dataset(KITTIPlugin())
