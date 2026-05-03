"""KITTI Object Detection dataset plugin.

Supports the Ultralytics YOLO-ready KITTI package for auto setup, while still
accepting the official KITTI object detection archive layout:
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
_KITTI_ZIP = "https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip"

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


def _find_yolo_img_dir(split: str) -> Path | None:
    for candidate in (_ROOT / "images" / split, DATASETS_DIR / "images" / split):
        if has_images(candidate):
            return candidate
    return None


def _find_yolo_label_dir(split: str) -> Path | None:
    for candidate in (_ROOT / "labels" / split, DATASETS_DIR / "labels" / split):
        if candidate.exists() and any(candidate.glob("*.txt")):
            return candidate
    return None


def _has_yolo_ready_layout() -> bool:
    return _find_yolo_img_dir("train") is not None or _find_yolo_img_dir("val") is not None


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _link_or_copy_fresh(src: Path, dst: Path) -> None:
    if _same_path(src, dst):
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    link_or_copy(src, dst)


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


def _parse_yolo_label_file(path: Path, img_w: int, img_h: int) -> list[dict]:
    anns: list[dict] = []
    if not path.exists():
        return anns
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if cls_id < 0 or cls_id >= len(_CLASSES) or bw <= 0 or bh <= 0:
            continue
        x = (cx - bw / 2.0) * img_w
        y = (cy - bh / 2.0) * img_h
        w = bw * img_w
        h = bh * img_h
        x = max(0.0, min(float(img_w), x))
        y = max(0.0, min(float(img_h), y))
        w = max(0.0, min(float(img_w) - x, w))
        h = max(0.0, min(float(img_h) - y, h))
        if w > 0 and h > 0:
            anns.append({"bbox": [x, y, w, h], "category_id": cls_id})
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
    def category_id_map(self) -> dict[int, str]:
        return {i: name for i, name in enumerate(_CLASSES)}

    def _cat_id_to_contiguous(self) -> dict[int, int]:
        return {i: i for i in range(len(_CLASSES))}

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
        return False

    @property
    def upload_instructions(self) -> str:
        return (
            "Auto setup downloads the Ultralytics KITTI package.\n\n"
            "Manual alternative:\n"
            "1. Go to https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d\n"
            "2. Download the official object detection image and label archives\n"
            "3. Upload data_object_image_2.zip and data_object_label_2.zip, or place both extracted folders in data/datasets/kitti/\n"
            "\nExpected source layout:\n"
            "  training/image_2/*.png\n"
            "  training/label_2/*.txt\n"
            "  testing/image_2/*.png (optional)\n\n"
            "Ultralytics layout is also supported:\n"
            "  kitti/images/train/* or images/train/*\n"
            "  kitti/images/val/* or images/val/*\n"
            "  kitti/labels/train/*.txt or labels/train/*.txt\n"
            "  kitti/labels/val/*.txt or labels/val/*.txt"
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

        if _has_yolo_ready_layout():
            self._rebuild_yolo_ready_index()
            return

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

    def _rebuild_yolo_ready_index(self) -> None:
        split_indices: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        split_paths: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

        for split in ("train", "val", "test"):
            img_dir = _find_yolo_img_dir(split)
            if img_dir is None:
                continue
            label_dir = _find_yolo_label_dir(split)
            dst_img_dir = _ROOT / "images" / split
            dst_label_dir = _ROOT / "labels" / split
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_label_dir.mkdir(parents=True, exist_ok=True)

            for src_img in image_files(img_dir):
                dst_img = dst_img_dir / src_img.name
                _link_or_copy_fresh(src_img, dst_img)

                size = image_size(dst_img)
                if size is None:
                    continue
                img_w, img_h = size

                dst_label = dst_label_dir / f"{src_img.stem}.txt"
                if label_dir is not None:
                    src_label = label_dir / f"{src_img.stem}.txt"
                    if src_label.exists():
                        _link_or_copy_fresh(src_label, dst_label)

                anns = _parse_yolo_label_file(dst_label, img_w, img_h)
                split_paths[split].append(dst_img)
                split_indices[split].append({
                    "file": dst_img.name,
                    "w": img_w,
                    "h": img_h,
                    "anns": anns,
                })

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
        from app.utils.download import download_and_extract

        state.setdefault("files", {})
        state["message"] = "Downloading KITTI..."
        state["progress"] = 0

        download_and_extract(
            _KITTI_ZIP,
            str(DATASETS_DIR),
            state,
            "KITTI",
            file_key="kitti.zip",
        )

        state["message"] = "Building KITTI index..."
        state["progress"] = 90
        self.rebuild_index()

        if not self.is_available():
            raise RuntimeError("KITTI download completed, but no train split was detected.")

        state["message"] = "Complete"
        state["progress"] = 100

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
