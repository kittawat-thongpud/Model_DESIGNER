"""IDD (India Driving Dataset) Detection plugin.

Supports the IDD Detection dataset from https://idd.insaan.iiit.ac.in/
which uses COCO-style JSON annotations.

Expected directory structure after upload/extraction:
  data/datasets/idd/
    images/
      train/   (or train2017/)
      val/     (or val2017/)
    annotations/
      instances_train.json   (or idd_detection_train.json / train.json)
      instances_val.json     (or idd_detection_val.json / val.json)

The plugin auto-detects annotation file names and image subdirectory names.
"""
from __future__ import annotations
import json
import threading
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image as PILImage

from app.config import DATASETS_DIR
from ..base import DatasetPlugin
from ..loader import register_dataset

_IDD_ROOT = DATASETS_DIR / "idd"
_INDEX_DIR = _IDD_ROOT / "_index"

# Bump this when the index schema changes so stale indexes are auto-rebuilt.
_INDEX_VERSION = 2

# IDD Detection classes (15 classes based on official IDD detection benchmark)
_IDD_CLASSES = [
    "person", "rider", "car", "truck", "bus", "motorcycle",
    "bicycle", "autorickshaw", "animal", "traffic light",
    "traffic sign", "utility pole", "misc", "drivable area", "non-drivable area",
]


# ── Auto-detect annotation and image paths ────────────────────────────────────

def _find_ann_file(split: str) -> Path | None:
    """Try common annotation file name patterns for a given split."""
    ann_dir = _IDD_ROOT / "annotations"
    if not ann_dir.exists():
        ann_dir = _IDD_ROOT / "Annotations"
    if not ann_dir.exists():
        return None

    candidates = [
        f"instances_{split}.json",
        f"instances_{split}2017.json",
        f"idd_detection_{split}.json",
        f"{split}.json",
        f"{split}2017.json",
        f"idd_{split}.json",
    ]
    for c in candidates:
        p = ann_dir / c
        if p.exists():
            return p
    # Fallback: any json containing the split name
    for p in ann_dir.glob("*.json"):
        if split in p.stem.lower():
            return p
    return None


def _find_img_dir(split: str) -> Path | None:
    """Try common image directory name patterns for a given split."""
    img_base = _IDD_ROOT / "images"
    if not img_base.exists():
        img_base = _IDD_ROOT / "JPEGImages"
    if not img_base.exists():
        return None

    candidates = [split, f"{split}2017", f"{split}2019"]
    for c in candidates:
        p = img_base / c
        if p.exists():
            return p
    # Single flat images dir (no split subdirs)
    if img_base.exists() and not any((img_base / c).exists() for c in candidates):
        return img_base
    return None


# ── Index builder ─────────────────────────────────────────────────────────────

def _build_index(
    ann_path: Path,
    img_dir: Path,
    index_path: Path,
    data: dict | None = None,
) -> list[dict]:
    """Build lightweight per-image index from COCO-style JSON.

    *data* may be passed in pre-loaded to avoid a redundant ``json.load`` when
    the caller already has the annotation dict in memory.
    """
    if data is None:
        with open(ann_path) as f:
            data = json.load(f)

    # Build image lookup
    img_lookup: dict[int, dict] = {}
    for img_info in data.get("images", []):
        img_lookup[img_info["id"]] = {
            "file": img_info["file_name"],
            "w": img_info.get("width", 0),
            "h": img_info.get("height", 0),
            "anns": [],
        }

    # Attach annotations
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id in img_lookup:
            img_lookup[img_id]["anns"].append({
                "bbox": ann["bbox"],
                "category_id": ann["category_id"],
                "area": ann.get("area", 0),
                "iscrowd": ann.get("iscrowd", 0),
            })

    # Build category id → contiguous index map
    cats = data.get("categories", [])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}

    # Remap category_ids to contiguous indices in each annotation
    index = []
    for entry in img_lookup.values():
        file_raw = str(entry["file"])

        # Resolve the canonical relative path under img_dir (like COCO does).
        # Try 1: use the path as-is.
        img_file = img_dir / file_raw
        resolved_rel = file_raw
        if not img_file.exists():
            # Try 2: basename only (strips any leading subdir in file_name).
            base = Path(file_raw).name
            img_file = img_dir / base
            resolved_rel = base
        if not img_file.exists():
            continue

        # Resolve w/h at build time so __getitem__ never needs to open files.
        w, h = entry["w"], entry["h"]
        if w == 0 or h == 0:
            try:
                with PILImage.open(img_file) as im:
                    w, h = im.size
            except Exception:
                continue

        remapped_anns = []
        for ann in entry["anns"]:
            cat_idx = cat_id_to_idx.get(ann["category_id"])
            if cat_idx is None:
                continue
            remapped_anns.append({
                "bbox": ann["bbox"],
                "category_id": cat_idx,
                "area": ann["area"],
                "iscrowd": ann["iscrowd"],
            })

        index.append({
            "file": resolved_rel,   # canonical path guaranteed to exist under img_dir
            "w": w,
            "h": h,
            "anns": remapped_anns,
        })

    index.sort(key=lambda e: e["file"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    # Wrap in versioned envelope (same pattern as COCO categories.json side-file).
    payload = {"version": _INDEX_VERSION, "images": index}
    with open(index_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    # Save categories alongside index so callers don't need to re-parse the full JSON.
    cats_path = index_path.parent / "categories.json"
    if not cats_path.exists():
        with open(cats_path, "w") as f:
            json.dump(sorted(data.get("categories", []), key=lambda c: c["id"]), f, indent=1)

    return index


# ── Raw dataset ───────────────────────────────────────────────────────────────

class IDDRawDataset(Dataset):
    """Returns (PIL_image, [annotation_dicts]) in COCO-like format."""

    def __init__(self, img_dir: Path, index: list[dict], transform=None):
        self._img_dir = img_dir
        self._index = index
        self._transform = transform

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        entry = self._index[idx]
        # Index stores canonical resolved paths (guaranteed by _build_index).
        # Direct lookup — no rglob fallback needed.
        p = self._img_dir / entry["file"]
        img = PILImage.open(p).convert("RGB")
        if self._transform is not None:
            img = self._transform(img)
        return img, entry["anns"]

    @property
    def orig_sizes(self) -> list[tuple[int, int]]:
        return [(e["w"], e["h"]) for e in self._index]


# ── Training wrapper ──────────────────────────────────────────────────────────

class _IDDDetectionWrapper(Dataset):
    """Converts raw (image, [ann_dicts]) → (image_tensor, {boxes, labels})."""

    def __init__(self, raw_ds: IDDRawDataset):
        self._ds = raw_ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        img, anns = self._ds[idx]
        if isinstance(img, torch.Tensor):
            _, img_h, img_w = img.shape
        else:
            img_w, img_h = img.size

        boxes, labels = [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x_min, y_min, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            cx = (x_min + bw / 2) / img_w
            cy = (y_min + bh / 2) / img_h
            boxes.append([cx, cy, bw / img_w, bh / img_h])
            labels.append(ann["category_id"])

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)

        return img, {"boxes": boxes_t, "labels": labels_t}


# ── Plugin ────────────────────────────────────────────────────────────────────

class IDDPlugin(DatasetPlugin):

    def __init__(self):
        self._index_cache: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        # Categories are read from annotation JSON on first access
        self._categories: list[dict] | None = None

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "idd"

    @property
    def display_name(self) -> str:
        return "IDD Detection (India Driving)"

    @property
    def task_type(self) -> str:
        return "detection"

    @property
    def input_shape(self) -> list[int]:
        return [3, 1280, 720]

    # ── Categories ────────────────────────────────────────────────────────────

    def _load_categories(self) -> list[dict]:
        if self._categories is not None:
            return self._categories
        # 1. Prefer pre-built categories.json (fast, same pattern as COCO).
        cats_path = _INDEX_DIR / "categories.json"
        if cats_path.exists():
            try:
                with open(cats_path) as f:
                    cats = json.load(f)
                if cats:
                    self._categories = cats
                    return self._categories
            except Exception:
                pass
        # 2. Parse annotation JSON once and cache the result.
        for split in ("train", "val"):
            ann_path = _find_ann_file(split)
            if ann_path:
                try:
                    with open(ann_path) as f:
                        data = json.load(f)
                    cats = data.get("categories", [])
                    if cats:
                        self._categories = sorted(cats, key=lambda c: c["id"])
                        return self._categories
                except Exception:
                    pass
        # 3. Fallback to hardcoded IDD classes.
        self._categories = [{"id": i, "name": n} for i, n in enumerate(_IDD_CLASSES)]
        return self._categories

    @property
    def num_classes(self) -> int:
        return len(self._load_categories())

    @property
    def class_names(self) -> list[str]:
        return [c["name"] for c in self._load_categories()]

    def category_id_to_name(self) -> dict[int, str]:
        return {i: c["name"] for i, c in enumerate(self._load_categories())}

    @property
    def category_id_map(self) -> dict[int, str]:
        return self.category_id_to_name()

    @property
    def train_size(self) -> int:
        idx = self._get_index("train")
        return len(idx)

    @property
    def val_size(self) -> int:
        idx = self._get_index("val")
        return len(idx)

    @property
    def test_size(self) -> int:
        return 0

    @property
    def normalization(self) -> tuple[tuple, tuple]:
        return ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # ── Availability ──────────────────────────────────────────────────────────

    @property
    def data_dirs(self) -> list[str]:
        return ["idd"]

    @property
    def manual_download(self) -> bool:
        return True

    @property
    def upload_instructions(self) -> str:
        return (
            "1. Register at https://idd.insaan.iiit.ac.in/dataset/download/\n"
            "2. Download the Detection dataset (zip or tar.gz)\n"
            "3. Upload the archive using the Upload button below\n"
            "\nExpected structure after extraction:\n"
            "  images/train/  + images/val/\n"
            "  annotations/instances_train.json + instances_val.json"
        )

    def is_available(self) -> bool:
        """Fast availability check — no rglob, no full JSON parse.

        Strategy (cheapest first):
        1. Pre-built index exists and is non-empty  → available.
        2. Annotation JSON exists (size > 1 KB) AND images dir exists → available.
           (After conversion the annotation JSON is always present and named
           idd_detection_train.json; its existence is a reliable sentinel.)
        """
        if not _IDD_ROOT.exists():
            return False

        # 1. Index already built — fastest possible check (single stat call).
        index_path = _INDEX_DIR / "train_index.json"
        if index_path.exists() and index_path.stat().st_size > 256:
            return True

        # 2. Annotation sentinel + images dir exist.
        ann_path = _find_ann_file("train")
        if ann_path and ann_path.exists() and ann_path.stat().st_size > 1024:
            img_dir = _find_img_dir("train")
            if img_dir and img_dir.exists():
                # Check just the directory is non-empty (single iterdir call)
                try:
                    next(img_dir.iterdir())
                    return True
                except StopIteration:
                    pass
        return False

    # ── Index management ──────────────────────────────────────────────────────

    def _get_index(self, split: str) -> list[dict]:
        with self._lock:
            if split in self._index_cache:
                return self._index_cache[split]

        index_path = _INDEX_DIR / f"{split}_index.json"
        index = None
        if index_path.exists():
            try:
                with open(index_path) as f:
                    raw = json.load(f)
                # Support both versioned envelope {"version": N, "images": [...]} and
                # legacy bare list format — rebuild if version is outdated.
                if isinstance(raw, dict):
                    if raw.get("version", 0) >= _INDEX_VERSION:
                        index = raw["images"]
                    # else: version mismatch → fall through to rebuild
                else:
                    # Legacy bare list — rebuild to get resolved paths.
                    pass
            except Exception:
                pass

        if index is None:
            ann_path = _find_ann_file(split)
            img_dir = _find_img_dir(split)
            if not ann_path or not img_dir:
                return []
            index = _build_index(ann_path, img_dir, index_path)

        with self._lock:
            self._index_cache[split] = index
        return index

    def rebuild_index(self):
        """Force rebuild all split indices (called after upload/extraction)."""
        with self._lock:
            self._index_cache.clear()
            self._categories = None
        if _INDEX_DIR.exists():
            import shutil
            shutil.rmtree(_INDEX_DIR)
        for split in ("train", "val"):
            self._get_index(split)

    def rebuild_index_with_data(self, split_data: dict[str, dict]) -> None:
        """Rebuild indices using pre-loaded annotation dicts to avoid re-parsing JSON.

        *split_data* maps split name (e.g. ``"train"``, ``"val"``) to the
        already-parsed COCO annotation dict for that split.  Any split absent
        from the mapping is built normally from disk.
        """
        with self._lock:
            self._index_cache.clear()
            self._categories = None
        if _INDEX_DIR.exists():
            import shutil
            shutil.rmtree(_INDEX_DIR)
        for split in ("train", "val"):
            index_path = _INDEX_DIR / f"{split}_index.json"
            ann_path = _find_ann_file(split)
            img_dir = _find_img_dir(split)
            if not ann_path or not img_dir:
                continue
            preloaded = split_data.get(split)
            index = _build_index(ann_path, img_dir, index_path, data=preloaded)
            with self._lock:
                self._index_cache[split] = index

    # ── Scan splits ───────────────────────────────────────────────────────────

    def scan_splits(self) -> dict[str, dict]:
        result = {}
        for split in ("train", "val", "test"):
            if split == "test":
                result[split] = {"total": 0, "labeled": 0}
                continue
            index = self._get_index(split)
            total = len(index)
            labeled = sum(1 for e in index if e.get("anns"))
            result[split] = {"total": total, "labeled": labeled}
        return result

    # ── Download (not supported — upload only) ────────────────────────────────

    def download(self, state: dict) -> None:
        raise NotImplementedError(
            "IDD requires manual download from https://idd.insaan.iiit.ac.in/. "
            "Please upload the dataset archive via the Upload button."
        )

    def clear_data(self) -> list[str]:
        import shutil
        deleted = []
        if _IDD_ROOT.exists():
            shutil.rmtree(_IDD_ROOT)
            deleted.append(str(_IDD_ROOT))
        with self._lock:
            self._index_cache.clear()
            self._categories = None
        return deleted

    # ── Load splits ───────────────────────────────────────────────────────────

    def load_train(self, transform=None):
        index = self._get_index("train")
        img_dir = _find_img_dir("train")
        if not index or not img_dir:
            return None
        return IDDRawDataset(img_dir, index, transform=transform)

    def load_val(self, transform=None):
        index = self._get_index("val")
        img_dir = _find_img_dir("val")
        if not index or not img_dir:
            return None
        return IDDRawDataset(img_dir, index, transform=transform)

    def load_test(self, transform=None):
        return None

    def wrap_for_training(self, dataset) -> "_IDDDetectionWrapper":
        return _IDDDetectionWrapper(dataset)


register_dataset(IDDPlugin())
