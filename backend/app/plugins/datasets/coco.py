"""COCO 2017 Detection dataset plugin.

Architecture:
- COCORawDataset: lightweight dataset backed by a per-image JSON index.
  Images are read lazily from disk (PIL). Returns (PIL_image, [annotation_dicts]).
  Used by BOTH the dataset browser and training — single shared representation.
- _build_index(): one-time conversion of COCO annotation JSON into a compact
  per-image index file. Much faster to load than the full 450 MB annotation JSON.
- _COCODetectionWrapper: training-only wrapper that converts raw annotations
  into normalized tensor format {boxes, labels}. Applied via wrap_for_training().
"""
from __future__ import annotations
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image as PILImage

from app.config import DATASETS_DIR
from ..base import DatasetPlugin
from ..loader import register_dataset

_COCO_ROOT = DATASETS_DIR / "coco"
_INDEX_DIR = _COCO_ROOT / "_index"

_IMAGE_ZIPS = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
]
_ANN_ZIPS = [
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
]


# ── Lightweight per-image index builder ────────────────────────────────────────

def _build_index(ann_path: Path, img_dir: Path, index_path: Path) -> list[dict]:
    """One-time: convert a COCO annotation JSON into a lightweight per-image index.

    The index is a JSON list of:
      {"file": "000000000001.jpg", "w": 640, "h": 480,
       "anns": [{"bbox": [x,y,w,h], "category_id": 1, "area": 1234, "iscrowd": 0}, ...]}

    Loading this index is ~50× faster than pycocotools parsing the full COCO JSON.
    """
    with open(ann_path) as f:
        data = json.load(f)

    # Build image_id → image info lookup
    img_lookup: dict[int, dict] = {}
    for img_info in data.get("images", []):
        img_id = img_info["id"]
        img_lookup[img_id] = {
            "file": img_info["file_name"],
            "w": img_info["width"],
            "h": img_info["height"],
            "anns": [],
        }

    # Attach annotations to their images
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id in img_lookup:
            img_lookup[img_id]["anns"].append({
                "bbox": ann["bbox"],             # [x, y, w, h] pixels
                "category_id": ann["category_id"],
                "area": ann.get("area", 0),
                "iscrowd": ann.get("iscrowd", 0),
            })

    # Only keep images that exist on disk
    index = []
    for entry in img_lookup.values():
        if (img_dir / entry["file"]).exists():
            index.append(entry)
    index.sort(key=lambda e: e["file"])

    # Save
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(index, f, separators=(",", ":"))  # compact

    # Also extract categories into a separate small file
    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    cats_path = index_path.parent / "categories.json"
    if not cats_path.exists():
        with open(cats_path, "w") as f:
            json.dump(cats, f, indent=1)

    return index


# ── Lightweight raw dataset ───────────────────────────────────────────────────

class COCORawDataset(Dataset):
    """Lightweight COCO dataset backed by a per-image index.

    - Images read lazily from disk (PIL) — never pre-loaded into RAM.
    - Annotations are plain dicts in COCO format.
    - Returns (PIL_image, [annotation_dicts]) — same raw format for
      both the dataset browser and training pipeline.
    - No dependency on torchvision or pycocotools for loading.
    """

    def __init__(self, img_dir: Path, index: list[dict], transform=None):
        self._img_dir = img_dir
        self._index = index
        self._transform = transform

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        entry = self._index[idx]
        img_path = self._img_dir / entry["file"]
        img = PILImage.open(img_path).convert("RGB")

        if self._transform is not None:
            img = self._transform(img)

        return img, entry["anns"]

    @property
    def orig_sizes(self) -> list[tuple[int, int]]:
        """(w, h) for each image from the index (no disk read)."""
        return [(e["w"], e["h"]) for e in self._index]


# ── Plugin ────────────────────────────────────────────────────────────────────

class COCOPlugin(DatasetPlugin):

    def __init__(self):
        self._categories: list[dict] | None = None
        self._cat_names: list[str] | None = None
        self._split_counts: dict[str, dict] | None = None
        # Cache loaded indices: split → list[dict]
        self._indices: dict[str, list[dict]] = {}
        self._index_lock = threading.Lock()

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "coco"

    @property
    def display_name(self) -> str:
        return "COCO 2017 (Detection)"

    @property
    def task_type(self) -> str:
        return "detection"

    @property
    def input_shape(self) -> list[int]:
        return [3, 640, 640]

    # ── Categories ────────────────────────────────────────────────────────────

    def _ensure_categories(self):
        """Lazily load categories from index or annotation JSON."""
        if self._categories is not None:
            return
        # Prefer the lightweight categories.json from our index
        cats_path = _INDEX_DIR / "categories.json"
        if cats_path.exists():
            with open(cats_path) as f:
                cats = json.load(f)
        else:
            ann_path = _COCO_ROOT / "annotations" / "instances_train2017.json"
            if not ann_path.exists():
                self._categories = []
                self._cat_names = []
                return
            with open(ann_path) as f:
                data = json.load(f)
            cats = sorted(data.get("categories", []), key=lambda c: c["id"])
        self._categories = cats
        self._cat_names = [c["name"] for c in cats]

    @property
    def num_classes(self) -> int:
        self._ensure_categories()
        return len(self._categories or [])

    @property
    def class_names(self) -> list[str]:
        self._ensure_categories()
        return self._cat_names or []

    @property
    def category_id_map(self) -> dict[int, str]:
        """Map COCO category_id (non-contiguous 1–90) → category name."""
        self._ensure_categories()
        return {c["id"]: c["name"] for c in (self._categories or [])}

    @property
    def train_size(self) -> int:
        if self._split_counts and "train" in self._split_counts:
            return self._split_counts["train"].get("labeled", 118287)
        return 118287

    @property
    def test_size(self) -> int:
        return 0

    @property
    def val_size(self) -> int:
        if self._split_counts and "val" in self._split_counts:
            return self._split_counts["val"].get("labeled", 5000)
        return 5000

    @property
    def normalization(self) -> tuple[tuple, tuple]:
        return ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # ── Data dirs & availability ─────────────────────────────────────────────

    @property
    def data_dirs(self) -> list[str]:
        return ["coco"]

    def is_available(self) -> bool:
        return (_COCO_ROOT / "images" / "train2017").exists()

    # ── Index management ─────────────────────────────────────────────────────

    _SPLIT_MAP = {
        "train": ("images/train2017", "annotations/instances_train2017.json", "train.json"),
        "val":   ("images/val2017",   "annotations/instances_val2017.json",   "val.json"),
    }

    def _get_index(self, split: str) -> list[dict]:
        """Return the per-image index for a split, building it if needed."""
        with self._index_lock:
            if split in self._indices:
                return self._indices[split]

        if split not in self._SPLIT_MAP:
            return []

        img_sub, ann_sub, idx_name = self._SPLIT_MAP[split]
        img_dir = _COCO_ROOT / img_sub
        ann_path = _COCO_ROOT / ann_sub
        index_path = _INDEX_DIR / idx_name

        # Load pre-built index if it exists
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
        elif ann_path.exists():
            # Build index from COCO annotation JSON (one-time)
            index = _build_index(ann_path, img_dir, index_path)
        else:
            index = []

        with self._index_lock:
            self._indices[split] = index
        return index

    # ── Scan splits ──────────────────────────────────────────────────────────

    def scan_splits(self) -> dict[str, dict]:
        splits: dict[str, dict] = {}
        for split_name in ("train", "val"):
            index = self._get_index(split_name)
            total = len(index)
            labeled = sum(1 for e in index if e.get("anns"))
            splits[split_name] = {"total": total, "labeled": labeled}
        splits["test"] = {"total": 0, "labeled": 0}
        self._split_counts = splits
        return splits

    # ── Download ─────────────────────────────────────────────────────────────

    def download(self, state: dict) -> None:
        from app.utils.download import download_and_extract

        coco_root = str(_COCO_ROOT)
        images_dir = os.path.join(coco_root, "images")
        os.makedirs(images_dir, exist_ok=True)

        total_files = len(_IMAGE_ZIPS) + len(_ANN_ZIPS)
        state["total_files"] = total_files
        state["completed_files"] = 0
        step_lock = threading.Lock()

        def _mark_done():
            with step_lock:
                state["completed_files"] = state.get("completed_files", 0) + 1

        def _dl_image_zip(url: str):
            folder = os.path.basename(url).replace(".zip", "")
            fk = folder
            if os.path.isdir(os.path.join(images_dir, folder)):
                _mark_done()
                state["message"] = f"{folder}/ already exists, skipping"
                return
            download_and_extract(url, images_dir, state, f"[?/{total_files}]", file_key=fk)
            _mark_done()

        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(_dl_image_zip, u): u for u in _IMAGE_ZIPS}
            for f in as_completed(futs):
                f.result()

        def _dl_ann_zip(url: str):
            fname = os.path.basename(url).replace(".zip", "")
            download_and_extract(url, coco_root, state, f"[?/{total_files}]", file_key=fname)
            _mark_done()

        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(_dl_ann_zip, u): u for u in _ANN_ZIPS}
            for f in as_completed(futs):
                f.result()

        state["files"] = {}
        state["completed_files"] = total_files

        # Build index immediately after download so first load is fast
        for split_name in ("train", "val"):
            self._get_index(split_name)

    # ── Loading ──────────────────────────────────────────────────────────────

    def _cat_id_to_contiguous(self) -> dict[int, int]:
        """Map non-contiguous COCO category_id (1-90) → contiguous 0-based index."""
        self._ensure_categories()
        return {c["id"]: idx for idx, c in enumerate(self._categories or [])}

    def _load_split_ds(self, split: str, transform=None) -> COCORawDataset | None:
        """Return a COCORawDataset for the given split."""
        if split not in self._SPLIT_MAP:
            return None
        img_sub = self._SPLIT_MAP[split][0]
        img_dir = _COCO_ROOT / img_sub
        if not img_dir.exists():
            return None
        index = self._get_index(split)
        if not index:
            return None
        return COCORawDataset(img_dir, index, transform=transform)

    def load_train(self, transform=None):
        return self._load_split_ds("train", transform)

    def load_val(self, transform=None):
        return self._load_split_ds("val", transform)

    def load_test(self, transform=None):
        return None

    def wrap_for_training(self, dataset) -> "_COCODetectionWrapper":
        """Wrap a raw COCORawDataset for training (normalized xywh tensors)."""
        return _COCODetectionWrapper(dataset, self._cat_id_to_contiguous())


class _COCODetectionWrapper(Dataset):
    """Training wrapper: converts raw (PIL_image, [ann_dicts]) → (image_tensor, target_dict).

    target_dict has:
      - boxes: (N, 4) tensor in normalized xywh (center-x, center-y, w, h) 0-1
      - labels: (N,) tensor of contiguous class indices
    """

    def __init__(self, coco_ds, cat_id_to_idx: dict[int, int]):
        self._ds = coco_ds
        self._cat_map = cat_id_to_idx

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        img, anns = self._ds[idx]

        # img is already transformed (PIL→Tensor via transform)
        # anns is a list of COCO annotation dicts
        boxes = []
        labels = []
        # Get image dimensions for normalization
        if isinstance(img, torch.Tensor):
            _, img_h, img_w = img.shape
        else:
            img_w, img_h = img.size  # PIL

        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            cat_id = ann["category_id"]
            if cat_id not in self._cat_map:
                continue
            # COCO bbox is [x_min, y_min, width, height] in pixels
            x_min, y_min, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            # Convert to center-x, center-y, w, h normalized 0-1
            cx = (x_min + bw / 2) / img_w
            cy = (y_min + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h
            boxes.append([cx, cy, nw, nh])
            labels.append(self._cat_map[cat_id])

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)

        return img, {"boxes": boxes_t, "labels": labels_t}


register_dataset(COCOPlugin())
