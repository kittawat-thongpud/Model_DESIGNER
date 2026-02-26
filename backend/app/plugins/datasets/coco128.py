"""COCO128 dataset plugin - small 128-image subset for quick testing.

COCO128 is a tiny version of COCO with 128 images from train2017.
Perfect for quick prototyping and testing detection models.
"""
from __future__ import annotations
import json
import os
import threading
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image as PILImage

from app.config import DATASETS_DIR
from ..base import DatasetPlugin
from ..loader import register_dataset

_COCO128_ROOT = DATASETS_DIR / "coco128"
_INDEX_PATH = _COCO128_ROOT / "_index.json"

# COCO128 download URL (Ultralytics hosted)
_COCO128_ZIP = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"


# ── Lightweight per-image index builder ────────────────────────────────────────

def _build_index() -> list[dict]:
    """Build a lightweight per-image index from COCO128 labels.
    
    COCO128 uses YOLO format labels (one .txt per image).
    Convert to our standard format: {"file": "...", "w": ..., "h": ..., "anns": [...]}
    """
    images_dir = _COCO128_ROOT / "images" / "train2017"
    labels_dir = _COCO128_ROOT / "labels" / "train2017"
    
    if not images_dir.exists():
        return []
    
    index = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        # Get image dimensions
        try:
            with PILImage.open(img_path) as img:
                w, h = img.size
        except:
            continue
        
        # Load YOLO format labels (class cx cy w h normalized)
        label_path = labels_dir / f"{img_path.stem}.txt"
        anns = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        # Convert YOLO normalized center format to pixel xywh
                        x_min = (cx - bw / 2) * w
                        y_min = (cy - bh / 2) * h
                        box_w = bw * w
                        box_h = bh * h
                        anns.append({
                            "bbox": [x_min, y_min, box_w, box_h],
                            "category_id": cls_id,
                            "area": box_w * box_h,
                            "iscrowd": 0,
                        })
        
        index.append({
            "file": img_path.name,
            "w": w,
            "h": h,
            "anns": anns,
        })
    
    # Save index
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_INDEX_PATH, "w") as f:
        json.dump(index, f, separators=(",", ":"))
    
    return index


# ── Lightweight raw dataset ───────────────────────────────────────────────────

class COCO128RawDataset(Dataset):
    """Lightweight COCO128 dataset backed by a per-image index.
    
    Returns (PIL_image, [annotation_dicts]) in COCO format.
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

class COCO128Plugin(DatasetPlugin):
    
    def __init__(self):
        # COCO128 uses the same 80 classes as COCO
        self._categories = [
            {"id": i, "name": name} for i, name in enumerate([
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            ])
        ]
        self._index: list[dict] | None = None
        self._index_lock = threading.Lock()
    
    # ── Identity ──────────────────────────────────────────────────────────────
    
    @property
    def name(self) -> str:
        return "coco128"
    
    @property
    def display_name(self) -> str:
        return "COCO128 (Detection)"
    
    @property
    def task_type(self) -> str:
        return "detection"
    
    @property
    def input_shape(self) -> list[int]:
        return [3, 640, 640]
    
    # ── Categories ────────────────────────────────────────────────────────────
    
    @property
    def num_classes(self) -> int:
        return 80
    
    @property
    def class_names(self) -> list[str]:
        return [c["name"] for c in self._categories]
    
    def category_id_to_name(self) -> dict[int, str]:
        return {c["id"]: c["name"] for c in self._categories}
    
    @property
    def category_id_map(self) -> dict[int, str]:
        """Map category_id → category name."""
        return {c["id"]: c["name"] for c in self._categories}
    
    @property
    def train_size(self) -> int:
        return 128
    
    @property
    def test_size(self) -> int:
        return 0
    
    @property
    def val_size(self) -> int:
        return 0
    
    @property
    def normalization(self) -> tuple[tuple, tuple]:
        return ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    # ── Data dirs & availability ─────────────────────────────────────────────
    
    @property
    def data_dirs(self) -> list[str]:
        return ["coco128"]
    
    def is_available(self) -> bool:
        return (_COCO128_ROOT / "images" / "train2017").exists()
    
    # ── Index management ─────────────────────────────────────────────────────
    
    def _get_index(self) -> list[dict]:
        """Return the per-image index, building it if needed."""
        with self._index_lock:
            if self._index is not None:
                return self._index
        
        # Load pre-built index if it exists
        if _INDEX_PATH.exists():
            with open(_INDEX_PATH) as f:
                index = json.load(f)
        elif (_COCO128_ROOT / "images" / "train2017").exists():
            # Build index from YOLO labels (one-time)
            index = _build_index()
        else:
            index = []
        
        with self._index_lock:
            self._index = index
        return index
    
    # ── Scan splits ──────────────────────────────────────────────────────────
    
    def scan_splits(self) -> dict[str, dict]:
        index = self._get_index()
        total = len(index)
        labeled = sum(1 for e in index if e.get("anns"))
        return {
            "train": {"total": total, "labeled": labeled},
            "val": {"total": 0, "labeled": 0},
            "test": {"total": 0, "labeled": 0},
        }
    
    # ── Download ─────────────────────────────────────────────────────────────
    
    def download(self, state: dict) -> None:
        from app.utils.download import download_and_extract
        
        state["message"] = "Downloading COCO128..."
        state["progress"] = 0
        
        # Download and extract (extracts to DATASETS_DIR)
        download_and_extract(
            _COCO128_ZIP,
            str(DATASETS_DIR),
            state,
            "COCO128"
        )
        
        state["message"] = "Building index..."
        state["progress"] = 90
        
        # Build index
        _build_index()
        
        state["message"] = "Complete"
        state["progress"] = 100
    
    def clear_data(self) -> list[str]:
        """Delete downloaded COCO128 dataset files and return list of deleted directories."""
        import shutil
        
        deleted = []
        if _COCO128_ROOT.exists():
            shutil.rmtree(_COCO128_ROOT)
            deleted.append(str(_COCO128_ROOT))
            # Clear cached index
            with self._index_lock:
                self._index = None
        
        return deleted
    
    # ── Load splits ──────────────────────────────────────────────────────────
    
    def _cat_id_to_contiguous(self) -> dict[int, int]:
        """Map category_id → contiguous 0-based index (identity for COCO128)."""
        return {c["id"]: c["id"] for c in self._categories}
    
    def load_train(self, transform=None):
        img_dir = _COCO128_ROOT / "images" / "train2017"
        if not img_dir.exists():
            return None
        index = self._get_index()
        if not index:
            return None
        return COCO128RawDataset(img_dir, index, transform=transform)
    
    def load_val(self, transform=None):
        return None
    
    def load_test(self, transform=None):
        return None
    
    def wrap_for_training(self, dataset) -> "_COCO128DetectionWrapper":
        """Wrap a raw COCO128RawDataset for training (normalized xywh tensors)."""
        return _COCO128DetectionWrapper(dataset, self._cat_id_to_contiguous())


class _COCO128DetectionWrapper(Dataset):
    """Training wrapper: converts raw (PIL_image, [ann_dicts]) → (image_tensor, target_dict).
    
    target_dict has:
      - boxes: (N, 4) tensor in normalized xywh (center-x, center-y, w, h) 0-1
      - labels: (N,) tensor of contiguous class indices
    """
    
    def __init__(self, coco128_ds, cat_id_to_idx: dict[int, int]):
        self._ds = coco128_ds
        self._cat_map = cat_id_to_idx
    
    def __len__(self):
        return len(self._ds)
    
    def __getitem__(self, idx):
        img, anns = self._ds[idx]
        
        # img is already transformed (PIL→Tensor via transform)
        # anns is a list of annotation dicts
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
            # bbox is [x_min, y_min, width, height] in pixels
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


register_dataset(COCO128Plugin())
