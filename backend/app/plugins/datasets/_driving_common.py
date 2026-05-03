"""Shared helpers for driving-scene detection dataset plugins."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(root: Path | None) -> list[Path]:
    if root is None or not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def has_images(root: Path | None) -> bool:
    return bool(image_files(root))


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    """Create a canonical dataset view using hardlink, symlink, then copy fallback."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def write_split_txt(root: Path, split: str, paths: list[Path]) -> None:
    txt_path = root / f"{split}.txt"
    if paths:
        txt_path.write_text("\n".join(str(p.resolve()) for p in paths) + "\n")
    elif txt_path.exists():
        txt_path.unlink()


def write_yolo_label(
    path: Path,
    anns: list[dict],
    img_w: int,
    img_h: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
            continue
        cx = max(0.0, min(1.0, (x + w / 2.0) / img_w))
        cy = max(0.0, min(1.0, (y + h / 2.0) / img_h))
        nw = max(0.0, min(1.0, w / img_w))
        nh = max(0.0, min(1.0, h / img_h))
        if nw > 0 and nh > 0:
            lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    path.write_text("\n".join(lines))


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with PILImage.open(path) as im:
            return im.size
    except Exception:
        return None


class DrivingDetectionRawDataset(Dataset):
    """Returns ``(PIL_image, annotations)`` from a per-image plugin index."""

    def __init__(self, img_dir: Path, index: list[dict], transform=None):
        self._img_dir = img_dir
        self._index = index
        self._transform = transform

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        entry = self._index[idx]
        img = PILImage.open(self._img_dir / entry["file"]).convert("RGB")
        if self._transform is not None:
            img = self._transform(img)
        return img, entry.get("anns", [])

    @property
    def orig_sizes(self) -> list[tuple[int, int]]:
        return [(e["w"], e["h"]) for e in self._index]


class DrivingDetectionWrapper(Dataset):
    """Converts raw driving annotations to Ultralytics-style normalized targets."""

    def __init__(self, raw_ds: DrivingDetectionRawDataset):
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
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([
                (x + w / 2.0) / img_w,
                (y + h / 2.0) / img_h,
                w / img_w,
                h / img_h,
            ])
            labels.append(ann["category_id"])

        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)
        return img, {"boxes": boxes_t, "labels": labels_t}
