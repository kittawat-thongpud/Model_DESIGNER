"""Validation tests for IDD YOLO label correctness.

Run with:
    cd /workspace/Model_DESIGNER
    python -m pytest backend/tests/test_idd_labels.py -v

Tests verify:
  1. labels/train/ and labels/val/ exist (not flat labels/)
  2. Every image in images/train/ has a corresponding label file in labels/train/
  3. Every image in images/val/  has a corresponding label file in labels/val/
  4. No label file is in the flat labels/ root (would mean wrong split_key mapping)
  5. Label files that are non-empty contain valid YOLO rows (6 cols, cls in [0,14])
  6. COCO JSON file_name paths match the actual image files on disk
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

IDD_ROOT = Path(__file__).parent.parent / "data" / "datasets" / "idd"
ANNOTATIONS_DIR = IDD_ROOT / "annotations"
IMAGES_DIR = IDD_ROOT / "images"
LABELS_DIR = IDD_ROOT / "labels"
NC = 15  # IDD has 15 detection classes (0-14)


# ── helpers ──────────────────────────────────────────────────────────────────

def _img_to_label(img_path: Path) -> Path:
    """Mirror Ultralytics' image→label path resolution."""
    rel = img_path.relative_to(IMAGES_DIR)
    return LABELS_DIR / rel.with_suffix(".txt")


def _sample(iterable, n: int = 200):
    """Return up to *n* items from iterable without loading all into memory."""
    items = []
    for item in iterable:
        items.append(item)
        if len(items) >= n:
            break
    return items


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def val_images():
    return list((IMAGES_DIR / "val").rglob("*.jpg")) + list((IMAGES_DIR / "val").rglob("*.png"))


@pytest.fixture(scope="module")
def train_images():
    return list((IMAGES_DIR / "train").rglob("*.jpg")) + list((IMAGES_DIR / "train").rglob("*.png"))


# ── tests ─────────────────────────────────────────────────────────────────────

def test_idd_root_exists():
    assert IDD_ROOT.exists(), f"IDD dataset not found at {IDD_ROOT}"


def test_images_split_dirs_exist():
    assert (IMAGES_DIR / "train").is_dir(), "images/train/ missing"
    assert (IMAGES_DIR / "val").is_dir(),   "images/val/ missing"


def test_labels_split_dirs_exist():
    """labels/train/ and labels/val/ must exist — flat labels/ root is wrong."""
    assert (LABELS_DIR / "train").is_dir(), (
        "labels/train/ missing — coco_converter wrote flat labels, SPLIT_MAP bug"
    )
    assert (LABELS_DIR / "val").is_dir(), (
        "labels/val/ missing — coco_converter wrote flat labels, SPLIT_MAP bug"
    )


def test_no_flat_camera_labels_at_root():
    """Camera-angle dirs (frontFar, sideLeft, …) must NOT appear directly under labels/."""
    camera_dirs = {"frontFar", "frontNear", "sideLeft", "sideRight", "rearNear", "highquality_16k"}
    found = [d.name for d in LABELS_DIR.iterdir() if d.is_dir() and d.name in camera_dirs]
    assert not found, (
        f"Flat camera dirs found directly under labels/: {found}\n"
        "Labels were written without split subdir — SPLIT_MAP bug."
    )


def test_train_label_count_nonzero(train_images):
    assert len(train_images) > 0, "No images found in images/train/"
    label_count = sum(1 for _ in (LABELS_DIR / "train").rglob("*.txt"))
    assert label_count > 0, "No label .txt files found in labels/train/"


def test_val_label_count_nonzero(val_images):
    assert len(val_images) > 0, "No images found in images/val/"
    label_count = sum(1 for _ in (LABELS_DIR / "val").rglob("*.txt"))
    assert label_count > 0, "No label .txt files found in labels/val/"


def test_val_images_have_label_files(val_images):
    """Sample up to 200 val images and check each has a corresponding label file."""
    sample = _sample(val_images, 200)
    missing = [img for img in sample if not _img_to_label(img).exists()]
    assert not missing, (
        f"{len(missing)} val images have no label file.\nFirst 5: {missing[:5]}"
    )


def test_train_images_have_label_files(train_images):
    """Sample up to 200 train images and check each has a corresponding label file."""
    sample = _sample(train_images, 200)
    missing = [img for img in sample if not _img_to_label(img).exists()]
    assert not missing, (
        f"{len(missing)} train images have no label file.\nFirst 5: {missing[:5]}"
    )


def test_nonempty_label_files_are_valid_yolo(val_images):
    """Non-empty label files must have valid YOLO rows: cls cx cy w h (all floats, cls in range)."""
    sample = _sample(val_images, 200)
    bad = []
    for img in sample:
        lbl = _img_to_label(img)
        if not lbl.exists() or lbl.stat().st_size == 0:
            continue
        for lineno, line in enumerate(lbl.read_text().splitlines(), 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                bad.append(f"{lbl}:{lineno} — expected 5 cols, got {len(parts)}: {line!r}")
                continue
            try:
                cls = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                bad.append(f"{lbl}:{lineno} — non-numeric: {line!r}")
                continue
            if not (0 <= cls < NC):
                bad.append(f"{lbl}:{lineno} — class {cls} out of range [0,{NC-1}]: {line!r}")
            for v in coords:
                if not (0.0 <= v <= 1.0):
                    bad.append(f"{lbl}:{lineno} — coord {v} outside [0,1]: {line!r}")
    assert not bad, f"Invalid YOLO rows found:\n" + "\n".join(bad[:20])


def test_coco_json_filenames_match_images():
    """COCO JSON file_name entries must resolve to actual files under images/<split>/."""
    for split in ("train", "val"):
        ann_path = ANNOTATIONS_DIR / f"idd_detection_{split}.json"
        if not ann_path.exists():
            pytest.skip(f"{ann_path.name} not found")
        data = json.loads(ann_path.read_text())
        img_dir = IMAGES_DIR / split
        missing = []
        for entry in data["images"][:500]:  # sample first 500
            candidate = img_dir / entry["file_name"]
            if not candidate.exists():
                missing.append(str(candidate))
        assert not missing, (
            f"[{split}] {len(missing)} COCO file_name entries not found on disk.\n"
            f"First 5: {missing[:5]}"
        )


def test_label_val_coverage_ratio(val_images):
    """At least 95% of val images should have non-empty label files."""
    if not val_images:
        pytest.skip("No val images found")
    sample = _sample(val_images, 1000)
    non_empty = sum(
        1 for img in sample
        if _img_to_label(img).exists() and _img_to_label(img).stat().st_size > 0
    )
    ratio = non_empty / len(sample)
    assert ratio >= 0.95, (
        f"Only {ratio:.1%} of sampled val images have non-empty labels "
        f"({non_empty}/{len(sample)}). Expected ≥ 95%."
    )
