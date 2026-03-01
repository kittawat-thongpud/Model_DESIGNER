"""
COCO to YOLO Format Converter Service.

Automatically converts COCO JSON annotations to YOLO format text files
using Ultralytics' built-in converter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import DATASETS_DIR


def _get_conversion_marker_path(dataset_path: Path) -> Path:
    return dataset_path / "labels" / ".coco_yolo_conversion.json"


def _annotations_fingerprint(dataset_path: Path) -> dict[str, Any]:
    anno_dir = dataset_path / "annotations"
    # Include both COCO-style (instances_*.json) and IDD-style (idd_detection_*.json / *_train*.json)
    candidates = sorted(
        set(anno_dir.glob("instances_*.json")) |
        set(anno_dir.glob("idd_detection_*.json")) |
        set(anno_dir.glob("*_train*.json")) |
        set(anno_dir.glob("*_val*.json")),
        key=lambda p: p.name,
    )
    entries = []
    for f in candidates:
        st = f.stat()  # single stat() call per file
        entries.append({"name": f.name, "size": st.st_size, "mtime": st.st_mtime})
    return {"files": entries}


def has_coco_annotations(dataset_path: Path) -> bool:
    """Check if dataset has COCO JSON annotations."""
    anno_dir = dataset_path / "annotations"
    if not anno_dir.exists():
        return False
    
    # Look for COCO JSON files — instances_*.json or any *_train/val*.json
    return bool(
        list(anno_dir.glob("instances_*.json"))
        or list(anno_dir.glob("*_train*.json"))
        or list(anno_dir.glob("*_val*.json"))
    )


def is_already_converted(dataset_path: Path) -> bool:
    """Check if COCO dataset has already been converted to YOLO format.

    Requires that at least one .txt file has non-empty content (actual labels),
    not just that .txt files exist — a broken conversion leaves all files empty.
    """
    labels_dir = dataset_path / "labels"
    if not labels_dir.exists():
        return False

    marker_path = _get_conversion_marker_path(dataset_path)
    if marker_path.exists():
        try:
            marker = json.loads(marker_path.read_text())
            current_fp = _annotations_fingerprint(dataset_path)
            if marker.get("annotations_fingerprint") == current_fp:
                return True
        except Exception:
            pass

    for split in ["train2017", "val2017", "train", "val", "test"]:
        split_labels = labels_dir / split
        if not split_labels.exists():
            continue
        # Use rglob so nested label dirs (e.g. IDD labels/train/frontFar/.../*.txt)
        # are detected — flat glob("*.txt") misses them and causes re-conversion every run.
        checked = 0
        for txt in split_labels.rglob("*.txt"):
            if txt.stat().st_size > 0:
                return True
            checked += 1
            if checked >= 200:
                break

    return False


def convert_coco_to_yolo(
    dataset_name: str,
    *,
    use_segments: bool = False,
    use_keypoints: bool = False,
    cls91to80: bool = True,
) -> dict[str, Any]:
    """Convert COCO JSON annotations to YOLO .txt format.

    Writes labels directly to ``labels/<split>/`` mirroring Ultralytics'
    expected path convention (e.g. ``labels/train2017/000000000001.txt``).
    Does NOT depend on Ultralytics' ``convert_coco()`` whose output path is
    non-deterministic across versions.

    COCO category_id (1-based, non-contiguous 1-90) is remapped to a
    contiguous 0-based class index using the sorted category list from the
    annotation JSON.  When ``cls91to80=True`` the standard 91→80 map is
    applied first (images without a mapping are dropped).

    Returns:
        dict with keys: status, message, labels_dir, file_count (or error)
    """
    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")

    anno_dir = dataset_path / "annotations"
    if not anno_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {anno_dir}")

    if is_already_converted(dataset_path):
        return {
            "status": "already_converted",
            "message": "Dataset already has YOLO format labels",
            "labels_dir": str(dataset_path / "labels"),
        }

    # Standard COCO 91-class → 80-class remapping (0-indexed output)
    _CLS91TO80 = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
        11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
        20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
        31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
        39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
        48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
        56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
        64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
        76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
        85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
    }

    # Map: split name in annotation filename → output subdir name.
    # COCO-2017 style uses explicit subdir names that are already final.
    # For generic train/val/test splits we detect at runtime whether a
    # split-specific images/<split>/ directory exists:
    #   - If images/train/ exists  → labels must go to labels/train/   (YOLO path resolution)
    #   - If only images/ is flat  → labels go directly to labels/
    SPLIT_MAP = {
        "train2017": "train2017",
        "val2017":   "val2017",
        "test2017":  "test2017",
    }
    img_root = dataset_path / "images"
    for _sp in ("train", "val", "test"):
        if (img_root / _sp).is_dir():
            SPLIT_MAP[_sp] = _sp   # split-specific image dir → matching label subdir
        else:
            SPLIT_MAP[_sp] = ""    # flat image layout → flat label layout

    # Collect annotation JSON files: instances_*.json OR *_train*.json / *_val*.json
    instances_files = list(anno_dir.glob("instances_*.json"))
    if not instances_files:
        # Fallback: any JSON containing a split keyword in the name
        instances_files = [
            p for p in sorted(anno_dir.glob("*.json"))
            if any(kw in p.stem.lower() for kw in ("train", "val", "test"))
            and not p.name.startswith(".")
        ]
    if not instances_files:
        return {
            "status": "error",
            "message": "No annotation JSON files containing train/val/test found",
            "error": "No detection annotations found",
        }

    total_written = 0
    total_non_empty = 0
    labels_root = dataset_path / "labels"
    loaded_data: dict[str, Any] = {}  # split_key → parsed annotation dict

    try:
        for json_path in instances_files:
            # Determine split subdir from filename stem
            # e.g. instances_train2017 → train2017
            #      idd_detection_train → train
            #      idd_detection_val   → val
            stem = json_path.stem
            split_key = stem.replace("instances_", "")
            # For patterns like idd_detection_train, take the last token after "_"
            # that matches a known split keyword
            if split_key not in SPLIT_MAP:
                for kw in ("train2017", "val2017", "test2017", "train", "val", "test"):
                    if kw in split_key:
                        split_key = kw
                        break
            out_subdir = SPLIT_MAP.get(split_key, split_key)
            out_dir = labels_root / out_subdir if out_subdir else labels_root
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(json_path) as f:
                data = json.load(f)
            loaded_data[split_key] = data

            # Build contiguous class map from this annotation file's categories
            cats_sorted = sorted(data.get("categories", []), key=lambda c: c["id"])
            if cls91to80:
                cat_id_to_cls = _CLS91TO80
            else:
                cat_id_to_cls = {c["id"]: i for i, c in enumerate(cats_sorted)}

            # Build image_id → {file_name, width, height}
            img_info: dict[int, dict] = {}
            for img in data.get("images", []):
                img_info[img["id"]] = {
                    "file": img["file_name"],
                    "w": img["width"],
                    "h": img["height"],
                }

            # Group annotations by image_id
            ann_by_img: dict[int, list] = {img_id: [] for img_id in img_info}
            for ann in data.get("annotations", []):
                img_id = ann.get("image_id")
                if img_id not in ann_by_img:
                    continue
                if ann.get("iscrowd", 0):
                    continue
                cat_id = ann.get("category_id")
                cls_idx = cat_id_to_cls.get(cat_id)
                if cls_idx is None:
                    continue
                ann_by_img[img_id].append((cls_idx, ann["bbox"]))

            # ── Pass A: build all (txt_path, content) pairs in memory ──────
            # Collecting everything first lets us batch-mkdir unique parent dirs
            # and avoid interleaving mkdir+open+write syscalls on NFS.
            pending: list[tuple[Path, str]] = []
            for img_id, info in img_info.items():
                file_path = Path(info["file"])
                rel_dir = file_path.parent
                txt_path = out_dir / rel_dir / f"{file_path.stem}.txt"
                w, h = info["w"], info["h"]
                lines: list[str] = []
                for cls_idx, bbox in ann_by_img.get(img_id, []):
                    bx, by, bw, bh = bbox
                    cx = max(0.0, min(1.0, (bx + bw / 2) / w))
                    cy = max(0.0, min(1.0, (by + bh / 2) / h))
                    nw = max(0.0, min(1.0, bw / w))
                    nh = max(0.0, min(1.0, bh / h))
                    if nw > 0 and nh > 0:
                        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                pending.append((txt_path, "\n".join(lines)))

            # ── Pass B: mkdir unique parent dirs (batch, not per-file) ───────
            seen_parents: set[Path] = set()
            for txt_path, _ in pending:
                p = txt_path.parent
                if p not in seen_parents:
                    p.mkdir(parents=True, exist_ok=True)
                    seen_parents.add(p)

            # ── Pass C: write all files sequentially ──────────────────────────
            for txt_path, content in pending:
                txt_path.write_text(content)
                total_written += 1
                if content:
                    total_non_empty += 1

        try:
            marker_path = _get_conversion_marker_path(dataset_path)
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(
                json.dumps(
                    {
                        "annotations_fingerprint": _annotations_fingerprint(dataset_path),
                        "total_written": total_written,
                        "total_non_empty": total_non_empty,
                    }
                )
            )
        except Exception:
            pass

        return {
            "status": "success",
            "message": f"Converted {total_written} annotation files to YOLO format",
            "labels_dir": str(labels_root),
            "file_count": total_written,
            "loaded_data": loaded_data,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Conversion failed: {str(e)}",
            "error": str(e),
        }


def _repair_marker_if_needed(dataset_path: Path) -> bool:
    """Rewrite the conversion marker if labels exist but the marker is stale or has
    an empty fingerprint (caused by the old bug where only instances_*.json was scanned).

    Returns True if marker was repaired (caller can skip conversion entirely).
    """
    labels_dir = dataset_path / "labels"
    if not labels_dir.exists():
        return False

    # Fast check: do any non-empty .txt label files exist?
    has_labels = False
    for split in ("train2017", "val2017", "train", "val"):
        split_dir = labels_dir / split
        if not split_dir.exists():
            continue
        for txt in split_dir.rglob("*.txt"):
            if txt.stat().st_size > 0:
                has_labels = True
                break
        if has_labels:
            break

    if not has_labels:
        return False

    # Labels are valid — ensure marker reflects the correct fingerprint.
    current_fp = _annotations_fingerprint(dataset_path)
    marker_path = _get_conversion_marker_path(dataset_path)
    try:
        if marker_path.exists():
            existing = json.loads(marker_path.read_text())
            if existing.get("annotations_fingerprint") == current_fp:
                return True  # already correct
        # Rewrite with correct fingerprint
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps({
            "annotations_fingerprint": current_fp,
            "total_written": -1,   # unknown (repaired)
            "total_non_empty": -1,
            "repaired": True,
        }))
    except Exception:
        pass
    return True  # labels are valid, skip conversion


def auto_convert_if_needed(dataset_name: str) -> dict[str, Any] | None:
    """Automatically convert COCO dataset to YOLO format if needed.
    
    Called when dataset is loaded or scanned.
    
    Returns:
        Conversion result dict if conversion was performed, None otherwise
    """
    dataset_path = DATASETS_DIR / dataset_name
    
    # Check if dataset has COCO annotations
    if not has_coco_annotations(dataset_path):
        return None
    
    # Fast path: marker matches current fingerprint → already converted.
    if is_already_converted(dataset_path):
        return None

    # Repair path: labels exist but marker was stale (old fingerprint bug).
    # Rewrite marker and skip conversion — avoids re-converting every train run.
    if _repair_marker_if_needed(dataset_path):
        return None

    # IDD uses its own 15-class scheme — do NOT remap via COCO 91→80
    no_remap = {"idd", "idd_detection"}
    cls91to80 = dataset_name.lower() not in no_remap

    # Auto-convert — result may contain loaded_data for callers that want to reuse parsed JSON
    return convert_coco_to_yolo(dataset_name, cls91to80=cls91to80)
