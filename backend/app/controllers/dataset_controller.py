"""
Dataset Controller — thin universal layer delegating to dataset plugins.

All dataset-specific logic (loading, scanning, downloading, availability)
lives in plugins (backend/app/plugins/datasets/). This controller only
orchestrates and exposes REST endpoints.
"""
from __future__ import annotations
import json
import shutil
import threading

import numpy as np
import asyncio
import tempfile
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from torch.utils.data import Subset

from datetime import datetime, timezone

from ..config import DATASETS_DIR, SPLITS_DIR
from ..schemas.dataset_schema import DatasetInfo
from ..schemas.dataset_schema import (
    SplitConfig, UpdatePartitionMethod, CreatePartition,
    SplitPartitionItem, SplitPartitionBody,
)
from ..services.dataset_registry import get_all_datasets, get_dataset_info as _get_info
from ..services import coco_converter
from ..plugins.loader import get_dataset_plugin

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])


# ── Plugin helper ────────────────────────────────────────────────────────────

def _get_plugin(name: str):
    """Get the dataset plugin, raise 404 if not found."""
    plugin = get_dataset_plugin(name.lower())
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return plugin


# ── List / Info endpoints ────────────────────────────────────────────────────

@router.get("/", response_model=list[DatasetInfo], summary="List available datasets")
async def list_datasets():
    """Return info about all supported datasets."""
    return get_all_datasets()


@router.get("/{name}/info", response_model=DatasetInfo, summary="Get dataset metadata")
async def get_dataset_info(name: str):
    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return ds


@router.get("/{name}/preview", summary="Preview dataset samples")
async def preview_dataset(name: str, count: int = 8):
    """Return a few sample images as base64-encoded PNGs."""
    import base64
    import io

    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    plugin = _get_plugin(name)

    try:
        from PIL import Image
        dataset = plugin.load_train()
        if dataset is None:
            raise HTTPException(status_code=400, detail=f"Cannot load train split for '{name}'")

        samples = []
        for i in range(min(count, len(dataset))):
            img, label = dataset[i]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            samples.append({
                "index": i,
                "label": label,
                "class_name": ds.class_names[label] if isinstance(label, int) and label < len(ds.class_names) else str(label),
                "image_base64": b64,
            })
        return {"dataset": name, "count": len(samples), "samples": samples}
    except ImportError:
        raise HTTPException(status_code=500, detail="torchvision not available")


# ── Meta helpers ─────────────────────────────────────────────────────────────

def _meta_path(name: str):
    """Meta file lives inside the dataset's first data dir under DATASETS_DIR."""
    plugin = get_dataset_plugin(name.lower())
    if plugin:
        folder = DATASETS_DIR / plugin.data_dirs[0]
    else:
        folder = DATASETS_DIR / name.lower()
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "meta.json"


def _load_meta(name: str) -> dict | None:
    p = _meta_path(name)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _save_meta(name: str, meta: dict):
    _meta_path(name).write_text(json.dumps(meta, indent=2))


def _delete_meta(name: str):
    p = _meta_path(name)
    if p.exists():
        p.unlink()


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore
    return f"{n:.1f} PB"


def _scan_dataset_meta(name: str) -> dict:
    """Scan a dataset via its plugin and build metadata. Cached in meta.json."""
    key = name.lower()
    plugin = get_dataset_plugin(key)
    if not plugin:
        return {"name": key, "available": False}

    available = plugin.is_available()
    disk_bytes = plugin.disk_size_bytes() if available else 0

    # Get raw split counts from plugin (fast — no heavy loading)
    splits = plugin.scan_splits() if available else {}

    # Apply transfer config to compute effective split counts
    split_cfg = _load_split_config(key)
    raw_train = splits.get("train", {}).get("labeled", 0)
    raw_test = splits.get("test", {}).get("labeled", 0)
    raw_val = splits.get("val", {}).get("labeled", 0)
    counts = _compute_transfer_counts(split_cfg, raw_train, raw_test, raw_val)

    for s in ("train", "val", "test"):
        if s not in splits:
            splits[s] = {"total": 0, "labeled": 0}
        splits[s]["effective"] = counts[f"{s}_count"]

    meta = {
        "name": key,
        "available": available,
        "disk_size_bytes": disk_bytes,
        "disk_size_human": _human_size(disk_bytes),
        "splits": splits,
        "split_config": split_cfg,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_meta(key, meta)
    return meta


# ── Status / Meta / Scan / Delete endpoints ──────────────────────────────────

@router.get("/{name}/status", summary="Check dataset download status")
async def dataset_status(name: str):
    """Check whether a dataset is downloaded. Reads cached meta only (fast)."""
    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    key = name.lower()
    plugin = get_dataset_plugin(key)
    available = plugin.is_available() if plugin else False

    result: dict = {"name": key, "available": available}
    if plugin:
        result["manual_download"] = getattr(plugin, "manual_download", False)
        result["instructions"] = getattr(plugin, "upload_instructions", "")
    meta = _load_meta(key)
    if meta:
        result["meta"] = meta
    return result


@router.get("/{name}/meta", summary="Get dataset metadata")
async def get_meta(name: str):
    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    meta = _load_meta(name.lower())
    if not meta:
        meta = _scan_dataset_meta(name.lower())
    return meta


@router.post("/{name}/scan", summary="Force rescan dataset metadata")
async def scan_meta(name: str):
    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    
    # Auto-convert COCO to YOLO if needed
    coco_converter.auto_convert_if_needed(name.lower())
    
    return _scan_dataset_meta(name.lower())


@router.post("/{name}/convert-coco", summary="Convert COCO JSON to YOLO format")
async def convert_coco_dataset(
    name: str,
    use_segments: bool = False,
    use_keypoints: bool = False,
    cls91to80: bool = True,
):
    """Manually convert COCO JSON annotations to YOLO format text files."""
    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    
    result = coco_converter.convert_coco_to_yolo(
        name.lower(),
        use_segments=use_segments,
        use_keypoints=use_keypoints,
        cls91to80=cls91to80,
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result


@router.delete("/{name}/data", summary="Delete downloaded dataset files")
async def delete_dataset_data(name: str):
    plugin = _get_plugin(name)
    _invalidate_cache(name.lower())
    deleted = plugin.clear_data()

    meta = {
        "name": name.lower(),
        "available": False,
        "disk_size_bytes": 0,
        "disk_size_human": "0 B",
        "splits": {},
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_meta(name.lower(), meta)
    return {"name": name.lower(), "deleted_dirs": deleted, "available": False}



# ── NOTE: Sample browsing routes (/{name}/samples, /{name}/samples/{index})
# have been extracted to dataset_samples_controller.py to reduce file size.


# ── Download with progress ───────────────────────────────────────────────────

_download_tasks: dict[str, dict] = {}
_download_lock = threading.Lock()


@router.post("/{name}/download", summary="Start dataset download")
async def start_download(name: str):
    plugin = _get_plugin(name)
    key = name.lower()

    with _download_lock:
        existing = _download_tasks.get(key)
        if existing and existing["status"] == "downloading":
            return existing

        _download_tasks[key] = {
            "status": "downloading",
            "progress": 0,
            "message": "Preparing download...",
            "current_file": "",
            "bytes_downloaded": 0,
            "bytes_total": 0,
            "rate_bps": 0,
            "eta_seconds": -1,
            "files": {},
            "total_files": 0,
            "completed_files": 0,
        }

    thread = threading.Thread(target=_bg_download, args=(key,), daemon=True)
    thread.start()
    return _download_tasks[key]


@router.get("/{name}/download-status", summary="Poll dataset download progress")
async def download_status_endpoint(name: str):
    key = name.lower()
    state = _download_tasks.get(key)
    if not state:
        return {"status": "idle", "progress": 0, "message": "", "files": {}}
    return {k: v for k, v in state.items() if not k.startswith("_")}


def _bg_download(name: str):
    """Background thread: delegate download to plugin, then scan meta."""
    state = _download_tasks[name]
    try:
        plugin = get_dataset_plugin(name)
        if not plugin:
            raise ValueError(f"No plugin for dataset '{name}'")

        # Clear old data before fresh download
        plugin.clear_data()
        state["message"] = "Starting download..."

        # Plugin handles all download logic
        plugin.download(state)

        # Invalidate cached dataset objects and scan metadata
        _invalidate_cache(name)
        state["message"] = "Scanning metadata..."
        _scan_dataset_meta(name)
        state["status"] = "complete"
        state["progress"] = 100
        state["message"] = "Download complete!"
    except Exception as e:
        state["status"] = "error"
        state["progress"] = 0
        state["message"] = f"Error: {e}"


# ── Workspace scan + local import ────────────────────────────────────────────

@router.get("/{name}/workspace-scan", summary="Check if dataset files exist in workspace")
async def workspace_scan(name: str):
    """Fast scan: check top-level entries + detect pending _download_ archives."""
    _get_plugin(name)  # 404 if not registered
    key = name.lower()
    dest = DATASETS_DIR / key
    if not dest.exists():
        return {"found": False, "path": str(dest), "file_count": 0,
                "pending_archive": None}

    def _looks_like_archive(p: Path) -> bool:
        if not p.is_file():
            return False
        n = p.name.lower()
        if n.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            return True
        try:
            with open(p, "rb") as _mf:
                hdr = _mf.read(8)
            return hdr[:4] == b"PK\x03\x04" or hdr[:2] == b"\x1f\x8b" or hdr[:3] == b"BZh" or hdr[:5] == b"ustar"
        except Exception:
            return False

    # Fast: only iterate top-level entries (no recursive stat)
    top_entries = list(dest.iterdir())
    file_count = sum(1 for f in top_entries if f.is_file())
    dir_count  = sum(1 for f in top_entries if f.is_dir())

    # Detect any leftover _download_*.{tar.gz,zip,tgz,...} archive
    pending_archive = None
    for f in top_entries:
        if f.is_file() and f.name.startswith("_download_") and _looks_like_archive(f):
            pending_archive = {"path": str(f), "name": f.name, "size_bytes": f.stat().st_size}
            break
    if pending_archive is None:
        for f in top_entries:
            if _looks_like_archive(f):
                pending_archive = {"path": str(f), "name": f.name, "size_bytes": f.stat().st_size}
                break

    found = (file_count + dir_count) > 0
    return {
        "found": found,
        "path": str(dest),
        "file_count": file_count,
        "dir_count": dir_count,
        "pending_archive": pending_archive,
    }


@router.post("/{name}/resume-extract", summary="Extract a pending _download_ archive found in the dataset directory")
async def resume_extract(name: str):
    """If a leftover _download_* archive exists in the dataset dir, extract it now."""
    key = name.lower()
    plugin = _get_plugin(key)
    dest = DATASETS_DIR / key

    def _looks_like_archive(p: Path) -> bool:
        if not p.is_file():
            return False
        n = p.name.lower()
        if n.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            return True
        try:
            with open(p, "rb") as _mf:
                hdr = _mf.read(8)
            return hdr[:4] == b"PK\x03\x04" or hdr[:2] == b"\x1f\x8b" or hdr[:3] == b"BZh" or hdr[:5] == b"ustar"
        except Exception:
            return False

    # Find the pending archive
    pending = None
    if dest.exists():
        for f in dest.iterdir():
            if f.is_file() and f.name.startswith("_download_") and _looks_like_archive(f):
                pending = f
                break
        if not pending:
            for f in dest.iterdir():
                if _looks_like_archive(f):
                    pending = f
                    break

    if not pending:
        raise HTTPException(status_code=404,
                            detail="No pending _download_ archive found in dataset directory")

    with _upload_lock:
        existing = _upload_tasks.get(key)
        if existing and existing["status"] in ("uploading", "extracting"):
            raise HTTPException(status_code=409, detail="Extraction already in progress")
        state: dict = {
            "status": "extracting",
            "progress": 50,
            "message": f"Resuming extraction of {pending.name}…",
        }
        _upload_tasks[key] = state

    thread = threading.Thread(
        target=_bg_extract,
        args=(key, str(pending), state),
        daemon=True,
    )
    thread.start()
    return {"status": "extracting", "message": f"Extracting {pending.name}"}


@router.post("/{name}/import-local", summary="Index an already-extracted dataset from workspace")
async def import_local(name: str):
    """Build index and scan metadata for a dataset already present in DATASETS_DIR."""
    key = name.lower()
    plugin = _get_plugin(key)

    dest = DATASETS_DIR / key

    def _looks_like_archive(p: Path) -> bool:
        if not p.is_file():
            return False
        n = p.name.lower()
        if n.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            return True
        try:
            with open(p, "rb") as _mf:
                hdr = _mf.read(8)
            return hdr[:4] == b"PK\x03\x04" or hdr[:2] == b"\x1f\x8b" or hdr[:3] == b"BZh" or hdr[:5] == b"ustar"
        except Exception:
            return False

    archive_to_extract: Path | None = None
    if dest.exists():
        extracted_dirs = {p.name for p in dest.iterdir() if p.is_dir()}
        looks_extracted = any(d in extracted_dirs for d in ("JPEGImages", "Annotations", "images", "annotations"))

        for f in dest.iterdir():
            if looks_extracted:
                break
            if _looks_like_archive(f):
                archive_to_extract = f
                break

    if archive_to_extract is not None:
        state: dict = {
            "status": "extracting",
            "progress": 50,
            "message": f"Extracting {archive_to_extract.name}…",
        }
        _upload_tasks[key] = state
        threading.Thread(
            target=_bg_extract,
            args=(key, str(archive_to_extract), state),
            daemon=True,
        ).start()
        return {"status": "extracting", "message": f"Extracting {archive_to_extract.name}"}

    state: dict = {"status": "indexing", "progress": 10, "message": "Scanning workspace..."}
    _upload_tasks[key] = state

    def _bg():
        try:
            if key == "idd":
                try:
                    ann_dir = dest / "annotations"
                    has_json = ann_dir.exists() and any(p.suffix.lower() == ".json" for p in ann_dir.glob("*.json"))
                    yolo_train = (dest / "images" / "train")
                    yolo_val = (dest / "images" / "val")
                    has_yolo_images = yolo_train.exists() and yolo_val.exists()
                    if (not has_json) or (not has_yolo_images):
                        state["message"] = "Converting IDD dataset to COCO JSON..."
                        state["progress"] = 25
                        _auto_convert_idd_voc_to_coco(dest, state)
                except Exception:
                    pass

            state["message"] = "Building index..."
            state["progress"] = 40
            if hasattr(plugin, "rebuild_index"):
                plugin.rebuild_index()
            state["progress"] = 80
            state["message"] = "Scanning metadata..."
            _invalidate_cache(key)
            _scan_dataset_meta(key)
            state["status"] = "complete"
            state["progress"] = 100
            state["message"] = "Dataset ready!"
        except Exception as e:
            state["status"] = "error"
            state["progress"] = 0
            state["message"] = f"Import failed: {e}"

    threading.Thread(target=_bg, daemon=True).start()
    return {"status": "indexing", "message": "Indexing started"}


# ── URL download (manual datasets: Google Drive, direct link, etc.) ───────────

@router.post("/{name}/download-url", summary="Download dataset from a URL")
async def download_from_url(name: str, body: dict):
    """Start a background download from a user-supplied URL (e.g. Google Drive)."""
    key = name.lower()
    _get_plugin(key)
    url: str = body.get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    with _download_lock:
        existing = _download_tasks.get(key)
        if existing and existing["status"] == "downloading":
            raise HTTPException(status_code=409, detail="Download already in progress")
        _download_tasks[key] = {
            "status": "downloading", "progress": 0,
            "message": "Starting download...",
            "current_file": "", "bytes_downloaded": 0, "bytes_total": 0,
            "rate_bps": 0, "eta_seconds": -1, "files": {},
            "total_files": 1, "completed_files": 0,
        }

    threading.Thread(target=_bg_url_download, args=(key, url), daemon=True).start()
    return _download_tasks[key]


def _extract_nested_archives(dest_dir, state: dict, progress_start: int = 70, progress_end: int = 84) -> int:
    """Scan dest_dir for nested archive files and extract them in-place.

    Handles patterns like idd-detection.tar.gz sitting inside the extracted dir.
    Returns number of nested archives extracted.
    Only scans one level deep to avoid false positives on image/data files.
    Strips absolute/rooted paths inside archives to prevent path pollution.
    """
    import zipfile, tarfile as _tarfile

    ARCHIVE_SUFFIXES = {".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz"}

    # Collect candidate archive files — TOP-LEVEL ONLY (no rglob) to avoid
    # false-positives on image/binary files buried in subdirectories.
    candidates = []
    for p in sorted(dest_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith((".", "_download_")):
            continue
        name_lower = p.name.lower()
        if any(name_lower.endswith(s) for s in ARCHIVE_SUFFIXES):
            candidates.append(p)
            continue
        # Check magic bytes only for files without a recognised extension
        try:
            with open(p, "rb") as _mf:
                hdr = _mf.read(8)
            if (hdr[:4] == b"PK\x03\x04" or hdr[:2] == b"\x1f\x8b"
                    or hdr[:3] == b"BZh" or hdr[:5] == b"ustar"):
                candidates.append(p)
        except Exception:
            pass

    if not candidates:
        return 0

    def _safe_tar_extract(tf: "_tarfile.TarFile", extract_to: "Path") -> int:
        """Extract tar members, stripping absolute/rooted paths to prevent
        path pollution (e.g. archive built from /home/user/... would otherwise
        extract to dest/home/user/... creating unwanted nested directories)."""
        import posixpath
        count = 0
        for member in tf:
            # Strip leading slashes and any ../.. traversal
            safe_name = member.name.lstrip("/")
            # Remove leading path components that are absolute system paths
            # e.g. "home/rase/kittawat_ws/.../IDD_Detection/JPEGImages/..."
            # → keep only the meaningful relative portion after the archive root
            parts = safe_name.replace("\\", "/").split("/")
            # Drop leading parts that look like absolute system dirs (home, usr, etc.)
            while parts and parts[0] in ("home", "usr", "etc", "var", "opt", "root", "tmp"):
                parts = parts[1:]
            if not parts or not parts[0]:
                continue
            safe_name = posixpath.join(*parts)
            member.name = safe_name
            try:
                tf.extract(member, extract_to, filter="data")
            except Exception:
                pass
            count += 1
            if count % 500 == 0:
                state["message"] = f"Extracting nested archive: {count} files..."
        return count

    extracted = 0
    for idx, arc in enumerate(candidates):
        arc_lower = arc.name.lower()
        extract_to = arc.parent  # extract beside the archive file

        state["message"] = f"Extracting nested archive: {arc.name} ..."
        pct = progress_start + int((idx / len(candidates)) * (progress_end - progress_start))
        state["progress"] = pct

        try:
            with open(arc, "rb") as _mf:
                hdr = _mf.read(8)
            is_zip    = hdr[:4] == b"PK\x03\x04"
            is_tar_gz  = hdr[:2] == b"\x1f\x8b"
            is_tar_bz2 = hdr[:3] == b"BZh"
            is_tar_raw = hdr[:5] == b"ustar"
            is_tar_ext = arc_lower.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar", ".tar.xz"))

            if is_zip or arc_lower.endswith(".zip"):
                with zipfile.ZipFile(arc) as zf:
                    members = zf.infolist()
                    total_m = max(len(members), 1)
                    for i, info in enumerate(members):
                        # Strip absolute/rooted paths to prevent home/rase/... pollution
                        import posixpath as _pp
                        safe = info.filename.lstrip("/").replace("\\", "/")
                        parts = safe.split("/")
                        while parts and parts[0] in ("home", "usr", "etc", "var", "opt", "root", "tmp"):
                            parts = parts[1:]
                        if not parts or not parts[0]:
                            continue
                        info.filename = _pp.join(*parts)
                        zf.extract(info, extract_to)
                        if i % 500 == 0:
                            state["message"] = f"Extracting {arc.name}: {i}/{total_m} files..."
            elif is_tar_gz or is_tar_bz2 or is_tar_raw or is_tar_ext:
                with _tarfile.open(arc) as tf:
                    _safe_tar_extract(tf, extract_to)
            else:
                continue  # not a recognised archive despite name/magic — skip

            arc.unlink(missing_ok=True)
            extracted += 1
        except Exception as _ne:
            # Don't fail the whole job for one nested archive
            state["message"] = f"Warning: could not extract {arc.name}: {_ne}"

    return extracted


def _auto_convert_idd_voc_to_coco(root: Path, state: dict) -> bool:
    import json as _json
    import xml.etree.ElementTree as _ET
    import os as _os
    from pathlib import Path as _Path

    ann_dir = root / "Annotations"
    img_dir = root / "JPEGImages"
    train_list = root / "train.txt"
    val_list = root / "val.txt"
    if not (ann_dir.exists() and img_dir.exists() and train_list.exists() and val_list.exists()):
        return False

    coco_ann_dir = root / "annotations"
    coco_ann_dir.mkdir(parents=True, exist_ok=True)

    yolo_img_dir = root / "images"
    (yolo_img_dir / "train").mkdir(parents=True, exist_ok=True)
    (yolo_img_dir / "val").mkdir(parents=True, exist_ok=True)

    cats = [
        "person", "rider", "car", "truck", "bus", "motorcycle",
        "bicycle", "autorickshaw", "animal", "traffic light",
        "traffic sign", "utility pole", "misc", "drivable area", "non-drivable area",
    ]
    cat_name_to_id = {n: i for i, n in enumerate(cats)}
    categories = [{"id": i, "name": n} for i, n in enumerate(cats)]

    def _convert_split(split: str, list_path: Path) -> Path:
        lines = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip()]
        images = []
        annotations = []
        img_id = 1
        ann_id = 1

        for idx, rel in enumerate(lines):
            if idx % 500 == 0:
                state["message"] = f"Converting IDD VOC→COCO ({split}): {idx}/{len(lines)}"
                state["progress"] = min(state.get("progress", 70) + 1, 89)

            rel_path = _Path(rel)
            jpg_src = img_dir / (str(rel_path) + ".jpg")
            xml_src = ann_dir / (str(rel_path) + ".xml")
            if not jpg_src.exists() or not xml_src.exists():
                continue

            jpg_dest = yolo_img_dir / split / (str(rel_path) + ".jpg")
            jpg_dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not jpg_dest.exists():
                    import shutil as _shutil
                    _shutil.copy2(jpg_src, jpg_dest)
            except Exception:
                pass

            try:
                tree = _ET.parse(str(xml_src))
                xml_root = tree.getroot()
                size = xml_root.find("size")
                w = int(size.findtext("width", default="0")) if size is not None else 0
                h = int(size.findtext("height", default="0")) if size is not None else 0
            except Exception:
                continue

            images.append({
                "id": img_id,
                "file_name": str(rel_path) + ".jpg",
                "width": w,
                "height": h,
            })

            for obj in xml_root.findall("object"):
                name_node = obj.find("name")
                cls = (name_node.text or "").strip() if name_node is not None else ""
                cat_id = cat_name_to_id.get(cls)
                if cat_id is None:
                    continue
                bnd = obj.find("bndbox")
                if bnd is None:
                    continue
                try:
                    xmin = float(bnd.findtext("xmin"))
                    ymin = float(bnd.findtext("ymin"))
                    xmax = float(bnd.findtext("xmax"))
                    ymax = float(bnd.findtext("ymax"))
                except Exception:
                    continue
                bw = max(0.0, xmax - xmin)
                bh = max(0.0, ymax - ymin)
                if bw <= 0 or bh <= 0:
                    continue
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [xmin, ymin, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1

            img_id += 1

        out_path = coco_ann_dir / f"idd_detection_{split}.json"
        out = {
            "info": {"description": "IDD Detection"},
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        with open(out_path, "w") as f:
            _json.dump(out, f)
        return out_path

    out_train = _convert_split("train", train_list)
    out_val = _convert_split("val", val_list)
    
    success = out_train.exists() and out_val.exists()
    if success:
        # Cleanup raw files after successful reorganization
        try:
            import shutil as _shutil
            if ann_dir.exists():
                _shutil.rmtree(ann_dir)
            if img_dir.exists():
                _shutil.rmtree(img_dir)
            for txt_file in [train_list, val_list, root / "test.txt"]:
                if txt_file.exists():
                    txt_file.unlink()
            # Also remove IDD_Detection folder if it exists
            idd_det_dir = root / "IDD_Detection"
            if idd_det_dir.exists():
                _shutil.rmtree(idd_det_dir)
        except Exception:
            pass
    return success


def _gdrive_download(file_id: str, dest_path: str, state: dict) -> None:
    """Download a Google Drive file handling large-file virus-scan confirmation.

    Strategy:
    1. Try gdown (handles all edge-cases natively) if available.
    2. Fall back to requests session: first request may return HTML warning page;
       extract the confirm token from the page and retry with it.
    """
    import time as _time, re as _re

    state["message"] = "Connecting to Google Drive..."

    # ── Strategy 1: gdown ────────────────────────────────────────────────────
    try:
        import gdown  # type: ignore
        state["message"] = "Downloading via gdown..."

        def _hook(received: int, total: int):
            state["bytes_downloaded"] = received
            state["bytes_total"] = total
            elapsed = max(_time.time() - _hook._t0, 0.001)
            state["rate_bps"] = int(received / elapsed)
            state["progress"] = int(received / total * 50) if total else 10
            mb = received / (1024 * 1024)
            tmb = total / (1024 * 1024)
            state["message"] = f"Downloading {mb:.0f} / {tmb:.0f} MB"

        _hook._t0 = _time.time()
        gdown.download(id=file_id, output=dest_path, quiet=True,
                       fuzzy=False, resume=False)
        return
    except ImportError:
        pass  # gdown not installed, fall through
    except Exception as e:
        # gdown failed (e.g. quota exceeded) — fall through to requests
        state["message"] = f"gdown failed ({e}), retrying with requests..."

    # ── Strategy 2: requests session with confirm token ───────────────────────
    try:
        import requests as _req
    except ImportError:
        raise RuntimeError(
            "Neither 'gdown' nor 'requests' is installed. "
            "Install one: pip install gdown  OR  pip install requests"
        )

    session = _req.Session()
    base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = session.get(base_url, headers=headers, stream=True, timeout=60)

    # Google Drive returns HTML for large files (virus-scan warning)
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        html = resp.content.decode("utf-8", errors="replace")
        # Extract confirm token from the warning form
        m = _re.search(r'confirm=([0-9A-Za-z_\-]+)', html)
        if not m:
            # Newer Drive: look for a download form action with token
            m = _re.search(r'["\']([0-9A-Za-z_\-]{4,})["\'].*?virus', html)
        if m:
            confirm = m.group(1)
            dl_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm}"
        else:
            # Try the newer /uc bypass endpoint
            dl_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        resp = session.get(dl_url, headers=headers, stream=True, timeout=60)

    total = int(resp.headers.get("Content-Length", 0) or 0)
    state["bytes_total"] = total

    received = 0
    t0 = _time.time()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            received += len(chunk)
            elapsed = max(_time.time() - t0, 0.001)
            state["bytes_downloaded"] = received
            state["rate_bps"] = int(received / elapsed)
            state["progress"] = int(received / total * 50) if total else 10
            mb = received / (1024 * 1024)
            tmb = total / (1024 * 1024)
            state["message"] = f"Downloading {mb:.0f} / {tmb:.0f} MB"
            state["eta_seconds"] = (
                int((total - received) / state["rate_bps"])
                if state["rate_bps"] > 0 and total > 0 else -1
            )

    # Sanity check: if we got HTML instead of a real file, fail loudly
    with open(dest_path, "rb") as _sf:
        _magic = _sf.read(8)
    _is_html = _magic[:5] in (b"<!DOC", b"<html", b"<HTML") or _magic[:2] == b"<!"
    if _is_html or received < 1024:
        import os
        os.unlink(dest_path)
        raise RuntimeError(
            "Google Drive returned an HTML page instead of the file. "
            "The file is likely restricted or requires sign-in. "
            "Fix: share the file as 'Anyone with the link → Viewer', "
            "then retry. Alternatively, install gdown (pip install gdown) "
            "or download manually and use the Upload option."
        )


def _bg_url_download(name: str, url: str):
    """Download from URL, detect Google Drive links, extract, index."""
    import os, re
    from pathlib import Path as _Path

    state = _download_tasks[name]
    try:
        # Resolve Google Drive share links → direct download URL
        gdrive = re.match(r'https://drive\.google\.com/file/d/([^/?]+)', url)
        gdrive_folder = re.match(r'https://drive\.google\.com/drive/folders/([^?]+)', url)
        gdrive_open = re.match(r'https://drive\.google\.com/open\?id=([^&]+)', url)

        is_gdrive = bool(gdrive or gdrive_open)
        gdrive_file_id = None
        if gdrive:
            gdrive_file_id = gdrive.group(1)
        elif gdrive_open:
            gdrive_file_id = gdrive_open.group(1)
        elif gdrive_folder:
            state["status"] = "error"
            state["message"] = "Google Drive folder links not supported. Share individual file links."
            return

        state["message"] = "Connecting..."
        import urllib.request, time as _time

        dest_dir = DATASETS_DIR / name
        dest_dir.mkdir(parents=True, exist_ok=True)

        if is_gdrive and gdrive_file_id:
            # Use dedicated GDrive downloader that handles virus-scan confirmation
            fname = f"gdrive_{gdrive_file_id}.bin"
            tmp_path = dest_dir / f"_download_{fname}"
            _gdrive_download(gdrive_file_id, str(tmp_path), state)
        else:
            # Generic URL download
            with urllib.request.urlopen(url, timeout=30) as resp:
                cd = resp.headers.get("Content-Disposition", "")
                fname_match = re.search(r'filename[^;=\n]*=([\'"]?)([^\n;"\']+)\1', cd)
                if fname_match:
                    fname = fname_match.group(2).strip()
                else:
                    fname = url.split("?")[0].split("/")[-1] or "dataset.zip"
                total = int(resp.headers.get("Content-Length", 0) or 0)
                state["bytes_total"] = total

                tmp_path = dest_dir / f"_download_{fname}"
                received = 0
                t0 = _time.time()
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        received += len(chunk)
                        elapsed = max(_time.time() - t0, 0.001)
                        state["bytes_downloaded"] = received
                        state["rate_bps"] = int(received / elapsed)
                        state["progress"] = int(received / total * 50) if total else 10
                        state["message"] = f"Downloading {received // (1024*1024):.0f} / {total // (1024*1024):.0f} MB"
                        state["eta_seconds"] = int((total - received) / state["rate_bps"]) if state["rate_bps"] > 0 and total > 0 else -1

        state["progress"] = 50
        state["message"] = "Detecting archive format..."

        # Detect archive format from magic bytes only (no tarfile.is_tarfile — it reads whole file)
        import zipfile, tarfile
        with open(tmp_path, "rb") as _mf:
            magic = _mf.read(8)
        is_zip   = magic[:4] == b"PK\x03\x04"
        is_tar_gz  = magic[:2] == b"\x1f\x8b"   # gzip
        is_tar_bz2 = magic[:3] == b"BZh"         # bzip2
        is_tar_raw = magic[:5] == b"ustar"        # ustar posix tar (no compression)
        lower = str(tmp_path).lower()
        is_tar_ext = lower.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar"))

        # Detect HTML response (Google Drive login/virus-scan page returned instead of file)
        is_html = magic[:5] in (b"<!DOC", b"<html", b"<HTML") or magic[:2] == b"<!"
        if is_html:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                "Google Drive returned an HTML page instead of the file. "
                "The file may be restricted, require sign-in, or the sharing link has expired. "
                "Try: (1) Make sure the file is shared as 'Anyone with the link', "
                "(2) Install gdown on the server: pip install gdown, "
                "(3) Download the file manually and use the Upload option instead."
            )

        state["message"] = "Extracting..."

        def _strip_abs_path(name: str) -> str:
            """Strip leading absolute path components (home/usr/etc...) from archive member names."""
            import posixpath as _pp
            parts = name.lstrip("/").replace("\\", "/").split("/")
            while parts and parts[0] in ("home", "usr", "etc", "var", "opt", "root", "tmp"):
                parts = parts[1:]
            return _pp.join(*parts) if parts and parts[0] else ""

        if is_zip or lower.endswith(".zip"):
            with zipfile.ZipFile(tmp_path) as zf:
                members = zf.infolist()
                total_m = max(len(members), 1)
                for i, info in enumerate(members):
                    safe = _strip_abs_path(info.filename)
                    if not safe:
                        continue
                    info.filename = safe
                    zf.extract(info, dest_dir)
                    if i % 200 == 0:
                        pct = 50 + int((i / total_m) * 35)
                        state["progress"] = pct
                        state["message"] = f"Extracting {i}/{total_m} files..."
        elif is_tar_gz or is_tar_bz2 or is_tar_raw or is_tar_ext:
            with tarfile.open(tmp_path) as tf:
                i = 0
                for member in tf:
                    safe = _strip_abs_path(member.name)
                    if not safe:
                        continue
                    member.name = safe
                    tf.extract(member, dest_dir, filter="data")
                    i += 1
                    if i % 200 == 0:
                        pct = min(50 + int(i / 100), 84)
                        state["progress"] = pct
                        state["message"] = f"Extracting {i} files..."
        else:
            raise ValueError(
                f"Downloaded file does not appear to be a zip or tar archive "
                f"(magic bytes: {magic[:4].hex()}). "
                "If this is a Google Drive link, the file may require authentication."
            )

        tmp_path.unlink(missing_ok=True)

        # Flatten single root dir
        subdirs = [d for d in dest_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
        if len(subdirs) == 1:
            inner = subdirs[0]
            for item in list(inner.iterdir()):
                target = dest_dir / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            try:
                inner.rmdir()
            except Exception:
                pass

        # Extract any nested archives (e.g. idd-detection.tar.gz inside the zip)
        nested = _extract_nested_archives(dest_dir, state, progress_start=70, progress_end=84)
        if nested:
            state["message"] = f"Extracted {nested} nested archive(s). Building index..."

        state["progress"] = 85
        state["message"] = "Building index..."

        plugin = get_dataset_plugin(name)
        if plugin and hasattr(plugin, "rebuild_index"):
            plugin.rebuild_index()

        _invalidate_cache(name)
        _scan_dataset_meta(name)

        # Verify dataset is actually usable after extraction
        if plugin and not plugin.is_available():
            state["status"] = "error"
            state["progress"] = 0
            state["message"] = (
                "Extraction complete but dataset files are incomplete or invalid. "
                "Expected annotation JSON files containing 'idd'/'detection' keywords "
                "and at least one image file. Please check the archive structure and re-upload."
            )
            return

        state["status"] = "complete"
        state["progress"] = 100
        state["message"] = "Dataset ready!"

    except Exception as e:
        state["status"] = "error"
        state["progress"] = 0
        state["message"] = f"Download failed: {e}"


# ── Upload with progress ─────────────────────────────────────────────────────

_upload_tasks: dict[str, dict] = {}
_upload_lock = threading.Lock()


@router.post("/{name}/upload", summary="Upload a dataset archive (zip/tar)")
async def upload_dataset(
    name: str,
    file: UploadFile = File(...),
):
    """Stream-upload a zip/tar.gz archive, extract into the dataset directory."""
    key = name.lower()
    plugin = _get_plugin(key)

    with _upload_lock:
        existing = _upload_tasks.get(key)
        if existing and existing["status"] == "uploading":
            raise HTTPException(status_code=409, detail="Upload already in progress")
        _upload_tasks[key] = {
            "status": "uploading",
            "progress": 0,
            "message": "Receiving file...",
            "bytes_received": 0,
        }

    state = _upload_tasks[key]

    # Stream file to a temp location using large buffer for max local throughput
    suffix = (file.filename or "upload.bin")
    suffix = "." + suffix.split(".")[-1] if "." in suffix else ".bin"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            # Use shutil.copyfileobj with 4 MB buffer — fastest for local/LAN transfers
            await asyncio.to_thread(
                shutil.copyfileobj, file.file, tmp, 4 * 1024 * 1024
            )
            state["bytes_received"] = tmp.tell()
            state["message"] = f"Received {state['bytes_received'] // (1024*1024):.1f} MB"
    except Exception as e:
        state["status"] = "error"
        state["message"] = f"Upload failed: {e}"
        raise HTTPException(status_code=500, detail=str(e))

    state["message"] = "Extracting archive..."
    state["progress"] = 50

    thread = threading.Thread(
        target=_bg_extract,
        args=(key, tmp_path, state),
        daemon=True,
    )
    thread.start()
    return {"status": "extracting", "message": "Upload received, extracting..."}


@router.get("/{name}/upload-status", summary="Poll dataset upload/extract progress")
async def upload_status_endpoint(name: str):
    key = name.lower()
    state = _upload_tasks.get(key)
    if not state:
        return {"status": "idle", "progress": 0, "message": ""}
    return state


def _bg_extract(name: str, archive_path: str, state: dict):
    """Background: extract archive → dataset dir → scan meta."""
    import os
    from pathlib import Path as _Path

    try:
        dest = DATASETS_DIR / name
        dest.mkdir(parents=True, exist_ok=True)

        arc = _Path(archive_path)
        lower = arc.name.lower()

        state["message"] = "Detecting archive format..."
        state["progress"] = 55

        import zipfile, tarfile as _tarfile
        with open(arc, "rb") as _mf:
            magic = _mf.read(8)
        is_zip    = magic[:4] == b"PK\x03\x04"
        is_tar_gz  = magic[:2] == b"\x1f\x8b"
        is_tar_bz2 = magic[:3] == b"BZh"
        is_tar_raw = magic[:5] == b"ustar"
        is_tar_ext = lower.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar"))

        state["message"] = "Extracting..."

        def _strip_abs(n: str) -> str:
            import posixpath as _pp
            parts = n.lstrip("/").replace("\\", "/").split("/")
            while parts and parts[0] in ("home", "usr", "etc", "var", "opt", "root", "tmp"):
                parts = parts[1:]
            return _pp.join(*parts) if parts and parts[0] else ""

        is_zip_valid = zipfile.is_zipfile(str(arc)) if (is_zip or lower.endswith(".zip")) else False
        if is_zip_valid:
            try:
                with zipfile.ZipFile(arc) as zf:
                    members = zf.infolist()
                    total_m = max(len(members), 1)
                    for i, info in enumerate(members):
                        safe = _strip_abs(info.filename)
                        if not safe:
                            continue
                        info.filename = safe
                        zf.extract(info, dest)
                        if i % 200 == 0:
                            pct = 55 + int((i / total_m) * 33)
                            state["progress"] = pct
                            state["message"] = f"Extracting {i}/{total_m} files..."
            except zipfile.BadZipFile:
                is_zip_valid = False

        if not is_zip_valid:
            if is_tar_gz or is_tar_bz2 or is_tar_raw or is_tar_ext or is_zip or lower.endswith(".zip"):
                try:
                    with _tarfile.open(arc) as tf:
                        i = 0
                        for member in tf:
                            safe = _strip_abs(member.name)
                            if not safe:
                                continue
                            member.name = safe
                            tf.extract(member, dest, filter="data")
                            i += 1
                            if i % 200 == 0:
                                pct = min(55 + int(i / 100), 87)
                                state["progress"] = pct
                                state["message"] = f"Extracting {i} files..."
                except _tarfile.TarError as _te:
                    raise ValueError(
                        f"Archive '{arc.name}' is not a valid zip or tar archive. "
                        f"If the file extension is .zip, it may actually be a .tar.gz (or the file is corrupted). "
                        f"Details: {_te}"
                    )
            else:
                raise ValueError(
                    f"Unsupported archive format: {arc.name} "
                    f"(magic bytes: {magic[:4].hex()})"
                )
        state["progress"] = 88

        # Flatten single-root-dir extraction (e.g. idd/ inside zip)
        subdirs = [d for d in dest.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and not any(dest.iterdir().__next__() == d for d in [dest / "images", dest / "labels", dest / "annotations"]):
            inner = subdirs[0]
            # Move contents up one level if the subdir name matches dataset name or common names
            inner_items = list(inner.iterdir())
            for item in inner_items:
                target = dest / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            try:
                inner.rmdir()
            except Exception:
                pass

        # Extract any nested archives (e.g. idd-detection.tar.gz inside the zip)
        nested = _extract_nested_archives(dest, state, progress_start=70, progress_end=88)
        if nested:
            state["message"] = f"Extracted {nested} nested archive(s). Building index..."

        def _auto_convert_idd_voc_to_coco(_root: _Path, _state: dict) -> bool:
            import json as _json
            import xml.etree.ElementTree as _ET

            ann_dir = _root / "Annotations"
            img_dir = _root / "JPEGImages"
            train_list = _root / "train.txt"
            val_list = _root / "val.txt"
            if not (ann_dir.exists() and img_dir.exists() and train_list.exists() and val_list.exists()):
                return False

            coco_ann_dir = _root / "annotations"
            coco_ann_dir.mkdir(parents=True, exist_ok=True)

            cats = [
                "person", "rider", "car", "truck", "bus", "motorcycle",
                "bicycle", "autorickshaw", "animal", "traffic light",
                "traffic sign", "utility pole", "misc", "drivable area", "non-drivable area",
            ]
            cat_name_to_id = {n: i for i, n in enumerate(cats)}
            categories = [{"id": i, "name": n} for i, n in enumerate(cats)]

            def _convert_split(split: str, list_path: _Path) -> _Path:
                lines = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip()]
                images = []
                annotations = []
                img_id = 1
                ann_id = 1

                for idx, rel in enumerate(lines):
                    if idx % 500 == 0:
                        _state["message"] = f"Converting IDD VOC→COCO ({split}): {idx}/{len(lines)}"
                        _state["progress"] = min(_state.get("progress", 70) + 1, 89)

                    rel_path = _Path(rel)
                    jpg_src = img_dir / (str(rel_path) + ".jpg")
                    xml_src = ann_dir / (str(rel_path) + ".xml")
                    if not jpg_src.exists() or not xml_src.exists():
                        continue

                    # Copy image to YOLO structure
                    yolo_img_dir = _root / "images"
                    (yolo_img_dir / split).mkdir(parents=True, exist_ok=True)
                    jpg_dest = yolo_img_dir / split / (str(rel_path) + ".jpg")
                    try:
                        if not jpg_dest.exists():
                            import shutil as _shutil
                            _shutil.copy2(jpg_src, jpg_dest)
                    except Exception:
                        pass

                    try:
                        tree = _ET.parse(str(xml_src))
                        root = tree.getroot()
                        size = root.find("size")
                        w = int(size.findtext("width", default="0")) if size is not None else 0
                        h = int(size.findtext("height", default="0")) if size is not None else 0
                    except Exception:
                        continue

                    images.append({
                        "id": img_id,
                        "file_name": str(rel_path) + ".jpg",
                        "width": w,
                        "height": h,
                    })

                    for obj in root.findall("object"):
                        name_node = obj.find("name")
                        cls = (name_node.text or "").strip() if name_node is not None else ""
                        cat_id = cat_name_to_id.get(cls)
                        if cat_id is None:
                            continue
                        bnd = obj.find("bndbox")
                        if bnd is None:
                            continue
                        try:
                            xmin = float(bnd.findtext("xmin"))
                            ymin = float(bnd.findtext("ymin"))
                            xmax = float(bnd.findtext("xmax"))
                            ymax = float(bnd.findtext("ymax"))
                        except Exception:
                            continue
                        bw = max(0.0, xmax - xmin)
                        bh = max(0.0, ymax - ymin)
                        if bw <= 0 or bh <= 0:
                            continue
                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cat_id,
                            "bbox": [xmin, ymin, bw, bh],
                            "area": bw * bh,
                            "iscrowd": 0,
                        })
                        ann_id += 1

                    img_id += 1

                out_path = coco_ann_dir / f"idd_detection_{split}.json"
                out = {
                    "info": {"description": "IDD Detection"},
                    "images": images,
                    "annotations": annotations,
                    "categories": categories,
                }
                with open(out_path, "w") as f:
                    _json.dump(out, f)
                return out_path

            out_train = _convert_split("train", train_list)
            out_val = _convert_split("val", val_list)
            
            success = out_train.exists() and out_val.exists()
            if success:
                # Cleanup raw files after successful reorganization
                try:
                    import shutil as _shutil
                    if ann_dir.exists():
                        _shutil.rmtree(ann_dir)
                    if img_dir.exists():
                        _shutil.rmtree(img_dir)
                    for txt_file in [train_list, val_list, _root / "test.txt"]:
                        if txt_file.exists():
                            txt_file.unlink()
                    # Also remove IDD_Detection folder if it exists
                    idd_det_dir = _root / "IDD_Detection"
                    if idd_det_dir.exists():
                        _shutil.rmtree(idd_det_dir)
                except Exception:
                    pass
            return success

        if name.lower() == "idd":
            try:
                has_json = any(p.suffix.lower() == ".json" for p in (dest / "annotations").glob("*.json")) if (dest / "annotations").exists() else False
                if not has_json:
                    state["message"] = "Converting IDD dataset to COCO JSON..."
                    state["progress"] = max(state.get("progress", 70), 70)
                    _auto_convert_idd_voc_to_coco(dest, state)
                
                # Auto-convert COCO to YOLO format for training
                state["message"] = "Converting COCO annotations to YOLO format..."
                state["progress"] = 91
                coco_converter.auto_convert_if_needed("idd")
            except Exception:
                pass

        state["progress"] = 90
        state["message"] = "Building index..."

        # Let plugin rebuild index
        plugin = get_dataset_plugin(name)
        if plugin and hasattr(plugin, "rebuild_index"):
            plugin.rebuild_index()

        _invalidate_cache(name)
        _scan_dataset_meta(name)

        # Verify dataset is actually usable after extraction
        if plugin and not plugin.is_available():
            state["status"] = "error"
            state["progress"] = 0
            state["message"] = (
                "Extraction complete but dataset files are incomplete or invalid. "
                "Expected annotation JSON files containing 'idd'/'detection' keywords "
                "and at least one image file. Please check the archive structure and re-upload."
            )
            return

        state["status"] = "complete"
        state["progress"] = 100
        state["message"] = "Dataset ready!"

    except Exception as e:
        state["status"] = "error"
        state["progress"] = 0
        state["message"] = f"Extraction failed: {e}"
    finally:
        try:
            arc_path = _Path(archive_path)
            should_delete = False
            if not str(arc_path.resolve()).startswith(str((DATASETS_DIR / name).resolve())):
                should_delete = True
            if arc_path.name.startswith("_download_"):
                should_delete = True
            if should_delete:
                os.unlink(archive_path)
        except Exception:
            pass


# ── Split config helpers ─────────────────────────────────────────────────────

_DEFAULT_SPLIT = {
    "seed": 42,
    "train_to_val": 0, "train_to_test": 0,
    "test_to_train": 0, "test_to_val": 0,
    "val_to_train": 0, "val_to_test": 0,
}

_TRANSFER_KEYS = [
    "train_to_val", "train_to_test",
    "test_to_train", "test_to_val",
    "val_to_train", "val_to_test",
]


def _split_config_path(name: str):
    return SPLITS_DIR / f"{name}.json"


def _load_split_config(name: str) -> dict:
    """Load split config, migrating old formats if needed."""
    p = _split_config_path(name)
    if not p.exists():
        return dict(_DEFAULT_SPLIT)
    raw = json.loads(p.read_text())
    if "val_fraction" in raw and "train_to_val" not in raw:
        vf = raw.get("val_fraction", 0.0)
        return {**_DEFAULT_SPLIT, "seed": raw.get("seed", 42),
                "train_to_val": round(vf * 100)}
    if "train_pct" in raw and "train_to_val" not in raw:
        return {**_DEFAULT_SPLIT, "seed": raw.get("seed", 42)}
    return raw


def _save_split_config(name: str, cfg: dict):
    _split_config_path(name).write_text(json.dumps(cfg, indent=2))


def _compute_transfer_counts(cfg: dict, orig_train: int, orig_test: int, orig_val: int = 0) -> dict:
    """Compute effective split sizes after transfers."""
    t2v = int(orig_train * cfg.get("train_to_val", 0) / 100)
    t2te = int(orig_train * cfg.get("train_to_test", 0) / 100)
    if t2v + t2te > orig_train:
        ratio = orig_train / (t2v + t2te) if (t2v + t2te) else 0
        t2v = int(t2v * ratio)
        t2te = orig_train - t2v

    te2tr = int(orig_test * cfg.get("test_to_train", 0) / 100)
    te2v = int(orig_test * cfg.get("test_to_val", 0) / 100)
    if te2tr + te2v > orig_test:
        ratio = orig_test / (te2tr + te2v) if (te2tr + te2v) else 0
        te2tr = int(te2tr * ratio)
        te2v = orig_test - te2tr

    v2tr = int(orig_val * cfg.get("val_to_train", 0) / 100)
    v2te = int(orig_val * cfg.get("val_to_test", 0) / 100)
    if v2tr + v2te > orig_val:
        ratio = orig_val / (v2tr + v2te) if (v2tr + v2te) else 0
        v2tr = int(v2tr * ratio)
        v2te = orig_val - v2tr

    final_train = orig_train - t2v - t2te + te2tr + v2tr
    final_val = orig_val - v2tr - v2te + t2v + te2v
    final_test = orig_test - te2tr - te2v + t2te + v2te

    return {
        "orig_train": orig_train, "orig_test": orig_test, "orig_val": orig_val,
        "train_count": final_train, "val_count": final_val, "test_count": final_test,
    }


def _apply_transfers(cfg: dict, n_train: int, n_test: int, n_val: int = 0):
    """Return (train_indices, val_indices, test_indices) into a ConcatDataset([train, test, val])."""
    seed = cfg.get("seed", 42)
    rng = np.random.RandomState(seed)

    train_shuf = rng.permutation(n_train).tolist()
    test_shuf = (rng.permutation(n_test) + n_train).tolist()
    val_shuf = (rng.permutation(n_val) + n_train + n_test).tolist() if n_val > 0 else []

    t2v = int(n_train * cfg.get("train_to_val", 0) / 100)
    t2te = int(n_train * cfg.get("train_to_test", 0) / 100)
    if t2v + t2te > n_train:
        ratio = n_train / (t2v + t2te) if (t2v + t2te) else 0
        t2v = int(t2v * ratio); t2te = n_train - t2v

    te2tr = int(n_test * cfg.get("test_to_train", 0) / 100)
    te2v = int(n_test * cfg.get("test_to_val", 0) / 100)
    if te2tr + te2v > n_test:
        ratio = n_test / (te2tr + te2v) if (te2tr + te2v) else 0
        te2tr = int(te2tr * ratio); te2v = n_test - te2tr

    v2tr = int(n_val * cfg.get("val_to_train", 0) / 100)
    v2te = int(n_val * cfg.get("val_to_test", 0) / 100)
    if v2tr + v2te > n_val:
        ratio = n_val / (v2tr + v2te) if (v2tr + v2te) else 0
        v2tr = int(v2tr * ratio); v2te = n_val - v2tr

    train_give_val = train_shuf[:t2v]
    train_give_test = train_shuf[t2v:t2v + t2te]
    train_keep = train_shuf[t2v + t2te:]

    test_give_train = test_shuf[:te2tr]
    test_give_val = test_shuf[te2tr:te2tr + te2v]
    test_keep = test_shuf[te2tr + te2v:]

    val_give_train = val_shuf[:v2tr]
    val_give_test = val_shuf[v2tr:v2tr + v2te]
    val_keep = val_shuf[v2tr + v2te:]

    final_train = train_keep + test_give_train + val_give_train
    final_val = val_keep + train_give_val + test_give_val
    final_test = test_keep + train_give_test + val_give_test

    return final_train, final_val, final_test


@router.get("/{name}/splits", summary="Get dataset split config")
async def get_splits(name: str):
    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    cfg = _load_split_config(name.lower())
    val_size = getattr(ds, 'val_size', 0)
    counts = _compute_transfer_counts(cfg, ds.train_size, ds.test_size, val_size)
    return {**cfg, **counts}


@router.post("/{name}/splits", summary="Save dataset split config")
async def save_splits(name: str, body: SplitConfig):
    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    cfg = body.model_dump()
    _save_split_config(name.lower(), cfg)
    _invalidate_cache(name.lower())
    _invalidate_partition_cache(name.lower())
    _scan_dataset_meta(name.lower())
    val_size = getattr(ds, 'val_size', 0)
    counts = _compute_transfer_counts(cfg, ds.train_size, ds.test_size, val_size)
    return {**cfg, **counts}


# ── Partition system ──────────────────────────────────────────────────────────
#
# After the split config divides data into train/val/test, partitions further
# subdivide each split.  The "master" partition is always the remainder after
# all named partitions have been carved out.
#
# Config stored in  SPLITS_DIR/{dataset}_partitions.json :
# {
#   "seed": 42,
#   "partitions": [
#     {"id": "p_abc123", "name": "Batch 1", "percent": 30},
#     {"id": "p_def456", "name": "Batch 2", "percent": 20}
#   ]
# }
#
# master gets  100 - sum(percent) = 50%  of each split.

def _partition_config_path(name: str):
    return SPLITS_DIR / f"{name.lower()}_partitions.json"


PARTITION_METHODS = ("random", "stratified", "round_robin", "iterative")


def _load_partition_config(name: str) -> dict:
    p = _partition_config_path(name)
    if not p.exists():
        return {"seed": 42, "method": "stratified", "partitions": []}
    cfg = json.loads(p.read_text())
    cfg.setdefault("method", "stratified")
    return cfg


def _save_partition_config(name: str, cfg: dict):
    _partition_config_path(name).write_text(json.dumps(cfg, indent=2))


# ── Partition cache ──────────────────────────────────────────────────────────
#
# Indices are computed once when partition config is saved (create / delete /
# split / method change) and cached to disk.  Consumers (summary, samples
# filter, trainer) read from cache — no redundant recomputation.

def _partition_cache_path(name: str):
    return SPLITS_DIR / f"{name.lower()}_partition_cache.json"


def _extract_labels_from_ds(ds) -> list[int] | None:
    """Extract per-sample class labels from a dataset object.

    Works for torchvision classification datasets (.targets), detection
    wrappers (_COCODetectionWrapper / COCORawDataset), Subset, ConcatDataset.
    Returns None if labels cannot be determined.
    """
    import torch
    from collections import Counter
    from torch.utils.data import Subset as _Sub, ConcatDataset as _Cat

    if hasattr(ds, 'targets'):
        t = ds.targets
        return t.tolist() if isinstance(t, torch.Tensor) else list(t)

    # Detection: _COCODetectionWrapper
    if hasattr(ds, '_ds') and hasattr(ds, '_cat_map') and hasattr(ds._ds, '_index'):
        cat_map = ds._cat_map
        labels: list[int] = []
        for entry in ds._ds._index:
            anns = entry.get("anns", [])
            if not anns:
                labels.append(-1)
                continue
            counts: Counter = Counter()
            for ann in anns:
                cid = ann.get("category_id")
                if cid in cat_map:
                    counts[cat_map[cid]] += 1
            labels.append(counts.most_common(1)[0][0] if counts else -1)
        return labels

    # Detection: COCORawDataset
    if hasattr(ds, '_index') and isinstance(getattr(ds, '_index', None), list):
        idx = ds._index
        if idx and isinstance(idx[0], dict) and "anns" in idx[0]:
            labels = []
            for entry in idx:
                anns = entry.get("anns", [])
                if not anns:
                    labels.append(-1)
                    continue
                counts = Counter(a.get("category_id", 0) for a in anns)
                labels.append(counts.most_common(1)[0][0])
            return labels

    if isinstance(ds, _Sub):
        parent = _extract_labels_from_ds(ds.dataset)
        if parent is not None:
            return [parent[i] for i in ds.indices]
        return None

    if isinstance(ds, _Cat):
        all_l: list[int] = []
        for sub in ds.datasets:
            sub_l = _extract_labels_from_ds(sub)
            if sub_l is None:
                return None
            all_l.extend(sub_l)
        return all_l

    return None


def _get_contiguous_cat_map(plugin) -> dict[int, int] | None:
    """Return {original_category_id → contiguous_class_index} for a detection plugin."""
    if hasattr(plugin, '_cat_id_to_contiguous'):
        return plugin._cat_id_to_contiguous()
    return None


def _get_detection_anns_fast(dataset, idx) -> list[dict] | None:
    """Get annotations from a detection dataset without loading the image."""
    from bisect import bisect_right
    from torch.utils.data import Subset, ConcatDataset as _Cat

    raw = dataset
    real_idx = idx

    # Unwrap Subset layers
    while isinstance(raw, Subset):
        if real_idx < 0 or real_idx >= len(raw.indices):
            return None
        real_idx = raw.indices[real_idx]
        raw = raw.dataset

    # Handle ConcatDataset
    if isinstance(raw, _Cat):
        if not raw.cumulative_sizes or real_idx < 0:
            return None
        ds_idx = bisect_right(raw.cumulative_sizes, real_idx)
        if ds_idx >= len(raw.datasets):
            return None
        prev = raw.cumulative_sizes[ds_idx - 1] if ds_idx > 0 else 0
        raw = raw.datasets[ds_idx]
        real_idx = real_idx - prev
        while isinstance(raw, Subset):
            if real_idx < 0 or real_idx >= len(raw.indices):
                return None
            real_idx = raw.indices[real_idx]
            raw = raw.dataset

    # Access COCORawDataset's _index directly (has "anns" field)
    if hasattr(raw, '_index') and isinstance(raw._index, list):
        if 0 <= real_idx < len(raw._index):
            return raw._index[real_idx].get("anns", [])

    return None


def _build_partition_cache(name: str) -> dict | None:
    """Compute partition indices for all splits and persist to cache file.

    Returns the cache dict, or None if dataset not found.

    For **classification** datasets, class_counts = images per class.
    For **detection** datasets, class_counts = annotations per contiguous
    class index (so filter tags show how many bounding boxes of each class).
    """
    from collections import Counter

    ds_info = _get_info(name)
    if not ds_info:
        return None

    is_detection = getattr(ds_info, 'task_type', 'classification') == 'detection'

    cfg = _load_partition_config(name)
    seed = cfg.get("seed", 42)
    method = cfg.get("method", "stratified")
    parts = cfg.get("partitions", [])

    # Effective split sizes after transfers
    split_cfg = _load_split_config(name.lower())
    val_size = getattr(ds_info, 'val_size', 0)
    counts = _compute_transfer_counts(split_cfg, ds_info.train_size, ds_info.test_size, val_size)

    split_defs = [
        ("train", counts["train_count"], seed),
        ("val",   counts["val_count"],   seed + 1),
        ("test",  counts["test_count"],  seed + 2),
    ]

    # For detection, get contiguous category map
    det_cat_map: dict[int, int] | None = None
    if is_detection:
        plugin = get_dataset_plugin(name.lower())
        if plugin:
            det_cat_map = _get_contiguous_cat_map(plugin)

    # Try to load datasets to get labels + dataset objects for detection
    split_labels: dict[str, list[int] | None] = {}
    split_datasets: dict[str, object] = {}
    split_actual_n: dict[str, int] = {}
    for sname, sn, _ in split_defs:
        if sn <= 0:
            split_labels[sname] = None
            continue
        try:
            ds_obj = _load_dataset(name.lower(), sname)
            split_datasets[sname] = ds_obj
            split_actual_n[sname] = len(ds_obj)
            split_labels[sname] = _extract_labels_from_ds(ds_obj)
        except Exception:
            split_labels[sname] = None

    cache: dict = {"method": method, "splits": {}}

    for sname, sn, sseed in split_defs:
        # Use actual dataset size when available (theoretical may differ)
        sn = split_actual_n.get(sname, sn)
        labels = split_labels.get(sname)
        idx_map = compute_partition_indices(sn, sseed, parts,
                                            labels=labels, method=method)

        # Compute per-partition per-class counts
        class_counts: dict[str, dict[str, int]] = {}

        if is_detection and det_cat_map and sname in split_datasets:
            # Detection: count annotations per contiguous class per partition
            ds_obj = split_datasets[sname]
            for pid, indices in idx_map.items():
                cc: Counter = Counter()
                for i in indices:
                    anns = _get_detection_anns_fast(ds_obj, i)
                    if anns:
                        for a in anns:
                            cat_id = a.get("category_id")
                            if cat_id in det_cat_map:
                                cc[det_cat_map[cat_id]] += 1
                class_counts[pid] = {str(k): v for k, v in sorted(cc.items())}
        else:
            # Classification: count images per class
            for pid, indices in idx_map.items():
                if labels and len(labels) == sn:
                    cc = Counter()
                    for i in indices:
                        cc[labels[i]] += 1
                    class_counts[pid] = {str(k): v for k, v in sorted(cc.items())}
                else:
                    class_counts[pid] = {}

        cache["splits"][sname] = {
            "n": sn,
            "indices": idx_map,
            "class_counts": class_counts,
        }

    # Persist
    _partition_cache_path(name).write_text(json.dumps(cache))
    return cache


def _load_partition_cache(name: str) -> dict | None:
    """Load cached partition indices.  Returns None if not yet computed."""
    p = _partition_cache_path(name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _invalidate_partition_cache(name: str):
    """Delete the cache file so it will be rebuilt on next access."""
    p = _partition_cache_path(name)
    p.unlink(missing_ok=True)


def _expected_split_sizes(name: str) -> dict[str, int] | None:
    """Return expected split sizes from current split config + dataset info."""
    ds_info = _get_info(name)
    if not ds_info:
        return None
    split_cfg = _load_split_config(name.lower())
    val_size = getattr(ds_info, 'val_size', 0)
    counts = _compute_transfer_counts(split_cfg, ds_info.train_size, ds_info.test_size, val_size)
    return {
        "train": counts["train_count"],
        "val": counts["val_count"],
        "test": counts["test_count"],
    }


def _is_partition_cache_stale(name: str, cache: dict) -> bool:
    """True when cache split sizes no longer match current expected sizes."""
    expected = _expected_split_sizes(name)
    if expected is None:
        return False

    splits = cache.get("splits", {}) if isinstance(cache, dict) else {}
    for split_name in ("train", "val", "test"):
        cached_n = splits.get(split_name, {}).get("n")
        if cached_n is None:
            return True
        if int(cached_n) != int(expected[split_name]):
            return True

    return False


def _get_or_build_cache(name: str) -> dict | None:
    """Return cached partition data, building it if missing."""
    cache = _load_partition_cache(name)
    if cache is None or _is_partition_cache_stale(name, cache):
        cache = _build_partition_cache(name)
    return cache


# ── Partition index computation ──────────────────────────────────────────────

def compute_partition_indices(
    n: int,
    seed: int,
    partitions: list[dict],
    labels: list[int] | None = None,
    method: str = "stratified",
) -> dict[str, list[int]]:
    """Compute deterministic index sets for each partition + master.

    Parameters
    ----------
    n : int
        Total number of samples in the split.
    seed : int
        Random seed for shuffling.
    partitions : list[dict]
        Each dict has ``id`` and ``percent``.
    labels : list[int] | None
        Per-sample class labels (length == *n*).
        Required for ``stratified``, ``round_robin``, and ``iterative``.
        When ``None`` the method falls back to ``random``.
    method : str
        ``"random"``      – plain random shuffle (original behaviour).
        ``"stratified"``  – proportional class distribution per partition.
        ``"round_robin"`` – round-robin interleaving across classes.
        ``"iterative"``   – greedy iterative stratification.

    Returns
    -------
    dict mapping partition id (and ``"master"``) to sorted index lists.
    """
    rng = np.random.RandomState(seed)

    # If labels unavailable, force random regardless of requested method
    if labels is None or len(labels) != n:
        method = "random"

    if method == "random":
        return _partition_random(n, rng, partitions)
    if method == "stratified":
        return _partition_stratified(n, rng, partitions, labels)  # type: ignore[arg-type]
    if method == "round_robin":
        return _partition_round_robin(n, rng, partitions, labels)  # type: ignore[arg-type]
    if method == "iterative":
        return _partition_iterative(n, rng, partitions, labels)  # type: ignore[arg-type]

    # Unknown method → fall back to stratified
    return _partition_stratified(n, rng, partitions, labels)  # type: ignore[arg-type]


def _partition_random(
    n: int, rng: np.random.RandomState, partitions: list[dict],
) -> dict[str, list[int]]:
    """Plain random shuffle — original behaviour."""
    perm = rng.permutation(n).tolist()
    result: dict[str, list[int]] = {}
    offset = 0
    for part in partitions:
        count = int(n * part["percent"] / 100)
        result[part["id"]] = sorted(perm[offset:offset + count])
        offset += count
    result["master"] = sorted(perm[offset:])
    return result


def _partition_stratified(
    n: int, rng: np.random.RandomState, partitions: list[dict],
    labels: list[int],
) -> dict[str, list[int]]:
    """Proportional class distribution — each partition gets its share of every class."""
    from collections import defaultdict

    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        class_indices[lbl].append(idx)

    for cls in sorted(class_indices.keys()):
        rng.shuffle(class_indices[cls])

    result: dict[str, list[int]] = {p["id"]: [] for p in partitions}
    result["master"] = []

    for cls in sorted(class_indices.keys()):
        indices = class_indices[cls]
        cls_n = len(indices)
        offset = 0
        for part in partitions:
            count = int(cls_n * part["percent"] / 100)
            result[part["id"]].extend(indices[offset:offset + count])
            offset += count
        result["master"].extend(indices[offset:])

    for key in result:
        result[key] = sorted(result[key])
    return result


def _partition_round_robin(
    n: int, rng: np.random.RandomState, partitions: list[dict],
    labels: list[int],
) -> dict[str, list[int]]:
    """Round-robin interleaving — cycles through partitions class-by-class.

    Within each class the shuffled samples are dealt out one-by-one to
    partitions in proportion to their requested percentages.  This gives
    a maximally even spread even for very small classes.
    """
    from collections import defaultdict

    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        class_indices[lbl].append(idx)

    for cls in sorted(class_indices.keys()):
        rng.shuffle(class_indices[cls])

    # Build a slot sequence proportional to each partition's percent.
    # e.g. partitions=[30%, 20%] + master(50%) → 30 slots for p1, 20 for p2, 50 for master
    slot_ids: list[str] = []
    for part in partitions:
        slot_ids.extend([part["id"]] * part["percent"])
    master_pct = 100 - sum(p["percent"] for p in partitions)
    slot_ids.extend(["master"] * max(master_pct, 0))
    # Shuffle to interleave fairly (deterministic via rng)
    rng.shuffle(slot_ids)

    result: dict[str, list[int]] = {p["id"]: [] for p in partitions}
    result["master"] = []

    for cls in sorted(class_indices.keys()):
        indices = class_indices[cls]
        for i, idx in enumerate(indices):
            slot = slot_ids[i % len(slot_ids)]
            result[slot].append(idx)

    for key in result:
        result[key] = sorted(result[key])
    return result


def _partition_iterative(
    n: int, rng: np.random.RandomState, partitions: list[dict],
    labels: list[int],
) -> dict[str, list[int]]:
    """Iterative (greedy) stratification.

    Processes classes from **rarest to most common**.  For each sample the
    algorithm picks the partition whose current class-ratio deviates most
    from the desired proportion, guaranteeing the tightest possible
    balance — especially for long-tail / imbalanced datasets.
    """
    from collections import defaultdict, Counter

    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        class_indices[lbl].append(idx)

    for cls in sorted(class_indices.keys()):
        rng.shuffle(class_indices[cls])

    # Target fractions per partition (including master)
    all_ids: list[str] = [p["id"] for p in partitions] + ["master"]
    master_pct = 100 - sum(p["percent"] for p in partitions)
    target_frac: dict[str, float] = {}
    for part in partitions:
        target_frac[part["id"]] = part["percent"] / 100.0
    target_frac["master"] = max(master_pct, 0) / 100.0

    result: dict[str, list[int]] = {pid: [] for pid in all_ids}
    # Track per-partition per-class counts for greedy decisions
    part_class_counts: dict[str, Counter] = {pid: Counter() for pid in all_ids}
    part_totals: dict[str, int] = {pid: 0 for pid in all_ids}

    # Process classes from rarest to most common
    sorted_classes = sorted(class_indices.keys(),
                            key=lambda c: len(class_indices[c]))

    for cls in sorted_classes:
        indices = class_indices[cls]
        for idx in indices:
            # Pick partition with largest *deficit* for this class
            best_pid = all_ids[0]
            best_deficit = -float("inf")
            for pid in all_ids:
                desired = target_frac[pid] * n
                current = part_totals[pid]
                deficit = desired - current
                # Tie-break by class deficit
                cls_desired = target_frac[pid] * len(indices)
                cls_current = part_class_counts[pid][cls]
                cls_deficit = cls_desired - cls_current
                score = deficit + cls_deficit
                if score > best_deficit:
                    best_deficit = score
                    best_pid = pid

            result[best_pid].append(idx)
            part_class_counts[best_pid][cls] += 1
            part_totals[best_pid] += 1

    for key in result:
        result[key] = sorted(result[key])
    return result


def _partition_summary(name: str) -> dict:
    """Build full partition summary including per-split sizes.

    Reads from the partition cache (built once on save).  Falls back to
    building the cache on-the-fly if it doesn't exist yet.
    """
    ds = _get_info(name)
    if not ds:
        return {}
    cfg = _load_partition_config(name)
    seed = cfg.get("seed", 42)
    method = cfg.get("method", "stratified")
    parts = cfg.get("partitions", [])

    cache = _get_or_build_cache(name)

    # Extract index maps from cache (or empty fallback)
    def _idx(split_name: str) -> dict[str, list]:
        if cache and split_name in cache.get("splits", {}):
            return cache["splits"][split_name].get("indices", {})
        return {}

    def _cc(split_name: str) -> dict[str, dict[str, int]]:
        if cache and split_name in cache.get("splits", {}):
            return cache["splits"][split_name].get("class_counts", {})
        return {}

    def _n(split_name: str) -> int:
        if cache and split_name in cache.get("splits", {}):
            return cache["splits"][split_name].get("n", 0)
        return 0

    train_idx, val_idx, test_idx = _idx("train"), _idx("val"), _idx("test")
    train_n, val_n, test_n = _n("train"), _n("val"), _n("test")
    train_cc, val_cc, test_cc = _cc("train"), _cc("val"), _cc("test")

    partition_list = []
    for p in parts:
        pid = p["id"]
        partition_list.append({
            "id": pid,
            "name": p["name"],
            "percent": p["percent"],
            "train_count": len(train_idx.get(pid, [])),
            "val_count": len(val_idx.get(pid, [])),
            "test_count": len(test_idx.get(pid, [])),
            "class_counts": {
                "train": train_cc.get(pid, {}),
                "val": val_cc.get(pid, {}),
                "test": test_cc.get(pid, {}),
            },
        })

    master_pct = 100 - sum(p["percent"] for p in parts)
    master_entry = {
        "id": "master",
        "name": "Master",
        "percent": master_pct,
        "train_count": len(train_idx.get("master", [])),
        "val_count": len(val_idx.get("master", [])),
        "test_count": len(test_idx.get("master", [])),
        "class_counts": {
            "train": train_cc.get("master", {}),
            "val": val_cc.get("master", {}),
            "test": test_cc.get("master", {}),
        },
    }

    return {
        "seed": seed,
        "method": method,
        "available_methods": list(PARTITION_METHODS),
        "master": master_entry,
        "partitions": partition_list,
        "total_train": train_n,
        "total_val": val_n,
        "total_test": test_n,
    }


@router.put("/{name}/partitions/method", summary="Change partition method")
async def update_partition_method(name: str, body: UpdatePartitionMethod):
    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    if body.method not in PARTITION_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method '{body.method}'. "
                   f"Choose from: {', '.join(PARTITION_METHODS)}"
        )
    cfg = _load_partition_config(name.lower())
    cfg["method"] = body.method
    _save_partition_config(name.lower(), cfg)
    _build_partition_cache(name.lower())
    return _partition_summary(name.lower())


@router.get("/{name}/partitions", summary="Get dataset partitions")
async def get_partitions(name: str):
    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return _partition_summary(name.lower())


@router.post("/{name}/partitions", summary="Create a new partition")
async def create_partition(name: str, body: CreatePartition):
    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    cfg = _load_partition_config(name.lower())
    parts = cfg.get("partitions", [])

    used_pct = sum(p["percent"] for p in parts)
    if used_pct + body.percent > 99:
        raise HTTPException(
            status_code=400,
            detail=f"Total partition percent would be {used_pct + body.percent}% "
                   f"(max 99%, master needs at least 1%)"
        )

    import uuid
    new_id = f"p_{uuid.uuid4().hex[:8]}"
    parts.append({"id": new_id, "name": body.name, "percent": body.percent})
    cfg["partitions"] = parts
    _save_partition_config(name.lower(), cfg)
    _build_partition_cache(name.lower())
    return _partition_summary(name.lower())


@router.post("/{name}/partitions/{partition_id}/split",
             summary="Split a partition into sub-partitions (original is deleted)")
async def split_partition(name: str, partition_id: str, body: SplitPartitionBody):
    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    cfg = _load_partition_config(name.lower())
    parts = cfg.get("partitions", [])

    if partition_id == "master":
        # Master = remainder; its percent = 100 - sum(parts)
        parent_pct = 100 - sum(p["percent"] for p in parts)
        insert_idx = len(parts)  # append at end (master is always last in index space)
    else:
        # Find the partition to split
        found = None
        for i, p in enumerate(parts):
            if p["id"] == partition_id:
                found = (i, p)
                break
        if found is None:
            raise HTTPException(status_code=404, detail=f"Partition '{partition_id}' not found")
        insert_idx, parent = found
        parent_pct = parent["percent"]

    # Validate children sum == parent percent
    children_total = sum(c.percent for c in body.children)
    if children_total != parent_pct:
        raise HTTPException(
            status_code=400,
            detail=f"Children total ({children_total}%) must equal "
                   f"parent partition ({parent_pct}%)"
        )

    # Build new partition entries
    import uuid
    new_parts = []
    for c in body.children:
        new_parts.append({
            "id": f"p_{uuid.uuid4().hex[:8]}",
            "name": c.name,
            "percent": c.percent,
        })

    # Replace: remove original (if not master), insert children at same position
    if partition_id != "master":
        parts.pop(insert_idx)
    for j, np_ in enumerate(new_parts):
        parts.insert(insert_idx + j, np_)

    cfg["partitions"] = parts
    _save_partition_config(name.lower(), cfg)
    _invalidate_cache(name.lower())
    _build_partition_cache(name.lower())
    return _partition_summary(name.lower())


@router.delete("/{name}/partitions/{partition_id}", summary="Delete a partition (returns data to master)")
async def delete_partition(name: str, partition_id: str):
    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    if partition_id == "master":
        raise HTTPException(status_code=400, detail="Cannot delete the master partition")
    cfg = _load_partition_config(name.lower())
    parts = cfg.get("partitions", [])
    before = len(parts)
    parts = [p for p in parts if p["id"] != partition_id]
    if len(parts) == before:
        raise HTTPException(status_code=404, detail=f"Partition '{partition_id}' not found")
    cfg["partitions"] = parts
    _save_partition_config(name.lower(), cfg)
    _build_partition_cache(name.lower())
    return _partition_summary(name.lower())


# ── Dataset loading (universal — delegates to plugin) ────────────────────────

_dataset_cache: dict[tuple[str, str], object] = {}
_dataset_cache_lock = threading.Lock()


def _invalidate_cache(name: str | None = None):
    """Clear cached dataset objects. If name is None, clear all."""
    with _dataset_cache_lock:
        if name is None:
            _dataset_cache.clear()
        else:
            keys = [k for k in _dataset_cache if k[0] == name.lower()]
            for k in keys:
                del _dataset_cache[k]


def _load_raw_dataset(key: str, split: str = "train"):
    """Load a raw dataset split via plugin. Cached to avoid repeated heavy I/O."""
    cache_key = (key.lower(), split)
    with _dataset_cache_lock:
        cached = _dataset_cache.get(cache_key)
    if cached is not None:
        return cached

    plugin = get_dataset_plugin(key)
    if not plugin:
        raise ValueError(f"No loader for dataset '{key}'")
    ds = plugin.load_split(split=split)
    if ds is not None:
        with _dataset_cache_lock:
            _dataset_cache[cache_key] = ds
    return ds


def _load_dataset(key: str, split: str = "train"):
    """Load dataset with transfer-based splitting."""
    from torch.utils.data import ConcatDataset

    cfg = _load_split_config(key)

    has_transfers = any(cfg.get(k, 0) > 0 for k in _TRANSFER_KEYS)
    if not has_transfers:
        ds = _load_raw_dataset(key, split=split)
        if ds is None:
            fallback = _load_raw_dataset(key, split="train")
            return Subset(fallback, [])
        return ds

    orig_train = _load_raw_dataset(key, split="train")
    orig_test = _load_raw_dataset(key, split="test")
    orig_val = _load_raw_dataset(key, split="val")

    n_train = len(orig_train) if orig_train else 0
    n_test = len(orig_test) if orig_test else 0
    n_val = len(orig_val) if orig_val else 0

    parts = [ds for ds in [orig_train, orig_test, orig_val] if ds is not None]
    if not parts:
        raise ValueError(f"No data available for dataset '{key}'")
    pool = ConcatDataset(parts)

    train_idx, val_idx, test_idx = _apply_transfers(cfg, n_train, n_test, n_val)

    if split == "train":
        return Subset(pool, train_idx)
    if split == "val":
        return Subset(pool, val_idx)
    return Subset(pool, test_idx)
