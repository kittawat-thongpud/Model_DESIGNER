"""
Dataset Samples Controller — paginated sample browsing and single-sample detail.

Split from dataset_controller.py to reduce file size.
All endpoints share the /api/datasets prefix.
"""
from __future__ import annotations
import asyncio
import threading
from functools import partial

from fastapi import APIRouter, HTTPException
from torch.utils.data import Subset

from ..services.dataset_registry import get_dataset_info as _get_info
from ..plugins.loader import get_dataset_plugin

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])


# ── Image encoding helpers ────────────────────────────────────────────────────

_ENCODE_POOL = None
_ENCODE_POOL_LOCK = threading.Lock()


def _get_encode_pool():
    global _ENCODE_POOL
    if _ENCODE_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        with _ENCODE_POOL_LOCK:
            if _ENCODE_POOL is None:
                _ENCODE_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="img_enc")
    return _ENCODE_POOL


def _extract_label(raw_label) -> tuple[int, list | None]:
    """Normalize label from classification (int) or detection (list of dicts)."""
    if isinstance(raw_label, int):
        return raw_label, None
    if isinstance(raw_label, (list, tuple)):
        return -1, list(raw_label)
    try:
        return int(raw_label), None
    except (TypeError, ValueError):
        return -1, None


def _encode_sample(
    dataset, idx: int, ds_info, thumb_size: int | None, include_annotations: bool,
    cat_id_map: dict[int, str] | None = None,
) -> dict:
    """Load, optionally resize, and encode one sample. Thread-safe."""
    import base64
    import io
    from PIL import Image
    import numpy as _np

    img, raw_label = dataset[idx]
    lbl, annotations = _extract_label(raw_label)

    if not isinstance(img, Image.Image):
        if isinstance(img, _np.ndarray):
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(img.numpy())

    orig_w, orig_h = img.size

    if thumb_size and max(orig_w, orig_h) > thumb_size:
        img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)

    if img.mode in ("RGB", "RGBA") and (orig_w > 64 or orig_h > 64):
        fmt, mime = "JPEG", "image/jpeg"
        buf = io.BytesIO()
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=85)
    else:
        fmt, mime = "PNG", "image/png"
        buf = io.BytesIO()
        img.save(buf, format="PNG")

    b64 = base64.b64encode(buf.getvalue()).decode()
    thumb_bytes = buf.tell()

    if cat_id_map and lbl == -1:
        class_name = "multi"
    elif cat_id_map and lbl >= 0:
        class_name = cat_id_map.get(lbl, str(lbl))
    elif lbl >= 0 and lbl < len(ds_info.class_names):
        class_name = ds_info.class_names[lbl]
    elif lbl >= 0:
        class_name = str(lbl)
    else:
        class_name = "multi"

    result: dict = {
        "index": idx, "label": lbl, "class_name": class_name,
        "image_base64": b64, "mime": mime,
        "orig_w": orig_w, "orig_h": orig_h,
        "thumb_bytes": thumb_bytes,
    }

    if include_annotations and annotations:
        result["annotations"] = _format_annotations(annotations, cat_id_map or {})

    return result


def _format_annotations(annotations: list, cat_id_map: dict[int, str]) -> list[dict]:
    """Normalize detection annotations to a standard format."""
    formatted = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        bbox = ann.get("bbox", [])
        cat_id = ann.get("category_id", -1)
        cat_name = cat_id_map.get(cat_id, str(cat_id)) if isinstance(cat_id, int) else ""
        formatted.append({
            "bbox": bbox,
            "category_id": cat_id,
            "category_name": cat_name,
            "area": ann.get("area", 0),
        })
    return formatted


def _get_detection_anns_fast(dataset, idx) -> list[dict] | None:
    """Get annotations from a detection dataset without loading the image."""
    from bisect import bisect_right

    raw = dataset
    real_idx = idx

    while isinstance(raw, Subset):
        if real_idx < 0 or real_idx >= len(raw.indices):
            return None
        real_idx = raw.indices[real_idx]
        raw = raw.dataset

    from torch.utils.data import ConcatDataset as _Cat
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

    if hasattr(raw, '_index') and isinstance(raw._index, list):
        if 0 <= real_idx < len(raw._index):
            return raw._index[real_idx].get("anns", [])

    return None


def _get_contiguous_cat_map(plugin) -> dict[int, int] | None:
    """Return {original_category_id → contiguous_class_index} for a detection plugin."""
    if hasattr(plugin, '_cat_id_to_contiguous'):
        return plugin._cat_id_to_contiguous()
    return None


# ── Paginated samples ────────────────────────────────────────────────────────

@router.get("/{name}/samples", summary="Paginated dataset samples with class filter")
async def dataset_samples(
    name: str,
    page: int = 0,
    page_size: int = 50,
    class_idx: int | None = None,
    class_indices: str | None = None,
    split: str = "train",
    thumb_size: int | None = 128,
    include_annotations: bool = False,
    partition_id: str | None = None,
):
    # Import from main controller for shared helpers
    from .dataset_controller import _load_dataset, _get_or_build_cache

    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    page_size = max(1, min(page_size, 200))

    filter_set: set[int] | None = None
    if class_indices is not None and class_indices.strip():
        filter_set = {int(x.strip()) for x in class_indices.split(",") if x.strip().isdigit()}
    elif class_idx is not None:
        filter_set = {class_idx}

    partition_index_set: set[int] | None = None
    if partition_id:
        cache = _get_or_build_cache(name.lower())
        if cache and split in cache.get("splits", {}):
            pidx = cache["splits"][split].get("indices", {}).get(partition_id)
            if pidx is not None:
                partition_index_set = set(pidx)

    try:
        dataset = _load_dataset(name.lower(), split)
        total = len(dataset)

        is_detection = getattr(ds, 'task_type', 'classification') == 'detection'

        cat_id_map: dict[int, str] | None = None
        cat_id_to_contiguous: dict[int, int] | None = None
        plugin = None
        if is_detection:
            plugin = get_dataset_plugin(name.lower())
            if plugin:
                cat_id_map = plugin.category_id_map
                cat_id_to_contiguous = _get_contiguous_cat_map(plugin)

        if partition_index_set is not None:
            candidate_indices = sorted(partition_index_set)
        else:
            candidate_indices = list(range(total))

        if filter_set:
            if is_detection and cat_id_to_contiguous:
                contiguous_to_cat = {v: k for k, v in cat_id_to_contiguous.items()}
                filter_cat_ids = {contiguous_to_cat[ci] for ci in filter_set
                                  if ci in contiguous_to_cat}
                indices = []
                for i in candidate_indices:
                    anns = _get_detection_anns_fast(dataset, i)
                    if anns is None:
                        _, raw_label = dataset[i]
                        _, anns = _extract_label(raw_label)
                    if anns:
                        img_cats = {a.get("category_id") for a in anns
                                    if isinstance(a, dict)}
                        if img_cats & filter_cat_ids:
                            indices.append(i)
                filtered_total = len(indices)
                start = page * page_size
                end = min(start + page_size, filtered_total)
                page_indices = indices[start:end]
            else:
                indices = []
                for i in candidate_indices:
                    _, label = dataset[i]
                    lbl, _ = _extract_label(label)
                    if lbl in filter_set:
                        indices.append(i)
                filtered_total = len(indices)
                start = page * page_size
                end = min(start + page_size, filtered_total)
                page_indices = indices[start:end]
        else:
            filtered_total = len(candidate_indices)
            start = page * page_size
            end = min(start + page_size, filtered_total)
            page_indices = candidate_indices[start:end]

        loop = asyncio.get_event_loop()
        pool = _get_encode_pool()

        _enc = partial(_encode_sample, cat_id_map=cat_id_map)
        futures = [
            loop.run_in_executor(
                pool,
                _enc, dataset, i, ds, thumb_size, include_annotations,
            )
            for i in page_indices
        ]
        samples = await asyncio.gather(*futures)

        total_bytes = sum(s["thumb_bytes"] for s in samples) if samples else 0
        avg_thumb_bytes = total_bytes // len(samples) if samples else 0

        return {
            "dataset": name, "split": split, "page": page,
            "page_size": page_size, "total": filtered_total,
            "total_pages": max(1, (filtered_total + page_size - 1) // page_size),
            "class_idx": class_idx, "samples": list(samples),
            "avg_thumb_bytes": avg_thumb_bytes,
            "task_type": getattr(ds, 'task_type', 'classification'),
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="torchvision not available")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Single full-resolution image ─────────────────────────────────────────────

@router.get("/{name}/samples/{index}", summary="Get single sample at full resolution")
async def dataset_sample_detail(
    name: str,
    index: int,
    split: str = "train",
    include_annotations: bool = True,
):
    from .dataset_controller import _load_dataset

    ds = _get_info(name)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    is_detection = getattr(ds, 'task_type', 'classification') == 'detection'
    cat_id_map: dict[int, str] | None = None
    if is_detection:
        plugin = get_dataset_plugin(name.lower())
        if plugin:
            cat_id_map = plugin.category_id_map

    try:
        dataset = _load_dataset(name.lower(), split)
        if index < 0 or index >= len(dataset):
            raise HTTPException(status_code=404, detail=f"Sample index {index} out of range (0..{len(dataset)-1})")

        loop = asyncio.get_event_loop()
        pool = _get_encode_pool()
        _enc = partial(_encode_sample, cat_id_map=cat_id_map)
        result = await loop.run_in_executor(
            pool, _enc, dataset, index, ds, None, include_annotations,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
