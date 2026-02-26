"""
Weight snapshot storage for recording weight progression during training.
Stores per-layer tensor snapshots and statistics at configurable intervals.
"""
from __future__ import annotations
import json
import math
import re
import numpy as np
from pathlib import Path
from typing import Any

import torch

# Base directory for job snapshots
from ..config import JOBS_DIR


def _snapshots_dir(job_id: str) -> Path:
    d = JOBS_DIR / job_id / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _snapshot_path(job_id: str, epoch: int, layer_name: str) -> Path:
    safe_name = layer_name.replace(".", "__")
    return _snapshots_dir(job_id) / f"ep{epoch}_{safe_name}.pt"


def _stats_path(job_id: str) -> Path:
    return _snapshots_dir(job_id) / "stats.jsonl"


def _safe_float(v: float) -> float:
    """Replace NaN / Inf with 0.0 so JSON stays valid."""
    return 0.0 if (math.isnan(v) or math.isinf(v)) else round(v, 6)


def _compute_stats(epoch: int, layer_name: str, tensor: torch.Tensor) -> dict:
    """Compute per-layer statistics."""
    data = tensor.cpu().detach().float()
    return {
        "epoch": epoch,
        "layer": layer_name,
        "shape": list(tensor.shape),
        "min": _safe_float(float(data.min())),
        "max": _safe_float(float(data.max())),
        "mean": _safe_float(float(data.mean())),
        "std": _safe_float(float(data.std())),
        "norm": _safe_float(float(data.norm())),
        "num_params": int(data.numel()),
    }


def _append_stats(job_id: str, stats: dict) -> None:
    sp = _stats_path(job_id)
    with open(sp, "a") as f:
        f.write(json.dumps(stats) + "\n")


def save_snapshot(
    job_id: str,
    epoch: int,
    layer_name: str,
    tensor: torch.Tensor,
) -> None:
    """Save a single layer's weight tensor and append stats."""
    path = _snapshot_path(job_id, epoch, layer_name)
    torch.save(tensor.cpu().detach(), str(path))
    _append_stats(job_id, _compute_stats(epoch, layer_name, tensor))


def save_full_snapshot(
    job_id: str,
    epoch: int,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Save entire state_dict as one .pt file and compute per-layer stats."""
    snap_dir = _snapshots_dir(job_id)
    path = snap_dir / f"ep{epoch}.pt"
    torch.save(state_dict, str(path))

    # Append per-layer stats for visualization
    for name, tensor in state_dict.items():
        _append_stats(job_id, _compute_stats(epoch, name, tensor))


def get_epoch_thumbnails(job_id: str, epoch: int, max_size: int = 48) -> dict:
    """Load all layer tensors for an epoch and return downsampled thumbnails."""
    import torch.nn.functional as F

    snap_dir = _snapshots_dir(job_id)
    state_dict: dict[str, torch.Tensor] = {}

    # Try full state_dict file first
    full_path = snap_dir / f"ep{epoch}.pt"
    if full_path.exists():
        state_dict = torch.load(str(full_path), map_location="cpu")
    else:
        # Gather per-layer files
        prefix = f"ep{epoch}_"
        for f in snap_dir.iterdir():
            if f.suffix == ".pt" and f.stem.startswith(prefix):
                name = f.stem[len(prefix):].replace("__", ".")
                state_dict[name] = torch.load(str(f), map_location="cpu")

    result: dict[str, dict] = {}
    for name, tensor in state_dict.items():
        data = tensor.cpu().detach().float().nan_to_num(0.0)
        orig_shape = list(tensor.shape)

        if data.dim() == 0:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.dim() == 1:
            data = data.unsqueeze(0)
        elif data.dim() == 4:
            data = data.reshape(data.size(0), -1)
        elif data.dim() >= 3:
            data = data.reshape(data.size(0), -1)

        rows, cols = data.shape[0], data.shape[1]

        # Downsample if larger than max_size
        if rows > max_size or cols > max_size:
            scale = max_size / max(rows, cols)
            new_h = max(1, round(rows * scale))
            new_w = max(1, round(cols * scale))
            data = F.adaptive_avg_pool2d(
                data.unsqueeze(0).unsqueeze(0), (new_h, new_w)
            ).squeeze(0).squeeze(0)
            rows, cols = new_h, new_w

        result[name] = {
            "rows": int(rows),
            "cols": int(cols),
            "shape": orig_shape,
            "min": _safe_float(float(data.min())),
            "max": _safe_float(float(data.max())),
            "values": data.tolist(),
        }

    return result


def list_snapshots(job_id: str) -> list[dict]:
    """List all snapshot metadata for a job."""
    sp = _stats_path(job_id)
    if not sp.exists():
        return []

    entries: list[dict] = []
    # Regex to fix legacy NaN / Infinity tokens written by Python's json encoder
    _nan_re = re.compile(r'\bNaN\b')
    _inf_re = re.compile(r'\bInfinity\b')
    with open(sp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Replace invalid JSON tokens before parsing
                cleaned = _nan_re.sub('0.0', line)
                cleaned = _inf_re.sub('0.0', cleaned)
                entries.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    return entries


def get_snapshot_stats(job_id: str) -> dict:
    """Get organized stats timeline grouped by layer."""
    entries = list_snapshots(job_id)
    if not entries:
        return {"layers": [], "epochs": [], "timeline": {}}

    layers = sorted(set(e["layer"] for e in entries))
    epochs = sorted(set(e["epoch"] for e in entries))

    timeline: dict[str, list[dict]] = {}
    for layer in layers:
        timeline[layer] = [
            e for e in entries if e["layer"] == layer
        ]
        timeline[layer].sort(key=lambda x: x["epoch"])

    return {
        "layers": layers,
        "epochs": epochs,
        "timeline": timeline,
    }


def load_snapshot_data(
    job_id: str, epoch: int, layer_name: str
) -> dict | None:
    """Load a snapshot tensor and return as flattened data for visualization.
    Tries per-layer file first, then falls back to the full ep{N}.pt state_dict.
    """
    # 1. Try per-layer file
    path = _snapshot_path(job_id, epoch, layer_name)
    if path.exists():
        tensor = torch.load(str(path), map_location="cpu")
    else:
        # 2. Fall back to full state_dict ep{N}.pt
        full_path = _snapshots_dir(job_id) / f"ep{epoch}.pt"
        if not full_path.exists():
            return None
        state_dict = torch.load(str(full_path), map_location="cpu")
        if layer_name not in state_dict:
            return None
        tensor = state_dict[layer_name]

    data = tensor.float()

    # For 2D+ tensors, reshape to 2D for heatmap visualization
    shape = list(data.shape)
    if data.dim() == 1:
        matrix = data.unsqueeze(0)
    elif data.dim() == 2:
        matrix = data
    elif data.dim() == 4:
        # Conv weight: [out_ch, in_ch, kH, kW] → [out_ch, in_ch * kH * kW]
        matrix = data.reshape(data.size(0), -1)
    else:
        # Flatten higher dims into 2D
        matrix = data.reshape(data.size(0), -1)

    return {
        "epoch": epoch,
        "layer": layer_name,
        "shape": shape,
        "rows": int(matrix.size(0)),
        "cols": int(matrix.size(1)),
        "values": matrix.tolist(),
        "min": round(float(data.min()), 6),
        "max": round(float(data.max()), 6),
        "mean": round(float(data.mean()), 6),
        "std": round(float(data.std()), 6),
    }


def get_recorded_layers(job_id: str) -> list[str]:
    """Get list of unique layer names that have been recorded."""
    entries = list_snapshots(job_id)
    return sorted(set(e["layer"] for e in entries))


def get_recorded_epochs(job_id: str) -> list[int]:
    """Get list of epochs that have recorded snapshots."""
    entries = list_snapshots(job_id)
    return sorted(set(e["epoch"] for e in entries))
