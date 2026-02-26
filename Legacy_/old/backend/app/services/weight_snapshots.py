"""
Weight snapshot storage for recording weight progression during training.
Stores per-layer tensor snapshots and statistics at configurable intervals.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Any

import torch

# Base directory for job snapshots
JOBS_DIR = Path(__file__).parent.parent.parent / "jobs"


def _snapshots_dir(job_id: str) -> Path:
    d = JOBS_DIR / job_id / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _snapshot_path(job_id: str, epoch: int, layer_name: str) -> Path:
    safe_name = layer_name.replace(".", "__")
    return _snapshots_dir(job_id) / f"ep{epoch}_{safe_name}.pt"


def _stats_path(job_id: str) -> Path:
    return _snapshots_dir(job_id) / "stats.jsonl"


def save_snapshot(
    job_id: str,
    epoch: int,
    layer_name: str,
    tensor: torch.Tensor,
) -> None:
    """Save a single layer's weight tensor and append stats."""
    path = _snapshot_path(job_id, epoch, layer_name)
    torch.save(tensor.cpu().detach(), str(path))

    # Compute statistics
    data = tensor.cpu().detach().float()
    stats = {
        "epoch": epoch,
        "layer": layer_name,
        "shape": list(tensor.shape),
        "min": round(float(data.min()), 6),
        "max": round(float(data.max()), 6),
        "mean": round(float(data.mean()), 6),
        "std": round(float(data.std()), 6),
        "norm": round(float(data.norm()), 6),
        "num_params": int(data.numel()),
    }

    # Append to stats file
    sp = _stats_path(job_id)
    with open(sp, "a") as f:
        f.write(json.dumps(stats) + "\n")


def list_snapshots(job_id: str) -> list[dict]:
    """List all snapshot metadata for a job."""
    sp = _stats_path(job_id)
    if not sp.exists():
        return []

    entries: list[dict] = []
    with open(sp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
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
    """Load a snapshot tensor and return as flattened data for visualization."""
    path = _snapshot_path(job_id, epoch, layer_name)
    if not path.exists():
        return None

    tensor = torch.load(str(path), map_location="cpu")
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
