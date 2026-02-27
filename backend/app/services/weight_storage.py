"""
Persistent storage for trained weight metadata.

Directory layout (folder-per-weight):
  WEIGHTS_DIR/
    {weight_id}/
      weight.pt       ← model state_dict
      meta.json        ← weight metadata
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime
from pathlib import Path

from ..config import WEIGHTS_DIR
from .base_storage import BaseJsonStorage


# ── Storage instance (folder-per-weight, JSON file = meta.json) ────────────

_store = BaseJsonStorage(WEIGHTS_DIR, folder_mode="meta.json")


# ── Path helpers ────────────────────────────────────────────────────────────────────────

def weight_pt_path(weight_id: str) -> Path:
    """Public helper — returns the path to weight.pt for a given weight_id."""
    return _store.record_dir(weight_id) / "weight.pt"


# ── CRUD ────────────────────────────────────────────────────────────────────

def save_weight_meta(
    model_id: str,
    model_name: str,
    job_id: str | None,
    dataset: str,
    epochs_trained: int,
    final_accuracy: float | None,
    final_loss: float | None,
    weight_id: str | None = None,
    model_scale: str | None = None,
    parent_weight_id: str | None = None,
    total_time: float | None = None,
    device: str | None = None,
) -> str:
    """Save weight metadata alongside the .pt file. Returns weight_id.

    Builds a cumulative ``training_runs`` array:
      – inherits the parent weight's runs (if any)
      – appends the current training run
    This allows the full training lineage to be read from a single weight.
    """
    if weight_id is None:
        weight_id = uuid.uuid4().hex[:12]

    pt_path = weight_pt_path(weight_id)
    file_size = pt_path.stat().st_size if pt_path.exists() else 0

    # ── Build cumulative training_runs ──
    inherited_runs: list[dict] = []
    if parent_weight_id:
        parent_meta = load_weight_meta(parent_weight_id)
        if parent_meta:
            inherited_runs = list(parent_meta.get("training_runs", []))
            # If parent has no training_runs yet (legacy), synthesise one entry
            if not inherited_runs and parent_meta.get("job_id"):
                inherited_runs.append({
                    "run": 1,
                    "job_id": parent_meta["job_id"],
                    "weight_id": parent_weight_id,
                    "dataset": parent_meta.get("dataset", ""),
                    "epochs": parent_meta.get("epochs_trained", 0),
                    "accuracy": parent_meta.get("final_accuracy"),
                    "loss": parent_meta.get("final_loss"),
                    "total_time": parent_meta.get("total_time"),
                    "device": parent_meta.get("device"),
                    "created_at": parent_meta.get("created_at", ""),
                })

    current_run: dict = {
        "run": len(inherited_runs) + 1,
        "job_id": job_id,
        "weight_id": weight_id,
        "dataset": dataset,
        "epochs": epochs_trained,
        "accuracy": final_accuracy,
        "loss": final_loss,
        "total_time": total_time,
        "device": device,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    training_runs = inherited_runs + [current_run]

    # Compute cumulative totals
    total_epochs = sum(r.get("epochs", 0) for r in training_runs)

    meta = {
        "weight_id": weight_id,
        "model_id": model_id,
        "model_scale": model_scale or "n",
        "model_name": model_name,
        "job_id": job_id,
        "dataset": dataset,
        "epochs_trained": epochs_trained,
        "total_epochs": total_epochs,
        "final_accuracy": final_accuracy,
        "final_loss": final_loss,
        "file_size_bytes": file_size,
        "parent_weight_id": parent_weight_id,
        "training_runs": training_runs,
        "total_time": total_time,
        "device": device,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    _store.save(weight_id, meta)
    return weight_id


def load_weight_meta(weight_id: str) -> dict | None:
    """Load weight metadata."""
    return _store.load(weight_id)


def list_weights(model_id: str | None = None) -> list[dict]:
    """List all weight metadata, optionally filtered by model_id."""
    return _store.list_all(model_id=model_id)


def save_edit_meta(
    weight_id: str,
    edit_type: str,
    source_weight_id: str | None = None,
    mapping_count: int = 0,
    frozen_nodes: list[str] | None = None,
) -> None:
    """Append an edit operation record to the weight's metadata.

    This tracks weight edits (transfers, pruning, etc.) in the ``edits``
    array inside meta.json, preserving full edit history for lineage display.
    """
    meta = load_weight_meta(weight_id)
    if meta is None:
        return

    edits: list[dict] = meta.get("edits", [])
    edits.append({
        "edit_type": edit_type,
        "source_weight_id": source_weight_id,
        "mapping_count": mapping_count,
        "frozen_nodes": frozen_nodes or [],
        "created_at": datetime.utcnow().isoformat() + "Z",
    })
    meta["edits"] = edits
    _store.save(weight_id, meta)


def get_lineage(weight_id: str, max_depth: int = 20) -> list[dict]:
    """Walk the parent chain and return a list from oldest ancestor to current."""
    chain: list[dict] = []
    visited: set[str] = set()
    current_id: str | None = weight_id
    while current_id and current_id not in visited and len(chain) < max_depth:
        visited.add(current_id)
        meta = load_weight_meta(current_id)
        if not meta:
            break
        chain.append(meta)
        current_id = meta.get("parent_weight_id")
    chain.reverse()  # oldest first
    return chain


def resolve_dataset_name(meta: dict) -> str:
    """Return a human-readable dataset name for a weight record.

    Priority:
    1. meta["dataset"] if it looks like a plain name (no path separators)
    2. Extract from ``/datasets/<name>/`` pattern in meta["dataset"]
    3. Read the job data.yaml ``path:`` field and extract dataset dir name
    4. Fallback to empty string
    """
    import re as _re
    import yaml as _yaml

    raw = meta.get("dataset", "")
    if raw and "/" not in raw and "\\" not in raw:
        return raw

    if raw:
        normalized = raw.replace("\\", "/")
        m = _re.search(r"/datasets/([^/]+)/", normalized)
        if m:
            return m.group(1)

        # raw is a path to a job data.yaml — read it to get 'path:' field
        try:
            p = Path(raw)
            if p.exists():
                content = _yaml.safe_load(p.read_text())
                yaml_path_field = str(content.get("path", "")).replace("\\", "/")
                m2 = _re.search(r"/datasets/([^/]+)$", yaml_path_field)
                if m2:
                    return m2.group(1)
        except Exception:
            pass

    return ""


def delete_weight(weight_id: str) -> bool:
    """Delete entire weight folder."""
    existed = _store.delete(weight_id)
    # Also remove legacy flat files if they exist
    for legacy in (WEIGHTS_DIR / f"{weight_id}.pt", WEIGHTS_DIR / f"{weight_id}.meta.json"):
        if legacy.exists():
            legacy.unlink()
    return existed
