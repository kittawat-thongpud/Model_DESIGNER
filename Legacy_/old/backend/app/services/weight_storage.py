"""
Persistent storage for trained weight metadata.
Each weight has a .pt file + a .meta.json file in the weights/ directory.
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime
from pathlib import Path

from ..storage import WEIGHTS_DIR


def _meta_path(weight_id: str) -> Path:
    return WEIGHTS_DIR / f"{weight_id}.meta.json"


def save_weight_meta(
    model_id: str,
    model_name: str,
    job_id: str | None,
    dataset: str,
    epochs_trained: int,
    final_accuracy: float | None,
    final_loss: float | None,
    weight_id: str | None = None,
) -> str:
    """Save weight metadata alongside the .pt file. Returns weight_id."""
    if weight_id is None:
        weight_id = uuid.uuid4().hex[:12]

    pt_path = WEIGHTS_DIR / f"{weight_id}.pt"
    file_size = pt_path.stat().st_size if pt_path.exists() else 0

    meta = {
        "weight_id": weight_id,
        "model_id": model_id,
        "model_name": model_name,
        "job_id": job_id,
        "dataset": dataset,
        "epochs_trained": epochs_trained,
        "final_accuracy": final_accuracy,
        "final_loss": final_loss,
        "file_size_bytes": file_size,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    path = _meta_path(weight_id)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

    return weight_id


def load_weight_meta(weight_id: str) -> dict | None:
    """Load weight metadata."""
    path = _meta_path(weight_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def list_weights(model_id: str | None = None) -> list[dict]:
    """List all weight metadata, optionally filtered by model_id."""
    results: list[dict] = []
    for p in sorted(WEIGHTS_DIR.glob("*.meta.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p) as f:
                data = json.load(f)
            if model_id and data.get("model_id") != model_id:
                continue
            results.append(data)
        except Exception:
            continue
    return results


def delete_weight(weight_id: str) -> bool:
    """Delete weight .pt and .meta.json files."""
    deleted = False
    meta = _meta_path(weight_id)
    if meta.exists():
        meta.unlink()
        deleted = True
    pt = WEIGHTS_DIR / f"{weight_id}.pt"
    if pt.exists():
        pt.unlink()
        deleted = True
    return deleted
