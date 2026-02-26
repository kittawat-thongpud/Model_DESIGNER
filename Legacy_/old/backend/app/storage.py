"""
File I/O helpers for model persistence.
"""
from __future__ import annotations
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from .schemas.model_schema import ModelGraph, ModelSummary

STORAGE_DIR = Path(__file__).parent.parent / "models"
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def _model_path(model_id: str) -> Path:
    return STORAGE_DIR / f"{model_id}.json"


def save_model(graph: ModelGraph, model_id: str | None = None) -> str:
    """Save a model graph to disk. Returns the model id."""
    if model_id is None:
        model_id = uuid.uuid4().hex[:12]

    graph.meta.updated_at = datetime.utcnow()
    data = graph.model_dump(mode="json")
    data["_id"] = model_id

    path = _model_path(model_id)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return model_id


def load_model(model_id: str) -> tuple[str, ModelGraph]:
    """Load a model graph from disk. Returns (id, graph)."""
    path = _model_path(model_id)
    if not path.exists():
        raise FileNotFoundError(f"Model '{model_id}' not found")

    with open(path) as f:
        data = json.load(f)

    mid = data.pop("_id", model_id)
    return mid, ModelGraph(**data)


def list_models() -> list[ModelSummary]:
    """List all saved models."""
    results: list[ModelSummary] = []
    for p in sorted(STORAGE_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p) as f:
                data = json.load(f)
            mid = data.get("_id", p.stem)
            meta = data.get("meta", {})
            results.append(ModelSummary(
                id=mid,
                name=meta.get("name", "Untitled"),
                description=meta.get("description", ""),
                created_at=meta.get("created_at", datetime.utcnow()),
                updated_at=meta.get("updated_at", datetime.utcnow()),
                node_count=len(data.get("nodes", [])),
                edge_count=len(data.get("edges", [])),
            ))
        except Exception:
            continue
    return results


def find_model_by_name(name: str) -> dict | None:
    """Find a model by name (case-insensitive). Return basic info or None."""
    target = name.lower().strip()
    for p in STORAGE_DIR.glob("*.json"):
        try:
            with open(p) as f:
                data = json.load(f)
            meta_name = data.get("meta", {}).get("name", "").lower().strip()
            if meta_name == target:
                mid = data.get("id") or data.get("_id") or p.stem
                return {"id": mid, "name": data.get("meta", {}).get("name")}
        except Exception:
            continue
    return None


def delete_model(model_id: str) -> bool:
    path = _model_path(model_id)
    if path.exists():
        path.unlink()
        return True
    return False
