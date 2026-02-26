"""
File-based persistence for built model records.
Stores builds as JSON files in the builds/ directory.
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

BUILDS_DIR = Path(__file__).parent.parent.parent / "builds"
BUILDS_DIR.mkdir(parents=True, exist_ok=True)


def _build_path(build_id: str) -> Path:
    return BUILDS_DIR / f"{build_id}.json"


def save_build(record: dict[str, Any]) -> str:
    """Save a build record. Returns build_id."""
    build_id = record.get("build_id") or uuid.uuid4().hex[:12]
    record["build_id"] = build_id

    path = _build_path(build_id)
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return build_id


def load_build(build_id: str) -> dict[str, Any] | None:
    path = _build_path(build_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def list_builds() -> list[dict[str, Any]]:
    results = []
    for p in sorted(BUILDS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p) as f:
                data = json.load(f)
            results.append(data)
        except Exception:
            continue
    return results


def delete_build(build_id: str) -> bool:
    path = _build_path(build_id)
    if path.exists():
        path.unlink()
        return True
    return False


def find_build_by_name(model_name: str) -> dict[str, Any] | None:
    """Find existing build by model name (case-insensitive)."""
    for p in BUILDS_DIR.glob("*.json"):
        try:
            with open(p) as f:
                data = json.load(f)
            if data.get("model_name", "").lower() == model_name.lower():
                return data
        except Exception:
            continue
    return None
