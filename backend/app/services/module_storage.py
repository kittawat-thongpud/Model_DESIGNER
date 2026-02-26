"""
Storage for custom nn.Module definitions.

Each module is saved as a folder:
  MODULES_DIR/{module_id}/
    record.json   ← metadata + args
    code.py       ← Python source code
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime
from pathlib import Path

from ..config import MODULES_DIR


def _dir(module_id: str) -> Path:
    return MODULES_DIR / module_id


def save_module(name: str, code: str, args: list[dict],
                category: str = "custom", description: str = "",
                module_id: str | None = None) -> dict:
    """Create or update a custom module definition."""
    mid = module_id or uuid.uuid4().hex[:12]
    d = _dir(mid)
    d.mkdir(parents=True, exist_ok=True)

    now = datetime.utcnow().isoformat() + "Z"
    record_path = d / "record.json"

    # Preserve created_at on update
    created_at = now
    if record_path.exists():
        old = json.loads(record_path.read_text())
        created_at = old.get("created_at", now)

    record = {
        "module_id": mid,
        "name": name,
        "args": args,
        "category": category,
        "description": description,
        "created_at": created_at,
        "updated_at": now,
    }
    record_path.write_text(json.dumps(record, indent=2))
    (d / "code.py").write_text(code)
    return record


def load_module(module_id: str) -> dict | None:
    """Load a module record + code."""
    d = _dir(module_id)
    rp = d / "record.json"
    cp = d / "code.py"
    if not rp.exists():
        return None
    record = json.loads(rp.read_text())
    record["code"] = cp.read_text() if cp.exists() else ""
    return record


def list_modules() -> list[dict]:
    """List all custom modules (metadata only, no code)."""
    results = []
    if not MODULES_DIR.exists():
        return results
    for d in sorted(MODULES_DIR.iterdir()):
        rp = d / "record.json"
        if rp.exists():
            try:
                record = json.loads(rp.read_text())
                record["arg_count"] = len(record.get("args", []))
                results.append(record)
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def delete_module(module_id: str) -> bool:
    """Delete a custom module."""
    import shutil
    d = _dir(module_id)
    if d.exists():
        shutil.rmtree(d)
        return True
    return False
