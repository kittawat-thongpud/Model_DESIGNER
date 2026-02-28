"""
Storage for Ultralytics model YAML definitions.

Each model is saved as:
  MODELS_DIR/{model_id}/
    record.json   ← metadata (name, task, timestamps)
    model.yaml    ← Ultralytics YAML config
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime
from pathlib import Path

import yaml

from ..config import MODELS_DIR


def _dir(model_id: str) -> Path:
    return MODELS_DIR / model_id


def save_model(
    name: str,
    yaml_def: dict,
    task: str = "detect",
    description: str = "",
    model_id: str | None = None,
) -> dict:
    """Create or update a model definition."""
    mid = model_id or uuid.uuid4().hex[:12]
    d = _dir(mid)
    d.mkdir(parents=True, exist_ok=True)

    now = datetime.utcnow().isoformat() + "Z"
    record_path = d / "record.json"

    created_at = now
    if record_path.exists():
        old = json.loads(record_path.read_text())
        created_at = old.get("created_at", now)

    record = {
        "model_id": mid,
        "name": name,
        "description": description,
        "task": task,
        "created_at": created_at,
        "updated_at": now,
    }
    record_path.write_text(json.dumps(record, indent=2))

    # Extract _graph metadata and save separately
    graph_meta = yaml_def.pop("_graph", None)
    if graph_meta:
        (d / "graph.json").write_text(json.dumps(graph_meta, indent=2))

    # Write clean YAML (no _graph)
    yaml_str = _dict_to_yaml(yaml_def)
    (d / "model.yaml").write_text(yaml_str)

    # Return full yaml_def with _graph for response
    if graph_meta:
        yaml_def["_graph"] = graph_meta
    record["yaml_def"] = yaml_def
    return record


def load_model(model_id: str) -> dict | None:
    """Load a model record + YAML definition."""
    d = _dir(model_id)
    rp = d / "record.json"
    yp = d / "model.yaml"
    gp = d / "graph.json"
    if not rp.exists():
        return None
    record = json.loads(rp.read_text())
    if yp.exists():
        yd = yaml.safe_load(yp.read_text()) or {}
        # Normalize layers from YAML arrays to objects
        yd["backbone"] = _normalize_layers(yd.get("backbone", []))
        yd["head"] = _normalize_layers(yd.get("head", []))
        record["yaml_def"] = yd
    else:
        record["yaml_def"] = {}
    # Merge graph metadata if exists
    if gp.exists():
        try:
            record["yaml_def"]["_graph"] = json.loads(gp.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return record


def _sanitize_from(from_val):
    """Clean up from field: [-1, 15, 18, 21] → [15, 18, 21] (drop -1 when 2+ explicit indices)."""
    if isinstance(from_val, list):
        explicit = [f for f in from_val if f != -1]
        if len(explicit) >= 2 and -1 in from_val:
            from_val = explicit
        if len(from_val) == 1:
            from_val = from_val[0]
    return from_val


def _normalize_layers(layers: list) -> list[dict]:
    """Convert layers from YAML array format to object format."""
    result = []
    for layer in layers:
        if isinstance(layer, list):
            # [from, repeats, module, args]
            result.append({
                "from": _sanitize_from(layer[0] if len(layer) > 0 else -1),
                "repeats": layer[1] if len(layer) > 1 else 1,
                "module": str(layer[2]) if len(layer) > 2 else "Conv",
                "args": layer[3] if len(layer) > 3 and isinstance(layer[3], list) else [],
            })
        elif isinstance(layer, dict):
            # Already object format — normalize key alias
            result.append({
                "from": _sanitize_from(layer.get("from", layer.get("from_", -1))),
                "repeats": layer.get("repeats", 1),
                "module": layer.get("module", "Conv"),
                "args": layer.get("args", []),
            })
        else:
            continue
    return result


def load_model_yaml_path(model_id: str) -> Path | None:
    """Return the path to the model.yaml file."""
    p = _dir(model_id) / "model.yaml"
    return p if p.exists() else None


def list_models() -> list[dict]:
    """List all models (metadata only)."""
    results = []
    if not MODELS_DIR.exists():
        return results
    for d in MODELS_DIR.iterdir():
        if not d.is_dir():
            continue
        rp = d / "record.json"
        yp = d / "model.yaml"
        if not rp.exists():
            continue
        try:
            rec = json.loads(rp.read_text())
        except (json.JSONDecodeError, ValueError):
            continue
        
        # Count layers and determine input shape
        layer_count = 0
        input_shape = None
        if yp.exists():
            try:
                yd = yaml.safe_load(yp.read_text()) or {}
                layer_count = len(yd.get("backbone", [])) + len(yd.get("head", []))
                
                # Determine input shape based on task
                task = rec.get("task", "detect")
                if task in ["detect", "detection", "segment", "pose", "obb"]:
                    input_shape = [3, 640, 640]  # Standard YOLO input
                elif task in ["classify", "classification"]:
                    input_shape = [3, 224, 224]  # Standard classification input
                else:
                    input_shape = [3, 640, 640]  # Default
            except:
                pass
        
        results.append({
            "model_id": rec["model_id"],
            "name": rec["name"],
            "task": rec.get("task", "detect"),
            "layer_count": layer_count,
            "input_shape": input_shape,
            "params": rec.get("params"),
            "gradients": rec.get("gradients"),
            "flops": rec.get("flops"),
            "created_at": rec["created_at"],
            "updated_at": rec["updated_at"],
        })
    return results


def delete_model(model_id: str) -> bool:
    """Delete a model."""
    import shutil
    d = _dir(model_id)
    if d.exists():
        shutil.rmtree(d)
        return True
    return False


def _dict_to_yaml(yaml_def: dict) -> str:
    """Convert a model definition dict to Ultralytics-style YAML string."""
    lines: list[str] = []

    # Parameters
    if "nc" in yaml_def:
        lines.append(f"nc: {yaml_def['nc']}")

    # Write scales section when present — do NOT write depth_multiple/width_multiple
    # as they override scales in Ultralytics parse_model and break per-scale sizing
    scales = yaml_def.get("scales", {})
    if scales:
        lines.append("scales:")
        for key, vals in scales.items():
            lines.append(f"  {key}: {vals}")
    
    if "kpt_shape" in yaml_def and yaml_def["kpt_shape"]:
        lines.append(f"kpt_shape: {yaml_def['kpt_shape']}")

    lines.append("")

    # Backbone
    if "backbone" in yaml_def:
        lines.append("backbone:")
        lines.append("  # [from, repeats, module, args]")
        for i, layer in enumerate(yaml_def["backbone"]):
            line = _format_layer(layer, i)
            lines.append(f"  - {line}")

    lines.append("")

    # Head
    if "head" in yaml_def:
        lines.append("head:")
        lines.append("  # [from, repeats, module, args]")
        bb_len = len(yaml_def.get("backbone", []))
        for i, layer in enumerate(yaml_def["head"]):
            line = _format_layer(layer, bb_len + i)
            lines.append(f"  - {line}")

    return "\n".join(lines) + "\n"


def _format_layer(layer: dict | list, index: int) -> str:
    """Format a single layer as YAML list notation."""
    if isinstance(layer, list):
        # Already in [from, repeats, module, args] format
        return str(layer).replace("'", "")

    # Dict format from our schema
    from_ = layer.get("from", layer.get("from_", -1))
    repeats = layer.get("repeats", 1)
    module = layer.get("module", "Conv")
    args = layer.get("args", [])

    # Sanitize from: if list has -1 plus 2+ explicit indices, drop -1
    # e.g. [-1, 15, 18, 21] → [15, 18, 21] (Detect only takes skip inputs)
    # but keep [-1, 6] as-is (Concat takes seq + skip)
    if isinstance(from_, list):
        explicit = [f for f in from_ if f != -1]
        if len(explicit) >= 2 and -1 in from_:
            from_ = explicit
        if len(from_) == 1:
            from_ = from_[0]

    return f"[{from_}, {repeats}, {module}, {args}]".replace("'", "")
