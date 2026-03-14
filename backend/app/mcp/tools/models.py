"""
MCP tools — Model management.
Wraps model_storage and model_controller logic.
"""
from __future__ import annotations
from typing import Any

from ...services import model_storage
from ..filters import apply_view, apply_list_view, paginate
from ..serializers import ok, err, safe_dict


def list_models(
    task: str | None = None,
    view: str = "summary",
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """List all models.

    Args:
        task: Filter by task type (detect, segment, classify, pose, obb).
        view: "summary" (default) returns compact fields; "detail" returns full record.
        limit: Max items to return.
        offset: Items to skip.
    """
    try:
        records: list[dict] = model_storage.list_models()
        if task:
            records = [r for r in records if r.get("task") == task]
        records = paginate(records, limit=limit, offset=offset)
        items = apply_list_view(records, "model", view=view)
        return {"ok": True, "count": len(items), "items": items}
    except Exception as e:
        return err(str(e), "list_models_failed")


def get_model(
    model_id: str,
    view: str = "summary",
) -> dict[str, Any]:
    """Get a model by ID.

    Args:
        model_id: Target model ID.
        view: "summary" (default) or "detail" (includes yaml_def, graph metadata).
    """
    try:
        record = model_storage.load_model(model_id)
        if not record:
            return err(f"Model not found: {model_id}", "not_found")
        filtered = apply_view(record, "model", view=view)
        return ok(filtered)
    except Exception as e:
        return err(str(e), "get_model_failed")


def get_model_yaml(model_id: str) -> dict[str, Any]:
    """Get the raw YAML string for a model.

    Args:
        model_id: Target model ID.
    """
    try:
        p = model_storage.load_model_yaml_path(model_id)
        if not p:
            return err(f"Model not found: {model_id}", "not_found")
        return {"ok": True, "model_id": model_id, "yaml": p.read_text()}
    except Exception as e:
        return err(str(e), "get_model_yaml_failed")


def create_model(
    name: str,
    task: str,
    yaml_def: dict,
    description: str = "",
    replace: bool = False,
) -> dict[str, Any]:
    """Create or replace a model definition.

    Args:
        name: Human-readable model name.
        task: Task type (detect, segment, classify, pose, obb).
        yaml_def: Ultralytics-compatible YAML definition as dict.
        description: Optional description.
        replace: If True, overwrite an existing model with the same name.
    """
    try:
        if not replace:
            for m in model_storage.list_models():
                if m.get("name") == name:
                    return {"ok": True, "exists": True, "model_id": m["model_id"], "message": "Model already exists"}

        model_id = None
        if replace:
            for m in model_storage.list_models():
                if m.get("name") == name:
                    model_id = m["model_id"]
                    break

        record = model_storage.save_model(
            name=name,
            yaml_def=yaml_def,
            task=task,
            description=description,
            model_id=model_id,
        )
        return {"ok": True, "model_id": record["model_id"], "message": "Model saved"}
    except Exception as e:
        return err(str(e), "create_model_failed")


def validate_model(model_id: str, scale: str | None = None) -> dict[str, Any]:
    """Validate a model YAML with Ultralytics and return param/layer counts.

    Args:
        model_id: Target model ID.
        scale: Optional scale char (n, s, m, l, x) to validate a specific size.
    """
    try:
        yaml_path = model_storage.load_model_yaml_path(model_id)
        if not yaml_path:
            return err(f"Model not found: {model_id}", "not_found")

        record = model_storage.load_model(model_id)
        task = (record or {}).get("task", "detect")
        yaml_def = (record or {}).get("yaml_def", {})
        scales: dict = yaml_def.get("scales", {})

        if scale and scale in scales:
            selected_scale = scale
        elif scales:
            selected_scale = "m" if "m" in scales else next(iter(scales))
        else:
            selected_scale = None

        from ...utils.yaml_utils import prepare_model_yaml
        import hsg_det  # noqa: F401
        from ultralytics import YOLO
        import torch, os

        patched_yaml_path = prepare_model_yaml(yaml_path, scale=selected_scale)
        try:
            model = YOLO(patched_yaml_path, task=task)
        except Exception as e:
            return {"ok": True, "valid": False, "model_id": model_id, "message": str(e)}
        finally:
            try:
                os.remove(patched_yaml_path)
            except OSError:
                pass

        n_params = sum(p.numel() for p in model.model.parameters())
        n_layers = len(list(model.model.modules()))
        return {
            "ok": True,
            "valid": True,
            "model_id": model_id,
            "scale": selected_scale,
            "params": n_params,
            "layers": n_layers,
            "message": f"Valid — {n_params:,} params, {n_layers} layers",
        }
    except Exception as e:
        return err(str(e), "validate_model_failed")
