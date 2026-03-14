"""
MCP tools — Weight management.
Wraps weight_storage and weight-related services.
"""
from __future__ import annotations
from typing import Any

from ...services import weight_storage
from ..filters import apply_view, apply_list_view, paginate
from ..serializers import ok, err


def list_weights(
    model_id: str | None = None,
    view: str = "summary",
    limit: int | None = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List all saved weights.

    Args:
        model_id: Filter by parent model ID.
        view: "summary" (default) or "detail".
        limit: Max items to return (default 50).
        offset: Items to skip.
    """
    try:
        records = weight_storage.list_weights(model_id=model_id)
        for r in records:
            if not r.get("dataset_name"):
                r["dataset_name"] = weight_storage.resolve_dataset_name(r)
        records = paginate(records, limit=limit, offset=offset)
        items = apply_list_view(records, "weight", view=view)
        return {"ok": True, "count": len(items), "items": items}
    except Exception as e:
        return err(str(e), "list_weights_failed")


def get_weight(weight_id: str, view: str = "summary") -> dict[str, Any]:
    """Get weight metadata by ID.

    Args:
        weight_id: Target weight ID.
        view: "summary" (default) or "detail" (includes training_runs, edits).
    """
    try:
        record = weight_storage.load_weight_meta(weight_id)
        if not record:
            return err(f"Weight not found: {weight_id}", "not_found")
        if not record.get("dataset_name"):
            record["dataset_name"] = weight_storage.resolve_dataset_name(record)
        filtered = apply_view(record, "weight", view=view)
        return ok(filtered)
    except Exception as e:
        return err(str(e), "get_weight_failed")


def get_weight_info(weight_id: str) -> dict[str, Any]:
    """Get model param count and GFLOPs from a weight .pt file.

    Args:
        weight_id: Target weight ID.
    """
    try:
        pt_path = weight_storage.weight_pt_path(weight_id)
        if not pt_path.exists():
            return err(f"Weight file not found: {weight_id}", "not_found")

        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import get_flops
        yolo = YOLO(str(pt_path))
        params = sum(p.numel() for p in yolo.model.parameters())
        try:
            gflops = get_flops(yolo.model, imgsz=640)
        except Exception:
            gflops = None
        return {
            "ok": True,
            "weight_id": weight_id,
            "params": params,
            "gflops": round(gflops, 2) if gflops is not None else None,
        }
    except Exception as e:
        return err(str(e), "get_weight_info_failed")


def get_weight_lineage(weight_id: str, view: str = "summary") -> dict[str, Any]:
    """Get the training lineage chain for a weight (oldest ancestor first).

    Args:
        weight_id: Target weight ID.
        view: "summary" (default) or "detail".
    """
    try:
        record = weight_storage.load_weight_meta(weight_id)
        if not record:
            return err(f"Weight not found: {weight_id}", "not_found")
        chain = weight_storage.get_lineage(weight_id)
        if view == "summary":
            chain = apply_list_view(chain, "weight", view="summary")
        return {"ok": True, "weight_id": weight_id, "depth": len(chain), "lineage": chain}
    except Exception as e:
        return err(str(e), "get_weight_lineage_failed")


def create_empty_weight(
    model_id: str | None = None,
    yolo_model: str | None = None,
    use_pretrained: bool = False,
    model_scale: str | None = None,
    name: str = "",
) -> dict[str, Any]:
    """Create an empty weight from a model architecture or official YOLO checkpoint.

    Args:
        model_id: Custom model ID to use. Mutually exclusive with yolo_model.
        yolo_model: Official YOLO model key (e.g. 'yolov8n', 'yolov8s').
        use_pretrained: If True, loads COCO-pretrained weights; False gives random init.
        model_scale: Optional scale char (n, s, m, l, x) for custom models.
        name: Optional display name for the weight.
    """
    try:
        import torch
        from ultralytics import YOLO
        from ...services import model_storage as _ms

        if yolo_model:
            yolo_key = yolo_model.strip()
            if use_pretrained:
                model = YOLO(f"{yolo_key}.pt")
            else:
                model = YOLO(f"{yolo_key}.yaml")
            sd = model.model.state_dict()
            pretrained_label = "pretrained-coco" if use_pretrained else "random-init"
            display_name = name.strip() if name.strip() else f"{yolo_key}-{pretrained_label}"
            model_id_meta = f"yolo:{yolo_key}"
            weight_id = weight_storage.save_weight_meta(
                model_id=model_id_meta,
                model_name=display_name,
                job_id=None,
                dataset="COCO (pretrained)" if use_pretrained else "(empty)",
                epochs_trained=0,
                final_accuracy=None,
                final_loss=None,
            )
            pt_path = weight_storage.weight_pt_path(weight_id)
            ckpt = {
                "model": model.model,
                "epoch": -1,
                "optimizer": None,
                "train_args": {"model": f"{yolo_key}.pt" if use_pretrained else f"{yolo_key}.yaml"},
                "date": None,
                "version": None,
            }
            torch.save(ckpt, str(pt_path))
            meta = weight_storage.load_weight_meta(weight_id)
            if meta:
                meta["file_size_bytes"] = pt_path.stat().st_size
                meta["key_count"] = len(sd)
                meta["param_count"] = sum(v.numel() for v in sd.values())
                weight_storage._store.save(weight_id, meta)
            return {
                "ok": True,
                "weight_id": weight_id,
                "model_id": model_id_meta,
                "model_name": display_name,
                "key_count": len(sd),
                "file_size_bytes": pt_path.stat().st_size,
            }

        if not model_id:
            return err("Either model_id or yolo_model must be provided", "invalid_args")

        record = _ms.load_model(model_id)
        if not record:
            return err(f"Model not found: {model_id}", "not_found")

        yaml_path = _ms.load_model_yaml_path(model_id)
        if not yaml_path:
            return err("No YAML file for this model", "missing_yaml")

        from ...plugins.loader import find_arch_for_yaml
        from ...utils.yaml_utils import prepare_model_yaml

        arch_plugin = find_arch_for_yaml(str(yaml_path))
        if arch_plugin:
            arch_plugin.register_modules()

        effective_scale = model_scale or record.get("scale")
        patched_yaml = prepare_model_yaml(yaml_path, scale=effective_scale)
        model = YOLO(str(patched_yaml), task=record.get("task", "detect"))
        sd = model.model.state_dict()
        from pathlib import Path
        Path(patched_yaml).unlink(missing_ok=True)

        display_name = name.strip() if name.strip() else record.get("name", "Untitled")
        weight_id = weight_storage.save_weight_meta(
            model_id=model_id,
            model_name=display_name,
            job_id=None,
            dataset="(empty)",
            epochs_trained=0,
            final_accuracy=None,
            final_loss=None,
        )
        pt_path = weight_storage.weight_pt_path(weight_id)
        ckpt = {
            "model": model.model,
            "epoch": -1,
            "optimizer": None,
            "train_args": {},
            "date": None,
            "version": None,
        }
        torch.save(ckpt, str(pt_path))
        meta = weight_storage.load_weight_meta(weight_id)
        if meta:
            meta["file_size_bytes"] = pt_path.stat().st_size
            meta["key_count"] = len(sd)
            meta["param_count"] = sum(v.numel() for v in sd.values())
            weight_storage._store.save(weight_id, meta)
        return {
            "ok": True,
            "weight_id": weight_id,
            "model_id": model_id,
            "model_name": display_name,
            "key_count": len(sd),
            "file_size_bytes": pt_path.stat().st_size,
        }
    except Exception as e:
        return err(str(e), "create_empty_weight_failed")


def list_pretrained_weights(
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """List available pretrained models from all weight source plugins.

    Args:
        limit: Max items to return.
        offset: Items to skip.
    """
    try:
        from ...plugins.loader import all_weight_source_plugins
        result = []
        for plugin in all_weight_source_plugins():
            if plugin.has_pretrained_catalog:
                result.extend(plugin.list_pretrained())
        result = paginate(result, limit=limit, offset=offset)
        return {"ok": True, "count": len(result), "items": result}
    except Exception as e:
        return err(str(e), "list_pretrained_failed")


def download_pretrained_weight(model_key: str) -> dict[str, Any]:
    """Download a pretrained model by key and save it as a weight record.

    Args:
        model_key: Pretrained model key (e.g. 'yolov8n', 'yolov8s-seg').
    """
    try:
        from ...plugins.loader import all_weight_source_plugins
        for plugin in all_weight_source_plugins():
            if not plugin.has_pretrained_catalog:
                continue
            catalog_keys = {e["model_key"] for e in plugin.list_pretrained()}
            if model_key in catalog_keys:
                result = plugin.download_pretrained(model_key)
                return {"ok": True, **result}
        return err(f"Pretrained model not found: {model_key}", "not_found")
    except Exception as e:
        return err(str(e), "download_pretrained_failed")
