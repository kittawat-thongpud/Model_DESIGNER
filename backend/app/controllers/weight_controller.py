"""
Weight Controller — API endpoints for trained weight management.
Tagged as "Weights" in ReDoc.
"""
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from ..schemas.weight_schema import (
    ExtractRequest, TransferRequest, AutoMapRequest,
    ApplyMapRequest, CreateEmptyRequest,
)
from ..services import weight_storage, weight_transfer, weight_import, model_storage
from ..plugins.loader import all_weight_source_plugins
from .. import logging_service as logger

router = APIRouter(prefix="/api/weights", tags=["Weights"])


@router.post("/create-empty", summary="Create an empty weight from a model")
async def create_empty_weight(body: CreateEmptyRequest):
    """Generate a weight file from a model architecture or official YOLO checkpoint.

    - If ``yolo_model`` is set (e.g. 'yolov8n') uses the official Ultralytics model.
      ``use_pretrained=True`` loads COCO-pretrained weights; False gives random init.
    - Otherwise uses the custom model YAML identified by ``model_id``.
    """
    import torch
    from ultralytics import YOLO

    # ── Branch A: Official YOLO model ────────────────────────────────────────
    if body.yolo_model:
        yolo_key = body.yolo_model.strip()   # e.g. "yolov8n", "yolov8s"
        try:
            if body.use_pretrained:
                # Downloads pretrained weights from Ultralytics hub if not cached
                model = YOLO(f"{yolo_key}.pt")
            else:
                # Load architecture only (random init) via YAML
                model = YOLO(f"{yolo_key}.yaml")
            sd = model.model.state_dict()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load YOLO model '{yolo_key}': {e}")

        pretrained_label = "pretrained-coco" if body.use_pretrained else "random-init"
        display_name = body.name.strip() if body.name.strip() else f"{yolo_key}-{pretrained_label}"
        model_id_meta = f"yolo:{yolo_key}"
        weight_id = weight_storage.save_weight_meta(
            model_id=model_id_meta,
            model_name=display_name,
            job_id=None,
            dataset="COCO (pretrained)" if body.use_pretrained else "(empty)",
            epochs_trained=0,
            final_accuracy=None,
            final_loss=None,
        )
        pt_path = weight_storage.weight_pt_path(weight_id)
        ckpt = {
            "model": model.model,
            "epoch": -1,
            "optimizer": None,
            "train_args": {"model": f"{yolo_key}.pt" if body.use_pretrained else f"{yolo_key}.yaml"},
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

        logger.log("system", "INFO", "Official YOLO weight created", {
            "weight_id": weight_id, "yolo_model": yolo_key,
            "pretrained": body.use_pretrained, "keys": len(sd),
        })
        return {
            "weight_id": weight_id,
            "model_id": model_id_meta,
            "model_name": display_name,
            "key_count": len(sd),
            "file_size_bytes": pt_path.stat().st_size,
        }

    # ── Branch B: Custom model YAML ───────────────────────────────────────────
    if not body.model_id:
        raise HTTPException(status_code=400, detail="Either model_id or yolo_model must be provided")

    record = model_storage.load_model(body.model_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Model '{body.model_id}' not found")

    yaml_path = model_storage.load_model_yaml_path(body.model_id)
    if not yaml_path:
        raise HTTPException(status_code=400, detail="No YAML file for this model")

    try:
        from ..plugins.loader import find_arch_for_yaml
        from ..utils.yaml_utils import prepare_model_yaml

        arch_plugin = find_arch_for_yaml(str(yaml_path))
        if arch_plugin:
            arch_plugin.register_modules()

        effective_scale = body.model_scale or record.get("scale")
        patched_yaml = prepare_model_yaml(yaml_path, scale=effective_scale)
        model = YOLO(str(patched_yaml), task=record.get("task", "detect"))
        sd = model.model.state_dict()
        Path(patched_yaml).unlink(missing_ok=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model instantiation failed: {e}")

    display_name = body.name.strip() if body.name.strip() else record.get("name", "Untitled")
    weight_id = weight_storage.save_weight_meta(
        model_id=body.model_id,
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

    logger.log("system", "INFO", "Empty weight created", {
        "weight_id": weight_id, "model_id": body.model_id, "keys": len(sd),
    })
    return {
        "weight_id": weight_id,
        "model_id": body.model_id,
        "model_name": display_name,
        "key_count": len(sd),
        "file_size_bytes": pt_path.stat().st_size,
    }


@router.get("/", summary="List all saved weights")
async def list_weights(model_id: str | None = None):
    """Return all weight metadata, optionally filtered by parent model."""
    records = weight_storage.list_weights(model_id=model_id)
    for r in records:
        if not r.get("dataset_name"):
            r["dataset_name"] = weight_storage.resolve_dataset_name(r)
    return records


# ── Weight Source Plugins (must be before /{weight_id} to avoid shadowing) ──

@router.get("/sources", summary="List available weight source plugins")
async def list_sources():
    """Return all registered weight-source plugins (Ultralytics, torchvision, etc.)."""
    return [p.to_dict() for p in all_weight_source_plugins()]


@router.get("/pretrained", summary="List available pretrained models")
async def list_pretrained():
    """Aggregate pretrained model catalogs from all weight source plugins."""
    result = []
    for plugin in all_weight_source_plugins():
        if plugin.has_pretrained_catalog:
            result.extend(plugin.list_pretrained())
    return result


@router.post("/pretrained/download", summary="Download a pretrained model")
async def download_pretrained(body: dict):
    """Download a pretrained model by key. Returns the imported weight record.

    The actual download runs in a thread pool so the event loop stays responsive.
    """
    import asyncio

    model_key = body.get("model_key", "")
    if not model_key:
        raise HTTPException(status_code=400, detail="model_key is required")

    # Find the plugin that owns this model
    for plugin in all_weight_source_plugins():
        if not plugin.has_pretrained_catalog:
            continue
        catalog_keys = {e["model_key"] for e in plugin.list_pretrained()}
        if model_key in catalog_keys:
            try:
                # Run blocking download in thread pool to keep server responsive
                result = await asyncio.to_thread(plugin.download_pretrained, model_key)
                logger.log("system", "INFO", f"Pretrained model downloaded: {model_key}", result)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=404, detail=f"Pretrained model '{model_key}' not found in any plugin")


@router.get("/{weight_id}", summary="Get weight details")
async def get_weight(weight_id: str):
    """Return full weight metadata including parent model and job info."""
    record = weight_storage.load_weight_meta(weight_id)
    if record and not record.get("dataset_name"):
        record["dataset_name"] = weight_storage.resolve_dataset_name(record)
    if not record:
        raise HTTPException(status_code=404, detail=f"Weight '{weight_id}' not found")
    return record


@router.get("/{weight_id}/info", summary="Get model params and GFLOPs from weight file")
async def get_weight_info(weight_id: str):
    """Load the .pt file via Ultralytics YOLO and return parameter count + GFLOPs."""
    pt_path = weight_storage.weight_pt_path(weight_id)
    if not pt_path.exists():
        raise HTTPException(status_code=404, detail=f"Weight file not found: {weight_id}")
    try:
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import get_flops
        yolo = YOLO(str(pt_path))
        params = sum(p.numel() for p in yolo.model.parameters())
        try:
            gflops = get_flops(yolo.model, imgsz=640)
        except Exception:
            gflops = None
        return {"params": params, "gflops": round(gflops, 2) if gflops is not None else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{weight_id}/download", summary="Download weight .pt file")
async def download_weight(weight_id: str):
    """Stream the weight.pt file as a file download."""
    from fastapi.responses import FileResponse
    pt_path = weight_storage.weight_pt_path(weight_id)
    if not pt_path.exists():
        raise HTTPException(status_code=404, detail=f"Weight file not found: {weight_id}")
    meta = weight_storage.load_weight_meta(weight_id)
    model_name = (meta or {}).get("model_name", "weight")
    filename = f"{model_name}_{weight_id[:8]}.pt".replace(" ", "_")
    return FileResponse(
        path=str(pt_path),
        media_type="application/octet-stream",
        filename=filename,
    )


@router.get("/{weight_id}/lineage", summary="Get weight lineage chain")
async def get_lineage(weight_id: str):
    """Walk the parent_weight_id chain and return from oldest ancestor to current."""
    record = weight_storage.load_weight_meta(weight_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Weight '{weight_id}' not found")
    return weight_storage.get_lineage(weight_id)


@router.delete("/{weight_id}", summary="Delete a weight")
async def delete_weight(weight_id: str):
    """Delete weight .pt file and its metadata."""
    if weight_storage.delete_weight(weight_id):
        logger.log("system", "INFO", f"Weight deleted", {"weight_id": weight_id})
        return {"message": f"Weight '{weight_id}' deleted"}
    raise HTTPException(status_code=404, detail=f"Weight '{weight_id}' not found")


@router.patch("/{weight_id}/rename", summary="Rename a weight")
async def rename_weight(weight_id: str, body: dict):
    """Update model_name for a weight record."""
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    meta = weight_storage.load_weight_meta(weight_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Weight '{weight_id}' not found")
    meta["model_name"] = name
    weight_storage._store.save(weight_id, meta)
    logger.log("system", "INFO", "Weight renamed", {"weight_id": weight_id, "name": name})
    return meta


# ── Partial weight operations ──────────────────────────────────────────────

@router.get("/{weight_id}/keys", summary="Inspect weight state_dict keys")
async def inspect_keys(weight_id: str):
    """Return all state_dict keys grouped by node ID with shapes."""
    try:
        return weight_transfer.inspect_weight_keys(weight_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Weight '{weight_id}' not found")


@router.post("/{weight_id}/extract", summary="Extract partial weights by node IDs")
async def extract_partial(weight_id: str, body: ExtractRequest):
    """Create a new weight file containing only the specified nodes' parameters."""
    try:
        new_id, count = weight_transfer.extract_partial(weight_id, body.node_ids)
        logger.log("system", "INFO", "Partial weight extracted", {
            "source": weight_id, "new_id": new_id, "keys": count,
        })
        return {"weight_id": new_id, "keys_extracted": count}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Weight '{weight_id}' not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{weight_id}/transfer", summary="Transfer weights from another source")
async def transfer_weights(weight_id: str, body: TransferRequest):
    """Copy matching parameters from source weight into this weight (in-place).

    Matches by key name and shape. Optionally remap node IDs via node_id_map.
    """
    try:
        matched, total, keys = weight_transfer.transfer_weights(
            source_weight_id=body.source_weight_id,
            target_weight_id=weight_id,
            node_id_map=body.node_id_map,
        )
        logger.log("system", "INFO", "Weight transfer complete", {
            "source": body.source_weight_id, "target": weight_id,
            "matched": matched, "total": total,
        })
        return {
            "matched_keys": matched,
            "total_target_keys": total,
            "match_ratio": round(matched / total, 3) if total else 0,
            "keys": keys,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── External Weight Import ─────────────────────────────────────────────────

@router.post("/import", summary="Import an external weight file")
async def import_weight(
    file: UploadFile = File(...),
    name: str = Form("imported_weight"),
):
    """Upload a .pt/.pth file, auto-detect format via plugins, save as system weight."""
    # Save upload to a temp file
    suffix = Path(file.filename or "upload.pt").suffix or ".pt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(file.file, tmp)

    try:
        result = weight_import.import_external_weight(tmp_path, name)
        logger.log("system", "INFO", "External weight imported", {
            "weight_id": result["weight_id"],
            "source_plugin": result["source_plugin"],
            "key_count": result["key_count"],
        })
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Layer Groups ───────────────────────────────────────────────────────────

@router.get("/{weight_id}/groups", summary="Get layer groups for a weight")
async def get_groups(weight_id: str):
    """Return layer groups (for the visual mapping canvas)."""
    try:
        return weight_import.get_weight_groups(weight_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Weight '{weight_id}' not found")


@router.get("/{weight_id}/groups-annotated", summary="Get annotated layer groups")
async def get_groups_annotated(weight_id: str, model_id: str | None = None):
    """Return layer groups annotated with node labels from the model graph.

    If ``model_id`` is provided, loads the model graph and resolves
    ``node.label`` for each group prefix that matches a node ID.
    SubModel nodes are expanded into nested sub-groups with a ``children``
    array so the frontend can render them as collapsible groups.
    """
    try:
        groups = weight_import.get_weight_groups(weight_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Weight '{weight_id}' not found")

    # Build node label map + SubModel info from model graph
    label_map: dict[str, str] = {}
    submodel_ids: set[str] = set()
    # Map SubModel node_id → {child_node_id: {"type": ..., "label": ...}}
    submodel_child_info: dict[str, dict[str, dict]] = {}
    if model_id:
        try:
            record = model_storage.load_model(model_id)
            if record:
                yaml_def = record.get("yaml_def", {})
                # Build label map from YAML layers
                for section in ("backbone", "head"):
                    for i, layer in enumerate(yaml_def.get(section, [])):
                        if isinstance(layer, dict):
                            module = layer.get("module", "")
                            label_map[f"model.{i}"] = module
        except Exception:
            pass  # model not found or corrupt — proceed without labels

    # Annotate groups + expand SubModel groups into nested children
    annotated = []
    for g in groups:
        entry = dict(g) if isinstance(g, dict) else g
        prefix = entry.get("prefix", "")
        node_label = label_map.get(prefix, "")
        entry["node_label"] = node_label
        entry["is_submodel"] = prefix in submodel_ids

        # For SubModel groups, split keys into sub-groups by second-level prefix
        if prefix in submodel_ids and entry.get("keys"):
            from collections import OrderedDict
            sub_groups: dict[str, list[dict]] = OrderedDict()
            for k_info in entry["keys"]:
                key: str = k_info["key"]
                # key format: "node_X.sub_layer.weight" → sub prefix = "sub_layer"
                parts = key.split(".")
                if len(parts) >= 3:
                    sub_prefix = parts[1]
                else:
                    sub_prefix = parts[-1] if len(parts) > 1 else key
                sub_groups.setdefault(sub_prefix, []).append(k_info)

            child_info = submodel_child_info.get(prefix, {})
            children = []
            for sub_prefix, sub_keys in sub_groups.items():
                info = child_info.get(sub_prefix, {})
                children.append({
                    "prefix": f"{prefix}.{sub_prefix}",
                    "display_prefix": info.get("label") or sub_prefix,
                    "module_type": info.get("type", "unknown"),
                    "param_count": len(sub_keys),
                    "keys": sub_keys,
                })
            entry["children"] = children
        else:
            entry["children"] = []

        annotated.append(entry)

    return annotated


# ── Auto-Map ──────────────────────────────────────────────────────────────

@router.post("/{weight_id}/auto-map", summary="Auto-map source weight to target")
async def auto_map_weights(weight_id: str, body: AutoMapRequest):
    """Generate a mapping preview between source and target weights.

    Uses shape + suffix matching to find the best layer-to-layer mapping.
    Does NOT modify any weight files.
    """
    try:
        mapping = weight_transfer.auto_map(
            source_weight_id=body.source_weight_id,
            target_weight_id=weight_id,
        )
        matched = sum(1 for m in mapping if m["status"] == "matched")
        total = sum(1 for m in mapping if m["tgt_prefix"] is not None)
        return {
            "mapping": mapping,
            "matched_groups": matched,
            "total_target_groups": total,
            "match_ratio": round(matched / total, 3) if total else 0,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Apply Mapping ─────────────────────────────────────────────────────────

@router.post("/{weight_id}/apply-map", summary="Apply a confirmed weight mapping")
async def apply_map(weight_id: str, body: ApplyMapRequest):
    """Apply a user-confirmed mapping from source to target weight (in-place).

    Optionally returns freeze_node_ids for the trainer to freeze after transfer.
    """
    try:
        result = weight_transfer.apply_mapping(
            source_weight_id=body.source_weight_id,
            target_weight_id=weight_id,
            mapping=body.mapping,
        )
        result["freeze_node_ids"] = body.freeze_node_ids

        # Record edit in lineage
        weight_storage.save_edit_meta(
            weight_id=weight_id,
            edit_type="transfer",
            source_weight_id=body.source_weight_id,
            mapping_count=result.get("applied", 0),
            frozen_nodes=body.freeze_node_ids,
        )

        logger.log("system", "INFO", "Weight mapping applied", {
            "source": body.source_weight_id,
            "target": weight_id,
            "applied": result["applied"],
        })
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Compatibility Check ──────────────────────────────────────────────────

class CompatCheckRequest(BaseModel):
    source_weight_id: str
    src_prefix: str = Field(..., description="Source group prefix")
    tgt_prefix: str = Field(..., description="Target group prefix")


@router.post("/{weight_id}/compat-check", summary="Check compatibility between two weight groups")
async def compat_check(weight_id: str, body: CompatCheckRequest):
    """Compare source and target weight groups for shape/dtype compatibility.

    Returns per-key comparison with mismatch details and a severity summary
    explaining what would happen if the transfer is forced.
    """
    import torch

    src_path = weight_storage.weight_pt_path(body.source_weight_id)
    tgt_path = weight_storage.weight_pt_path(weight_id)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"Source weight not found: {body.source_weight_id}")
    if not tgt_path.exists():
        raise HTTPException(status_code=404, detail=f"Target weight not found: {weight_id}")

    src_sd: dict = torch.load(src_path, map_location="cpu", weights_only=True)
    tgt_sd: dict = torch.load(tgt_path, map_location="cpu", weights_only=True)

    from ..services.weight_transfer import _match_keys_smart

    # Filter to the requested prefixes
    src_keys = {k: v for k, v in src_sd.items() if k.startswith(f"{body.src_prefix}.")}
    tgt_keys = {k: v for k, v in tgt_sd.items() if k.startswith(f"{body.tgt_prefix}.")}

    # Use smart multi-level matching (exact suffix → leaf suffix → index+shape)
    smart_matches = _match_keys_smart(src_keys, tgt_keys, body.src_prefix, body.tgt_prefix)

    # Build a lookup for quick access
    matched_src_keys: set[str] = set()
    matched_tgt_keys: set[str] = set()
    pair_map: dict[str, dict] = {}  # tgt_key → match entry
    for m in smart_matches:
        if m["src_key"] and m["tgt_key"]:
            matched_src_keys.add(m["src_key"])
            matched_tgt_keys.add(m["tgt_key"])
            pair_map[m["tgt_key"]] = m

    def _suffix(key: str, prefix: str) -> str:
        return key[len(prefix) + 1:] if key.startswith(prefix + ".") else key

    issues: list[dict] = []
    keys_detail: list[dict] = []

    # Process matched pairs
    for m in smart_matches:
        sk, tk = m.get("src_key"), m.get("tgt_key")

        if sk and tk:
            sv, tv = src_sd[sk], tgt_sd[tk]
            src_shape = list(sv.shape)
            tgt_shape = list(tv.shape)
            src_dtype = str(sv.dtype)
            tgt_dtype = str(tv.dtype)
            suf = _suffix(tk, body.tgt_prefix)

            entry: dict = {
                "suffix": suf,
                "src_key": sk, "tgt_key": tk,
                "src_shape": src_shape, "tgt_shape": tgt_shape,
                "src_dtype": src_dtype, "tgt_dtype": tgt_dtype,
                "src_numel": sv.numel(), "tgt_numel": tv.numel(),
                "shape_match": src_shape == tgt_shape,
                "dtype_match": src_dtype == tgt_dtype,
                "status": "ok",
            }

            if src_shape != tgt_shape:
                if len(src_shape) != len(tgt_shape):
                    severity = "error"
                    msg = f"Dimension mismatch: source is {len(src_shape)}D, target is {len(tgt_shape)}D. Transfer will fail — tensors are structurally incompatible."
                elif sv.numel() != tv.numel():
                    severity = "error"
                    msg = f"Shape mismatch: {src_shape} → {tgt_shape} (element count differs: {sv.numel()} vs {tv.numel()}). Transfer will fail — cannot reshape without data loss."
                else:
                    severity = "warning"
                    msg = f"Shape mismatch but same element count: {src_shape} → {tgt_shape}. Transfer possible with reshape, but may produce incorrect results if dimensions have different semantics."
                entry["status"] = severity
                entry["message"] = msg
                issues.append({"suffix": suf, "severity": severity, "message": msg})
            elif src_dtype != tgt_dtype:
                entry["status"] = "warning"
                entry["message"] = f"Dtype mismatch: {src_dtype} → {tgt_dtype}. Values will be auto-cast, which may cause precision loss."
                issues.append({"suffix": suf, "severity": "warning", "message": entry["message"]})

            keys_detail.append(entry)

        elif sk and not tk:
            sv = src_sd[sk]
            suf = _suffix(sk, body.src_prefix)
            entry = {
                "suffix": suf,
                "src_key": sk, "tgt_key": None,
                "src_shape": list(sv.shape), "tgt_shape": None,
                "src_dtype": str(sv.dtype), "tgt_dtype": None,
                "shape_match": False, "dtype_match": False,
                "status": "extra_source",
                "message": "Source has this parameter but target does not. It will be ignored.",
            }
            issues.append({"suffix": suf, "severity": "info", "message": entry["message"]})
            keys_detail.append(entry)

        elif tk and not sk:
            tv = tgt_sd[tk]
            suf = _suffix(tk, body.tgt_prefix)
            entry = {
                "suffix": suf,
                "src_key": None, "tgt_key": tk,
                "src_shape": None, "tgt_shape": list(tv.shape),
                "src_dtype": None, "tgt_dtype": str(tv.dtype),
                "shape_match": False, "dtype_match": False,
                "status": "missing_source",
                "message": "Target has this parameter but source does not. It will keep its current values (random or pretrained).",
            }
            issues.append({"suffix": suf, "severity": "warning", "message": entry["message"]})
            keys_detail.append(entry)

    # Summary
    n_ok = sum(1 for d in keys_detail if d["status"] == "ok")
    n_error = sum(1 for i in issues if i["severity"] == "error")
    n_warning = sum(1 for i in issues if i["severity"] == "warning")
    n_info = sum(1 for i in issues if i["severity"] == "info")

    if n_error > 0:
        overall = "incompatible"
        summary = f"{n_error} parameter(s) have incompatible shapes and cannot be transferred. The rest ({n_ok} ok) can be transferred partially."
    elif n_warning > 0:
        overall = "partial"
        summary = f"All shapes transferable, but {n_warning} issue(s) detected (dtype mismatch or missing parameters). Transfer will work but may need attention."
    else:
        overall = "compatible"
        summary = f"All {n_ok} parameters are fully compatible. Transfer is safe."

    return {
        "overall": overall,
        "summary": summary,
        "ok_count": n_ok,
        "error_count": n_error,
        "warning_count": n_warning,
        "info_count": n_info,
        "total_keys": len(keys_detail),
        "issues": issues,
        "keys": keys_detail,
    }


# ── Layer Detail (histogram, stats, heatmap) ─────────────────────────────

@router.get("/{weight_id}/layer-detail", summary="Get detailed stats for a weight parameter")
async def layer_detail(weight_id: str, key: str, bins: int = 50):
    """Return histogram, statistics, and optional 2D heatmap data for a
    specific weight tensor identified by its state_dict key.

    Parameters
    ----------
    weight_id : str
        The weight ID to inspect.
    key : str
        The state_dict key (e.g. ``node_2.weight``).
    bins : int
        Number of histogram bins (default 50).
    """
    import torch
    import math

    pt_path = weight_storage.weight_pt_path(weight_id)
    if not pt_path.exists():
        raise HTTPException(status_code=404, detail=f"Weight not found: {weight_id}")

    sd: dict = torch.load(pt_path, map_location="cpu", weights_only=True)
    if key not in sd:
        raise HTTPException(status_code=404, detail=f"Key '{key}' not found in weight {weight_id}")

    raw_tensor = sd[key]
    tensor = raw_tensor.float()
    flat = tensor.flatten()
    n = int(tensor.numel())

    def _safe(v: float) -> float:
        """Replace NaN / Inf with 0 for JSON safety."""
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v

    # Basic stats (handle 0-dim scalars and single-element tensors)
    if n == 0:
        stats = {
            "shape": list(tensor.shape), "dtype": str(raw_tensor.dtype),
            "numel": 0, "min": 0, "max": 0, "mean": 0, "std": 0,
            "median": 0, "zeros_pct": 0, "near_zero_pct": 0,
        }
    elif n == 1:
        val = float(flat[0])
        stats = {
            "shape": list(tensor.shape), "dtype": str(raw_tensor.dtype),
            "numel": 1, "min": _safe(val), "max": _safe(val),
            "mean": _safe(val), "std": 0.0, "median": _safe(val),
            "zeros_pct": 100.0 if val == 0 else 0.0,
            "near_zero_pct": 100.0 if abs(val) < 1e-6 else 0.0,
        }
    else:
        stats = {
            "shape": list(tensor.shape),
            "dtype": str(raw_tensor.dtype),
            "numel": n,
            "min": _safe(float(flat.min())),
            "max": _safe(float(flat.max())),
            "mean": _safe(float(flat.mean())),
            "std": _safe(float(flat.std())),
            "median": _safe(float(flat.median())),
            "zeros_pct": _safe(float((flat == 0).float().mean() * 100)),
            "near_zero_pct": _safe(float((flat.abs() < 1e-6).float().mean() * 100)),
        }

    # Histogram
    if n >= 2:
        hist_counts = torch.histc(flat, bins=bins)
        bin_min = _safe(float(flat.min()))
        bin_max = _safe(float(flat.max()))
        bin_width = (bin_max - bin_min) / bins if bins > 0 and bin_max != bin_min else 1.0
        histogram = {
            "counts": [int(c) for c in hist_counts.tolist()],
            "bin_edges": [round(bin_min + i * bin_width, 6) for i in range(bins + 1)],
            "bin_min": bin_min,
            "bin_max": bin_max,
        }
    else:
        histogram = {"counts": [], "bin_edges": [], "bin_min": 0, "bin_max": 0}

    # ── Tensor Map (unified, shape-aware) ──
    # Builds a list of 2D slices appropriate to the tensor's actual dimensionality.
    # 0D → scalar (no slices)
    # 1D [N] → single slice, 1 row × N cols (horizontal strip)
    # 2D [H,W] → single slice
    # 3D [C,H,W] → C slices, each [H,W]
    # 4D [out,in,kH,kW] → out slices, each averaged across in → [kH,kW]
    def _slice_to_dict(label: str, mat):
        return {
            "label": label,
            "values": [[round(float(v), 6) for v in row] for row in mat.tolist()],
            "rows": int(mat.shape[0]),
            "cols": int(mat.shape[1]),
        }

    tm_slices: list[dict] = []
    tm_description = ""

    if tensor.ndim == 0:
        tm_description = "Scalar value"
    elif tensor.ndim == 1:
        # 1D → horizontal strip: 1 × N
        tm_slices = [_slice_to_dict("values", tensor.unsqueeze(0))]
        tm_description = f"1D tensor [{tensor.shape[0]}]"
    elif tensor.ndim == 2:
        h, w = tensor.shape
        tm_slices = [_slice_to_dict(f"[{h}×{w}]", tensor)]
        tm_description = f"2D tensor [{h}×{w}]"
    elif tensor.ndim == 3:
        C, H, W = tensor.shape
        for c in range(C):
            tm_slices.append(_slice_to_dict(f"ch_{c}", tensor[c]))
        tm_description = f"3D tensor [{C} channels × {H}×{W}]"
    elif tensor.ndim == 4:
        out_ch, in_ch, kH, kW = tensor.shape
        for oc in range(out_ch):
            avg = tensor[oc].mean(dim=0)  # average across in_channels → [kH, kW]
            tm_slices.append(_slice_to_dict(f"out_{oc}", avg))
        tm_description = f"4D conv [{out_ch} out × {in_ch} in × {kH}×{kW}]"
    else:
        # 5D+ → flatten leading dims, show first few slices
        reshaped = tensor.reshape(-1, *tensor.shape[-2:])
        limit = min(reshaped.shape[0], 64)
        for i in range(limit):
            tm_slices.append(_slice_to_dict(f"slice_{i}", reshaped[i]))
        tm_description = f"{tensor.ndim}D tensor [{' × '.join(str(s) for s in tensor.shape)}], showing {limit} slices"

    tensor_map = {
        "ndim": int(tensor.ndim),
        "shape": list(tensor.shape),
        "description": tm_description,
        "slices": tm_slices,
    }

    return {
        "key": key,
        "weight_id": weight_id,
        "stats": stats,
        "histogram": histogram,
        "tensor_map": tensor_map,
    }


# ── Export ─────────────────────────────────────────────────────────────────

class ExportWeightRequest(BaseModel):
    format: str = "onnx"            # onnx | torchscript | engine | tflite | coreml
    imgsz: int = 640
    device: str = ""                # "" = auto
    half: bool = False
    simplify: bool = True           # ONNX simplify


@router.post("/{weight_id}/export", summary="Export weight to ONNX / TorchScript / etc.")
async def export_weight(weight_id: str, body: ExportWeightRequest):
    """Export a weight .pt file using Ultralytics model.export()."""
    import asyncio

    meta = weight_storage.load_weight_meta(weight_id)
    if not meta:
        raise HTTPException(404, f"Weight '{weight_id}' not found")

    pt_path = weight_storage.weight_pt_path(weight_id)
    if not pt_path.exists():
        raise HTTPException(404, f"Weight file missing: {weight_id}")

    def _export():
        import torch
        from ultralytics import YOLO
        model = YOLO(str(pt_path))
        device = body.device
        if not device:
            device = "0" if torch.cuda.is_available() else "cpu"
        exported = model.export(
            format=body.format,
            imgsz=body.imgsz,
            device=device,
            half=body.half,
            simplify=body.simplify,
            verbose=False,
        )
        return str(exported)

    try:
        exported_path = await asyncio.to_thread(_export)
        logger.log("system", "INFO", "Weight exported", {
            "weight_id": weight_id, "format": body.format, "path": exported_path,
        })
        return {
            "weight_id": weight_id,
            "format": body.format,
            "exported_path": exported_path,
            "message": f"Exported as {body.format}",
        }
    except Exception as e:
        raise HTTPException(500, f"Export failed: {e}")


@router.get("/{weight_id}/export/download", summary="Download exported file")
async def download_exported(weight_id: str, fmt: str = "onnx"):
    """Stream an already-exported file for the given format."""
    from fastapi.responses import FileResponse
    pt_path = weight_storage.weight_pt_path(weight_id)
    ext_map = {
        "onnx": ".onnx", "torchscript": ".torchscript",
        "engine": ".engine", "tflite": ".tflite", "coreml": ".mlpackage",
    }
    ext = ext_map.get(fmt, f".{fmt}")
    exported = pt_path.with_suffix(ext)
    if not exported.exists():
        raise HTTPException(404, f"No exported {fmt} file found. Export it first.")
    meta = weight_storage.load_weight_meta(weight_id)
    name = (meta or {}).get("model_name", "weight")
    filename = f"{name}_{weight_id[:8]}{ext}".replace(" ", "_")
    return FileResponse(path=str(exported), media_type="application/octet-stream", filename=filename)
