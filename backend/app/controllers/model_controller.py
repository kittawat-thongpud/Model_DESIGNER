"""
Model Controller — CRUD for Ultralytics model YAML definitions.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from ..schemas.model import SaveModelRequest, ExportRequest
from ..services import model_storage
from ..services.yaml_to_graph import yaml_to_graph

router = APIRouter(prefix="/api/models", tags=["Models"])


class ImportYAMLRequest(BaseModel):
    """Request to import YAML content."""
    yaml_content: str
    name: str = "Imported Model"
    task: str = "detect"


@router.get("/", summary="List all models")
async def list_models():
    return model_storage.list_models()


@router.get("/{model_id}", summary="Load a model")
async def load_model(model_id: str):
    record = model_storage.load_model(model_id)
    if not record:
        raise HTTPException(404, f"Model not found: {model_id}")
    return record


@router.post("/", summary="Save a model")
async def save_model(req: SaveModelRequest, replace: bool = False):
    # Check for existing model with same name
    if not replace:
        for m in model_storage.list_models():
            if m["name"] == req.name:
                return {"exists": True, "model_id": m["model_id"]}

    # Find existing model_id if replacing by name
    model_id = None
    if replace:
        for m in model_storage.list_models():
            if m["name"] == req.name:
                model_id = m["model_id"]
                break

    record = model_storage.save_model(
        name=req.name,
        yaml_def=req.yaml_def,
        task=req.task,
        description=req.description,
        model_id=model_id,
    )
    return {"model_id": record["model_id"], "message": "Model saved"}


@router.delete("/{model_id}", summary="Delete a model")
async def delete_model(model_id: str):
    if not model_storage.delete_model(model_id):
        raise HTTPException(404, f"Model not found: {model_id}")
    return {"message": "Deleted"}


@router.post("/import/yaml", summary="Import YAML and convert to graph")
async def import_yaml(req: ImportYAMLRequest):
    """
    Import external YAML model definition and convert to graph format.
    
    Handles:
    - Standard detection models (YOLOv8, YOLO11)
    - Classification models (sequential head with Classify)
    - Segmentation, pose, OBB models
    """
    try:
        # Convert YAML to graph format
        result = yaml_to_graph(req.yaml_content)
        
        # Save as new model
        record = model_storage.save_model(
            name=req.name,
            yaml_def=result,
            task=req.task,
            description=f"Imported from YAML",
        )
        
        return {
            "model_id": record["model_id"],
            "name": req.name,
            "task": req.task,
            "message": "YAML imported successfully",
            "graph": result.get("_graph"),
        }
    except Exception as e:
        raise HTTPException(400, f"Import failed: {e}")


@router.post("/import/yaml/file", summary="Import YAML file")
async def import_yaml_file(file: UploadFile = File(...), name: str = "Imported Model", task: str = "detect"):
    """Import YAML file and convert to graph format."""
    try:
        content = await file.read()
        yaml_content = content.decode('utf-8')
        
        # Convert YAML to graph format
        result = yaml_to_graph(yaml_content)
        
        # Save as new model
        record = model_storage.save_model(
            name=name,
            yaml_def=result,
            task=task,
            description=f"Imported from {file.filename}",
        )
        
        return {
            "model_id": record["model_id"],
            "name": name,
            "task": task,
            "filename": file.filename,
            "message": "YAML file imported successfully",
            "graph": result.get("_graph"),
        }
    except Exception as e:
        raise HTTPException(400, f"Import failed: {e}")


@router.get("/{model_id}/yaml", summary="Get raw YAML string")
async def get_model_yaml(model_id: str):
    """Return the raw YAML string for a model."""
    p = model_storage.load_model_yaml_path(model_id)
    if not p:
        raise HTTPException(404, f"Model not found: {model_id}")
    return {"model_id": model_id, "yaml": p.read_text()}


@router.post("/{model_id}/validate", summary="Validate model YAML with Ultralytics")
async def validate_model(model_id: str, scale: str | None = None):
    """Try to parse the model YAML with YOLO and return param count + any errors.
    
    Args:
        model_id: Model ID to validate
        scale: Optional scale (n/s/m/l/x) to validate specific model size
    """
    yaml_path = model_storage.load_model_yaml_path(model_id)
    if not yaml_path:
        raise HTTPException(404, f"Model not found: {model_id}")
    record = model_storage.load_model(model_id)
    task = record.get("task", "detect") if record else "detect"
    # Pick a scale to avoid "no model scale passed" warning.
    # We write a temp YAML with only the chosen scale so YOLO doesn't warn.
    yaml_def = record.get("yaml_def", {}) if record else {}
    scales: dict = yaml_def.get("scales", {})
    
    # Use provided scale or pick default
    if scale and scale in scales:
        selected_scale = scale
    elif scales:
        selected_scale = "m" if "m" in scales else next(iter(scales))
    else:
        selected_scale = None
    try:
        import tempfile, shutil
        import hsg_det  # noqa: F401 — registers SparseGlobalBlock* into ultralytics.nn
        from ultralytics import YOLO
        import torch
        import yaml as _yaml

        # Build a patched YAML with only the chosen scale (suppresses the warning)
        # AND manually scale the args for SparseGlobalBlock* because parse_model won't do it
        # for modules not in its internal base_modules list.
        if scale:
            with open(str(yaml_path)) as f:
                yaml_data = _yaml.safe_load(f)
            
            # 1. Set explicit scale to suppress warning
            yaml_data["scales"] = {scale: scales[scale]}
            
            # 2. Manually scale custom module args (c, k) -> (scaled_c, k)
            # Logic mimics Ultralytics parse_model scaling
            depth, width, max_channels = scales[scale]
            
            def make_divisible(x, divisor):
                # Returns nearest x divisible by divisor
                import math
                return math.ceil(x / divisor) * divisor

            custom_modules = {"SparseGlobalBlock", "SparseGlobalBlockGated"}
            
            for section in ["backbone", "head"]:
                if section in yaml_data:
                    for i, layer in enumerate(yaml_data[section]):
                        # layer format: [from, repeats, module, args]
                        if len(layer) == 4:
                            m = layer[2]
                            args = layer[3]
                            if m in custom_modules and len(args) >= 1 and isinstance(args[0], (int, float)):
                                c2 = args[0]
                                # Scale c2 same way as YOLO scales c2
                                c2 = make_divisible(min(c2, max_channels) * width, 8)
                                args[0] = c2
                                yaml_data[section][i][3] = args

            # 3. Check for forward references (cycles) which cause YOLO "list index out of range"
            # Calculate total layers to map (section, i) -> global_index
            backbone_len = len(yaml_data.get("backbone", []))
            
            for section in ["backbone", "head"]:
                if section not in yaml_data: continue
                is_backbone = (section == "backbone")
                layer_list = yaml_data[section]
                
                for i, layer in enumerate(layer_list):
                    # Global index: if head, add backbone_len
                    current_idx = i if is_backbone else (backbone_len + i)
                    
                    # layer format: [from, repeats, module, args]
                    if len(layer) < 1: continue
                    f = layer[0]
                    
                    # Normalize from to list
                    if isinstance(f, int):
                        f = [f]
                    
                    if isinstance(f, list):
                        for src_idx in f:
                            if src_idx == -1: continue
                            # YOLO allows src_idx < current_idx (past layers)
                            # If src_idx >= current_idx, it's a forward reference -> Crash
        # Prepare YAML with validation and patching (shared logic)
        from ..utils.yaml_utils import prepare_model_yaml
        
        try:
            # This will raise ValueError if cycle detected
            # It also patches custom modules (channel scaling)
            patched_yaml_path = prepare_model_yaml(yaml_path, scale=selected_scale)
            
            # Load model using patched YAML
            model = YOLO(patched_yaml_path, task=task)
            
            # Clean up temp file
            import os
            try:
                os.remove(patched_yaml_path)
            except OSError:
                pass
                
        except ValueError as e:
            # Catch cycle errors or other validation issues
            return {"valid": False, "model_id": model_id, "params": 0, "layers": 0, "input_shape": None, "output_shape": None, "message": str(e)}
        except Exception as e:
            # Catch generic YOLO parsing errors
            import traceback
            traceback.print_exc()
            return {"valid": False, "model_id": model_id, "params": 0, "layers": 0, "input_shape": None, "output_shape": None, "message": str(e)}

        # Get model info (params, gradients, FLOPs)
        n_params = sum(p.numel() for p in model.model.parameters())
        n_gradients = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        n_layers = len(list(model.model.modules()))
        
        # Get FLOPs from model.info() - it prints but also returns the value
        flops = None
        try:
            import io
            import sys
            from contextlib import redirect_stdout
            
            # Capture model.info() output to extract FLOPs
            f = io.StringIO()
            with redirect_stdout(f):
                model.info(verbose=False, imgsz=640)
            
            # Parse output for GFLOPs
            output = f.getvalue()
            print(f"[DEBUG] model.info() output:\n{output}")
            
            for line in output.split('\n'):
                if 'GFLOPs' in line or 'GFLOPS' in line or 'GFLops' in line:
                    # Extract number before GFLOPs
                    import re
                    match = re.search(r'([\d.]+)\s*GFLOPs?', line, re.IGNORECASE)
                    if match:
                        flops = float(match.group(1))
                        print(f"[DEBUG] Extracted FLOPs: {flops}")
                        break
            
            # Fallback: Try thop library for detailed per-layer FLOPs (optional)
            if flops is None:
                print(f"[DEBUG] Trying thop library for detailed FLOPs calculation...")
                try:
                    from thop import profile, clever_format
                    dummy_input = torch.randn(1, 3, 640, 640)
                    flops_count, params_count = profile(model.model, inputs=(dummy_input,), verbose=False)
                    # Convert to GFLOPs
                    flops = flops_count / 1e9
                    print(f"[DEBUG] thop FLOPs: {flops:.2f} GFLOPs")
                except ImportError:
                    print(f"[DEBUG] thop library not installed (optional for detailed FLOPs)")
                except Exception as e:
                    print(f"[DEBUG] thop calculation failed: {e}")
        except Exception as e:
            print(f"[ERROR] FLOPs calculation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Get input/output shapes
        input_shape = None
        output_shape = None
        try:
            # Try to infer shapes with a dummy input
            dummy_input = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                output = model.model(dummy_input)
            input_shape = [3, 640, 640]
            if isinstance(output, (list, tuple)):
                output_shape = [list(o.shape[1:]) for o in output]
            else:
                output_shape = list(output.shape[1:])
        except Exception:
            pass
        
        # Save params, gradients, and flops to model metadata (exclude yaml_def)
        if record:
            import json
            record_path = model_storage._dir(model_id) / "record.json"
            # Create clean record without yaml_def (which belongs in graph.json)
            clean_record = {k: v for k, v in record.items() if k != 'yaml_def'}
            clean_record["params"] = n_params
            clean_record["gradients"] = n_gradients
            if flops is not None:
                clean_record["flops"] = flops
            
            print(f"[DEBUG] Saving to {record_path}")
            print(f"[DEBUG] params={n_params}, gradients={n_gradients}, flops={flops}")
            
            try:
                record_path.write_text(json.dumps(clean_record, indent=2, default=str))
                print(f"[DEBUG] Successfully saved record.json")
            except Exception as e:
                print(f"[ERROR] Failed to save record.json: {e}")
                import traceback
                traceback.print_exc()
        
        message = f"Valid — {n_params:,} params, {n_gradients:,} gradients, {n_layers} layers"
        if flops is not None:
            message += f", {flops:.1f} GFLOPs"
        
        return {
            "valid": True,
            "model_id": model_id,
            "params": n_params,
            "gradients": n_gradients,
            "flops": flops,
            "layers": n_layers,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "message": message,
        }
    except Exception as e:
        return {
            "valid": False,
            "model_id": model_id,
            "params": 0,
            "layers": 0,
            "input_shape": None,
            "output_shape": None,
            "message": str(e),
        }


@router.post("/export", summary="Export a model")
async def export_model(req: ExportRequest):
    """Export model in various formats (yaml, onnx, torchscript, etc.)."""
    record = model_storage.load_model(req.model_id)
    if not record:
        raise HTTPException(404, f"Model not found: {req.model_id}")

    yaml_path = model_storage.load_model_yaml_path(req.model_id)
    if not yaml_path:
        raise HTTPException(400, "No YAML file for this model")

    if req.format == "yaml":
        return {
            "model_id": req.model_id,
            "format": "yaml",
            "content": yaml_path.read_text(),
            "message": "YAML exported",
        }

    # Ultralytics export formats
    try:
        from ultralytics import YOLO
        model = YOLO(str(yaml_path), task=record.get("task", "detect"))

        if req.weight_id:
            from ..services.weight_storage import weight_pt_path
            wp = weight_pt_path(req.weight_id)
            if wp.exists():
                model.load(str(wp))

        exported = model.export(format=req.format, imgsz=req.imgsz)
        return {
            "model_id": req.model_id,
            "format": req.format,
            "file_path": str(exported),
            "message": f"Exported as {req.format}",
        }
    except Exception as e:
        raise HTTPException(400, f"Export failed: {e}")
