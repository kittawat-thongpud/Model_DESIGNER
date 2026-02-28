"""
Inference Controller — Run predictions on images/video using trained weights.
"""
from __future__ import annotations
import asyncio
import base64
import io
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import DATA_DIR
from ..services import weight_storage, model_storage
from .. import logging_service as logger

router = APIRouter(prefix="/api/inference", tags=["Inference"])

INFERENCE_DIR = DATA_DIR / "inference"
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = INFERENCE_DIR / "history.jsonl"
MAX_HISTORY = 200


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_yolo(weight_id: str):
    """Load YOLO model from weight_id."""
    from ultralytics import YOLO
    meta = weight_storage.load_weight_meta(weight_id)
    if not meta:
        raise ValueError(f"Weight '{weight_id}' not found")
    pt_path = weight_storage.weight_pt_path(weight_id)
    if not pt_path.exists():
        raise ValueError(f"Weight file not found: {weight_id}")
    return YOLO(str(pt_path))


def _append_history(entry: dict) -> None:
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _load_history(limit: int = 50) -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    lines = HISTORY_FILE.read_text().splitlines()
    entries = []
    for line in reversed(lines):
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
        if len(entries) >= limit:
            break
    return entries


def _extract_top_classes(boxes_data, i: int, names: dict, top_k: int = 5) -> list[dict]:
    """
    Extract top-k class scores for detection i.

    Ultralytics stores boxes as:
      - Standard NMS output: [x1,y1,x2,y2, conf, cls]  → shape col=6  (no full dist)
      - When num_classes>1 raw preds survive NMS with all class scores appended:
        [x1,y1,x2,y2, cls_score_0, ..., cls_score_N]  → shape col=4+N
    We also check r._raw_preds / r.extra for some versions.
    """
    try:
        import torch
        data = boxes_data.data  # tensor [N, cols]
        n_cols = data.shape[1]
        if n_cols > 6:
            # columns 4..end are per-class scores (after x1y1x2y2 + ... format varies)
            # Ultralytics v8 agnostic=False: [x1,y1,x2,y2, conf, cls, cls_s0..cls_sN]
            # Try: columns 6 onward are per-class scores
            class_scores = data[i, 6:]
            if class_scores.numel() == 0:
                # Try columns 4 onward minus last 2 (conf, cls)
                class_scores = data[i, 4:-2]
            if class_scores.numel() > 0 and class_scores.numel() == len(names):
                # Softmax to normalize if raw logits
                scores = class_scores.float()
                topk = min(top_k, len(names))
                vals, idxs = torch.topk(scores, topk)
                return [
                    {
                        "class_id": int(idx.item()),
                        "class_name": names.get(int(idx.item()), str(int(idx.item()))),
                        "score": round(float(val.item()), 4),
                    }
                    for val, idx in zip(vals, idxs)
                    if float(val.item()) > 0
                ]
    except Exception:
        pass
    return []


def _results_to_response(results, weight_id: str, source_name: str,
                          elapsed_ms: float, image_count: int,
                          top_k: int = 5) -> dict:
    """Convert Ultralytics Results list to API response dict."""
    detections_per_image = []
    all_classes: dict[int, dict] = {}

    for r in results:
        img_dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes_data = r.boxes
            names = r.names or {}
            for i in range(len(boxes_data)):
                cls_id = int(boxes_data.cls[i].item())
                conf = float(boxes_data.conf[i].item())
                xyxy = boxes_data.xyxy[i].tolist()

                top_classes = _extract_top_classes(boxes_data, i, names, top_k)
                # If no full dist available, include at least the detected class
                if not top_classes:
                    top_classes = [{
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                        "score": round(conf, 4),
                    }]

                det = {
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": round(conf, 4),
                    "bbox": [round(v, 1) for v in xyxy],
                    "top_classes": top_classes,
                }
                img_dets.append(det)

                if cls_id not in all_classes:
                    all_classes[cls_id] = {
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                        "count": 0,
                        "max_conf": 0.0,
                        "confidences": [],
                    }
                all_classes[cls_id]["count"] += 1
                all_classes[cls_id]["confidences"].append(round(conf, 4))
                all_classes[cls_id]["max_conf"] = round(
                    max(all_classes[cls_id]["max_conf"], conf), 4
                )

        # Encode result image to base64
        img_b64 = None
        try:
            import numpy as np
            from PIL import Image as PILImage
            annotated = r.plot()  # numpy BGR
            if annotated is not None:
                img_rgb = annotated[:, :, ::-1]
                pil = PILImage.fromarray(img_rgb)
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=85)
                img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass

        # Latency per image
        speed = getattr(r, "speed", {}) or {}
        detections_per_image.append({
            "detections": img_dets,
            "image_b64": img_b64,
            "preprocess_ms": round(speed.get("preprocess", 0), 2),
            "inference_ms": round(speed.get("inference", 0), 2),
            "postprocess_ms": round(speed.get("postprocess", 0), 2),
            "total_ms": round(
                speed.get("preprocess", 0) +
                speed.get("inference", 0) +
                speed.get("postprocess", 0), 2
            ),
        })

    # Summary per class
    class_summary = sorted(all_classes.values(), key=lambda x: -x["count"])
    for c in class_summary:
        confs = c.pop("confidences")
        c["avg_conf"] = round(sum(confs) / len(confs), 4) if confs else 0.0

    avg_latency = (
        sum(d["total_ms"] for d in detections_per_image) / len(detections_per_image)
        if detections_per_image else 0.0
    )
    fps = round(1000.0 / avg_latency, 1) if avg_latency > 0 else 0.0

    return {
        "weight_id": weight_id,
        "source_name": source_name,
        "image_count": image_count,
        "total_detections": sum(len(d["detections"]) for d in detections_per_image),
        "elapsed_ms": round(elapsed_ms, 1),
        "avg_latency_ms": round(avg_latency, 2),
        "fps": fps,
        "class_summary": class_summary,
        "images": detections_per_image,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/predict", summary="Run inference on uploaded images")
async def predict_images(
    weight_id: str = Form(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    top_k: int = Form(5),
    files: list[UploadFile] = File(...),
):
    """Run YOLO prediction on one or more uploaded images. Returns annotated images + detections."""
    if not files:
        raise HTTPException(400, "No files uploaded")
    if len(files) > 32:
        raise HTTPException(400, "Max 32 images per request")

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        saved_paths = []
        names = []
        for f in files:
            suffix = Path(f.filename or "img.jpg").suffix or ".jpg"
            dest = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved_paths.append(str(dest))
            names.append(f.filename or dest.name)

        model = await asyncio.to_thread(_load_yolo, weight_id)

        t0 = time.time()
        results = await asyncio.to_thread(
            model.predict,
            saved_paths,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            stream=False,
        )
        elapsed_ms = (time.time() - t0) * 1000

        resp = _results_to_response(results, weight_id, ", ".join(names[:3]), elapsed_ms, len(files), top_k=top_k)

        # Save to history (without images to keep file small)
        history_entry = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "weight_id": weight_id,
            "source_name": ", ".join(names[:3]),
            "image_count": len(files),
            "total_detections": resp["total_detections"],
            "avg_latency_ms": resp["avg_latency_ms"],
            "fps": resp["fps"],
            "class_summary": resp["class_summary"],
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "type": "image",
        }
        _append_history(history_entry)

        return resp
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("/history", summary="Get inference history")
async def get_history(limit: int = 50):
    """Return recent inference runs (newest first)."""
    return _load_history(limit)


@router.delete("/history", summary="Clear inference history")
async def clear_history():
    """Delete all inference history."""
    HISTORY_FILE.unlink(missing_ok=True)
    return {"message": "History cleared"}


@router.post("/infer", summary="Run inference — bboxes, labels, confidence + annotated image")
async def infer(
    weight_id: str = Form(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    visualize_sgbg: bool = Form(False),
    file: UploadFile = File(...),
):
    """
    POST /api/inference/infer

    Upload a single image, run YOLO inference, return:
      • detections   : list of {class_id, class_name, confidence, bbox [x1,y1,x2,y2]}
      • image_b64    : annotated JPEG (data:image/jpeg;base64,...)
      • speed        : {preprocess_ms, inference_ms, postprocess_ms}
      • sgbg_vis     : (optional, HSG-DET only) per-scale selection/attn/delta heatmaps

    Form fields:
      weight_id       : weight ID (required)
      conf            : confidence threshold (default 0.25)
      iou             : NMS IoU threshold (default 0.45)
      imgsz           : inference image size (default 640)
      visualize_sgbg  : include SGBG feature maps (default false, slower)
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
        img_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
        with img_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        # ── Load model ──────────────────────────────────────────────────────
        model = await asyncio.to_thread(_load_yolo, weight_id)

        # ── Optionally attach SGBG hooks ─────────────────────────────────
        sgbg_hooks = None
        if visualize_sgbg:
            try:
                from hsg_det.nn.sparse_global import _register_into_ultralytics
                _register_into_ultralytics()
                from hsg_det.tools.visualize_sgbg import SGBGHooks
                sgbg_hooks = SGBGHooks()
                sgbg_hooks.register(model.model)
                if not sgbg_hooks.captures:
                    sgbg_hooks = None  # not an HSG-DET model — skip silently
            except Exception:
                sgbg_hooks = None

        # ── Run inference ────────────────────────────────────────────────
        t0 = time.time()
        results = await asyncio.to_thread(
            model.predict,
            str(img_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            stream=False,
        )
        elapsed_ms = (time.time() - t0) * 1000

        if sgbg_hooks:
            sgbg_hooks.remove()

        if not results:
            raise HTTPException(500, "No inference results returned")

        r = results[0]

        # ── Build detections list ────────────────────────────────────────
        detections = []
        names = r.names or {}
        if r.boxes is not None and len(r.boxes) > 0:
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i].item())
                conf_val = float(r.boxes.conf[i].item())
                xyxy = [round(v, 1) for v in r.boxes.xyxy[i].tolist()]
                detections.append({
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": round(conf_val, 4),
                    "bbox": xyxy,  # [x1, y1, x2, y2] absolute pixels
                })
        # Sort by confidence descending
        detections.sort(key=lambda d: -d["confidence"])

        # ── Annotated image → base64 ─────────────────────────────────────
        image_b64 = None
        try:
            import numpy as _np
            from PIL import Image as _PIL
            annotated = r.plot()  # BGR numpy
            if annotated is not None:
                pil = _PIL.fromarray(annotated[:, :, ::-1])
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=88)
                image_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass

        # ── Speed summary ────────────────────────────────────────────────
        speed = getattr(r, "speed", {}) or {}

        # ── SGBG visualization ───────────────────────────────────────────
        sgbg_vis: dict | None = None
        if sgbg_hooks and sgbg_hooks.captures:
            try:
                import cv2 as _cv2
                import numpy as _np
                from hsg_det.tools.visualize_sgbg import (
                    _infer_scale_label,
                    vis_selection_map,
                    vis_attention_heatmap,
                    vis_refinement_delta,
                )

                # Read original image for overlays
                img_bgr = _cv2.imread(str(img_path))
                ih, iw = img_bgr.shape[:2]
                scale_f = imgsz / max(ih, iw)
                display = _cv2.resize(img_bgr, (int(iw * scale_f), int(ih * scale_f)))

                # Only keep top-level Gated modules — skip inner .block submodules
                # (they share the same captures but produce duplicate slugs)
                from hsg_det.nn.sparse_global import SparseGlobalBlockGated as _SGBG
                top_level_names = [
                    n for n, m in model.model.named_modules()
                    if isinstance(m, _SGBG) and n in sgbg_hooks.captures
                ]
                # Fallback: if hook only captured inner blocks, use all with topk_idx
                if not top_level_names:
                    top_level_names = [
                        n for n, store in sgbg_hooks.captures.items()
                        if "topk_idx" in store
                    ]

                all_names = top_level_names
                sgbg_vis = {}

                def _panel_b64(arr_bgr: _np.ndarray) -> str:
                    from PIL import Image as _PI
                    pil = _PI.fromarray(arr_bgr[:, :, ::-1])
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=82)
                    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

                # Sort modules by feature map area: smallest=highest stride=P5, largest=P3
                _valid = [
                    (n, sgbg_hooks.captures[n]) for n in top_level_names
                    if "topk_idx" in sgbg_hooks.captures[n]
                ]
                _valid.sort(key=lambda x: x[1]["feature_hw"][0] * x[1]["feature_hw"][1])
                # slugs from deepest to shallowest: p5, p4, p3, p2
                _slugs_from_deep = ["p5", "p4", "p3", "p2"]
                _mod_slug = {
                    name: _slugs_from_deep[i] if i < len(_slugs_from_deep) else f"s{i}"
                    for i, (name, _) in enumerate(_valid)
                }
                for mod_name in top_level_names:
                    store = sgbg_hooks.captures[mod_name]
                    if "topk_idx" not in store:
                        continue
                    slug = _mod_slug[mod_name]
                    H, W = store["feature_hw"]
                    scale_label = f"{slug.upper()} ({H}×{W})"  # show actual hw instead of fake stride

                    vis_tmp = Path(tempfile.mkdtemp())
                    try:
                        p_sel   = vis_tmp / "sel.jpg"
                        p_attn  = vis_tmp / "attn.jpg"
                        p_delta = vis_tmp / "delta.jpg"

                        pan_sel   = vis_selection_map(
                            store["topk_idx"], store["feature_hw"], display, scale_label, p_sel)
                        pan_attn  = vis_attention_heatmap(
                            store["attn"], store["topk_idx"], store["feature_hw"],
                            display, scale_label, p_attn)
                        pan_delta = vis_refinement_delta(
                            store["delta"], display, scale_label, p_delta,
                            store.get("gate_value", 1.0))

                        sgbg_vis[slug] = {
                            "scale":      scale_label,
                            "feature_hw": list(store["feature_hw"]),
                            "k":          store["k_actual"],
                            "gate":       round(store.get("gate_value", 1.0), 6),
                            "selection":  _panel_b64(pan_sel),
                            "attention":  _panel_b64(pan_attn),
                            "delta":      _panel_b64(pan_delta),
                        }
                    finally:
                        shutil.rmtree(str(vis_tmp), ignore_errors=True)

            except Exception as _ve:
                sgbg_vis = {"error": str(_ve)}

        # ── History entry ────────────────────────────────────────────────
        _append_history({
            "id": uuid.uuid4().hex[:12],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "weight_id": weight_id,
            "source_name": file.filename or img_path.name,
            "image_count": 1,
            "total_detections": len(detections),
            "avg_latency_ms": round(elapsed_ms, 2),
            "fps": round(1000.0 / elapsed_ms, 1) if elapsed_ms > 0 else 0.0,
            "class_summary": {},
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "type": "infer",
        })

        # is_hsg_det = True if SGBG hooks found modules (even if visualize_sgbg=False)
        _has_sgbg = False
        try:
            from hsg_det.nn.sparse_global import SparseGlobalBlockGated as _SGBG2
            _has_sgbg = any(isinstance(m, _SGBG2) for _, m in model.model.named_modules())
        except Exception:
            pass

        return {
            "weight_id": weight_id,
            "filename": file.filename,
            "elapsed_ms": round(elapsed_ms, 1),
            "total_detections": len(detections),
            "is_hsg_det": _has_sgbg,
            "detections": detections,
            "image_b64": image_b64,
            "speed": {
                "preprocess_ms":  round(speed.get("preprocess", 0), 2),
                "inference_ms":   round(speed.get("inference", 0), 2),
                "postprocess_ms": round(speed.get("postprocess", 0), 2),
            },
            "sgbg_vis": sgbg_vis,
        }

    except ValueError as e:
        raise HTTPException(404, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


@router.post("/infer/attention", summary="SGBG attention map for a specific detection bbox")
async def infer_attention(
    weight_id: str = Form(...),
    imgsz: int = Form(640),
    bbox_x1: float = Form(...),
    bbox_y1: float = Form(...),
    bbox_x2: float = Form(...),
    bbox_y2: float = Form(...),
    det_label: str = Form(""),        # "car 89%" for title
    file: UploadFile = File(...),
):
    """
    POST /api/inference/infer/attention

    Run inference with SGBG hooks, then render attention heatmaps where the
    query token is the selected token nearest to the detection bbox centroid.

    Returns per-scale attention images (base64 JPEG) keyed by p3/p4/p5.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
        img_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
        with img_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        model = await asyncio.to_thread(_load_yolo, weight_id)

        # Attach hooks
        try:
            from hsg_det.nn.sparse_global import _register_into_ultralytics, SparseGlobalBlockGated as _SGBG
            _register_into_ultralytics()
            from hsg_det.tools.visualize_sgbg import SGBGHooks, vis_attention_heatmap
            sgbg_hooks = SGBGHooks()
            sgbg_hooks.register(model.model)
            if not sgbg_hooks.captures:
                raise HTTPException(400, "Model has no SparseGlobalBlockGated modules")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Failed to attach SGBG hooks: {e}")

        # Run inference
        await asyncio.to_thread(
            model.predict, str(img_path),
            imgsz=imgsz, verbose=False, stream=False,
        )
        sgbg_hooks.remove()

        # Compute bbox centroid in image pixels
        # Read display image for overlays
        import cv2 as _cv2
        img_bgr = _cv2.imread(str(img_path))
        ih, iw = img_bgr.shape[:2]
        scale_f = imgsz / max(ih, iw)
        display = _cv2.resize(img_bgr, (int(iw * scale_f), int(ih * scale_f)))
        d_ih, d_iw = display.shape[:2]

        # bbox is in original image pixels → scale to display
        cx = ((bbox_x1 + bbox_x2) / 2) * (d_iw / iw)
        cy = ((bbox_y1 + bbox_y2) / 2) * (d_ih / ih)

        # Filter top-level Gated modules
        top_level_names = [
            n for n, m in model.model.named_modules()
            if isinstance(m, _SGBG) and n in sgbg_hooks.captures
        ]
        if not top_level_names:
            top_level_names = [
                n for n, s in sgbg_hooks.captures.items() if "topk_idx" in s
            ]

        def _b64(arr) -> str:
            from PIL import Image as _PI
            pil = _PI.fromarray(arr[:, :, ::-1])
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        _valid_attn = [
            (n, sgbg_hooks.captures[n]) for n in top_level_names
            if "topk_idx" in sgbg_hooks.captures[n]
        ]
        _valid_attn.sort(key=lambda x: x[1]["feature_hw"][0] * x[1]["feature_hw"][1])
        _slugs_from_deep = ["p5", "p4", "p3", "p2"]
        _mod_slug_attn = {
            name: _slugs_from_deep[i] if i < len(_slugs_from_deep) else f"s{i}"
            for i, (name, _) in enumerate(_valid_attn)
        }
        result_scales = {}

        for mod_name in top_level_names:
            store = sgbg_hooks.captures[mod_name]
            if "topk_idx" not in store:
                continue
            slug = _mod_slug_attn[mod_name]
            H, W = store["feature_hw"]
            scale_label = f"{slug.upper()} ({H}×{W})"

            vis_tmp = Path(tempfile.mkdtemp())
            try:
                p_attn = vis_tmp / "attn.jpg"
                pan = vis_attention_heatmap(
                    store["attn"], store["topk_idx"],
                    store["feature_hw"], display, scale_label, p_attn,
                    query_pixel=(cx, cy),
                    query_label=det_label,
                )
                result_scales[slug] = {
                    "scale": scale_label,
                    "feature_hw": list(store["feature_hw"]),
                    "query_pixel": [round(cx, 1), round(cy, 1)],
                    "attention": _b64(pan),
                }
            finally:
                shutil.rmtree(str(vis_tmp), ignore_errors=True)

        return {
            "weight_id": weight_id,
            "det_label": det_label,
            "bbox_centroid": [round(cx, 1), round(cy, 1)],
            "scales": result_scales,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Attention query failed: {e}")
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


@router.delete("/history/{entry_id}", summary="Delete a history entry")
async def delete_history_entry(entry_id: str):
    """Delete a single history entry by id."""
    if not HISTORY_FILE.exists():
        raise HTTPException(404, "No history")
    lines = HISTORY_FILE.read_text().splitlines()
    new_lines = [l for l in lines if l and json.loads(l).get("id") != entry_id]
    HISTORY_FILE.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
    return {"message": f"Entry '{entry_id}' deleted"}
