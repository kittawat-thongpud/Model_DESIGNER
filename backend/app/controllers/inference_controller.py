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


@router.delete("/history/{entry_id}", summary="Delete a history entry")
async def delete_history_entry(entry_id: str):
    """Delete a single history entry by id."""
    if not HISTORY_FILE.exists():
        raise HTTPException(404, "No history")
    lines = HISTORY_FILE.read_text().splitlines()
    new_lines = [l for l in lines if l and json.loads(l).get("id") != entry_id]
    HISTORY_FILE.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
    return {"message": f"Entry '{entry_id}' deleted"}
