"""
Ultralytics YOLO weight source plugin.

Handles checkpoints saved by the Ultralytics library (YOLOv5, v8, v11, etc.).
These are typically dicts with a ``"model"`` key containing the full nn.Module,
plus ``"ema"``, ``"names"``, ``"epoch"`` metadata.

Also provides a pretrained model catalog — users can browse and download
official Ultralytics pretrained weights directly from the Weight Editor.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import torch

from ..base import WeightSourcePlugin
from ..loader import register_weight_source


# ── Pretrained Model Catalog ──────────────────────────────────────────────────

_YOLO_PRETRAINED: list[dict] = [
    # YOLOv8 Detection
    {"model_key": "yolov8n", "display_name": "YOLOv8n (Nano)", "task": "detection",
     "param_count": 3_200_000, "description": "YOLOv8 Nano — 3.2M params, fastest"},
    {"model_key": "yolov8s", "display_name": "YOLOv8s (Small)", "task": "detection",
     "param_count": 11_200_000, "description": "YOLOv8 Small — 11.2M params"},
    {"model_key": "yolov8m", "display_name": "YOLOv8m (Medium)", "task": "detection",
     "param_count": 25_900_000, "description": "YOLOv8 Medium — 25.9M params"},
    {"model_key": "yolov8l", "display_name": "YOLOv8l (Large)", "task": "detection",
     "param_count": 43_700_000, "description": "YOLOv8 Large — 43.7M params"},
    {"model_key": "yolov8x", "display_name": "YOLOv8x (XLarge)", "task": "detection",
     "param_count": 68_200_000, "description": "YOLOv8 XLarge — 68.2M params, most accurate"},
    # YOLOv5 Detection
    {"model_key": "yolov5nu", "display_name": "YOLOv5n (Nano)", "task": "detection",
     "param_count": 2_600_000, "description": "YOLOv5 Nano updated — 2.6M params"},
    {"model_key": "yolov5su", "display_name": "YOLOv5s (Small)", "task": "detection",
     "param_count": 9_100_000, "description": "YOLOv5 Small updated — 9.1M params"},
    # YOLOv8 Classification
    {"model_key": "yolov8n-cls", "display_name": "YOLOv8n-cls", "task": "classification",
     "param_count": 2_700_000, "description": "YOLOv8 Nano classifier — ImageNet"},
    {"model_key": "yolov8s-cls", "display_name": "YOLOv8s-cls", "task": "classification",
     "param_count": 6_400_000, "description": "YOLOv8 Small classifier — ImageNet"},
    # YOLOv8 Segmentation
    {"model_key": "yolov8n-seg", "display_name": "YOLOv8n-seg", "task": "segmentation",
     "param_count": 3_400_000, "description": "YOLOv8 Nano segmentation — COCO"},
    {"model_key": "yolov8s-seg", "display_name": "YOLOv8s-seg", "task": "segmentation",
     "param_count": 11_800_000, "description": "YOLOv8 Small segmentation — COCO"},
]

_DOWNLOAD_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/{key}.pt"


class UltralyticsWeightSource(WeightSourcePlugin):

    @property
    def name(self) -> str:
        return "ultralytics"

    @property
    def display_name(self) -> str:
        return "Ultralytics YOLO"

    @property
    def file_extensions(self) -> list[str]:
        return [".pt"]

    # ── Detection ─────────────────────────────────────────────────────────────

    def can_parse(self, data: Any) -> bool:
        """Ultralytics checkpoints have a 'model' key that is an nn.Module."""
        if not isinstance(data, dict):
            return False
        model = data.get("model")
        if model is None:
            return False
        # The model object should be an nn.Module (or have state_dict)
        if hasattr(model, "state_dict"):
            return True
        # Some versions store the model already as a state_dict under "model"
        if isinstance(model, dict) and any(
            isinstance(v, torch.Tensor) for v in list(model.values())[:5]
        ):
            return True
        return False

    # ── Extraction ────────────────────────────────────────────────────────────

    def extract_state_dict(self, data: Any) -> dict[str, torch.Tensor]:
        model = data["model"]

        # nn.Module path (most common)
        if hasattr(model, "state_dict"):
            # .float() converts FP16 weights to FP32
            try:
                sd = model.float().state_dict()
            except Exception:
                sd = model.state_dict()
        elif isinstance(model, dict):
            sd = model
        else:
            raise ValueError("Cannot extract state_dict from Ultralytics checkpoint")

        # Strip the leading "model." prefix if present
        # Ultralytics keys look like: model.0.conv.weight → we want 0.conv.weight
        cleaned: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            new_key = k[len("model."):] if k.startswith("model.") else k
            cleaned[new_key] = v

        return cleaned

    # ── Layer Groups ──────────────────────────────────────────────────────────

    def get_layer_groups(self, sd: dict[str, torch.Tensor]) -> list[dict]:
        """Group by numeric layer index (0, 1, 2, ...) typical of YOLO."""
        from collections import OrderedDict
        groups: dict[str, list[dict]] = OrderedDict()

        for k, v in sd.items():
            # Keys look like: 0.conv.weight, 1.cv1.conv.weight, 22.dfl.conv.weight
            prefix = k.split(".")[0] if "." in k else k
            groups.setdefault(prefix, []).append({
                "key": k,
                "shape": list(v.shape),
                "dtype": str(v.dtype),
            })

        result = []
        for prefix, keys in groups.items():
            result.append({
                "prefix": prefix,
                "module_type": self._detect_yolo_module(keys),
                "param_count": len(keys),
                "keys": keys,
            })
        return result

    # ── Pretrained Catalog ───────────────────────────────────────────────────

    @property
    def has_pretrained_catalog(self) -> bool:
        return True

    def list_pretrained(self) -> list[dict]:
        from ...services import weight_storage
        # Check which models are already downloaded
        existing = weight_storage.list_weights()
        downloaded_keys: set[str] = set()
        for w in existing:
            src = w.get("source_plugin", "")
            orig = w.get("original_filename", "")
            if src == "ultralytics" and orig:
                # original_filename is like "yolov8n.pt"
                key = orig.rsplit(".", 1)[0]
                downloaded_keys.add(key)

        result = []
        for entry in _YOLO_PRETRAINED:
            result.append({
                **entry,
                "source_plugin": self.name,
                "downloaded": entry["model_key"] in downloaded_keys,
            })
        return result

    def download_pretrained(self, model_key: str) -> dict:
        """Download a pretrained YOLO model and import as system weight.

        Uses the ``ultralytics`` library which caches downloads in
        ``~/.config/Ultralytics/`` — second downloads are instant.
        """
        import logging
        from ...services import weight_import

        log = logging.getLogger(__name__)

        catalog = {e["model_key"]: e for e in _YOLO_PRETRAINED}
        if model_key not in catalog:
            raise ValueError(f"Unknown pretrained model: {model_key}")

        entry = catalog[model_key]

        # Use ultralytics YOLO() which auto-downloads + caches the .pt file
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(f"{model_key}.pt")
            # The .pt file is now at yolo_model.ckpt_path (or model_key.pt in cwd)
            pt_path = Path(yolo_model.ckpt_path) if hasattr(yolo_model, 'ckpt_path') else Path(f"{model_key}.pt")
            if not pt_path.exists():
                pt_path = Path(f"{model_key}.pt")
            log.info("Ultralytics model loaded: %s at %s", model_key, pt_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download {model_key}: {e}")

        # Import using the existing weight import pipeline
        result = weight_import.import_external_weight(pt_path, name=entry["display_name"])

        # Clean up local .pt if ultralytics left one in cwd
        cwd_pt = Path(f"{model_key}.pt")
        if cwd_pt.exists() and cwd_pt != pt_path:
            try:
                cwd_pt.unlink()
            except Exception:
                pass

        return {
            **result,
            "model_key": model_key,
            "task": entry["task"],
        }

    @staticmethod
    def _detect_yolo_module(keys: list[dict]) -> str:
        """Detect YOLO module type from sub-key patterns."""
        suffixes = {k["key"].split(".", 1)[-1] if "." in k["key"] else "" for k in keys}

        # C2f / C3 / CSP blocks have cv1, cv2, and m.* (bottleneck chain)
        has_cv1 = any("cv1." in s for s in suffixes)
        has_cv2 = any("cv2." in s for s in suffixes)
        has_m = any(s.startswith("m.") or ".m." in s for s in suffixes)

        if has_cv1 and has_cv2 and has_m:
            return "C2f/CSP"
        if has_cv1 and has_cv2:
            return "Bottleneck"

        # SPPF: cv1, cv2, but no m.* — distinguished by having maxpool ref
        # Actually SPPF has cv1 + cv2 but no m, same as Bottleneck.
        # We look for specific patterns:
        if any("cv." in s for s in suffixes) and len(keys) <= 6:
            return "Conv"

        # Simple Conv block: conv.weight + bn.weight/bias/running_mean/var
        if any(s.startswith("conv.") for s in suffixes):
            return "Conv"

        # Detection head: has cv2, cv3 lists
        if any("cv2." in s for s in suffixes) and any("cv3." in s for s in suffixes):
            return "Detect"

        # DFL
        if any("dfl." in s for s in suffixes):
            return "DFL"

        return "unknown"


# ── Auto-register ─────────────────────────────────────────────────────────────
register_weight_source(UltralyticsWeightSource())
