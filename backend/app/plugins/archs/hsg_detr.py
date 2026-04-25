"""
HSG-DETR Model Architecture Plugin.

Registers four scale variants (N / S / M / L) into the plugin system.
Modules are defined locally (no external repo dependency).

Registers as: "hsg_detr_n", "hsg_detr_s", "hsg_detr_m", "hsg_detr_l"
"""
from __future__ import annotations
from pathlib import Path

from ..base import ModelArchPlugin
from ..loader import register_arch


_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "hsg_detr" / "configs"

_SCALE_DESCRIPTIONS = {
    "n": ("Nano",    "~2–3 M",  "~5–8 GFLOPs",  "100 queries"),
    "s": ("Small",   "~4–5 M",  "~8–12 GFLOPs", "150 queries"),
    "m": ("Medium",  "~6–8 M",  "~14–18 GFLOPs","200 queries"),
    "l": ("Large",   "~11–15 M","~25–40 GFLOPs","300 queries"),
}


class HSGDetRPlugin(ModelArchPlugin):
    """Plugin for one HSG-DETR scale variant (N, S, M, or L)."""

    def __init__(self, scale: str):
        self._scale = scale.lower()

    @property
    def name(self) -> str:
        return f"hsg_detr_{self._scale}"

    @property
    def display_name(self) -> str:
        label, params, flops, queries = _SCALE_DESCRIPTIONS[self._scale]
        return f"HSG-DETR {label} ({params}, {flops})"

    @property
    def family(self) -> str:
        return "hsg_detr"

    @property
    def family_display_name(self) -> str:
        return "HSG-DETR"

    @property
    def scale(self) -> str:
        return self._scale

    @property
    def scale_label(self) -> str:
        label, params, flops, queries = _SCALE_DESCRIPTIONS[self._scale]
        return f"{label} ({params}, {flops}) — {queries}"

    @property
    def task_type(self) -> str:
        return "detect"

    @property
    def description(self) -> str:
        return (
            "HSG-DETR — Sparse-Token SGB encoder feeding RT-DETR decoder. "
            "Top-k token selection before sparse global self-attention, "
            "scatter-back with gated residual, RTDETRDecoder head. "
            "SGB-centric architecture: sparse global reasoning is the core "
            "feature representation, not an add-on. "
            "Warm-start from YOLOv8 backbone weights."
        )

    def yaml_path(self) -> Path:
        return _CONFIGS_DIR / f"hsg_detr_{self._scale}.yaml"

    def register_modules(self) -> None:
        """Inject HSG-DETR blocks into ultralytics.nn.modules."""
        try:
            import hsg_detr.nn  # noqa: F401  — triggers register() on import
        except ImportError as e:
            raise ImportError(
                "Could not import hsg_detr package. "
                "Ensure backend/hsg_detr/ is on PYTHONPATH."
            ) from e

    def pretrain_key(self) -> str | None:
        return None

    def warm_start(self, model, log_fn=None, model_scale: str | None = None) -> dict:
        """Transfer shape-matching backbone layers from YOLOv8 into model."""
        from pathlib import Path
        import torch

        def _log(msg: str):
            if log_fn:
                log_fn(msg)

        _SCALE_MAP = {
            "n": "yolov8n", "s": "yolov8s", "m": "yolov8m",
            "l": "yolov8l", "x": "yolov8x",
        }
        scale = (model_scale or "").lower()
        yolo_key = _SCALE_MAP.get(scale)
        if yolo_key is None:
            _log("Backbone warm-start: no model_scale provided — skipping")
            return {"transferred": 0, "skipped": 0, "total_src": 0, "total_tgt": 0, "matched_layers": []}

        _log(f"Backbone warm-start: HSG-DETR scale='{scale}' → source={yolo_key}.pt")

        cache_candidates = [
            Path.home() / ".config" / "Ultralytics" / f"{yolo_key}.pt",
            Path.home() / "AppData" / "Roaming" / "Ultralytics" / f"{yolo_key}.pt",
        ]
        src_pt: Path | None = next((p for p in cache_candidates if p.exists()), None)

        if src_pt is None:
            _log(f"Backbone warm-start: {yolo_key}.pt not in cache — attempting download...")
            try:
                from ultralytics import YOLO as _YOLO
                _tmp = _YOLO(f"{yolo_key}.pt")
                src_pt = next((p for p in cache_candidates if p.exists()), None)
                if src_pt is None:
                    cwd_pt = Path(f"{yolo_key}.pt")
                    if cwd_pt.exists():
                        src_pt = cwd_pt
                if src_pt is None:
                    _log(f"Backbone warm-start: cannot locate {yolo_key}.pt after download — skipping")
                    return {"transferred": 0, "skipped": 0, "total_src": 0, "total_tgt": 0, "matched_layers": []}
            except Exception as e:
                _log(f"Backbone warm-start: download failed ({e}) — training from scratch")
                return {"transferred": 0, "skipped": 0, "total_src": 0, "total_tgt": 0, "matched_layers": []}

        _log(f"Backbone warm-start: loading source weights from {src_pt}")

        try:
            raw = torch.load(src_pt, map_location="cpu", weights_only=False)
        except Exception as e:
            _log(f"Backbone warm-start: failed to load {src_pt} ({e}) — skipping")
            return {"transferred": 0, "skipped": 0, "total_src": 0, "total_tgt": 0, "matched_layers": []}

        if isinstance(raw, dict) and "model" in raw:
            src_obj = raw["model"]
            src_sd_raw = src_obj.float().state_dict() if hasattr(src_obj, "state_dict") else src_obj
        elif isinstance(raw, dict):
            src_sd_raw = raw
        else:
            _log("Backbone warm-start: unrecognised checkpoint format — skipping")
            return {"transferred": 0, "skipped": 0, "total_src": 0, "total_tgt": 0, "matched_layers": []}

        _PFX = "model."
        src_sd = {
            (k[len(_PFX):] if k.startswith(_PFX) else k): v
            for k, v in src_sd_raw.items()
            if isinstance(v, torch.Tensor)
        }

        try:
            tgt_nn = model.model
            tgt_sd_raw = tgt_nn.state_dict()
        except Exception as e:
            _log(f"Backbone warm-start: cannot read target state_dict ({e}) — skipping")
            return {"transferred": 0, "skipped": 0, "total_src": 0, "total_tgt": 0, "matched_layers": []}

        tgt_sd = {
            (k[len(_PFX):] if k.startswith(_PFX) else k): v
            for k, v in tgt_sd_raw.items()
        }

        transferred = 0
        skipped = 0
        matched_layers: set[str] = set()
        skipped_layers: set[str] = set()

        new_tgt = dict(tgt_sd)
        for key, tgt_tensor in tgt_sd.items():
            layer_id = key.split(".")[0]
            if key not in src_sd:
                skipped += 1
                continue
            src_tensor = src_sd[key]
            if src_tensor.shape == tgt_tensor.shape:
                new_tgt[key] = src_tensor.to(tgt_tensor.dtype)
                transferred += 1
                matched_layers.add(layer_id)
            else:
                skipped += 1
                skipped_layers.add(layer_id)

        restored_sd = {
            (_PFX + k if not k.startswith(_PFX) else k): v
            for k, v in new_tgt.items()
        }
        tgt_nn.load_state_dict(restored_sd, strict=False)

        total_tgt_keys = len(tgt_sd)
        pct = transferred / total_tgt_keys * 100 if total_tgt_keys else 0
        _log(f"Warm-start: {transferred}/{total_tgt_keys} tensors ({pct:.1f}%) transferred")
        _log(f"  matched layers : {sorted(matched_layers, key=lambda x: int(x) if x.isdigit() else 999)}")
        _log(f"  skipped layers : {sorted(skipped_layers, key=lambda x: int(x) if x.isdigit() else 999)}")

        return {
            "transferred": transferred,
            "skipped": skipped,
            "total_src": len(src_sd),
            "total_tgt": len(tgt_sd),
            "matched_layers": sorted(matched_layers),
        }


# ── Auto-register all four scales ──────────────────────────────────────────
for _scale in ("n", "s", "m", "l"):
    register_arch(HSGDetRPlugin(_scale))
