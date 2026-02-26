"""
HSG-DET Model Architecture Plugin.

Registers the HSG-DET (Hybrid Sparse-Global Detector) model architecture
into the backend plugin system so the training pipeline can discover,
configure, and train it exactly like any other architecture.

Registers as: "hsg_det"
"""
from __future__ import annotations
from pathlib import Path

from ..base import ModelArchPlugin
from ..loader import register_arch


class HSGDetPlugin(ModelArchPlugin):
    """
    Plugin for the HSG-DET (Hybrid Sparse-Global Detector) architecture.

    Behaviour
    ---------
    - ``register_modules()`` imports ``hsg_det.nn`` which auto-injects
      ``SparseGlobalBlock`` and ``SparseGlobalBlockGated`` into the
      Ultralytics ``nn.modules`` namespace before any YAML parsing.
    - ``yaml_path()`` returns the path to ``hsg_det_m.yaml`` which the
      training pipeline passes directly to ``YOLO(yaml_path)``.
    - ``pretrain_key()`` returns ``"yolov8m"`` so the backbone can be
      warm-started from the YOLOv8-m pretrained checkpoint, reducing
      convergence time by ~60–70% compared to random init.
    """

    @property
    def name(self) -> str:
        return "hsg_det"

    @property
    def display_name(self) -> str:
        return "HSG-DET (Hybrid Sparse-Global Detector)"

    @property
    def task_type(self) -> str:
        return "detect"

    @property
    def description(self) -> str:
        return (
            "Conservative hybrid CNN + Sparse Global Attention detector (~38.5M params). "
            "Targets 1080p dense-object scenes (IDD, VisDrone). "
            "Adds SparseGlobalBlock at PAN P4/P5 (~680 GFLOPs, +4–6 ms vs YOLO-L). "
            "Warm-start from YOLOv8-m backbone weights."
        )

    def yaml_path(self) -> Path:
        """Return the absolute path to hsg_det_m.yaml."""
        # This file lives at backend/hsg_det/configs/hsg_det_m.yaml
        # We resolve relative to this plugin file → app/plugins/archs/hsg_det.py
        # → go up 4 levels (archs, plugins, app, backend root) → hsg_det/configs/
        backend_root = Path(__file__).resolve().parents[3]
        return backend_root / "hsg_det" / "configs" / "hsg_det_m.yaml"

    def register_modules(self) -> None:
        """Inject SparseGlobalBlock/Gated into ultralytics.nn.modules."""
        # hsg_det.nn.__init__ calls _register_into_ultralytics() on import.
        # We just ensure the import happens here; it is idempotent.
        try:
            import hsg_det.nn  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Could not import hsg_det package. "
                "Ensure the backend/hsg_det/ directory is on PYTHONPATH."
            ) from e

    def pretrain_key(self) -> str | None:
        return "yolov8m"

    def warm_start(self, model, log_fn=None, model_scale: str | None = None) -> dict:
        """Transfer shape-matching backbone layers from YOLOv8 into model.

        Selects the YOLOv8 variant that matches model_scale (n→yolov8n,
        s→yolov8s, m→yolov8m, l→yolov8l, x→yolov8x) so channel dims align.
        Only tensors with identical shapes are copied (Option A partial
        transfer). Works offline if the .pt is already in Ultralytics cache.
        """
        from pathlib import Path
        import torch

        def _log(msg: str):
            if log_fn:
                log_fn(msg)

        # ── 0. Pick YOLOv8 scale matching model_scale ────────────────────────
        # HSG-DET backbone is CSP/C2f — same family as YOLOv8.
        # Always select the same scale so channel dims align exactly.
        # If scale is unknown, skip warm-start rather than defaulting to a
        # mismatched checkpoint (wrong channel dims = zero transfers anyway).
        _SCALE_MAP = {"n": "yolov8n", "s": "yolov8s", "m": "yolov8m",
                      "l": "yolov8l", "x": "yolov8x"}
        scale = (model_scale or "").lower()
        yolo_key = _SCALE_MAP.get(scale)
        if yolo_key is None:
            _log(f"Backbone warm-start: no model_scale provided — skipping warm-start")
            return {"transferred": 0, "skipped": 0, "total_src": 0, "total_tgt": 0, "matched_layers": []}
        _log(f"Backbone warm-start: HSG-DET scale='{scale}' → pretrain source={yolo_key}.pt (CSP/C2f family match)")

        # ── 1. Locate or download .pt ─────────────────────────────────────────
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
                # Ultralytics saves to ~/.config/Ultralytics/ after download
                src_pt = next((p for p in cache_candidates if p.exists()), None)
                if src_pt is None:
                    # fallback: check cwd
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

        # ── 2. Load source state_dict, strip "model." prefix ─────────────────
        try:
            raw = torch.load(src_pt, map_location="cpu", weights_only=False)
        except Exception as e:
            _log(f"Backbone warm-start: failed to load {src_pt} ({e}) — training from scratch")
            return {"transferred": 0, "skipped": 0, "total_src": 0, "total_tgt": 0, "matched_layers": []}

        if isinstance(raw, dict) and "model" in raw:
            src_obj = raw["model"]
            src_sd_raw: dict = src_obj.float().state_dict() if hasattr(src_obj, "state_dict") else src_obj
        elif isinstance(raw, dict):
            src_sd_raw = raw
        else:
            _log(f"Backbone warm-start: unrecognised checkpoint format — skipping")
            return {"transferred": 0, "skipped": 0, "total_src": 0, "total_tgt": 0, "matched_layers": []}

        # Strip leading "model." — Ultralytics keys: "model.0.conv.weight" → "0.conv.weight"
        _PFX = "model."
        src_sd: dict = {
            (k[len(_PFX):] if k.startswith(_PFX) else k): v
            for k, v in src_sd_raw.items()
            if isinstance(v, torch.Tensor)
        }

        # ── 3. Get target state_dict, also strip "model." prefix ──────────────
        # DetectionModel.state_dict() also uses "model.0.conv.weight" keys
        try:
            tgt_nn = model.model  # ultralytics.YOLO.model → DetectionModel
            tgt_sd_raw: dict = tgt_nn.state_dict()
        except Exception as e:
            _log(f"Backbone warm-start: cannot read target state_dict ({e}) — skipping")
            return {"transferred": 0, "skipped": 0, "total_src": 0, "total_tgt": 0, "matched_layers": []}

        tgt_sd: dict = {
            (k[len(_PFX):] if k.startswith(_PFX) else k): v
            for k, v in tgt_sd_raw.items()
        }

        # ── 4. Transfer shape-matching keys ──────────────────────────────────
        transferred = 0
        skipped = 0
        matched_layers: set[str] = set()
        skipped_layers: set[str] = set()
        # Detailed records for compatibility report
        _compat: list[dict] = []  # {key, src_shape, tgt_shape, status}

        new_tgt: dict = dict(tgt_sd)
        for key, tgt_tensor in tgt_sd.items():
            layer_id = key.split(".")[0]
            if key not in src_sd:
                skipped += 1
                _compat.append({"key": key, "src": None, "tgt": tuple(tgt_tensor.shape), "status": "not_in_src"})
                continue
            src_tensor = src_sd[key]
            if src_tensor.shape == tgt_tensor.shape:
                new_tgt[key] = src_tensor.to(tgt_tensor.dtype)
                transferred += 1
                matched_layers.add(layer_id)
                _compat.append({"key": key, "src": tuple(src_tensor.shape), "tgt": tuple(tgt_tensor.shape), "status": "ok"})
            else:
                skipped += 1
                skipped_layers.add(layer_id)
                _compat.append({"key": key, "src": tuple(src_tensor.shape), "tgt": tuple(tgt_tensor.shape), "status": "shape_mismatch"})

        # ── 5. Apply merged weights directly on tgt_nn ───────────────────────
        # warm_start() is called from on_pretrain_routine_end callback, AFTER
        # Ultralytics setup_model() has created the final nn.Module — so
        # load_state_dict() here will persist through the entire training run.
        restored_sd = {
            (_PFX + k if not k.startswith(_PFX) else k): v
            for k, v in new_tgt.items()
        }
        tgt_nn.load_state_dict(restored_sd, strict=False)

        # ── 6. Log detailed compatibility report ─────────────────────────────
        total_tgt_keys = len(tgt_sd)
        pct = transferred / total_tgt_keys * 100 if total_tgt_keys else 0
        _log(f"")
        _log(f"{'─'*60}")
        _log(f" Warm-start Compatibility Report: {yolo_key}.pt → HSG-DET-{scale}")
        _log(f"{'─'*60}")
        _log(f" Source  : {yolo_key}.pt  ({len(src_sd)} tensors)")
        _log(f" Target  : HSG-DET-{scale} ({total_tgt_keys} tensors)")
        _log(f" Transferred : {transferred} tensors ({pct:.1f}%)")
        _log(f" Skipped     : {skipped} tensors")
        _log(f" Layers matched : {sorted(matched_layers, key=lambda x: int(x) if x.isdigit() else 999)}")
        _log(f" Layers skipped : {sorted(skipped_layers, key=lambda x: int(x) if x.isdigit() else 999)}")
        _log(f"{'─'*60}")

        # Per-layer breakdown grouped by layer index
        from itertools import groupby
        sorted_compat = sorted(_compat, key=lambda r: (
            int(r["key"].split(".")[0]) if r["key"].split(".")[0].isdigit() else 999,
            r["key"]
        ))
        cur_layer = None
        for r in sorted_compat:
            layer_id = r["key"].split(".")[0]
            if layer_id != cur_layer:
                cur_layer = layer_id
                _log(f" Layer {layer_id}:")
            param_name = ".".join(r["key"].split(".")[1:])  # strip layer prefix
            if r["status"] == "ok":
                _log(f"   ✓ {param_name:40s}  {str(r['src']):20s} → matched")
            elif r["status"] == "shape_mismatch":
                _log(f"   ✗ {param_name:40s}  src={r['src']} tgt={r['tgt']} [SHAPE MISMATCH]")
            else:
                _log(f"   – {param_name:40s}  (not in source — HSG-DET only)")
        _log(f"{'─'*60}")
        _log(f"")

        return {
            "transferred": transferred,
            "skipped": skipped,
            "total_src": len(src_sd),
            "total_tgt": len(tgt_sd),
            "matched_layers": sorted(matched_layers),
        }


# ── Auto-register ─────────────────────────────────────────────────────────────
register_arch(HSGDetPlugin())
