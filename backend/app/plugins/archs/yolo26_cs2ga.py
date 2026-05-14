"""
YOLO26 + CS²GA feature-enhancer plugin.

Architecture: YOLO26 backbone/neck (layers 0–22) + CrossScaleSGA neck
enhancer (layer 23) + YOLO26 E2E Detect head (layer 24).

Warm-start strategy:
  • Layers 0–22 (backbone + neck): copied directly from yolo26n.pt by key match
  • Layer 24 (Detect): remapped from source layer 23 (same Detect, shift +1)
  • Layer 23 (CrossScaleSGA): skipped — init fresh with LayerScale=1e-4

This means 100% of the pretrained backbone/neck transfers, and CS²GA is the
only component that trains from scratch. Any mAP gain is attributable to CS²GA.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..base import ModelArchPlugin
from ..loader import register_arch


_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "hsg_detr" / "configs"

_SCALE_MAP = {
    "n": "yolo26n",
    "s": "yolo26s",
    "m": "yolo26m",
    "l": "yolo26l",
    "x": "yolo26x",
}

# Layer index of CS²GA in our YAML — weights in source don't exist for this
_CS2GA_LAYER = 23
# Detect layer in our YAML vs source (source=23, ours=24)
_DETECT_SRC_LAYER = 23
_DETECT_TGT_LAYER = 24


class YOLO26CS2GAPlugin(ModelArchPlugin):
    """YOLO26-N with CS²GA cross-scale feature enhancer."""

    def __init__(self, scale: str):
        self._scale = scale.lower()

    @property
    def name(self) -> str:
        return f"yolo26_cs2ga_{self._scale}"

    @property
    def display_name(self) -> str:
        return f"YOLO26-CS²GA {self._scale.upper()}"

    @property
    def family(self) -> str:
        return "yolo26_cs2ga"

    @property
    def family_display_name(self) -> str:
        return "YOLO26 + CS²GA"

    @property
    def scale(self) -> str:
        return self._scale

    @property
    def scale_label(self) -> str:
        return f"{self._scale.upper()}-scale — YOLO26 base + CS²GA enhancer"

    @property
    def task_type(self) -> str:
        return "detect"

    @property
    def description(self) -> str:
        return (
            "YOLO26 backbone/neck (pretrained) with CS²GA cross-scale sparse attention "
            "inserted as a plug-in feature enhancer after the FPN/PAN outputs. "
            "CS²GA is the only new component — all other weights transfer from yolo26n.pt. "
            "Uses YOLO26's E2E one-to-one + one-to-many loss with decay schedule."
        )

    def yaml_path(self) -> Path:
        return _CONFIGS_DIR / f"yolo26_cs2ga_{self._scale}.yaml"

    def register_modules(self) -> None:
        try:
            import hsg_detr.nn  # noqa: F401 — registers CrossScaleSGA into parse_model
        except ImportError as e:
            raise ImportError(
                "Could not import hsg_detr package. Ensure backend/hsg_detr/ is on PYTHONPATH."
            ) from e

    def pretrain_key(self) -> str | None:
        # warm_start handles download directly — no separate pretrain step needed
        return None

    def get_config_options(self) -> dict[str, object]:
        # YOLO26 official training recipe for N-scale (from Ultralytics docs)
        # https://docs.ultralytics.com/guides/yolo26-training-recipe
        return {
            "amp": False,  # GradScaler 65536 still overflows backbone early batches
            "enable_metrics": True,
            # YOLO26-N official optimizer settings
            "optimizer": "MuSGD",
            "lr0": 0.0054,
            "lrf": 0.0495,
            "momentum": 0.947,
            "weight_decay": 0.00064,
            "warmup_epochs": 0.98,
            # YOLO26-N loss weights
            "box": 5.63,
            "cls": 0.56,
            "dfl": 9.04,
            # YOLO26-N augmentation
            "mosaic": 0.909,
            "mixup": 0.012,
            "copy_paste": 0.075,
            "scale": 0.562,
            "fliplr": 0.606,
            "degrees": 1.11,
            "shear": 1.46,
            "translate": 0.071,
            "hsv_h": 0.014,
            "hsv_s": 0.645,
            "hsv_v": 0.566,
            "bgr": 0.106,
            "close_mosaic": 10,
        }

    # ------------------------------------------------------------------ #

    def warm_start(self, model, log_fn=None, model_scale: str | None = None) -> dict:
        """Transfer YOLO26 pretrained weights into layers 0–22 + Detect.

        Layer mapping:
          source 0–22  → target 0–22   (backbone + neck, direct key match)
          source 23.*  → target 24.*   (Detect head, layer-index remap)
          target 23.*  → skip          (CrossScaleSGA, init fresh)
        """
        def _log(msg: str):
            if log_fn:
                log_fn(msg)

        scale = (model_scale or self._scale).lower()
        yolo_key = _SCALE_MAP.get(scale)
        if yolo_key is None:
            _log(f"Warm-start: unknown scale '{scale}' — skipping")
            return _empty()

        _log(f"Warm-start: YOLO26-CS²GA scale='{scale}' → source={yolo_key}.pt")

        # ── 1. Locate / download checkpoint ──────────────────────────────
        src_pt = _find_or_download(yolo_key, _log)
        if src_pt is None:
            return _empty()

        # ── 2. Load source state_dict ─────────────────────────────────────
        try:
            raw = torch.load(src_pt, map_location="cpu", weights_only=False)
        except Exception as e:
            _log(f"Warm-start: failed to load {src_pt} ({e}) — skipping")
            return _empty()

        src_sd_raw: dict = {}
        if isinstance(raw, dict) and "model" in raw:
            obj = raw["model"]
            src_sd_raw = obj.float().state_dict() if hasattr(obj, "state_dict") else {}
        elif isinstance(raw, dict):
            src_sd_raw = raw

        _PFX = "model."
        src_sd: dict[str, torch.Tensor] = {
            (k[len(_PFX):] if k.startswith(_PFX) else k): v
            for k, v in src_sd_raw.items()
            if isinstance(v, torch.Tensor)
        }

        # ── 3. Build remapped source: shift Detect 23.* → 24.* ───────────
        src_remap: dict[str, torch.Tensor] = {}
        src_str = str(_DETECT_SRC_LAYER) + "."
        tgt_str = str(_DETECT_TGT_LAYER) + "."
        for k, v in src_sd.items():
            if k.startswith(src_str):
                src_remap[tgt_str + k[len(src_str):]] = v  # 23.x → 24.x
            else:
                src_remap[k] = v                           # all others unchanged

        # ── 4. Get target state_dict ──────────────────────────────────────
        try:
            tgt_nn = model.model
            tgt_sd_raw = tgt_nn.state_dict()
        except Exception as e:
            _log(f"Warm-start: cannot read target state_dict ({e}) — skipping")
            return _empty()

        tgt_sd: dict[str, torch.Tensor] = {
            (k[len(_PFX):] if k.startswith(_PFX) else k): v
            for k, v in tgt_sd_raw.items()
        }

        # ── 5. Transfer shape-matching keys ──────────────────────────────
        transferred = skipped = 0
        matched_layers: set[str] = set()
        skipped_layers: set[str] = set()
        compat: list[dict] = []

        new_tgt = dict(tgt_sd)
        cs2ga_prefix = str(_CS2GA_LAYER) + "."

        for key, tgt_t in tgt_sd.items():
            # Skip CS²GA — always init fresh
            if key.startswith(cs2ga_prefix):
                skipped += 1
                skipped_layers.add(str(_CS2GA_LAYER))
                compat.append({"key": key, "src": None, "tgt": tuple(tgt_t.shape),
                                "status": "cs2ga_skip"})
                continue

            if key not in src_remap:
                skipped += 1
                compat.append({"key": key, "src": None, "tgt": tuple(tgt_t.shape),
                                "status": "not_in_src"})
                continue

            src_t = src_remap[key]
            if src_t.shape == tgt_t.shape:
                new_tgt[key] = src_t.to(tgt_t.dtype)
                transferred += 1
                matched_layers.add(key.split(".")[0])
                compat.append({"key": key, "src": tuple(src_t.shape),
                                "tgt": tuple(tgt_t.shape), "status": "ok"})
            else:
                skipped += 1
                skipped_layers.add(key.split(".")[0])
                compat.append({"key": key, "src": tuple(src_t.shape),
                                "tgt": tuple(tgt_t.shape), "status": "shape_mismatch"})

        # ── 6. Apply ──────────────────────────────────────────────────────
        restored = {(_PFX + k if not k.startswith(_PFX) else k): v
                    for k, v in new_tgt.items()}
        tgt_nn.load_state_dict(restored, strict=False)

        # ── 7. Report ─────────────────────────────────────────────────────
        total = len(tgt_sd)
        pct = transferred / total * 100 if total else 0
        _log("")
        _log("─" * 62)
        _log(f" Warm-start: {yolo_key}.pt → YOLO26-CS²GA-{scale.upper()}")
        _log("─" * 62)
        _log(f" Source  : {yolo_key}.pt  ({len(src_sd)} tensors)")
        _log(f" Target  : YOLO26-CS²GA-{scale} ({total} tensors)")
        _log(f" Transferred : {transferred}  ({pct:.1f}%)")
        _log(f" Skipped     : {skipped}  (CS²GA=fresh, rest=shape-mismatch)")
        _log(f" Layers transferred : {sorted(matched_layers, key=_layer_sort)}")
        _log(f" Layers skipped     : {sorted(skipped_layers, key=_layer_sort)}")
        _log("─" * 62)
        _log("")

        return {
            "transferred": transferred,
            "skipped": skipped,
            "total_src": len(src_sd),
            "total_tgt": total,
            "matched_layers": sorted(matched_layers, key=_layer_sort),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty() -> dict:
    return {"transferred": 0, "skipped": 0, "total_src": 0,
            "total_tgt": 0, "matched_layers": []}


def _layer_sort(x: str) -> int:
    try:
        return int(x)
    except ValueError:
        return 999


def _find_or_download(yolo_key: str, log_fn) -> Path | None:
    candidates = [
        Path.home() / ".config" / "Ultralytics" / f"{yolo_key}.pt",
        Path.home() / "AppData" / "Roaming" / "Ultralytics" / f"{yolo_key}.pt",
        Path(f"{yolo_key}.pt"),
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found:
        log_fn(f"Warm-start: found {found}")
        return found

    log_fn(f"Warm-start: {yolo_key}.pt not in cache — downloading...")
    try:
        from ultralytics import YOLO as _YOLO
        _YOLO(f"{yolo_key}.pt")
        found = next((p for p in candidates if p.exists()), None)
        if found:
            return found
        log_fn(f"Warm-start: cannot locate {yolo_key}.pt after download — skipping")
    except Exception as e:
        log_fn(f"Warm-start: download failed ({e}) — training from scratch")
    return None


# ── Auto-register ─────────────────────────────────────────────────────────────
for _scale in ("n",):
    register_arch(YOLO26CS2GAPlugin(_scale))
