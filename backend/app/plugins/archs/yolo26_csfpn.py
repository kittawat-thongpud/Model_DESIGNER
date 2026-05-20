"""
YOLO26 + CrossScaleFPN arch plugin.

Architecture: YOLO26 backbone (layers 0–10) + CrossScaleFPN neck (layers 11–20)
              + YOLO26 E2E Detect head (layer 21).

CrossScaleFPN replaces the FPN top-down Upsample+Concat path with cross-scale
attention, making attention the PRIMARY aggregation mechanism (not additive
decoration as in CS²GA).

Two warm-start sources are supported (auto-detected by layer count):

A) yolo26n.pt (25 layers incl. Detect at 23):
   Source 0–10 → Target 0–10   backbone (COCO-pretrained)
   Source 13   → Target 12     C3k2 P4-fused  ← shape mismatch → FRESH
   Source 16   → Target 14     C3k2 P3-fused  ← shape mismatch → FRESH
   Source 17   → Target 15     Conv bottom-up ✓
   Source 19   → Target 17     C3k2 P4-out    ✓
   Source 20   → Target 18     Conv bottom-up ✓
   Source 22   → Target 20     C3k2 P5-out    ✓
   Source 23   → Target 21     Detect 80cls   ← shape mismatch → FRESH

B) yolo26_cs2ga best.pt (25 layers, Detect at 24):
   Source 0–10 → Target 0–10   backbone (IDD-adapted, BETTER)
   Source 17   → Target 15     Conv bottom-up ✓ (IDD-adapted)
   Source 19   → Target 17     C3k2 P4-out    ✓ (IDD-adapted)
   Source 20   → Target 18     Conv bottom-up ✓ (IDD-adapted)
   Source 22   → Target 20     C3k2 P5-out    ✓ (IDD-adapted)
   Source 24   → Target 21     Detect 15cls   ✓ (IDD-adapted, BEST)
   Source 23   (CS²GA)         → SKIP (different arch)

RECOMMENDATION: Use CS²GA best.pt (weight_id 27e921014373).
Backbone is IDD-specialised, PAN is IDD-tuned, and Detect head transfers
directly (15 cls). yolo26n.pt Detect has 80 cls → shape mismatch → fresh init.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..base import ModelArchPlugin
from ..loader import register_arch


_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "hsg_detr" / "configs"

_SCALE_MAP = {
    "n": "yolo26n",
}

# ── Layer remapping: source → target ─────────────────────────────────────────
# Source A: yolo26n.pt (no CS²GA, Detect at layer 23)
_REMAP_FROM_YOLO26: dict[int, int] = {
    # Backbone 0-10: direct (same index)
    13: 12,   # C3k2 P4-fused   — shape MISMATCH (src:384ch, tgt:128ch) → skipped
    16: 14,   # C3k2 P3-fused   — shape MISMATCH (src:256ch, tgt:128ch) → skipped
    17: 15,   # Conv bottom-up  ✓
    19: 17,   # C3k2 P4-out     ✓
    20: 18,   # Conv bottom-up  ✓
    22: 20,   # C3k2 P5-out     ✓
    23: 21,   # Detect 80cls    — shape MISMATCH (80 vs 15 cls) → skipped
}

# Source B: yolo26_cs2ga best.pt (CS²GA at 23, Detect at 24)
_REMAP_FROM_CS2GA: dict[int, int] = {
    # Backbone 0-10: direct (same index, IDD-adapted ✓)
    # C3k2 13→12 and 16→14: shape mismatch → skipped (same reason as yolo26n)
    17: 15,   # Conv bottom-up  ✓ IDD-adapted
    19: 17,   # C3k2 P4-out     ✓ IDD-adapted
    20: 18,   # Conv bottom-up  ✓ IDD-adapted
    22: 20,   # C3k2 P5-out     ✓ IDD-adapted
    24: 21,   # Detect 15cls    ✓ IDD-adapted  ← KEY ADVANTAGE over yolo26n
    # 23 = CS²GA → skip (different architecture)
}

# Target layers always initialised fresh (CrossScaleFPN)
_FRESH_LAYERS = {11, 13}


class YOLO26CSFPNPlugin(ModelArchPlugin):
    """YOLO26 with CrossScaleFPN — attention as core feature aggregation."""

    def __init__(self, scale: str):
        self._scale = scale.lower()

    @property
    def name(self) -> str:
        return f"yolo26_csfpn_{self._scale}"

    @property
    def display_name(self) -> str:
        return f"YOLO26-CSFPN {self._scale.upper()}"

    @property
    def family(self) -> str:
        return "yolo26_csfpn"

    @property
    def family_display_name(self) -> str:
        return "YOLO26 + CrossScaleFPN"

    @property
    def scale(self) -> str:
        return self._scale

    @property
    def scale_label(self) -> str:
        return f"{self._scale.upper()}-scale — YOLO26 base + CrossScaleFPN neck"

    @property
    def task_type(self) -> str:
        return "detect"

    @property
    def description(self) -> str:
        return (
            "YOLO26 backbone (pretrained) with CrossScaleFPN replacing the FPN "
            "top-down path. Instead of Upsample+Concat, cross-scale attention "
            "fuses P5→P4 and P4→P3 by learning which positions are relevant. "
            "Attention is the PRIMARY aggregation mechanism — 100% gradient "
            "flows through it (vs CS²GA where 93% bypasses). "
            "~90% of yolo26n.pt weights transfer; only the two CrossScaleFPN "
            "modules (layers 11 and 13) train from scratch as pure identity."
        )

    def yaml_path(self) -> Path:
        return _CONFIGS_DIR / f"yolo26_csfpn_{self._scale}.yaml"

    def register_modules(self) -> None:
        try:
            import hsg_detr.nn  # noqa: F401 — registers CrossScaleFPN + CrossScaleSGA
        except ImportError as e:
            raise ImportError(
                "Could not import hsg_detr package. Ensure backend/hsg_detr/ "
                "is on PYTHONPATH."
            ) from e

    def pretrain_key(self) -> str | None:
        return None  # warm_start handles download directly

    def get_config_options(self) -> dict[str, object]:
        # AdamW recommended for attention-based neck (anisotropic loss surface).
        # Lower LR than SGD; longer warmup so attention specialises before LR drops.
        return {
            "optimizer": "AdamW",
            "lr0": 0.0005,
            "lrf": 0.1,
            "momentum": 0.9,        # AdamW beta1
            "weight_decay": 0.01,   # higher than SGD — attention overfits faster
            "warmup_epochs": 10,    # critical: attention starts near-uniform
            "cos_lr": True,
            "amp": True,
            "ema": True,
            # Loss weights same as YOLO26-N official recipe
            "box": 5.63,
            "cls": 0.56,
            "dfl": 9.04,
            # Augmentation (same as CS²GA baseline for fair comparison)
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

    def warm_start(self, model, log_fn=None, model_scale: str | None = None,
                   pretrained_path: str | None = None) -> dict:
        """Transfer weights into backbone + PAN + Detect.

        Auto-detects source type by inspecting the checkpoint's layer count:
          - 24 layers (yolo26n, Detect@23): uses _REMAP_FROM_YOLO26
          - 25 layers (yolo26_cs2ga, Detect@24): uses _REMAP_FROM_CS2GA

        Recommended source: yolo26_cs2ga best.pt — backbone is IDD-adapted and
        Detect head transfers directly (15cls). yolo26n Detect has 80cls → fresh.

        Args:
            pretrained_path: optional path to a .pt file to use instead of the
                             default yolo26n download. Pass the resolved path of
                             weight_id 27e921014373 to use CS²GA best.pt.
        """
        def _log(msg: str):
            if log_fn:
                log_fn(msg)

        scale = (model_scale or self._scale).lower()

        # ── 1. Locate checkpoint ──────────────────────────────────────────
        if pretrained_path and Path(pretrained_path).exists():
            src_pt = Path(pretrained_path)
            _log(f"Warm-start: using provided checkpoint: {src_pt}")
        else:
            yolo_key = _SCALE_MAP.get(scale)
            if yolo_key is None:
                _log(f"Warm-start: unknown scale '{scale}' — skipping")
                return _empty()
            _log(f"Warm-start: YOLO26-CSFPN scale='{scale}' ← {yolo_key}.pt")
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

        # ── 3. Auto-detect source type and choose remap table ────────────
        # Count max layer index in source to distinguish yolo26n vs CS²GA
        max_src_layer = -1
        for k in src_sd:
            part = k.split(".")[0]
            try:
                max_src_layer = max(max_src_layer, int(part))
            except ValueError:
                pass

        if max_src_layer >= 24:
            # CS²GA source (25 layers, Detect@24) — IDD-adapted, RECOMMENDED
            layer_remap = _REMAP_FROM_CS2GA
            source_label = "yolo26_cs2ga (IDD-adapted)"
        else:
            # yolo26n source (24 layers, Detect@23)
            layer_remap = _REMAP_FROM_YOLO26
            source_label = "yolo26n (COCO)"

        _log(f"Warm-start: detected source = {source_label} (max_layer={max_src_layer})")

        # ── 4. Build remapped source dict ─────────────────────────────────
        src_remap: dict[str, torch.Tensor] = {}
        for k, v in src_sd.items():
            layer_str = k.split(".")[0]
            try:
                layer_idx = int(layer_str)
            except ValueError:
                src_remap[k] = v
                continue

            if layer_idx in layer_remap:
                tgt_idx = layer_remap[layer_idx]
                new_key = str(tgt_idx) + k[len(layer_str):]
                src_remap[new_key] = v
            else:
                # Backbone (0-10) and unmapped → keep as-is
                src_remap[k] = v

        # ── 5. Get target state_dict ──────────────────────────────────────
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

        # ── 6. Transfer shape-matching keys ──────────────────────────────
        transferred = skipped = 0
        matched_layers: set[str] = set()
        skipped_layers: set[str] = set()

        new_tgt = dict(tgt_sd)

        for key, tgt_t in tgt_sd.items():
            layer_str = key.split(".")[0]
            try:
                layer_idx = int(layer_str)
            except ValueError:
                layer_idx = -1

            # Fresh layers — always skip
            if layer_idx in _FRESH_LAYERS:
                skipped += 1
                skipped_layers.add(layer_str)
                continue

            if key not in src_remap:
                skipped += 1
                continue

            src_t = src_remap[key]
            if src_t.shape == tgt_t.shape:
                new_tgt[key] = src_t.to(tgt_t.dtype)
                transferred += 1
                matched_layers.add(layer_str)
            else:
                skipped += 1
                skipped_layers.add(layer_str)

        # ── 7. Apply ──────────────────────────────────────────────────────
        restored = {
            (_PFX + k if not k.startswith(_PFX) else k): v
            for k, v in new_tgt.items()
        }
        tgt_nn.load_state_dict(restored, strict=False)

        # ── 8. Report ─────────────────────────────────────────────────────
        total = len(tgt_sd)
        pct = transferred / total * 100 if total else 0
        _log("")
        _log("─" * 64)
        _log(f" Warm-start: {yolo_key}.pt → YOLO26-CSFPN-{scale.upper()}")
        _log("─" * 64)
        _log(f" Source  : {yolo_key}.pt  ({len(src_sd)} tensors)")
        _log(f" Target  : YOLO26-CSFPN-{scale} ({total} tensors)")
        _log(f" Transferred : {transferred}  ({pct:.1f}%)")
        _log(f" Skipped     : {skipped}")
        _log(f" Fresh layers: {sorted(_FRESH_LAYERS)} (CrossScaleFPN, zero-init)")
        _log(f" Layers transferred: {sorted(matched_layers, key=_layer_sort)}")
        _log(f" Layers skipped    : {sorted(skipped_layers, key=_layer_sort)}")
        _log("─" * 64)
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
    register_arch(YOLO26CSFPNPlugin(_scale))
