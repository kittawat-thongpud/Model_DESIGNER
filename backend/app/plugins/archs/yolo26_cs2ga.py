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
from ..training_profile import TrainingProfile, TrainingConfigField, SelectOption


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
            "enable_deep_metrics": True,  # Enable CS²GA debug metrics collection
            "nan_retries": 16,  # Higher retry count for CS²GA AMP sensitivity
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

    def get_config_fields(self) -> list[TrainingConfigField]:
        """Arch-specific config fields rendered as a "Model" tab in the UI.

        Values flow into the training config dict unchanged and are read by
        the trainer (training_mode → profile resolution + freeze,
        cs2ga_lr_* → build_optimizer LR multipliers).
        """
        return [
            # ── Training Mode ─────────────────────────────────────────────
            TrainingConfigField(
                key="training_mode",
                label="Training Mode",
                field_type="select",
                default="full",
                description=(
                    "Controls which parts of the model are trained. "
                    "'full' trains everything end-to-end; 'attention_only' freezes the "
                    "YOLO26 backbone/neck so only CS²GA learns; 'joint_finetune' lets "
                    "backbone adapt slowly while CS²GA drives the gradient."
                ),
                group="Training Mode",
                options=[
                    SelectOption("full", "Full Training",
                                 "All layers unfrozen — standard end-to-end training"),
                    SelectOption("attention_only", "Attention-Only",
                                 "Freeze backbone + neck (layers 0-22); train only CS²GA + Detect head. "
                                 "Isolates CS²GA contribution."),
                    SelectOption("joint_finetune", "Joint Fine-Tune",
                                 "All layers unfrozen; backbone gets 0.2× LR, CS²GA gets 15×. "
                                 "Best as phase-2 after Attention-Only."),
                ],
            ),

            # ── CS²GA Learning Rate multipliers ──────────────────────────
            TrainingConfigField(
                key="cs2ga_lr_sparse",
                label="Projection LR ×",
                field_type="slider",
                default=10.0,
                description=(
                    "LR multiplier for CS²GA projection layers (q/k/v/out_proj, scale_embed) "
                    "relative to base lr0. Higher values compensate for backbone gradient dominance."
                ),
                group="CS²GA Learning Rate",
                min_val=1.0, max_val=50.0, step=0.5, unit="×",
            ),
            TrainingConfigField(
                key="cs2ga_lr_gamma",
                label="LayerScale LR ×",
                field_type="slider",
                default=20.0,
                description=(
                    "LR multiplier for LayerScale (ls_p3/p4/p5) parameters. "
                    "LayerScale gates how much attention contributes to the residual — "
                    "needs higher LR to grow from the small init value."
                ),
                group="CS²GA Learning Rate",
                min_val=1.0, max_val=50.0, step=0.5, unit="×",
            ),
            TrainingConfigField(
                key="cs2ga_lr_norm",
                label="Norm LR ×",
                field_type="slider",
                default=5.0,
                description=(
                    "LR multiplier for CS²GA pre-norm and output-norm layers. "
                    "Elevated so norms adapt at the same rate as the attention projections."
                ),
                group="CS²GA Learning Rate",
                min_val=1.0, max_val=20.0, step=0.5, unit="×",
            ),
            TrainingConfigField(
                key="cs2ga_lr_backbone",
                label="Backbone LR ×",
                field_type="slider",
                default=1.0,
                description=(
                    "LR multiplier for the backbone/neck base layers. "
                    "Set below 1.0 in joint_finetune mode to slow backbone adaptation "
                    "and let CS²GA drive the gradient."
                ),
                group="CS²GA Learning Rate",
                min_val=0.01, max_val=1.0, step=0.01, unit="×",
            ),

        ]

    # ------------------------------------------------------------------ #

    def get_training_profiles(self) -> list[TrainingProfile]:
        """Named training profiles for YOLO26-CS²GA.

        Profile descriptions
        --------------------
        full (default)
            Standard end-to-end training with all layers unfrozen.
            Uses the official YOLO26-N recipe + elevated CS²GA LR groups
            (10-20× base) to compensate for backbone-gradient dominance.

        attention_only
            Freeze YOLO26 backbone + neck (layers 0-22); train ONLY the
            CS²GA block (layer 23) and the Detect head (layer 24).
            This isolates the CS²GA contribution: any mAP delta from a
            run with this profile is attributable exclusively to CS²GA.
            Useful as a first phase before joint fine-tuning.

        joint_finetune
            All layers unfrozen, but backbone/neck get 0.2× LR while
            CS²GA projections get 15× LR.  Allows gradual backbone
            adaptation while letting CS²GA drive the gradient.  Ideal
            as a second phase after attention_only.
        """
        # Backbone + neck layers in the YAML are indices 0-22.
        # Ultralytics stores them as model.{i}.* in state_dict / named_parameters.
        backbone_neck_prefixes = [f"model.model.{i}." for i in range(23)]

        return [
            TrainingProfile(
                name="full",
                display_name="Full Training",
                description=(
                    "Train all layers end-to-end with the official YOLO26-N recipe. "
                    "CS²GA projections and LayerScale params get 10-20× elevated LR to "
                    "compensate for backbone gradient dominance."
                ),
                is_default=True,
                badge_color="blue",
                tags=["all layers", "recommended"],
            ),
            TrainingProfile(
                name="attention_only",
                display_name="Attention-Only",
                description=(
                    "Freeze the YOLO26 backbone and neck (layers 0-22); train only the "
                    "CS²GA attention block (layer 23) and the Detect head (layer 24). "
                    "Use this to isolate the CS²GA contribution — any mAP gain comes "
                    "exclusively from the attention mechanism."
                ),
                is_default=False,
                badge_color="orange",
                freeze_param_prefixes=backbone_neck_prefixes,
                unfreeze_param_prefixes=["model.model.23.", "model.model.24."],
                lr_group_overrides={
                    "sgb_sparse":     15.0,   # CS²GA projections
                    "sgb_gamma":      25.0,   # LayerScale params
                    "sgb_norm_group": 8.0,    # CS²GA pre_norm layers
                },
                config_overrides={
                    "lr0": 0.001,            # lower base LR — backbone frozen, CS²GA drives
                    "lrf": 0.05,
                    "warmup_epochs": 0.5,    # shorter warmup when backbone is fixed
                    "epochs": 50,            # shorter run — only attention block trains
                },
                tags=["freeze backbone", "attention isolation", "fast"],
            ),
            TrainingProfile(
                name="joint_finetune",
                display_name="Joint Fine-Tune",
                description=(
                    "All layers unfrozen. Backbone/neck get 0.2× LR to adapt slowly "
                    "while CS²GA projections get 15× LR to drive the gradient. "
                    "Use this as phase-2 after an attention_only run, or as the main "
                    "training mode when you want both backbone and attention to co-adapt."
                ),
                is_default=False,
                badge_color="green",
                lr_group_overrides={
                    "base":           0.2,    # slow backbone adaptation
                    "sgb_sparse":     15.0,   # CS²GA projections lead
                    "sgb_gamma":      25.0,   # LayerScale
                    "sgb_norm_group": 8.0,    # CS²GA norms
                    "norm_bias":      0.5,    # backbone norms — slightly slower
                },
                tags=["co-adapt", "phase 2"],
            ),
        ]

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
