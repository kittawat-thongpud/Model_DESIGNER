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
    "n": ("Nano",    "~5.9 M",  "~9.6 GFLOPs",  "100 queries"),
    "s": ("Small",   "~20.9 M", "~41.8 GFLOPs", "150 queries"),
    "m": ("Medium",  "~33.8 M", "~87.1 GFLOPs","200 queries"),
    "l": ("Large",   "~56.3 M","~184.8 GFLOPs","300 queries"),
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
            "HSG-DETR — TGSR (Token-Guided Spatial Recalibration) SGB encoder "
            "feeding an RT-DETR decoder. Saliency top-k drives a k×k self-"
            "attention head whose output multiplicatively recalibrates the "
            "feature map: a tanh-bounded channel gate (always active, fast "
            "warmup) and an alpha-scheduled spatial refinement. The "
            "multiplicative paradigm has no `gamma → 0` bypass and stays in a "
            "stable AMP/bf16 range. Scratch training recommended; legacy "
            "additive/gamma checkpoints are auto-stripped on load."
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
        """Warm-start disabled for stability debugging."""
        if log_fn:
            log_fn(
                "Backbone warm-start: disabled for HSG-DETR stability debug — "
                "training from scratch"
            )

        return {
            "transferred": 0,
            "skipped": 0,
            "total_src": 0,
            "total_tgt": 0,
            "matched_layers": [],
        }


# ── Auto-register all four scales ──────────────────────────────────────────
for _scale in ("n", "s", "m", "l"):
    register_arch(HSGDetRPlugin(_scale))
