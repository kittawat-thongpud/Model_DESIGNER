"""
HSG-DETR V3-CS²GA architecture plugin.

Registers N-scale variant with CrossScaleSGA cross-scale sparse attention.
Uses standard YOLO Detect head for baseline comparison.
"""
from __future__ import annotations

from pathlib import Path

from ..base import ModelArchPlugin
from ..loader import register_arch


_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "hsg_detr" / "configs"


class HSGDetRV3CS2GAPlugin(ModelArchPlugin):
    """HSG-DETR V3-CS²GA N-scale variant."""

    def __init__(self, scale: str):
        self._scale = scale.lower()

    @property
    def name(self) -> str:
        return f"hsg_detr_v3_cs2ga_{self._scale}"

    @property
    def display_name(self) -> str:
        return f"HSG-DETR V3-CS²GA {self._scale.upper()}"

    @property
    def family(self) -> str:
        return "hsg_detr_v3_cs2ga"

    @property
    def family_display_name(self) -> str:
        return "HSG-DETR V3-CS²GA"

    @property
    def scale(self) -> str:
        return self._scale

    @property
    def scale_label(self) -> str:
        return f"N-scale — CS²GA cross-scale attention"

    @property
    def task_type(self) -> str:
        return "detect"

    @property
    def description(self) -> str:
        return (
            "HSG-DETR V3-CS²GA — Cross-Scale Sparse Global Attention. "
            "Replaces per-scale SGB with a single module that processes P3/P4/P5 "
            "simultaneously with learned scale embeddings and gated residuals. "
            "Uses standard YOLO Detect head for fair baseline comparison."
        )

    def yaml_path(self) -> Path:
        return _CONFIGS_DIR / f"hsg_detr_v3_cs2ga_{self._scale}.yaml"

    def register_modules(self) -> None:
        try:
            import hsg_detr.nn  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Could not import hsg_detr package. Ensure backend/hsg_detr/ is on PYTHONPATH."
            ) from e

    def pretrain_key(self) -> str | None:
        return None

    def warm_start(self, model, log_fn=None, model_scale: str | None = None) -> dict:
        if log_fn:
            log_fn("Backbone warm-start: disabled for HSG-DETR V3-CS²GA — training from scratch")
        return {"transferred": 0, "skipped": 0, "total_src": 0, "total_tgt": 0, "matched_layers": []}

    def get_config_options(self) -> dict[str, object]:
        return {
            "shared_dim": 256,
            "ratio_p3": 0.05,
            "ratio_p4": 0.10,
            "ratio_p5": 0.25,
            "enable_metrics": True,
            # AMP disabled: GradScaler starts at 65536 → overflows backbone
            # in early batches before scale backs down. Safe to re-enable once
            # training is stable (monitor grad/has_nan + amp_scale).
            "amp": False,
        }


for _scale in ("n",):
    register_arch(HSGDetRV3CS2GAPlugin(_scale))
