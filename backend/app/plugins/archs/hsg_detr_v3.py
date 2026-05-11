"""
HSG-DETR V3 architecture plugin.

Registers scale variants n/s/m/l. V3 uses SGTokenBlockV2 with Top-M
soft-hard training enabled by default and RTDETRDecoderV2 with stability
localization quality. Decoder-aware DAM is metric-only by default.
"""
from __future__ import annotations

from pathlib import Path

from ..base import ModelArchPlugin
from ..loader import register_arch


_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "hsg_detr" / "configs"

_SCALE_DESCRIPTIONS = {
    "n": ("Nano", "~10.5 M", "~19 GFLOPs", "100 queries"),
    "lean_n": ("Lean Nano", "~5.9 M", "~10.2 GFLOPs", "100 queries; hd128, no SE, hard top-k"),
    "ultra_n": ("Ultra Nano", "~5.9 M", "~10.0 GFLOPs", "100 queries; hd128, no SE, no P3-SGB"),
    "s": ("Small", "~21 M", "~42 GFLOPs", "150 queries"),
    "m": ("Medium", "~34 M", "~87 GFLOPs", "200 queries"),
    "l": ("Large", "~56 M", "~185 GFLOPs", "300 queries"),
}


class HSGDetRV3Plugin(ModelArchPlugin):
    """HSG-DETR V3 scale variant."""

    def __init__(self, scale: str):
        self._scale = scale.lower()

    @property
    def name(self) -> str:
        return f"hsg_detr_v3_{self._scale}"

    @property
    def display_name(self) -> str:
        label, params, flops, queries = _SCALE_DESCRIPTIONS[self._scale]
        return f"HSG-DETR V3 {label} ({params}, {flops})"

    @property
    def family(self) -> str:
        return "hsg_detr_v3"

    @property
    def family_display_name(self) -> str:
        return "HSG-DETR V3"

    @property
    def scale(self) -> str:
        return self._scale

    @property
    def yaml_scale(self) -> str:
        return "n" if self._scale in {"lean_n", "ultra_n"} else self._scale

    @property
    def scale_label(self) -> str:
        label, params, flops, queries = _SCALE_DESCRIPTIONS[self._scale]
        return f"{label} ({params}, {flops}) — {queries}"

    @property
    def task_type(self) -> str:
        return "detect"

    @property
    def description(self) -> str:
        if self._scale == "lean_n":
            return (
                "HSG-DETR V3 Lean-N — lightweight V3 for fair edge comparison. "
                "It keeps three-scale SGB and RTDETRDecoderV2 stability query selection, "
                "but reduces decoder hidden dim to 128, disables channel SE, lowers SGB "
                "ratios to P3/P4/P5=0.080/0.160/0.320, and uses hard top-k routing."
            )
        if self._scale == "ultra_n":
            return (
                "HSG-DETR V3 Ultra-N — most aggressive efficiency ablation near 10 GFLOPs. "
                "It uses decoder hd=128, disables channel SE and Top-M, keeps only P4/P5 SGB "
                "with lower ratios, and replaces P3 SGB with Identity."
            )
        return (
            "HSG-DETR V3 — full selective detector. It uses SGTokenBlockV2 with "
            "Top-M soft-hard sparse selection during training, channel-selective SE "
            "gating on sparse deltas, RTDETRDecoderV2 stability query selection, and "
            "decoder-aware approximate DAM metrics. KL selector loss is opt-in."
        )

    def yaml_path(self) -> Path:
        return _CONFIGS_DIR / f"hsg_detr_v3_{self._scale}.yaml"

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
            log_fn("Backbone warm-start: disabled for HSG-DETR V3 — training from scratch")
        return {"transferred": 0, "skipped": 0, "total_src": 0, "total_tgt": 0, "matched_layers": []}

    def get_config_options(self) -> dict[str, object]:
        if self._scale in {"lean_n", "ultra_n"}:
            return {
                "loc_quality_mode": "stability",
                "alpha_u": 0.3,
                "beta_s": 0.0,
                "soft_hard": False,
                "top_m_ratio": 1.0,
                "max_top_k": 768,
                "max_top_m": 1024,
                "tau": 1.0,
                "lambda_soft": 0.0,
                "eta": 0.0,
                "enable_query_metrics": True,
                "enable_gt_metrics": False,
                "enable_dam_metrics": False,
            }
        return {
            "loc_quality_mode": "stability",
            "alpha_u": 0.3,
            "beta_s": 0.0,
            "soft_hard": True,
            "top_m_ratio": 1.5,
            "max_top_k": 768,
            "max_top_m": 1024,
            "tau": 1.0,
            "lambda_soft": 0.10,
            "eta": 0.0,
            "enable_query_metrics": True,
            "enable_gt_metrics": False,
            "enable_dam_metrics": True,
        }


for _scale in ("ultra_n", "lean_n", "n", "s", "m", "l"):
    register_arch(HSGDetRV3Plugin(_scale))
