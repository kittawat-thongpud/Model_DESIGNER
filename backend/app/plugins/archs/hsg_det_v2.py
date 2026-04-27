"""
HSG-DET V2 Model Architecture Plugin.

CNN hybrid detector with SparseGlobalTokenBlock at P3/P4/P5.
Registers four scale variants: n, s, m, l.

Registers as: "hsg_det_v2_n", "hsg_det_v2_s", "hsg_det_v2_m", "hsg_det_v2_l"
"""
from __future__ import annotations
from pathlib import Path

from .hsg_det import HSGDetPlugin
from ..loader import register_arch


_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "hsg_det" / "configs"

_SCALE_DESCRIPTIONS = {
    "n": ("Nano",   "~3.5 M",  "~8 GFLOPs"),
    "s": ("Small",  "~12 M",   "~28 GFLOPs"),
    "m": ("Medium", "~26 M",   "~60 GFLOPs"),
    "l": ("Large",  "~44 M",   "~110 GFLOPs"),
}


class HSGDetV2Plugin(HSGDetPlugin):
    """Plugin for one HSG-DET V2 scale variant (n, s, m, or l)."""

    def __init__(self, scale: str):
        self._scale = scale.lower()

    @property
    def name(self) -> str:
        return f"hsg_det_v2_{self._scale}"

    @property
    def display_name(self) -> str:
        label, params, flops = _SCALE_DESCRIPTIONS[self._scale]
        return f"HSG-DET V2 {label} ({params}, {flops})"

    @property
    def family(self) -> str:
        return "hsg_det_v2"

    @property
    def family_display_name(self) -> str:
        return "HSG-DET V2"

    @property
    def scale(self) -> str:
        return self._scale

    @property
    def scale_label(self) -> str:
        label, params, flops = _SCALE_DESCRIPTIONS[self._scale]
        return f"{label} ({params}, {flops})"

    @property
    def description(self) -> str:
        return (
            "HSG-DET V2: YOLOv8 CSP backbone + FPN/PAN neck with SparseGlobalTokenBlock "
            "at P3/P4/P5. Ratio-based token selection scales with imgsz. "
            "Sigmoid gate (init=0.5) contributes from epoch 1. "
            "Warm-start from matching YOLOv8 backbone weights."
        )

    def yaml_path(self) -> Path:
        return _CONFIGS_DIR / f"hsg_det_{self._scale}_v2.yaml"


for _s in ("n", "s", "m", "l"):
    register_arch(HSGDetV2Plugin(_s))
