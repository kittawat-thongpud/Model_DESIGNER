"""
HSG-DET v2 Model Architecture Plugin.

Variant of HSG-DET Medium with SparseGlobalBlockGated on P3/P4/P5.
Registers as: "hsg_det_v2"
"""
from __future__ import annotations
from pathlib import Path

from .hsg_det import HSGDetPlugin
from ..loader import register_arch


class HSGDetV2Plugin(HSGDetPlugin):
    @property
    def name(self) -> str:
        return "hsg_det_v2"

    @property
    def display_name(self) -> str:
        return "HSG-DET v2 (P3/P4/P5 SGBG)"

    @property
    def family(self) -> str:
        return "hsg_det_v2"

    @property
    def family_display_name(self) -> str:
        return "HSG-DET v2"

    @property
    def scale(self) -> str:
        return "m"

    @property
    def scale_label(self) -> str:
        return "Medium v2 (P3/P4/P5 SGBG)"

    @property
    def description(self) -> str:
        return (
            "HSG-DET v2 adds SparseGlobalBlockGated at P3 in addition to P4/P5 "
            "for stronger small-object context modeling. Warm-start from YOLOv8-m backbone weights."
        )

    def yaml_path(self) -> Path:
        backend_root = Path(__file__).resolve().parents[3]
        return backend_root / "hsg_det" / "configs" / "hsg_det_m_v2.yaml"


register_arch(HSGDetV2Plugin())
