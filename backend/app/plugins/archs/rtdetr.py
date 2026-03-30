"""
RT-DETR Model Architecture Plugins.

Wraps the Ultralytics built-in RT-DETR-l and RT-DETR-x YAML configs as
ModelArchPlugin instances so the training pipeline discovers them alongside
other custom architectures.

All modules used (HGStem, HGBlock, AIFI, RepC3, RTDETRDecoder) are already
part of ultralytics.nn.modules — no custom module registration needed.

Registers as: "rtdetr_l", "rtdetr_x"
"""
from __future__ import annotations
from pathlib import Path

from ..base import ModelArchPlugin
from ..loader import register_arch


def _ultra_rtdetr_yaml(filename: str) -> Path:
    """Resolve an RT-DETR YAML path from the installed Ultralytics package."""
    try:
        import ultralytics
        root = Path(ultralytics.__file__).resolve().parent
        p = root / "cfg" / "models" / "rt-detr" / filename
        if p.exists():
            return p
    except ImportError:
        pass
    raise FileNotFoundError(
        f"Cannot find {filename} inside the Ultralytics package. "
        "Ensure ultralytics is installed in the active environment."
    )


_RTDETR_SCALES = {
    "l":         ("Large",       "~31.8 M",  "~108 GFLOPs",  "HGNetv2-L backbone",  "rtdetr-l.yaml"),
    "x":         ("Extra-Large", "~67.4 M",  "~234 GFLOPs",  "HGNetv2-X backbone",  "rtdetr-x.yaml"),
    "resnet50":  ("ResNet-50",   "~42.3 M",  "~136 GFLOPs",  "ResNet-50 backbone",  "rtdetr-resnet50.yaml"),
    "resnet101": ("ResNet-101",  "~76.0 M",  "~259 GFLOPs",  "ResNet-101 backbone", "rtdetr-resnet101.yaml"),
}


class RTDETRPlugin(ModelArchPlugin):
    """Plugin for one RT-DETR scale variant (L or X)."""

    def __init__(self, scale: str):
        self._scale = scale.lower()

    @property
    def name(self) -> str:
        return f"rtdetr_{self._scale}"

    @property
    def display_name(self) -> str:
        label, params, flops, backbone, _ = _RTDETR_SCALES[self._scale]
        return f"RT-DETR {label} ({params}, {flops})"

    @property
    def family(self) -> str:
        return "rtdetr"

    @property
    def family_display_name(self) -> str:
        return "RT-DETR"

    @property
    def scale(self) -> str:
        return self._scale

    @property
    def scale_label(self) -> str:
        label, params, flops, backbone, _ = _RTDETR_SCALES[self._scale]
        return f"{label} ({params}, {flops}) — {backbone}"

    @property
    def task_type(self) -> str:
        return "detect"

    @property
    def description(self) -> str:
        return (
            "RT-DETR — hybrid CNN-Transformer detector. "
            "HGNetv2 backbone + AIFI Transformer encoder + RTDETRDecoder. "
            "No NMS at inference; end-to-end predictions. "
            "Built-in Ultralytics architecture — no extra dependencies."
        )

    def yaml_path(self) -> Path:
        _, _, _, _, yaml_file = _RTDETR_SCALES[self._scale]
        return _ultra_rtdetr_yaml(yaml_file)

    def register_modules(self) -> None:
        pass

    def pretrain_key(self) -> str | None:
        return None


# ── Auto-register all available scale variants ───────────────────────────────
for _scale in _RTDETR_SCALES:
    register_arch(RTDETRPlugin(_scale))
