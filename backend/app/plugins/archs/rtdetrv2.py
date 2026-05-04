"""
RT-DETRv2 upstream architecture plugin.

This registers the official lyuwenyu/RT-DETR RT-DETRv2 variants as a separate
architecture family.  It deliberately does not import Mamba-YOLO's vendored
``ultralytics.models.rtdetr`` package; Mamba-YOLO only needs its own module
loader, while RT-DETRv2 is tracked as an upstream architecture here.
"""
from __future__ import annotations

from pathlib import Path

from ..base import ModelArchPlugin
from ..loader import register_arch


_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "rtdetrv2" / "configs"

_RTDETRV2_SCALES = {
    "s": (
        "Small",
        "~20 M",
        "~60 GFLOPs",
        "ResNet-18vd",
        "rtdetrv2_s.yml",
        "rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",
    ),
    "m": (
        "Medium",
        "~36 M",
        "~100 GFLOPs",
        "ResNet-50vd-m",
        "rtdetrv2_m.yml",
        "rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml",
    ),
    "l": (
        "Large",
        "~42 M",
        "~136 GFLOPs",
        "ResNet-50vd",
        "rtdetrv2_l.yml",
        "rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml",
    ),
    "x": (
        "Extra-Large",
        "~76 M",
        "~259 GFLOPs",
        "ResNet-101vd",
        "rtdetrv2_x.yml",
        "rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",
    ),
}


class RTDETRv2Plugin(ModelArchPlugin):
    """Discovery wrapper for one upstream RT-DETRv2 scale."""

    def __init__(self, scale: str):
        self._scale = scale.lower()

    @property
    def name(self) -> str:
        return f"rtdetrv2_{self._scale}"

    @property
    def display_name(self) -> str:
        label, params, flops, backbone, _, _ = _RTDETRV2_SCALES[self._scale]
        return f"RT-DETRv2 {label} ({params}, {flops})"

    @property
    def family(self) -> str:
        return "rtdetrv2"

    @property
    def family_display_name(self) -> str:
        return "RT-DETRv2"

    @property
    def scale(self) -> str:
        return self._scale

    @property
    def scale_label(self) -> str:
        label, params, flops, backbone, _, _ = _RTDETRV2_SCALES[self._scale]
        return f"{label} ({params}, {flops}) - {backbone}"

    @property
    def task_type(self) -> str:
        return "detect"

    @property
    def description(self) -> str:
        _, _, _, _, _, upstream_config = _RTDETRV2_SCALES[self._scale]
        return (
            "RT-DETRv2 upstream architecture from lyuwenyu/RT-DETR. "
            f"Config: {upstream_config}. "
            "Registered separately from Mamba-YOLO's vendored RT-DETR files."
        )

    def yaml_path(self) -> Path:
        _, _, _, _, metadata_file, _ = _RTDETRV2_SCALES[self._scale]
        return _CONFIGS_DIR / metadata_file

    def preflight_check(self) -> str | None:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            import yaml  # noqa: F401
            import pycocotools  # noqa: F401
            import tensorboard  # noqa: F401
            import faster_coco_eval  # noqa: F401
        except ImportError as exc:
            return f"RT-DETRv2 dependency missing: {exc}"
        return None

    def register_modules(self) -> None:
        # No import side effects. In particular, do not import
        # backend/data/vendor/Mamba-YOLO/ultralytics/models/rtdetr.
        return None

    def pretrain_key(self) -> str | None:
        return None


for _scale in _RTDETRV2_SCALES:
    register_arch(RTDETRv2Plugin(_scale))
