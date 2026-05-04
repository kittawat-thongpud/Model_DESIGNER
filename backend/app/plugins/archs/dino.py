"""DINO architecture plugins.

``dino_resnet50`` is the IDEA-Research/DINO detector (DETR with denoising
anchors) and supports COCO mAP benchmarking.  The ViT variants remain
facebookresearch/DINO self-supervised backbones.
"""
from __future__ import annotations

from pathlib import Path

from ..base import ModelArchPlugin
from ..loader import register_arch


_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "dino" / "configs"

_DINO_SCALES = {
    "vits16": ("ViT-S/16", "vit_small", 16, "dino_vits16.yml"),
    "vits8": ("ViT-S/8", "vit_small", 8, "dino_vits8.yml"),
    "vitb16": ("ViT-B/16", "vit_base", 16, "dino_vitb16.yml"),
    "vitb8": ("ViT-B/8", "vit_base", 8, "dino_vitb8.yml"),
    "resnet50": ("R50-4scale", "dino_detector", 4, "dino_resnet50.yml"),
}


class DINOPlugin(ModelArchPlugin):
    """Discovery wrapper for one official DINO backbone variant."""

    def __init__(self, scale: str):
        self._scale = scale.lower()

    @property
    def name(self) -> str:
        return f"dino_{self._scale}"

    @property
    def display_name(self) -> str:
        label, _, _, _ = _DINO_SCALES[self._scale]
        return f"DINO {label}"

    @property
    def family(self) -> str:
        return "dino"

    @property
    def family_display_name(self) -> str:
        return "DINO"

    @property
    def scale(self) -> str:
        return self._scale

    @property
    def scale_label(self) -> str:
        label, arch, patch_size, _ = _DINO_SCALES[self._scale]
        if self._scale == "resnet50":
            return "ResNet-50 4-scale detector"
        if arch.startswith("vit_"):
            return f"{label} ({arch}, patch {patch_size})"
        return label

    @property
    def task_type(self) -> str:
        if self._scale == "resnet50":
            return "detect"
        return "classify"

    @property
    def description(self) -> str:
        label, arch, patch_size, _ = _DINO_SCALES[self._scale]
        if self._scale == "resnet50":
            return (
                "Official IDEA-Research/DINO object detector "
                "(DETR with Improved DeNoising Anchor Boxes), ResNet-50 4-scale. "
                "Supports COCO mAP, per-class metrics, and confusion matrix benchmarking."
            )
        return (
            "Official facebookresearch/DINO self-supervised backbone training. "
            f"Variant: {label}, upstream arch={arch}, patch_size={patch_size}. "
            "Labels are ignored; Model Designer converts image datasets to ImageFolder."
        )

    def yaml_path(self) -> Path:
        _, _, _, metadata_file = _DINO_SCALES[self._scale]
        return _CONFIGS_DIR / metadata_file

    def preflight_check(self) -> str | None:
        try:
            import torch
            import torchvision  # noqa: F401
            import PIL  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:
            return f"DINO dependency missing: {exc}"
        if not torch.cuda.is_available():
            return "DINO upstream trainer requires CUDA."
        if self._scale == "resnet50":
            try:
                import torch  # noqa: F401
                from dino.detector_installer import ensure_installed

                ensure_installed(build_ops=True)
            except Exception as exc:
                return f"DINO detector preflight failed: {exc}"
        return None

    def register_modules(self) -> None:
        return None

    def pretrain_key(self) -> str | None:
        return None


for _scale in _DINO_SCALES:
    register_arch(DINOPlugin(_scale))
