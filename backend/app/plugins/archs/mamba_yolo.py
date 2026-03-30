"""
Mamba-YOLO Model Architecture Plugin.

Registers three scale variants (T / B / L) into the plugin system.
Modules are loaded directly from the HZAI-ZJNU/Mamba-YOLO repository
cloned by the installer — no custom re-implementations.

Registers as: "mamba_yolo_t", "mamba_yolo_b", "mamba_yolo_l"
"""
from __future__ import annotations
from pathlib import Path

from ..base import ModelArchPlugin
from ..loader import register_arch


_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "mamba_yolo" / "configs"

_SCALE_DESCRIPTIONS = {
    "t": ("Tiny",  "~5 M",  "~10 GFLOPs",  "fastest, edge/mobile"),
    "b": ("Base",  "~12 M", "~28 GFLOPs",  "balanced accuracy/speed"),
    "l": ("Large", "~26 M", "~65 GFLOPs",  "highest accuracy"),
}


class MambaYOLOPlugin(ModelArchPlugin):
    """Single plugin instance for one Mamba-YOLO scale (T, B, or L)."""

    def __init__(self, scale: str):
        self._scale = scale.lower()

    @property
    def name(self) -> str:
        return f"mamba_yolo_{self._scale}"

    @property
    def display_name(self) -> str:
        label, params, flops, note = _SCALE_DESCRIPTIONS[self._scale]
        return f"Mamba-YOLO {label} ({params}, {flops})"

    @property
    def family(self) -> str:
        return "mamba_yolo"

    @property
    def family_display_name(self) -> str:
        return "Mamba-YOLO"

    @property
    def scale(self) -> str:
        return self._scale

    @property
    def scale_label(self) -> str:
        label, params, flops, note = _SCALE_DESCRIPTIONS[self._scale]
        return f"{label} ({params}, {flops})"

    @property
    def task_type(self) -> str:
        return "detect"

    @property
    def description(self) -> str:
        return (
            "Mamba-YOLO — SSM-based YOLO variant using Visual State Space (VSS) blocks. "
            "Modules loaded directly from HZAI-ZJNU/Mamba-YOLO (official source). "
            "Supports selective_scan CUDA extension for best performance; "
            "falls back to PyTorch automatically when CUDA extension is unavailable. "
            "Based on HZAI-ZJNU/Mamba-YOLO (Apache-2.0)."
        )

    def yaml_path(self) -> Path:
        return _CONFIGS_DIR / "mamba_yolo.yaml"

    def preflight_check(self) -> str | None:
        """Fail fast if the repo is not cloned or module imports fail."""
        from mamba_yolo.installer import _modules_loadable
        if not _modules_loadable():
            return (
                "Mamba-YOLO repo not installed. "
                "Go to the Mamba-YOLO plugin page and click Install."
            )
        try:
            from mamba_yolo.nn import register
            register()
        except ImportError as exc:
            return (
                f"Mamba-YOLO modules could not be loaded: {exc}. "
                "Run the installer again to rebuild selective_scan."
            )
        import sys as _sys
        _sscc = _sys.modules.get("selective_scan_cuda_core")
        if _sscc is None or not hasattr(_sscc, "fwd"):
            return (
                "selective_scan CUDA extension is not built. "
                "Go to the Plugins page and click Install to build it."
            )
        try:
            import torch
            if not torch.cuda.is_available():
                return (
                    "Mamba-YOLO requires an NVIDIA CUDA device. "
                    "Current runtime has no available CUDA GPU."
                )
        except Exception:
            return (
                "PyTorch CUDA runtime check failed for Mamba-YOLO. "
                "Verify torch CUDA installation and GPU visibility."
            )
        return None

    def register_modules(self) -> None:
        """Load modules from the cloned repo and patch ultralytics parse_model."""
        from mamba_yolo.nn import register
        register()

    def pretrain_key(self) -> str | None:
        return None


# ── Auto-register all three scales ────────────────────────────────────────────
for _scale in ("t", "b", "l"):
    register_arch(MambaYOLOPlugin(_scale))
