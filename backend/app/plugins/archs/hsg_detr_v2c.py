"""
HSG-DETR V2c Model Architecture Plugin.

CNN hybrid detector with SparseGlobalTokenBlockV2 at P3/P4/P5.
Channel-selective squeeze-excitation gate in V2c.
Old Phase 1-2 experiment YAMLs remain readable for job reproducibility, but
only the stable base V2c architecture is registered for new jobs.
"""
from __future__ import annotations
from pathlib import Path

from .hsg_det import HSGDetPlugin
from ..loader import register_arch


_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "hsg_detr" / "configs"

_SCALE_DESCRIPTIONS = {
    "n": ("Nano",   "~3.5 M",  "~8 GFLOPs"),
}

_VARIANT_DESCRIPTIONS = {
    "": ("V2c", "Channel-selective sparse global attention with uncertainty-minimal query selection"),
    "phase1_lq_stability": ("Phase 1 LQ-B", "Localization quality: stability proxy exp(-abs(bbox_delta).mean())"),
    "phase1_lq_consistency": ("Phase 1 LQ-C", "Localization quality: class*localization consistency"),
    "phase2_beta0": ("Phase 2 β=0", "SGB role separation: no saliency in query selection"),
    "phase2_beta05": ("Phase 2 β=0.05", "SGB role separation: small saliency contribution"),
    "phase2_beta15": ("Phase 2 β=0.15", "SGB role separation: high saliency contribution"),
}


class HSGDETRV2cPlugin(HSGDetPlugin):
    """Plugin for HSG-DETR V2c with experimental variants."""

    def __init__(self, variant: str = ""):
        self._variant = variant

    @property
    def name(self) -> str:
        if self._variant:
            return f"hsg_detr_v2c_{self._variant}"
        return "hsg_detr_v2c"

    @property
    def display_name(self) -> str:
        label, params, flops = _SCALE_DESCRIPTIONS["n"]
        if self._variant:
            variant_label, _ = _VARIANT_DESCRIPTIONS[self._variant]
            return f"HSG-DETR V2c {label} - {variant_label} ({params}, {flops})"
        else:
            return f"HSG-DETR V2c {label} ({params}, {flops})"

    @property
    def family(self) -> str:
        return "hsg_detr_v2c"

    @property
    def family_display_name(self) -> str:
        return "HSG-DETR V2c"

    @property
    def scale(self) -> str:
        # Use variant as UI scale key (for frontend grouping), base uses "n"
        return self._variant if self._variant else "n"

    @property
    def scale_label(self) -> str:
        label, params, flops = _SCALE_DESCRIPTIONS["n"]
        if self._variant:
            variant_label, _ = _VARIANT_DESCRIPTIONS[self._variant]
            return f"{variant_label} ({params}, {flops})"
        return f"{label} ({params}, {flops})"

    @property
    def yaml_scale(self) -> str:
        """Always 'n' for YAML patching — all variants are Nano."""
        return "n"

    def pretrain_key(self) -> str | None:
        """No YOLO pretrained by default — HSG-DETR V2c trains from scratch."""
        return None

    def register_modules(self) -> None:
        """Inject HSG-DETR V2c custom modules into ultralytics.nn.modules."""
        try:
            import hsg_detr.nn  # noqa: F401 — triggers register() on import
        except ImportError as e:
            raise ImportError(
                "Could not import hsg_detr package. "
                "Ensure the backend/hsg_detr/ directory is on PYTHONPATH."
            ) from e

    @property
    def description(self) -> str:
        if self._variant:
            _, variant_desc = _VARIANT_DESCRIPTIONS[self._variant]
            return (
                f"HSG-DETR V2c: {variant_desc}. "
                "Channel-selective sparse global attention with uncertainty-minimal query selection. "
                "Nano scale optimized for edge deployment."
            )
        else:
            return (
                "HSG-DETR V2c: Channel-selective sparse global attention with uncertainty-minimal query selection. "
                "Nano scale optimized for edge deployment."
            )

    def yaml_path(self) -> Path:
        if self._variant:
            return _CONFIGS_DIR / f"hsg_detr_v2c_{self._variant}.yaml"
        return _CONFIGS_DIR / "hsg_detr_v2c_n.yaml"

    def get_config_options(self) -> dict[str, any]:
        """Return configurable options for this variant."""
        if self._variant.startswith("phase1"):
            return {
                "loc_quality_mode": "stability" if "stability" in self._variant else "cls_consistency",
                "alpha_u": 0.3,
                "beta_s": 0.0,
            }
        elif self._variant.startswith("phase2"):
            beta_map = {
                "beta0": 0.0,
                "beta05": 0.05,
                "beta15": 0.15,
            }
            beta = beta_map.get(self._variant.replace("phase2_", ""), 0.0)
            return {
                "loc_quality_mode": "area",
                "alpha_u": 0.3,
                "beta_s": beta,
            }
        else:
            return {
                "loc_quality_mode": "area",
                "alpha_u": 0.3,
                "beta_s": 0.0,
            }


# Register stable base only. Phase experiment variants are archived/hidden
# from new API/plugin discovery to keep the V2 surface clean.
register_arch(HSGDETRV2cPlugin(""))  # Base V2c
