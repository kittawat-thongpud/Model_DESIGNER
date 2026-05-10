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

    def __init__(self, scale: str, variant: str = "legacy"):
        self._scale = scale.lower()
        self._variant = variant.lower()

    @property
    def name(self) -> str:
        if self._variant == "v2c":
            return f"hsg_detr_v2c_{self._scale}"
        if self._variant == "v2":
            return f"hsg_detr_v2_{self._scale}"
        return f"hsg_detr_{self._scale}"

    @property
    def display_name(self) -> str:
        label, params, flops, queries = _SCALE_DESCRIPTIONS[self._scale]
        if self._variant == "v2c":
            return f"HSG-DETR V2c {label} ({params}, {flops})"
        if self._variant == "v2":
            return f"HSG-DETR V2 {label} ({params}, {flops})"
        return f"HSG-DETR {label} ({params}, {flops})"

    @property
    def family(self) -> str:
        if self._variant == "v2c":
            return "hsg_detr_v2c"
        if self._variant == "v2":
            return "hsg_detr_v2"
        return "hsg_detr"

    @property
    def family_display_name(self) -> str:
        if self._variant == "v2c":
            return "HSG-DETR V2c"
        if self._variant == "v2":
            return "HSG-DETR V2"
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
        if self._variant == "v2c":
            return (
                "HSG-DETR V2c — V2 sparse-token encoder and RT-DETR decoder with "
                "a channel-selective SE gate on sparse attention deltas for CS-SGA "
                "experiments. N-scale only."
            )
        if self._variant == "v2":
            return (
                "HSG-DETR V2 — AMP-safe sparse-token encoder with the legacy "
                "topology plus RT-DETR decoder changes: Look-Forward-Twice box "
                "refinement and uncertainty-minimal query selection. N-scale only."
            )
        return (
            "HSG-DETR — legacy Sparse-Token SGB encoder feeding RT-DETR decoder. "
            "Uses parameter-free L2 top-k token selection, sparse global "
            "self-attention, zero-canvas scatter-back, and small LayerScale "
            "residual fusion. This is the stable fallback baseline."
        )

    def yaml_path(self) -> Path:
        if self._variant == "v2c":
            return _CONFIGS_DIR / f"hsg_detr_v2c_{self._scale}.yaml"
        if self._variant == "v2":
            return _CONFIGS_DIR / f"hsg_detr_v2_{self._scale}.yaml"
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

# V2 is currently installed for N-scale only while it is being validated.
register_arch(HSGDetRPlugin("n", variant="v2"))

# V2c variants are now registered separately in hsg_detr_v2c.py with Phase 1-2 experimental options
# register_arch(HSGDetRPlugin("n", variant="v2c"))  # Removed - replaced by hsg_detr_v2c.py

# V2B-N: gamma_init=0.01 debug variant
class HSGDetRV2BPlugin(HSGDetRPlugin):
    """V2B variant with gamma_init=0.01 for gamma convergence debugging."""
    
    def __init__(self):
        super().__init__("n", variant="v2")
    
    @property
    def name(self) -> str:
        return "hsg_detr_v2_v2b"
    
    @property
    def scale(self) -> str:
        return "v2b"
    
    @property
    def scale_label(self) -> str:
        return "V2B (gamma_init=0.01 debug)"
    
    @property
    def family(self) -> str:
        return "hsg_detr_v2"
    
    @property
    def description(self) -> str:
        return (
            "HSG-DETR V2B — gamma_init=0.01 debug variant for gamma convergence testing. "
            "Same architecture as V2-N but starts gamma at 0.01 instead of 1e-4 to test "
            "whether gamma can grow beyond the 0.02 saturation point observed in V2 runs."
        )
    
    def yaml_path(self) -> Path:
        return _CONFIGS_DIR / "hsg_detr_v2b_n.yaml"

register_arch(HSGDetRV2BPlugin())
