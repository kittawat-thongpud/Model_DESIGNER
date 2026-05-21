"""Training Profile — named training configuration presets per arch plugin.

A TrainingProfile bundles together:
  - Which parameters to freeze (by layer prefix or module class name)
  - LR multiplier overrides per parameter group role
  - Hyperparameter overrides (applied on top of get_config_options())
  - Display metadata (name, description, badge color for UI)

Profiles let you train a model in different modes without changing the
arch YAML. Example: "attention_only" freezes the YOLO26 backbone/neck
and trains only the CS²GA block, proving isolation of the attention gain.

Usage
-----
1. Define profiles in an arch plugin::

    from app.plugins.training_profile import TrainingProfile

    class MyArchPlugin(ModelArchPlugin):
        def get_training_profiles(self) -> list[TrainingProfile]:
            return [
                TrainingProfile(
                    name="full",
                    display_name="Full Training",
                    description="Train all layers end-to-end.",
                    is_default=True,
                    badge_color="blue",
                ),
                TrainingProfile(
                    name="attention_only",
                    display_name="Attention-Only",
                    description="Freeze backbone/neck; train only the CS²GA block.",
                    freeze_param_prefixes=["model.0.", "model.1.", ...],
                    lr_group_overrides={"sgb_sparse": 10.0, "sgb_gamma": 20.0},
                    config_overrides={"epochs": 50, "lr0": 0.001},
                    badge_color="orange",
                    tags=["freeze backbone", "fast"],
                ),
            ]

2. Pass the selected profile name when starting a job (``training_profile``
   field in ``TrainRequest``).  The trainer will automatically freeze the
   correct parameters and apply LR / hyper-parameter overrides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingProfile:
    """Named training configuration preset for an arch plugin.

    Parameters
    ----------
    name : str
        Machine-readable key, e.g. ``"full"``, ``"attention_only"``.
    display_name : str
        Short UI label, e.g. ``"Full Training"``.
    description : str
        One-to-two sentence description shown below the profile selector.
    is_default : bool
        If True, this profile is auto-selected when the model is chosen.
    badge_color : str
        UI badge color. One of: ``"blue"``, ``"green"``, ``"orange"``,
        ``"red"``, ``"purple"``, ``"gray"``.

    freeze_module_names : list[str]
        Module class names whose *all* parameters are frozen.
        E.g. ``["Conv", "C3k2", "SPPF"]`` freezes all such layers.
    freeze_param_prefixes : list[str]
        Parameter name prefixes to freeze, matched against the full
        ``name`` from ``model.named_parameters()``.
        E.g. ``["model.0.", "model.1."]`` freezes layers 0 and 1.
    unfreeze_param_prefixes : list[str]
        Parameter name prefixes to always keep trainable (overrides the
        freeze lists above).  Use to carve out a sub-module inside an
        otherwise frozen region.

    lr_group_overrides : dict[str, float]
        Per-group LR multiplier overrides (relative to ``lr0``).
        Keys must match param group names in ``build_optimizer``:
        ``"base"``, ``"sgb_sparse"``, ``"sgb_gamma"``,
        ``"sgb_norm_group"``, ``"norm_bias"``, ``"decoder"``.
        Only the listed groups are changed; unlisted groups keep their
        defaults.

    config_overrides : dict[str, Any]
        Hyperparameter key-value pairs applied on top of the arch plugin's
        ``get_config_options()``, and in turn on top of whatever the user
        sets in the UI.  Useful for per-profile defaults such as a lower
        ``lr0`` when training only the attention block.

    tags : list[str]
        Short descriptor chips shown in the UI alongside the profile
        name, e.g. ``["freeze backbone", "fast"]``.
    """

    # Identity / display
    name: str
    display_name: str
    description: str
    is_default: bool = False
    badge_color: str = "blue"

    # Freeze control
    freeze_module_names: list[str] = field(default_factory=list)
    freeze_param_prefixes: list[str] = field(default_factory=list)
    unfreeze_param_prefixes: list[str] = field(default_factory=list)

    # LR overrides per param group role
    lr_group_overrides: dict[str, float] = field(default_factory=dict)

    # Hyper-parameter overrides
    config_overrides: dict[str, Any] = field(default_factory=dict)

    # UI
    tags: list[str] = field(default_factory=list)

    # ── helpers ──────────────────────────────────────────────────────────────

    def apply_freeze(self, model) -> tuple[int, int]:
        """Set ``requires_grad`` on model parameters according to this profile.

        Returns
        -------
        (frozen_count, trainable_count)
        """
        if not self.freeze_param_prefixes and not self.freeze_module_names:
            # Nothing to freeze — all params stay trainable
            total = sum(1 for _ in model.parameters())
            return 0, total

        # Collect all param ids that must stay unfrozen
        unfreeze_ids: set[int] = set()
        for pfx in self.unfreeze_param_prefixes:
            for n, p in model.named_parameters():
                if n.startswith(pfx):
                    unfreeze_ids.add(id(p))

        # Collect param ids to freeze via module class names
        freeze_ids: set[int] = set()
        if self.freeze_module_names:
            module_set = set(self.freeze_module_names)
            for module in model.modules():
                if module.__class__.__name__ in module_set:
                    for p in module.parameters():
                        freeze_ids.add(id(p))

        # Collect param ids to freeze via param name prefixes
        for pfx in self.freeze_param_prefixes:
            for n, p in model.named_parameters():
                if n.startswith(pfx):
                    freeze_ids.add(id(p))

        frozen = trainable = 0
        for p in model.parameters():
            pid = id(p)
            if pid in freeze_ids and pid not in unfreeze_ids:
                p.requires_grad_(False)
                frozen += 1
            else:
                p.requires_grad_(True)
                trainable += 1

        return frozen, trainable

    def effective_lr_multipliers(self, defaults: dict[str, float]) -> dict[str, float]:
        """Merge default multipliers with this profile's overrides."""
        merged = dict(defaults)
        merged.update(self.lr_group_overrides)
        return merged

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation (for API responses)."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "is_default": self.is_default,
            "badge_color": self.badge_color,
            "freeze_module_names": self.freeze_module_names,
            "freeze_param_prefixes": self.freeze_param_prefixes,
            "unfreeze_param_prefixes": self.unfreeze_param_prefixes,
            "lr_group_overrides": self.lr_group_overrides,
            "config_overrides": self.config_overrides,
            "tags": self.tags,
        }
