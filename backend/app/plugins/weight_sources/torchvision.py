"""
Torchvision / generic PyTorch weight source plugin.

Handles:
- Raw state_dicts (dict of str → Tensor)
- Wrapped checkpoints: {"state_dict": ...}, {"model_state_dict": ...}
- torchvision model zoo checkpoints

This plugin has the *lowest* priority — it acts as a fallback when no
more specific plugin (Ultralytics, timm, etc.) matches.
"""
from __future__ import annotations
from typing import Any

import torch

from ..base import WeightSourcePlugin
from ..loader import register_weight_source


class TorchvisionWeightSource(WeightSourcePlugin):

    @property
    def name(self) -> str:
        return "torchvision"

    @property
    def display_name(self) -> str:
        return "Torchvision / Generic PyTorch"

    @property
    def file_extensions(self) -> list[str]:
        return [".pt", ".pth", ".bin"]

    # ── Detection ─────────────────────────────────────────────────────────────

    def can_parse(self, data: Any) -> bool:
        """Accept any dict whose values are mostly Tensors (raw state_dict)
        or a dict that wraps one under a known key."""
        if not isinstance(data, dict):
            return False

        # Already a raw state_dict?
        if self._looks_like_state_dict(data):
            return True

        # Wrapped under a common key?
        for key in ("state_dict", "model_state_dict", "model"):
            inner = data.get(key)
            if isinstance(inner, dict) and self._looks_like_state_dict(inner):
                return True

        return False

    @staticmethod
    def _looks_like_state_dict(d: dict) -> bool:
        """Return True if ≥50% of values are Tensors (heuristic)."""
        if not d:
            return False
        sample = list(d.values())[:20]
        tensor_count = sum(1 for v in sample if isinstance(v, torch.Tensor))
        return tensor_count >= len(sample) * 0.5

    # ── Extraction ────────────────────────────────────────────────────────────

    def extract_state_dict(self, data: Any) -> dict[str, torch.Tensor]:
        sd = self._find_state_dict(data)
        # Filter to tensors only
        return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}

    @staticmethod
    def _find_state_dict(data: dict) -> dict:
        """Unwrap common checkpoint wrappers."""
        for key in ("state_dict", "model_state_dict", "model"):
            inner = data.get(key)
            if isinstance(inner, dict):
                # Check if it's actually a state_dict (not an nn.Module)
                sample = list(inner.values())[:5]
                if sample and isinstance(sample[0], torch.Tensor):
                    return inner
        # Assume top-level is already a state_dict
        return data

    # ── Layer Groups ──────────────────────────────────────────────────────────

    def get_layer_groups(self, sd: dict[str, torch.Tensor]) -> list[dict]:
        """Group by first dotted segment (e.g. 'features', 'classifier', 'layer1')."""
        from collections import OrderedDict
        groups: dict[str, list[dict]] = OrderedDict()

        for k, v in sd.items():
            # Try to get a meaningful prefix
            parts = k.split(".")
            if len(parts) >= 2:
                # Use first segment, but if it's a number try first two
                prefix = parts[0]
                if prefix.isdigit() and len(parts) >= 3:
                    prefix = f"{parts[0]}.{parts[1]}"
            else:
                prefix = k
            groups.setdefault(prefix, []).append({
                "key": k,
                "shape": list(v.shape),
                "dtype": str(v.dtype),
            })

        result = []
        for prefix, keys in groups.items():
            result.append({
                "prefix": prefix,
                "module_type": self._guess_module_type(keys),
                "param_count": len(keys),
                "keys": keys,
            })
        return result


# ── Auto-register ─────────────────────────────────────────────────────────────
register_weight_source(TorchvisionWeightSource())
