"""
Dataset Registry — thin façade over DatasetPlugin instances.

All metadata (class names, normalization, task type, etc.) is sourced
exclusively from registered DatasetPlugin objects.  No static dicts.

All services (trainer, validator, inference, dataset_controller) import
the helper functions below — their signatures are unchanged.
"""
from __future__ import annotations

from ..schemas.dataset_schema import DatasetInfo


# ─── Plugin lookup (lazy to avoid circular imports) ──────────────────────────

def _try_plugin(dataset: str):
    """Return the DatasetPlugin for *dataset*, or None."""
    try:
        from ..plugins.loader import get_dataset_plugin
        return get_dataset_plugin(dataset.lower())
    except Exception:
        return None


# ─── Public helpers (signatures kept identical for callers) ──────────────────

def get_class_names(dataset: str) -> list[str]:
    """Return class names for a dataset."""
    plugin = _try_plugin(dataset)
    return plugin.class_names if plugin else []


def get_normalization(dataset: str) -> tuple[tuple, tuple]:
    """Return (mean, std) normalization tuples for a dataset."""
    plugin = _try_plugin(dataset)
    return plugin.normalization if plugin else ((0.5,), (0.5,))


def get_task_type(dataset: str) -> str:
    """Return 'classification' or 'detection' for a dataset."""
    plugin = _try_plugin(dataset)
    return plugin.task_type if plugin else "classification"


def is_image_dataset(dataset: str) -> bool:
    """Return True if a dataset plugin is registered (all are image-based)."""
    return _try_plugin(dataset) is not None


def get_dataset_info(dataset: str) -> DatasetInfo | None:
    """Return DatasetInfo for a dataset, or None if unknown."""
    plugin = _try_plugin(dataset)
    if plugin:
        return DatasetInfo(**plugin.to_info_dict())
    return None


def get_all_datasets() -> list[DatasetInfo]:
    """Return DatasetInfo for every registered dataset plugin."""
    try:
        from ..plugins.loader import all_dataset_plugins
        return [DatasetInfo(**p.to_info_dict()) for p in all_dataset_plugins()]
    except Exception:
        return []


# ─── Backward-compat aliases (used by trainer.py import) ─────────────────────
# trainer.py does: ``from .dataset_registry import CLASS_NAMES as _CLASS_NAMES``
# We provide a lazy-loading dict-like proxy so the import doesn't break,
# but all data comes from plugins.

class _PluginBackedDict:
    """Read-only dict-like object that delegates .get() to plugins."""

    def __init__(self, accessor):
        self._accessor = accessor

    def get(self, key, default=None):
        result = self._accessor(key)
        return result if result else default

    def __getitem__(self, key):
        result = self._accessor(key)
        if not result:
            raise KeyError(key)
        return result

    def __contains__(self, key):
        return self._accessor(key) is not None


CLASS_NAMES = _PluginBackedDict(lambda k: get_class_names(k) or None)
