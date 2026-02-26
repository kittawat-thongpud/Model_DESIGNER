"""
Plugin loader — discovers dataset, weight source, and model arch plugins.
"""
from __future__ import annotations
from typing import Any

from .base import DatasetPlugin, WeightSourcePlugin, ModelArchPlugin


# ── Registries ───────────────────────────────────────────────────────────────

_dataset_plugins: dict[str, DatasetPlugin] = {}
_weight_source_plugins: dict[str, WeightSourcePlugin] = {}
_arch_plugins: dict[str, ModelArchPlugin] = {}


def register_dataset(plugin: DatasetPlugin) -> None:
    _dataset_plugins[plugin.name.lower()] = plugin


def register_weight_source(plugin: WeightSourcePlugin) -> None:
    _weight_source_plugins[plugin.source_name.lower()] = plugin


def register_arch(plugin: ModelArchPlugin) -> None:
    """Register a model architecture plugin by its name."""
    _arch_plugins[plugin.name.lower()] = plugin


# ── Accessors ────────────────────────────────────────────────────────────────

def get_dataset_plugin(name: str) -> DatasetPlugin | None:
    return _dataset_plugins.get(name.lower())


def get_arch_plugin(name: str) -> ModelArchPlugin | None:
    """Return a model arch plugin by name, or None if not found."""
    return _arch_plugins.get(name.lower())


def all_dataset_plugins() -> list[DatasetPlugin]:
    return list(_dataset_plugins.values())


def all_weight_source_plugins() -> list[WeightSourcePlugin]:
    return list(_weight_source_plugins.values())


def all_arch_plugins() -> list[ModelArchPlugin]:
    """Return all registered model architecture plugins."""
    return list(_arch_plugins.values())


def find_arch_for_yaml(yaml_path: str) -> "ModelArchPlugin | None":
    """Return the arch plugin whose yaml_path() matches yaml_path, or None.

    Strategy (in order):
    1. Exact resolved-path match.
    2. YAML text contains custom module names that the arch plugin registers
       (detected via plugin.register_modules() side-effects on nn namespace).
    """
    from pathlib import Path
    try:
        target = Path(yaml_path).resolve()
    except Exception:
        return None

    # ── Strategy 1: exact path match ─────────────────────────────────────────
    for plugin in _arch_plugins.values():
        try:
            if Path(plugin.yaml_path()).resolve() == target:
                return plugin
        except Exception:
            continue

    # ── Strategy 2: module-name scan in YAML text ────────────────────────────
    try:
        yaml_text = target.read_text(encoding="utf-8")
    except Exception:
        return None

    import re
    # Match module names as they appear in YAML layer entries:
    # e.g.  "- [-1, 1, SparseGlobalBlockGated, [512, 512]]"
    #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    _LAYER_MODULE_RE = re.compile(r'-\s+\[.*?,\s*\d+,\s*([A-Za-z][A-Za-z0-9_.]+)\s*,')
    _STANDARD_MODULES = {
        "Conv", "C2f", "C3", "SPPF", "SPP", "Bottleneck", "Concat",
        "Detect", "Segment", "Pose", "nn.Upsample", "nn.Sequential",
        "nn.Identity",
    }

    target_modules = set(_LAYER_MODULE_RE.findall(yaml_text))

    for plugin in _arch_plugins.values():
        try:
            plugin_yaml = Path(plugin.yaml_path()).resolve()
            plugin_text = plugin_yaml.read_text(encoding="utf-8")
        except Exception:
            continue

        plugin_modules = set(_LAYER_MODULE_RE.findall(plugin_text))
        custom_modules = plugin_modules - _STANDARD_MODULES
        if not custom_modules:
            continue

        # Target YAML uses at least one of the plugin's custom layer modules
        if custom_modules & target_modules:
            return plugin

    return None


# ── Discovery ────────────────────────────────────────────────────────────────

def discover_plugins() -> dict[str, int]:
    """Import all plugin modules so they self-register."""
    import importlib
    import pkgutil
    from . import datasets, weight_sources

    # Import archs package lazily (may not exist before Step 3)
    try:
        from . import archs as _archs_pkg
        arch_pkg: Any = _archs_pkg
    except ImportError:
        arch_pkg = None

    for pkg, key in [(datasets, "datasets"), (weight_sources, "weight_sources")]:
        for importer, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
            try:
                importlib.import_module(f"{pkg.__name__}.{modname}")
            except Exception:
                pass

    if arch_pkg is not None:
        for importer, modname, ispkg in pkgutil.iter_modules(arch_pkg.__path__):
            try:
                importlib.import_module(f"{arch_pkg.__name__}.{modname}")
            except Exception:
                pass

    return {
        "datasets": len(_dataset_plugins),
        "weight_sources": len(_weight_source_plugins),
        "archs": len(_arch_plugins),
    }

