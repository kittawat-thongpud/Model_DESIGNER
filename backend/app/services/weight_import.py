"""
External weight import service — generic core.

Iterates all registered WeightSourcePlugins to detect the checkpoint format,
extract a state_dict, and save it as a system weight.  No format-specific
code lives here; all intelligence is in the plugins.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from ..config import WEIGHTS_DIR
from ..plugins.loader import all_weight_source_plugins
from . import weight_storage


def import_external_weight(
    file_path: Path,
    name: str,
    *,
    force_plugin: str | None = None,
) -> dict:
    """Import an external ``.pt`` / ``.pth`` file as a system weight.

    Parameters
    ----------
    file_path : Path
        Path to the uploaded file on disk.
    name : str
        Human-readable name for the imported weight.
    force_plugin : str | None
        If set, only try the plugin with this name.

    Returns
    -------
    dict with keys:
        weight_id, source_plugin, key_count, groups, original_filename
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the raw checkpoint (allow all types for Ultralytics nn.Module compat)
    data = torch.load(str(file_path), map_location="cpu", weights_only=False)

    plugins = all_weight_source_plugins()
    if force_plugin:
        plugins = [p for p in plugins if p.name == force_plugin]

    # Try each plugin — first match wins
    for plugin in plugins:
        try:
            if not plugin.can_parse(data):
                continue
        except Exception:
            continue

        sd = plugin.extract_state_dict(data)
        if not sd:
            continue

        groups = plugin.get_layer_groups(sd)

        weight_id = _save_imported_weight(
            sd=sd,
            name=name,
            source_plugin=plugin.name,
            original_filename=file_path.name,
        )

        return {
            "weight_id": weight_id,
            "source_plugin": plugin.name,
            "source_display_name": plugin.display_name,
            "key_count": len(sd),
            "groups": groups,
            "original_filename": file_path.name,
        }

    # No plugin matched — raise helpful error
    raise ValueError(
        f"No weight-source plugin could parse '{file_path.name}'. "
        f"Available plugins: {[p.name for p in all_weight_source_plugins()]}"
    )


def _save_imported_weight(
    sd: dict[str, torch.Tensor],
    name: str,
    source_plugin: str,
    original_filename: str,
) -> str:
    """Save the extracted state_dict as a system weight with metadata."""
    weight_id = uuid.uuid4().hex[:12]
    wdir = WEIGHTS_DIR / weight_id
    wdir.mkdir(parents=True, exist_ok=True)

    # Save state_dict
    pt_path = wdir / "weight.pt"
    torch.save(sd, pt_path)

    # Save metadata
    param_count = sum(v.numel() for v in sd.values())
    meta = {
        "weight_id": weight_id,
        "model_id": "",
        "model_name": name,
        "source": "external",
        "source_plugin": source_plugin,
        "original_filename": original_filename,
        "key_count": len(sd),
        "param_count": param_count,
        "file_size_bytes": pt_path.stat().st_size,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(wdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return weight_id


def get_weight_groups(weight_id: str) -> list[dict]:
    """Load a system weight and return its layer groups.

    Tries to detect the source plugin from metadata; falls back to
    generic grouping.
    """
    pt_path = weight_storage.weight_pt_path(weight_id)
    if not pt_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_id}")

    # Try weights_only=True first (safe), fall back to False for full checkpoints
    try:
        raw = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    except Exception:
        raw = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    # Try to use the original source plugin for better grouping + extraction
    meta = weight_storage.load_weight_meta(weight_id)
    if meta:
        source_name = meta.get("source_plugin", "")
        if source_name:
            from ..plugins.loader import get_weight_source_plugin
            plugin = get_weight_source_plugin(source_name)
            if plugin and plugin.can_parse(raw):
                sd = plugin.extract_state_dict(raw)
                return plugin.get_layer_groups(sd)

    # If raw is already a state_dict (plain tensor dict), use it directly
    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in list(raw.values())[:5]):
        sd = raw
    else:
        # Try all plugins as fallback
        from ..plugins.loader import all_weight_source_plugins
        for plugin in all_weight_source_plugins():
            try:
                if plugin.can_parse(raw):
                    sd = plugin.extract_state_dict(raw)
                    return plugin.get_layer_groups(sd)
            except Exception:
                continue
        # Last resort: use raw as-is if it's a dict
        sd = raw if isinstance(raw, dict) else {}

    # Fallback: generic grouping by first key segment
    from collections import OrderedDict
    groups: dict[str, list[dict]] = OrderedDict()
    for k, v in sd.items():
        prefix = k.split(".")[0] if "." in k else k
        groups.setdefault(prefix, []).append({
            "key": k,
            "shape": list(v.shape),
            "dtype": str(v.dtype),
        })

    def _infer_module_type(keys: list[dict]) -> str:
        """Infer PyTorch module type from the suffixes of state_dict keys."""
        suffixes = {k["key"].split(".")[-1] for k in keys}
        shapes = {k["key"].split(".")[-1]: k["shape"] for k in keys}
        if {"running_mean", "running_var"} & suffixes:
            return "BatchNorm"
        if "weight" in suffixes and "bias" in suffixes:
            w_shape = shapes.get("weight", [])
            if len(w_shape) == 4:
                return "Conv2d"
            if len(w_shape) == 2:
                return "Linear"
            if len(w_shape) == 1:
                return "Linear"  # could be LayerNorm etc.
        if "weight" in suffixes:
            w_shape = shapes.get("weight", [])
            if len(w_shape) == 4:
                return "Conv2d"
            if len(w_shape) == 2:
                return "Linear"
        if len(keys) == 1:
            return "Parameter"
        return "Module"

    result = []
    for prefix, keys in groups.items():
        result.append({
            "prefix": prefix,
            "module_type": _infer_module_type(keys),
            "param_count": len(keys),
            "keys": keys,
        })
    return result
