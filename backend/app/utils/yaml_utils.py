"""
YAML utilities for model processing.
Handles patching of custom modules and validation of graph structure.
"""
from __future__ import annotations
import math
import yaml
import tempfile
from pathlib import Path
from typing import Any

from ..services import module_registry

def make_divisible(x: int | float, divisor: int) -> int:
    """Returns nearest x divisible by divisor."""
    return math.ceil(x / divisor) * divisor

def prepare_model_yaml(
    yaml_path: str | Path,
    scale: str | None = None
) -> str:
    """
    Read model YAML, apply patches for custom modules, check for cycles, and return path to temp file.
    
    Args:
        yaml_path: Path to the original model.yaml
        scale: Scale char ('n', 's', 'm', 'l', 'x') to use for channel scaling.
               If None, tries to detect from YAML or defaults to 'n' (if available).
    
    Returns:
        Path to the temporary patched YAML file.
    
    Raises:
        ValueError: If a cycle (forward reference) is detected or other validation errors.
    """
    path = Path(yaml_path)
    with open(path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # 1. Determine Scale
    scales = yaml_data.get("scales", {})
    if scales:
        if not scale:
            # Default to 'n' or first available
            scale = "n" if "n" in scales else next(iter(scales))
        
        # Set explicit scale to suppress YOLO warning
        yaml_data["scales"] = {scale: scales[scale]}
        
        # Remove top-level depth_multiple/width_multiple — they take precedence over
        # scales in Ultralytics parse_model and prevent proper scale application
        yaml_data.pop("depth_multiple", None)
        yaml_data.pop("width_multiple", None)
        
        # Get scale params for manual patching
        depth, width, max_channels = scales[scale]
    else:
        # No scales - likely user deleted them or old model
        width = 1.0
        max_channels = 65535

    # 2. Patch Custom Modules (Channel Scaling)
    # YOLO's parse_model only auto-scales modules in its internal 'base_modules'.
    # We must manually scale args for our custom modules.
    custom_modules = {"SparseGlobalBlock", "SparseGlobalBlockGated"}
    
    for section in ["backbone", "head"]:
        if section in yaml_data:
            for i, layer in enumerate(yaml_data[section]):
                # layer format: [from, repeats, module, args]
                if len(layer) >= 4:
                    m = layer[2]
                    args = layer[3]
                    # Check if it is our custom module that needs scaling
                    if m in custom_modules and len(args) >= 1 and isinstance(args[0], (int, float)):
                        c2 = args[0]
                        # Scale c2 same way as YOLO scales c2
                        c2 = make_divisible(min(c2, max_channels) * width, 8)
                        args[0] = c2
                        yaml_data[section][i][3] = args

    # 3. Check for forward references (Cycles)
    # Ultralytics parse_model assumes feed-forward. Forward refs cause IndexError (crash).
    backbone_len = len(yaml_data.get("backbone", []))
    
    for section in ["backbone", "head"]:
        if section not in yaml_data: continue
        is_backbone = (section == "backbone")
        layer_list = yaml_data[section]
        
        for i, layer in enumerate(layer_list):
            # Global index: if head, add backbone_len
            current_idx = i if is_backbone else (backbone_len + i)
            
            # layer format: [from, repeats, module, args]
            if len(layer) < 1: continue
            f = layer[0]
            
            # Normalize from to list
            if isinstance(f, int):
                f = [f]
            
            if isinstance(f, list):
                for src_idx in f:
                    if src_idx == -1: continue
                    # YOLO allows src_idx < current_idx (past layers)
                    # If src_idx >= current_idx, it's a forward reference -> Crash
                    if src_idx >= current_idx:
                        raise ValueError(
                            f"Invalid cycle detected at {section} layer {i} ({layer[2]}). "
                            f"Refers to future layer index {src_idx} (current is {current_idx}). "
                            "Feedback loops are not supported."
                        )

    # 4. Write to temp file with scale suffix
    # IMPORTANT: Ultralytics detects scale from filename (e.g. "model_n.yaml" → scale 'n')
    # We must include the scale in the filename so YOLO uses the correct scale
    suffix = f"_{scale}.yaml" if scale else ".yaml"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, text=True)
    with open(fd, 'w') as tmp:
        yaml.dump(yaml_data, tmp, default_flow_style=False)
    
    return tmp_path
