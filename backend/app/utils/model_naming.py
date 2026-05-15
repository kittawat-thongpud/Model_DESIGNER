"""
Centralized model naming utilities.

This module provides consistent naming logic for all model types
to ensure no hardcoded naming conventions across the codebase.
"""

def format_official_yolo_model_name(yolo_model: str) -> str:
    """
    Format official YOLO model name with consistent naming convention.
    
    Args:
        yolo_model: Model identifier (e.g., "yolov8n", "yolov9s", "yolo26n")
        
    Returns:
        Formatted model name following consistent pattern with space before scale
        
    Examples:
        - "yolov8n" -> "YOLOV8 N"
        - "yolov9s" -> "YOLOV9 S"
        - "yolo26n" -> "YOLO26 N"
        - "yolo11x" -> "YOLO11 X"
    """
    # Extract base name and scale letter
    if len(yolo_model) > 0 and yolo_model[-1].isalpha():
        base_name = yolo_model[:-1].upper()
        scale = yolo_model[-1].upper()
        return f"{base_name} {scale}"
    else:
        return yolo_model.upper()


def format_arch_plugin_model_name(arch_plugin, scale: str) -> str:
    """
    Format arch plugin model name with scale.
    
    Args:
        arch_plugin: Architecture plugin object
        scale: Scale character (e.g., "n", "s", "m", "l", "x")
        
    Returns:
        Formatted model name with scale
        
    Examples:
        - mamba_yolo + "n" -> "Mamba-YOLO N"
        - rtdetr + "l" -> "RT-DETR L"
        - yolo26_cs2ga + "n" -> "YOLO26 + CS²GA N"
    """
    base_name = arch_plugin.display_name
    if scale:
        return f"{base_name} {scale.upper()}"
    return base_name


def get_model_display_name(model_id: str, model_scale: str = None, arch_plugin=None) -> str:
    """
    Get standardized display name for any model.
    
    Args:
        model_id: Model identifier (e.g., "yolo:yolov8n", "arch:mamba_yolo")
        model_scale: Scale for arch plugins (e.g., "n", "s", "m")
        arch_plugin: Architecture plugin object for arch models
        
    Returns:
        Standardized model display name
    """
    if model_id.startswith("yolo:"):
        # Official YOLO model
        yolo_model = model_id.split(":")[1]
        return format_official_yolo_model_name(yolo_model)
    elif model_id.startswith("arch:") and arch_plugin:
        # Architecture plugin model
        return format_arch_plugin_model_name(arch_plugin, model_scale or "")
    else:
        # Custom model - use as-is or generate from model_id
        return model_id.replace("_", " ").replace("-", " ").title()
