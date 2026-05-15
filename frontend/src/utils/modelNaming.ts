/**
 * Centralized model naming utilities for frontend.
 * 
 * This provides consistent naming logic across the frontend to match
 * the backend naming conventions.
 */

export function formatOfficialYoloModelName(yoloModel: string): string {
  /**
   * Format official YOLO model name with consistent naming convention.
   * 
   * Args:
   *   yoloModel: Model identifier (e.g., "yolov8n", "yolov9s", "yolo26n")
   * 
   * Returns:
   *   Formatted model name following consistent pattern with space before scale
   * 
   * Examples:
   *   - "yolov8n" -> "YOLOV8 N"
   *   - "yolov9s" -> "YOLOV9 S"
   *   - "yolo26n" -> "YOLO26 N"
   *   - "yolo11x" -> "YOLO11 X"
   */
  // Extract base name and scale letter
  if (yoloModel.length > 0 && /[a-zA-Z]/.test(yoloModel[yoloModel.length - 1])) {
    const baseName = yoloModel.slice(0, -1).toUpperCase();
    const scale = yoloModel[yoloModel.length - 1].toUpperCase();
    return `${baseName} ${scale}`;
  } else {
    return yoloModel.toUpperCase();
  }
}

export function formatArchPluginModelName(displayName: string, scale: string): string {
  /**
   * Format arch plugin model name with scale.
   * 
   * Args:
   *   displayName: Plugin display name (e.g., "Mamba-YOLO", "RT-DETR")
   *   scale: Scale character (e.g., "n", "s", "m", "l", "x")
   * 
   * Returns:
   *   Formatted model name with scale
   * 
   * Examples:
   *   - "Mamba-YOLO" + "n" -> "Mamba-YOLO N"
   *   - "RT-DETR" + "l" -> "RT-DETR L"
   *   - "YOLO26 + CS²GA" + "n" -> "YOLO26 + CS²GA N"
   */
  if (scale) {
    return `${displayName} ${scale.toUpperCase()}`;
  }
  return displayName;
}

export function getModelDisplayName(modelId: string, modelScale?: string, archDisplayName?: string): string {
  /**
   * Get standardized display name for any model.
   * 
   * Args:
   *   modelId: Model identifier (e.g., "yolo:yolov8n", "arch:mamba_yolo")
   *   modelScale: Scale for arch plugins (e.g., "n", "s", "m")
   *   archDisplayName: Display name for arch plugins
   * 
   * Returns:
   *   Standardized model display name
   */
  if (modelId.startsWith('yolo:')) {
    // Official YOLO model
    const yoloModel = modelId.split(':')[1];
    return formatOfficialYoloModelName(yoloModel);
  } else if (modelId.startsWith('arch:') && archDisplayName) {
    // Architecture plugin model
    return formatArchPluginModelName(archDisplayName, modelScale || '');
  } else {
    // Custom model - use as-is or generate from model_id
    return modelId.replace(/_/g, ' ').replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }
}
