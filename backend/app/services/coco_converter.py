"""
COCO to YOLO Format Converter Service.

Automatically converts COCO JSON annotations to YOLO format text files
using Ultralytics' built-in converter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import DATASETS_DIR


def has_coco_annotations(dataset_path: Path) -> bool:
    """Check if dataset has COCO JSON annotations."""
    anno_dir = dataset_path / "annotations"
    if not anno_dir.exists():
        return False
    
    # Look for COCO JSON files
    json_files = list(anno_dir.glob("instances_*.json"))
    return len(json_files) > 0


def is_already_converted(dataset_path: Path) -> bool:
    """Check if COCO dataset has already been converted to YOLO format."""
    # Check if labels directory exists with .txt files
    labels_dir = dataset_path / "labels"
    if not labels_dir.exists():
        return False
    
    # Check for train/val subdirectories with .txt files
    for split in ["train", "val", "test"]:
        split_labels = labels_dir / split
        if split_labels.exists() and any(split_labels.glob("*.txt")):
            return True
    
    # Check for train2017/val2017 subdirectories (COCO format)
    for split in ["train2017", "val2017"]:
        split_labels = labels_dir / split
        if split_labels.exists() and any(split_labels.glob("*.txt")):
            return True
    
    return False


def convert_coco_to_yolo(
    dataset_name: str,
    *,
    use_segments: bool = False,
    use_keypoints: bool = False,
    cls91to80: bool = True,
) -> dict[str, Any]:
    """Convert COCO JSON annotations to YOLO format.
    
    Args:
        dataset_name: Name of the dataset (e.g. 'coco')
        use_segments: Convert segmentation masks instead of bounding boxes
        use_keypoints: Convert keypoints for pose estimation
        cls91to80: Map 91 COCO classes to 80 common ones
    
    Returns:
        Dictionary with conversion results
    """
    from ultralytics.data.converter import convert_coco
    import shutil
    
    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")
    
    anno_dir = dataset_path / "annotations"
    if not anno_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {anno_dir}")
    
    # Check if already converted
    if is_already_converted(dataset_path):
        return {
            "status": "already_converted",
            "message": "Dataset already has YOLO format labels",
            "labels_dir": str(dataset_path / "labels"),
        }
    
    # Create labels directory
    labels_dir = dataset_path / "labels"
    labels_dir.mkdir(exist_ok=True)
    
    # Create temp directory with only instances_*.json files
    # This avoids converting captions, person_keypoints, etc.
    temp_anno_dir = dataset_path / "_temp_annotations"
    temp_anno_dir.mkdir(exist_ok=True)
    
    try:
        # Copy only instances_*.json files
        instances_files = list(anno_dir.glob("instances_*.json"))
        if not instances_files:
            return {
                "status": "error",
                "message": "No instances_*.json files found in annotations directory",
                "error": "No detection annotations found",
            }
        
        for json_file in instances_files:
            shutil.copy2(json_file, temp_anno_dir / json_file.name)
        
        # Convert COCO to YOLO using temp directory
        # Note: Ultralytics creates numbered directories (coco2, coco3, etc.) in parent dir
        parent_dir = dataset_path.parent
        
        convert_coco(
            labels_dir=str(temp_anno_dir),
            save_dir=str(dataset_path),
            use_segments=use_segments,
            use_keypoints=use_keypoints,
            cls91to80=cls91to80,
        )
        
        # Find the actual output directory created by Ultralytics
        # It creates numbered directories like coco2, coco3, etc. in the parent directory
        dataset_base_name = dataset_path.name
        created_dirs = sorted(parent_dir.glob(f"{dataset_base_name}[0-9]*"))
        
        actual_labels_dir = None
        temp_dataset_dir = None
        
        # Check created directories for labels
        for created_dir in created_dirs:
            labels_subdir = created_dir / "labels"
            if labels_subdir.exists() and any(labels_subdir.rglob("*.txt")):
                actual_labels_dir = labels_subdir
                temp_dataset_dir = created_dir
                break
        
        if not actual_labels_dir:
            # Cleanup
            shutil.rmtree(temp_anno_dir, ignore_errors=True)
            for created_dir in created_dirs:
                shutil.rmtree(created_dir, ignore_errors=True)
            return {
                "status": "error",
                "message": "Conversion completed but no labels were generated",
                "error": "No .txt files found in output",
            }
        
        # Move the labels to the correct location
        if actual_labels_dir != labels_dir:
            if labels_dir.exists():
                shutil.rmtree(labels_dir)
            shutil.move(str(actual_labels_dir), str(labels_dir))
        
        # Count converted files
        txt_files = list(labels_dir.rglob("*.txt"))
        
        # Cleanup temp directories
        shutil.rmtree(temp_anno_dir, ignore_errors=True)
        if temp_dataset_dir and temp_dataset_dir.exists():
            shutil.rmtree(temp_dataset_dir, ignore_errors=True)
        
        return {
            "status": "success",
            "message": f"Converted {len(txt_files)} annotation files to YOLO format",
            "labels_dir": str(labels_dir),
            "file_count": len(txt_files),
        }
    except Exception as e:
        # Cleanup temp directories on error
        shutil.rmtree(temp_anno_dir, ignore_errors=True)
        # Cleanup any numbered dataset directories created by Ultralytics
        parent_dir = dataset_path.parent
        dataset_base_name = dataset_path.name
        for created_dir in parent_dir.glob(f"{dataset_base_name}[0-9]*"):
            shutil.rmtree(created_dir, ignore_errors=True)
        return {
            "status": "error",
            "message": f"Conversion failed: {str(e)}",
            "error": str(e),
        }


def auto_convert_if_needed(dataset_name: str) -> dict[str, Any] | None:
    """Automatically convert COCO dataset to YOLO format if needed.
    
    Called when dataset is loaded or scanned.
    
    Returns:
        Conversion result dict if conversion was performed, None otherwise
    """
    dataset_path = DATASETS_DIR / dataset_name
    
    # Check if dataset has COCO annotations
    if not has_coco_annotations(dataset_path):
        return None
    
    # Check if already converted
    if is_already_converted(dataset_path):
        return None
    
    # Auto-convert
    return convert_coco_to_yolo(dataset_name)
