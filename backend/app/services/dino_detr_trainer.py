"""DINO-DETR detection trainer.

This module handles DINO-DETR object detection training using the IDEA-Research/DINO-DETR codebase.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import JOBS_DIR
from app.services import job_storage, weight_storage
from app.services.dataset_yaml import write_data_yaml


def _set_job(job_id: str, **updates: Any) -> dict | None:
    """Update job record and publish event."""
    job = job_storage.load_job(job_id)
    if not job:
        return None
    job.update(updates)
    job_storage.save_job(job)
    return job


def _log(job_id: str, level: str, message: str, data: dict = None) -> None:
    """Append log to job."""
    job_storage.append_job_log(job_id, level, message, data or {})


def repo_dir() -> Path:
    """Get DINO-DETR vendor directory."""
    from app.config import DATA_DIR
    return DATA_DIR / "vendor" / "DINO-DETR"


def _convert_yolo_to_coco(job_id: str, dataset_dir: Path) -> Path:
    """Convert YOLO annotations to COCO format for DINO-DETR training."""
    _log(job_id, "INFO", "Converting YOLO annotations to COCO format for DINO-DETR")
    
    # Create annotations directory
    annotations_dir = dataset_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if COCO annotations already exist
    train_ann = annotations_dir / "instances_train2017.json"
    val_ann = annotations_dir / "instances_val2017.json"
    
    if train_ann.exists() and val_ann.exists():
        _log(job_id, "INFO", "COCO annotations already exist, skipping conversion")
        return annotations_dir
    
    # Load YOLO annotations
    train_labels = dataset_dir / "labels" / "train"
    val_labels = dataset_dir / "labels" / "val"
    
    # Simple conversion - create minimal COCO annotations
    # This is a basic implementation that may need refinement
    
    def create_coco_annotation(labels_dir: Path, images_dir: Path, output_path: Path, split_name: str):
        """Create COCO annotation file from YOLO labels."""
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "Car", "supercategory": "vehicle"},
                {"id": 1, "name": "Pedestrian", "supercategory": "person"},
                {"id": 2, "name": "Cyclist", "supercategory": "person"},
            ]
        }
        
        annotation_id = 0
        image_id = 0
        
        if not labels_dir.exists():
            _log(job_id, "WARNING", f"Labels directory not found: {labels_dir}")
            return
        
        for label_file in labels_dir.glob("*.txt"):
            # Get corresponding image
            image_file = label_file.stem
            image_path = images_dir / f"{image_file}.jpg"
            if not image_path.exists():
                image_path = images_dir / f"{image_file}.png"
            
            if not image_path.exists():
                continue
            
            # Get image dimensions
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception:
                img_width, img_height = 640, 480  # Default dimensions
            
            # Add image to COCO
            coco_output["images"].append({
                "id": image_id,
                "file_name": f"{split_name}/{image_file}.jpg",
                "width": img_width,
                "height": img_height,
            })
            
            # Parse YOLO annotations
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert YOLO to COCO format
                    x_min = (x_center - width / 2) * img_width
                    y_min = (y_center - height / 2) * img_height
                    box_width = width * img_width
                    box_height = height * img_height
                    
                    coco_output["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id if class_id < 3 else 0,
                        "bbox": [x_min, y_min, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                    })
                    annotation_id += 1
            
            image_id += 1
        
        # Write COCO annotation file
        with open(output_path, 'w') as f:
            json.dump(coco_output, f, indent=2)
        
        _log(job_id, "INFO", f"Created COCO annotation: {output_path} with {len(coco_output['images'])} images")
    
    # Convert train and val splits
    train_images = dataset_dir / "images" / "train"
    val_images = dataset_dir / "images" / "val"
    
    if train_labels.exists() and train_images.exists():
        create_coco_annotation(train_labels, train_images, train_ann, "train")
    
    if val_labels.exists() and val_images.exists():
        create_coco_annotation(val_labels, val_images, val_ann, "val")
    
    return annotations_dir


def _patch_vendor_code(root: Path, job_id: str) -> None:
    """Patch DINO-DETR vendor code to fix IndentationError in slconfig.py."""
    slconfig_path = root / "util" / "slconfig.py"
    if slconfig_path.exists():
        try:
            original_content = slconfig_path.read_text(encoding="utf-8")
            # Fix the specific IndentationError pattern: multiple nested try statements without indentation
            # Pattern to match: "try:" repeated multiple times on consecutive lines without indentation
            lines = original_content.split('\n')
            fixed_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                # Check if this is a sequence of "try:" statements
                if line.strip() == 'try:':
                    # Count consecutive "try:" statements
                    try_count = 0
                    j = i
                    while j < len(lines) and lines[j].strip() == 'try:':
                        try_count += 1
                        j += 1
                    # If we have multiple "try:" statements, keep only the first one with proper indentation
                    if try_count > 1:
                        # Keep the first "try:" with proper indentation
                        fixed_lines.append(lines[i])
                        i += 1
                        # Skip the duplicate "try:" statements
                        i += (try_count - 1)
                        # Add the actual try block content (should be indented)
                        while i < len(lines) and lines[i].strip() == '':
                            fixed_lines.append(lines[i])
                            i += 1
                        # The next line should be the actual content (indented)
                        if i < len(lines):
                            fixed_lines.append(lines[i])
                            i += 1
                        continue
                fixed_lines.append(line)
                i += 1
            
            fixed_content = '\n'.join(fixed_lines)
            
            # Also remove verify parameter from FormatCode calls
            fixed_content = re.sub(
                r'FormatCode\([^,]+,\s*style_config=yapf_style,\s*verify=True\)',
                'FormatCode(text, style_config=yapf_style)',
                fixed_content
            )
            
            slconfig_path.write_text(fixed_content, encoding="utf-8")
            _log(job_id, "INFO", "Patched DINO-DETR slconfig.py IndentationError and verify parameter")
            return original_content
        except Exception as e:
            _log(job_id, "WARNING", f"Failed to patch slconfig.py: {e}")
            return None
    return None


def _restore_vendor_code(root: Path, original_content: str) -> None:
    """Restore original vendor code after training."""
    if original_content:
        slconfig_path = root / "util" / "slconfig.py"
        if slconfig_path.exists():
            try:
                slconfig_path.write_text(original_content, encoding="utf-8")
            except Exception:
                pass


def run_worker(payload: dict[str, Any]) -> None:
    """Run one DINO-DETR detection training job."""
    job_id = str(payload["job_id"])
    config = dict(payload.get("config") or {})
    model_scale = str(payload.get("model_scale") or "").lower() or "resnet50"
    
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _set_job(
        job_id,
        status="running",
        started_at=datetime.utcnow().isoformat() + "Z",
        message="Preparing DINO-DETR detection training...",
    )
    
    def _log_fn(msg: str) -> None:
        _log(job_id, "INFO", msg)
    
    # Ensure DINO-DETR is installed
    from dino_detr.installer import ensure_installed
    root = ensure_installed(log_fn=_log_fn)
    if not (root / "main.py").exists():
        raise RuntimeError(f"DINO-DETR main.py not found: {root / 'main.py'}")
    
    # Get dataset
    data_arg = str(config.get("data") or "")
    if not data_arg:
        raise ValueError("Dataset name (config.data) is required for DINO-DETR training")
    
    from app.config import DATASETS_DIR
    dataset_dir = DATASETS_DIR / data_arg
    if not dataset_dir.exists():
        raise ValueError(f"Dataset not found: {dataset_dir}")
    
    # Convert YOLO annotations to COCO format (non-destructive)
    # COCO annotations are created in a separate 'annotations' directory
    # Original YOLO labels remain unchanged for other models to use
    annotations_dir = _convert_yolo_to_coco(job_id, dataset_dir)
    
    # Create data.yaml for DINO-DETR
    data_yaml = job_dir / "data.yaml"
    src_data_yaml = dataset_dir / "data.yaml"
    if src_data_yaml.exists():
        import shutil
        shutil.copy2(src_data_yaml, data_yaml)
    else:
        # Create minimal data.yaml
        import yaml
        data_yaml_content = {
            "path": str(dataset_dir),
            "train": "train.txt",
            "val": "val.txt",
            "names": ["Car", "Pedestrian", "Cyclist", "Truck", "Van", "Tram", "Misc"],
        }
        with data_yaml.open("w") as f:
            yaml.dump(data_yaml_content, f)
    
    # Training parameters
    epochs = int(config.get("epochs", 300))
    batch = int(config.get("batch", 16))
    workers = int(config.get("workers", 8))
    lr = float(config.get("lr0", 0.0001))
    weight_decay = float(config.get("weight_decay", 0.05))
    
    _log_fn(f"DINO-DETR detection training: epochs={epochs}, batch={batch}, workers={workers}, lr={lr}")
    
    # Patch vendor code to fix IndentationError
    original_slconfig = _patch_vendor_code(root, job_id)
    
    # Build DINO-DETR training command
    cmd = [
        sys.executable,
        "main.py",
        "-c",
        "config/DINO/DINO_4scale.py",
        "--coco_path",
        str(dataset_dir),
        "--output_dir",
        str(job_dir / "runs" / "dino_detr"),
        "--num_workers",
        str(workers),
        "--options",
        f"epochs={epochs}",
        f"batch_size={batch}",
        f"lr={lr}",
        f"weight_decay={weight_decay}",
    ]
    
    if config.get("amp", True):
        cmd.append("--amp")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root), env.get("PYTHONPATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    _log_fn(f"Running DINO-DETR training: {' '.join(cmd)}")
    
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    started = time.time()
    
    # Stream output
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        _log(job_id, "INFO", line)
    
    proc.wait()
    
    elapsed = time.time() - started
    
    # Restore original vendor code
    _restore_vendor_code(root, original_slconfig)
    
    if proc.returncode != 0:
        _set_job(
            job_id,
            status="failed",
            message=f"DINO-DETR training failed with code {proc.returncode}",
            completed_at=datetime.utcnow().isoformat() + "Z",
        )
        raise RuntimeError(f"DINO-DETR training failed with code {proc.returncode}")
    
    # Save weight
    checkpoint_path = job_dir / "runs" / "dino_detr" / "checkpoint.pth"
    if checkpoint_path.exists():
        weight_id = weight_storage.save_weight_meta(
            model_id=payload.get("model_id", ""),
            model_name=payload.get("model_name", "DINO-DETR"),
            model_scale=model_scale,
            job_id=job_id,
            dataset=data_arg,
            epochs_trained=epochs,
            final_accuracy=None,
            final_loss=None,
            weight_id=None,
        )
        _log_fn(f"Saved weight: {weight_id}")
    else:
        weight_id = None
        _log_fn("Warning: No checkpoint found")
    
    _set_job(
        job_id,
        status="completed",
        epoch=epochs,
        message="DINO-DETR detection training complete",
        weight_id=weight_id,
        completed_at=datetime.utcnow().isoformat() + "Z",
    )
    _log_fn(f"DINO-DETR detection training completed in {elapsed:.1f}s")
