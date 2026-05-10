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

import torch

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


def _emit_extended_metrics(job_id: str, epoch: int, metrics: dict[str, Any]) -> None:
    """Emit epoch metrics to extended_metrics.jsonl for frontend charts.
    
    Field names MUST match what job_storage.get_job_history() expects:
      train_box_loss, train_cls_loss, train_dfl_loss,
      map50, map (=mAP50-95), map75, precision, recall, lr, etc.
    """
    ext_metrics_path = JOBS_DIR / job_id / "extended_metrics.jsonl"
    try:
        epoch_data: dict[str, Any] = {
            "epoch": epoch,
            "timestamp": time.time(),
        }
        
        # Training losses (map DINO-DETR → Ultralytics-style names)
        if "loss_bbox" in metrics:
            epoch_data["train_box_loss"] = metrics["loss_bbox"]
        if "loss_ce" in metrics:
            epoch_data["train_cls_loss"] = metrics["loss_ce"]
        if "loss_giou" in metrics:
            epoch_data["train_dfl_loss"] = metrics["loss_giou"]
        
        # Validation metrics (from evaluation)
        if "map" in metrics:
            epoch_data["map"] = metrics["map"]  # mAP50-95
        if "map50" in metrics:
            epoch_data["map50"] = metrics["map50"]
        if "map75" in metrics:
            epoch_data["map75"] = metrics["map75"]
        
        # Learning rate
        if "lr" in metrics:
            epoch_data["lr"] = metrics["lr"]
        
        # Remove None values
        epoch_data = {k: v for k, v in epoch_data.items() if v is not None}
        
        # Write to extended_metrics.jsonl
        with ext_metrics_path.open("a") as mf:
            mf.write(json.dumps(epoch_data) + "\n")
    except Exception as e:
        _log(job_id, "WARNING", f"Failed to write extended_metrics.jsonl: {e}")


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
            image_ext = ".jpg"
            if not image_path.exists():
                image_path = images_dir / f"{image_file}.png"
                image_ext = ".png"
            
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
                "file_name": f"{image_file}{image_ext}",
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
                    
                    # Validate bounding box coordinates
                    # Skip invalid boxes (negative dimensions or out of bounds)
                    if box_width <= 0 or box_height <= 0:
                        continue
                    
                    # Clamp to image boundaries
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img_width, x_min + box_width)
                    y_max = min(img_height, y_min + box_height)
                    
                    # Recalculate width/height after clamping
                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    
                    # Skip if box becomes invalid after clamping
                    if box_width <= 0 or box_height <= 0:
                        continue
                    
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
    
    # Create symlinks for COCO directory structure (train2017, val2017)
    # DINO-DETR expects images in train2017/ and val2017/ directories
    train2017_link = dataset_dir / "train2017"
    val2017_link = dataset_dir / "val2017"
    
    if not train2017_link.exists():
        train2017_link.symlink_to(train_images)
        _log(job_id, "INFO", f"Created symlink: {train2017_link} -> {train_images}")
    
    if not val2017_link.exists():
        val2017_link.symlink_to(val_images)
        _log(job_id, "INFO", f"Created symlink: {val2017_link} -> {val_images}")
    
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


def _parse_dino_detr_metrics(line: str) -> dict[str, Any] | None:
    """Parse DINO-DETR stdout log line for metrics.

    Format 1: "Test:  [ 160/5985]  eta: 0:10:53  lr: 0.000100  class_error: 66.67  loss: 10.2455 (10.7039)  ..."
    Format 2: "Epoch: [0]  [1040/5985]  eta: 0:08:22  lr: 0.000100  ..."
    """
    metrics: dict[str, Any] = {}

    # Only parse lines that match the DINO-DETR training output format
    # Must start with "Test:" or "Epoch:" followed by "[iter/total]"
    m = re.match(r'(?:Test:|Epoch:)\s*\[\s*(\d+)/(\d+)\]', line)
    if not m:
        return None  # Not a DINO-DETR training progress line
    
    # Parse iteration and total_iterations
    metrics["iteration"] = int(m.group(1))
    metrics["total_iterations"] = int(m.group(2))
    
    # Epoch is not in the line - will be tracked externally via iteration count

    # Parse ETA from DINO-DETR: "eta: 0:10:53" (format: H:MM:SS)
    m = re.search(r'eta:\s*([\d:]+)', line)
    if m:
        eta_str = m.group(1)
        # Parse H:MM:SS format
        parts = eta_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            metrics["eta_s"] = hours * 3600 + minutes * 60 + seconds

    # Parse class_error: "class_error: 0.00"
    m = re.search(r"class_error:\s*([\d.]+)", line)
    if m:
        metrics["class_error"] = float(m.group(1))

    # Parse loss (current and running average): "loss: 13.8408 (15.2805)"
    m = re.search(r"loss:\s*([\d.]+)\s*\(([\d.]+)\)", line)
    if m:
        metrics["train_loss"] = float(m.group(1))
        metrics["train_loss_avg"] = float(m.group(2))

    # Parse key DINO-DETR loss components (simplified metrics like hsg-detr)
    # Only extract main losses: loss_ce, loss_bbox, loss_giou (skip detailed decoder-specific losses)
    m = re.search(r"loss_ce:\s*([\d.]+)\s*\(([\d.]+)\)", line)
    if m:
        metrics["loss_ce"] = float(m.group(1))

    m = re.search(r"loss_bbox:\s*([\d.]+)\s*\(([\d.]+)\)", line)
    if m:
        metrics["loss_bbox"] = float(m.group(1))

    m = re.search(r"loss_giou:\s*([\d.]+)\s*\(([\d.]+)\)", line)
    if m:
        metrics["loss_giou"] = float(m.group(1))

    # Parse time per iteration: "time: 0.0955"
    m = re.search(r"time:\s*([\d.]+)", line)
    if m:
        metrics["time_per_iter"] = float(m.group(1))

    # Parse data time: "data: 0.0017"
    m = re.search(r"data:\s*([\d.]+)", line)
    if m:
        metrics["data_time"] = float(m.group(1))

    # Parse max memory: "max mem: 3792"
    m = re.search(r"max mem:\s*(\d+)", line)
    if m:
        metrics["max_mem_mb"] = int(m.group(1))

    return metrics


def _parse_dino_detr_eval_metrics(line: str) -> dict[str, Any] | None:
    """Parse DINO-DETR evaluation output for mAP metrics.

    Format: "Average Precision  (AP) @[ IoU=0.50:0.95 | area=  all | maxDets=100 ] = 0.123"
    """
    metrics: dict[str, Any] = {}

    # Parse AP (mAP@[0.50:0.95]): "Average Precision  (AP) @[ IoU=0.50:0.95 | area=  all | maxDets=100 ] = 0.123"
    if "IoU=0.50:0.95" in line and "] =" in line:
        m = re.search(r"\]\s*=\s*([\d.]+)", line)
        if m:
            metrics["map"] = float(m.group(1))

    # Parse AP50: "Average Precision  (AP) @[ IoU=0.50      | area=  all | maxDets=100 ] = 0.345"
    # Match line with "IoU=0.50" followed by spaces and "|" (not ":0.95")
    # Use word boundary or check that "IoU=0.50" is not followed by ":0.95"
    if "IoU=0.50" in line and "IoU=0.50:0.95" not in line and "] =" in line:
        m = re.search(r"\]\s*=\s*([\d.]+)", line)
        if m:
            metrics["map50"] = float(m.group(1))

    # Parse AP75: "Average Precision  (AP) @[ IoU=0.75      | area=  all | maxDets=100 ] = 0.567"
    if "IoU=0.75" in line and "] =" in line:
        m = re.search(r"\]\s*=\s*([\d.]+)", line)
        if m:
            metrics["map75"] = float(m.group(1))

    # Only return metrics if we found at least one
    if metrics:
        return metrics
    return None


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
    
    # Get DINO-DETR venv Python
    venv_python = root / "venv" / "bin" / "python3"
    if not venv_python.exists():
        venv_python = Path(sys.executable)  # Fallback to system Python
    
    # Training parameters
    epochs = int(config.get("epochs", 300))
    batch = int(config.get("batch", 16))
    workers = int(config.get("workers", 8))
    lr = float(config.get("lr0", 0.0001))
    weight_decay = float(config.get("weight_decay", 0.05))
    
    # Clamp learning rate to safe range for DINO-DETR (0.0001 is default)
    if lr > 0.001:
        _log_fn(f"WARNING: Learning rate {lr} is too high for DINO-DETR. Clamping to 0.0001.")
        lr = 0.0001
    
    _log_fn(f"DINO-DETR detection training: epochs={epochs}, batch={batch}, workers={workers}, lr={lr}")
    
    # Disable periodic checkpoints to prevent bloat (keep only checkpoint.pth)
    # DINO-DETR saves checkpoint.pth by default, we don't need periodic saves
    save_period = epochs + 1  # Save only at the end
    
    # Patch vendor code to fix IndentationError
    original_slconfig = _patch_vendor_code(root, job_id)
    
    # Build DINO-DETR training command
    # Note: DINO-DETR doesn't support imgsz parameter like YOLO
    # It uses data_aug_scales in config file, but --options doesn't support list parsing
    # We'll ignore imgsz parameter and use default config
    imgsz_val = config.get("imgsz")
    if imgsz_val is not None:
        _log_fn(f"WARNING: imgsz parameter ({imgsz_val}) is not supported for DINO-DETR. Using default data_aug_scales from config file.")
    
    cmd = [
        str(venv_python),
        "main.py",
        "-c", "config/DINO/DINO_4scale.py",
        "--coco_path", str(dataset_dir),
        "--output_dir", str(job_dir / "runs" / "dino_detr"),
        "--num_workers", str(config.get("workers", 2)),
        "--options", f"epochs={epochs}", f"batch_size={batch}", f"lr={lr}", f"weight_decay={weight_decay}",
        "--amp",  # Enable mixed precision
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
    seen_epochs = set()  # Track seen epochs for dedup
    current_epoch = 0  # Track current epoch externally (not in log line)
    last_completed_epoch = 0  # Track last completed epoch for evaluation metrics matching
    accumulated_eval_metrics = {}  # Accumulate evaluation metrics for the current epoch
    
    # Stream output and parse metrics
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        _log(job_id, "INFO", line)
        
        # Parse metrics from stdout
        metrics = _parse_dino_detr_metrics(line)
        eval_metrics = _parse_dino_detr_eval_metrics(line)
        
        if metrics:
            # Calculate epoch from iteration count (DINO-DETR doesn't output epoch in Test: lines)
            iteration = metrics.get("iteration")
            total_iterations = metrics.get("total_iterations")
            
            # Estimate epoch based on iteration count
            # Assuming total_iterations is the total iterations per epoch
            if iteration and total_iterations:
                # Calculate which epoch we're in (1-indexed)
                epoch = ((iteration - 1) // total_iterations) + 1
                # Update current_epoch if it increased
                if epoch > current_epoch:
                    # Track the last completed epoch before incrementing
                    last_completed_epoch = current_epoch
                    current_epoch = epoch
            else:
                epoch = current_epoch
            
            metrics["epoch"] = epoch
            
            # Update job record with current epoch
            if epoch is not None:
                _set_job(job_id, epoch=epoch)
            
            # Emit PROGRESS log entry → log.jsonl → SSE via stream_controller
            if epoch is not None:
                import psutil
                system_res = {
                    "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                    "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                }
                now = time.time()
                total_elapsed_s = round(now - started, 1)
                time_per_iter = metrics.get("time_per_iter")
                imgs_per_sec = round(batch / time_per_iter, 1) if time_per_iter and time_per_iter > 0 else None
                
                # Use iteration for progress within epoch if available
                iteration = metrics.get("iteration")
                total_iterations = metrics.get("total_iterations")
                if iteration and total_iterations:
                    # Calculate overall progress: (completed epochs + current epoch progress) / total epochs
                    epoch_progress = iteration / total_iterations
                    pct = round(((epoch - 1 + epoch_progress) / epochs) * 100, 1)
                else:
                    pct = round((epoch / epochs) * 100, 1) if epoch and epochs else 0.0
                
                # Compute avg_epoch_s and total ETA
                # Use ETA from DINO-DETR logs if available, otherwise calculate
                eta_s = metrics.get("eta_s")
                if eta_s:
                    # DINO-DETR's ETA is for the current epoch, multiply by remaining epochs for total
                    eta_s = round(eta_s * (epochs - epoch + 1), 0)
                else:
                    # Calculate from elapsed time
                    if epoch > 0 and pct > 0:
                        avg_epoch_s = total_elapsed_s / epoch
                        eta_s = round(avg_epoch_s * (epochs - epoch + 1), 0)
                    else:
                        eta_s = None
                
                # Get learning rate
                lr = metrics.get("lr")
                if lr is None:
                    # Try to parse lr from DINO-DETR logs
                    m = re.search(r"lr:\s*([\d.]+)", line)
                    if m:
                        lr = float(m.group(1))
                
                progress_data = {
                    "type": "progress",
                    "phase": "train",
                    "epoch": f"{epoch}/{epochs}",
                    "batch": f"{iteration}/{total_iterations}" if iteration and total_iterations else "0/0",
                    "percent": pct,
                    "losses": {
                        "total": metrics.get("train_loss"),
                        "ce": metrics.get("loss_ce"),
                        "bbox": metrics.get("loss_bbox"),
                        "giou": metrics.get("loss_giou"),
                        "class_error": metrics.get("class_error"),
                    },
                    "val_map50": None,
                    "val_map": None,
                    "val_map75": None,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "ram_gb": system_res["ram_used_gb"],
                    "ram_total_gb": system_res["ram_total_gb"],
                    "gpu_mem_gb": None,
                    "total_elapsed_s": total_elapsed_s,
                    "epoch_elapsed_s": None,
                    "avg_epoch_s": total_elapsed_s / epoch if epoch > 0 else None,
                    "eta_s": eta_s,
                    "imgs_per_sec": imgs_per_sec,
                    "lr": lr,
                }
                job_storage.append_job_log(
                    job_id,
                    "PROGRESS",
                    f"Epoch {epoch}/{epochs} | {pct}%",
                    progress_data,
                )

            # Emit extended_metrics.jsonl entry when epoch changes
            if epoch not in seen_epochs:
                seen_epochs.add(epoch)
                
                # Emit extended_metrics.jsonl entry for frontend charts
                extended_data = {
                    "epoch": epoch,
                    "loss_bbox": metrics.get("loss_bbox"),
                    "loss_ce": metrics.get("loss_ce"),
                    "loss_giou": metrics.get("loss_giou"),
                    "lr": lr,
                }
                _emit_extended_metrics(job_id, epoch, extended_data)
        
        # Accumulate evaluation metrics instead of emitting immediately
        if eval_metrics:
            accumulated_eval_metrics.update(eval_metrics)
        
        # Detect end of evaluation and emit accumulated metrics
        # Evaluation typically ends after all IoU thresholds are reported
        # We emit when we have all three metrics (map, map50, map75)
        if accumulated_eval_metrics and "map" in accumulated_eval_metrics and "map50" in accumulated_eval_metrics and "map75" in accumulated_eval_metrics:
            # Update the existing extended_metrics entry for this epoch with validation metrics
            # Read existing extended_metrics.jsonl and update the entry
            ext_metrics_path = JOBS_DIR / job_id / "extended_metrics.jsonl"
            try:
                updated_lines = []
                with ext_metrics_path.open("r") as mf:
                    for line in mf:
                        entry = json.loads(line)
                        if entry.get("epoch") == last_completed_epoch:
                            # Update with validation metrics
                            entry.update(accumulated_eval_metrics)
                        updated_lines.append(json.dumps(entry))
                # Write back updated content
                with ext_metrics_path.open("w") as mf:
                    for line in updated_lines:
                        mf.write(line + "\n")
            except Exception as e:
                _log(job_id, "WARNING", f"Failed to update extended_metrics.jsonl with validation metrics: {e}")
            # Clear accumulated metrics
            accumulated_eval_metrics = {}
    
    proc.wait()
    
    elapsed = time.time() - started
    
    # Restore original vendor code
    _restore_vendor_code(root, original_slconfig)
    
    if proc.returncode != 0:
        # Check if error was due to CUDA OOM
        # Read log.jsonl to check for OOM error
        log_path = job_dir / "log.jsonl"
        is_oom = False
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    for line in f:
                        if "CUDA out of memory" in line:
                            is_oom = True
                            break
            except Exception:
                pass
        
        if is_oom:
            error_msg = "DINO-DETR training failed: CUDA out of memory (OOM). GPU memory insufficient for the current configuration. Try reducing batch size, image size, or using a smaller model."
            _log_fn(f"ERROR: {error_msg}")
        else:
            error_msg = f"DINO-DETR training failed with code {proc.returncode}"
        
        _set_job(
            job_id,
            status="failed",
            message=error_msg,
            completed_at=datetime.utcnow().isoformat() + "Z",
        )
        raise RuntimeError(error_msg)
    
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
