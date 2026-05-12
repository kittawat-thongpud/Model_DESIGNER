"""DINO-DETR detection trainer.

This module handles DINO-DETR object detection training using the IDEA-Research/DINO-DETR codebase.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from app.config import JOBS_DIR
from app.constants import job_channel, train_channel
from app.services import event_bus, job_storage, weight_storage
from app.services.dataset_yaml import write_data_yaml


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _publish(job_id: str, job: dict) -> None:
    """Publish DINO-DETR job state to the same channels as other trainers."""
    event_bus.publish_sync(train_channel(job_id), {"type": "status", **job})
    event_bus.publish_sync(job_channel(job_id), {
        "type": "job_update",
        "job_id": job_id,
        "status": job.get("status", "running"),
        "epoch": job.get("epoch", 0),
        "total_epochs": job.get("total_epochs", 0),
        "message": job.get("message", ""),
        "best_fitness": job.get("best_fitness"),
        "best_mAP50": job.get("best_mAP50"),
        "best_mAP50_95": job.get("best_mAP50_95"),
    })


def _set_job(job_id: str, **updates: Any) -> dict | None:
    """Update job record and publish event."""
    job = job_storage.load_job(job_id)
    if not job:
        return None
    job.update(updates)
    job_storage.save_job(job)
    _publish(job_id, job)
    return job


def _log(job_id: str, level: str, message: str, data: dict = None) -> None:
    """Append log to job."""
    job_storage.append_job_log(job_id, level, message, data or {})


def repo_dir() -> Path:
    """Get DINO-DETR vendor directory."""
    from app.config import DATA_DIR
    return DATA_DIR / "vendor" / "DINO-DETR"


def _json_safe(value: Any) -> Any:
    """Convert values to JSON-safe scalars while dropping NaN/Inf."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        out = {str(k): _json_safe(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _upsert_extended_metrics(job_id: str, epoch: int, metrics: dict[str, Any]) -> None:
    """Upsert epoch metrics to extended_metrics.jsonl for frontend charts.

    Field names MUST match what job_storage.get_job_history() expects:
      train_box_loss, train_cls_loss, train_dfl_loss,
      map50, map (=mAP50-95), map75, precision, recall, lr, etc.
    """
    ext_metrics_path = JOBS_DIR / job_id / "extended_metrics.jsonl"
    try:
        rows: dict[int, dict[str, Any]] = {}
        if ext_metrics_path.exists():
            with ext_metrics_path.open("r", encoding="utf-8") as mf:
                for raw in mf:
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    row_epoch = row.get("epoch")
                    if row_epoch is not None:
                        rows[int(row_epoch)] = row

        epoch_data = rows.get(epoch, {"epoch": epoch})
        epoch_data["timestamp"] = time.time()

        # Training losses. DINO-DETR loss rows expose both current and running
        # average values; epoch charts should use the running average.
        if "loss_bbox_avg" in metrics or "loss_bbox" in metrics:
            epoch_data["train_box_loss"] = metrics.get("loss_bbox_avg", metrics.get("loss_bbox"))
        if "loss_ce_avg" in metrics or "loss_ce" in metrics:
            epoch_data["train_cls_loss"] = metrics.get("loss_ce_avg", metrics.get("loss_ce"))
        if "loss_giou_avg" in metrics or "loss_giou" in metrics:
            epoch_data["train_dfl_loss"] = metrics.get("loss_giou_avg", metrics.get("loss_giou"))
            epoch_data["train_giou_loss"] = metrics.get("loss_giou_avg", metrics.get("loss_giou"))
        if "loss_avg" in metrics or "train_loss_avg" in metrics:
            epoch_data["train_loss"] = metrics.get("loss_avg", metrics.get("train_loss_avg"))

        # Validation losses from Test rows.
        if "val_loss_bbox_avg" in metrics or "val_loss_bbox" in metrics:
            epoch_data["val_box_loss"] = metrics.get("val_loss_bbox_avg", metrics.get("val_loss_bbox"))
        if "val_loss_ce_avg" in metrics or "val_loss_ce" in metrics:
            epoch_data["val_cls_loss"] = metrics.get("val_loss_ce_avg", metrics.get("val_loss_ce"))
        if "val_loss_giou_avg" in metrics or "val_loss_giou" in metrics:
            epoch_data["val_dfl_loss"] = metrics.get("val_loss_giou_avg", metrics.get("val_loss_giou"))
            epoch_data["val_giou_loss"] = metrics.get("val_loss_giou_avg", metrics.get("val_loss_giou"))

        for key in ("map", "map50", "map75", "precision", "recall", "fitness", "lr"):
            if key in metrics:
                epoch_data[key] = metrics[key]

        if "time_per_iter" in metrics:
            epoch_data["train_time_per_iter_s"] = metrics["time_per_iter"]
        if "val_time_per_iter" in metrics:
            epoch_data["val_time_per_iter_s"] = metrics["val_time_per_iter"]
        if "max_mem_mb" in metrics:
            epoch_data["gpu_mem_gb"] = round(float(metrics["max_mem_mb"]) / 1024.0, 3)

        dino_payload = metrics.get("dino_detr")
        if isinstance(dino_payload, dict) and dino_payload:
            existing_payload = epoch_data.get("dino_detr")
            if isinstance(existing_payload, dict):
                merged_payload = dict(existing_payload)
                phase = str(dino_payload.get("phase") or "metrics")
                merged_payload[phase] = dino_payload
                epoch_data["dino_detr"] = merged_payload
            else:
                phase = str(dino_payload.get("phase") or "metrics")
                epoch_data["dino_detr"] = {phase: dino_payload}

        epoch_data = {k: _json_safe(v) for k, v in epoch_data.items()}
        epoch_data = {k: v for k, v in epoch_data.items() if v is not None}
        rows[epoch] = epoch_data

        with ext_metrics_path.open("w", encoding="utf-8") as mf:
            for row_epoch in sorted(rows):
                mf.write(json.dumps(rows[row_epoch]) + "\n")
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

    def _read_data_yaml() -> dict[str, Any]:
        try:
            import yaml
            data_yaml = dataset_dir / "data.yaml"
            if data_yaml.exists():
                return yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        except Exception as e:
            _log(job_id, "WARNING", f"Failed to read data.yaml for DINO-DETR conversion: {e}")
        return {}

    data_cfg = _read_data_yaml()

    def _resolve_split_path(split: str) -> Path:
        raw_value = data_cfg.get(split)
        if raw_value is None:
            return dataset_dir / "images" / split
        if isinstance(raw_value, list):
            raw_value = raw_value[0] if raw_value else ""
        raw = str(raw_value).split("#", 1)[0].strip()
        path = Path(raw)
        if not path.is_absolute():
            base = Path(str(data_cfg.get("path") or dataset_dir))
            path = base / path
        return path

    def _image_files_from_split(split_path: Path) -> list[Path]:
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if split_path.is_file():
            base = split_path.parent
            files: list[Path] = []
            for raw in split_path.read_text(encoding="utf-8").splitlines():
                item = raw.split("#", 1)[0].strip()
                if not item:
                    continue
                p = Path(item)
                if not p.is_absolute():
                    p = base / p
                if p.suffix.lower() in image_exts:
                    files.append(p)
            return files
        if split_path.is_dir():
            return sorted(p for p in split_path.rglob("*") if p.suffix.lower() in image_exts)
        return []

    def _label_for_image(image_path: Path) -> Path:
        parts = list(image_path.parts)
        if "images" in parts:
            idx = parts.index("images")
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
        return dataset_dir / "labels" / image_path.with_suffix(".txt").name

    def _categories() -> list[dict[str, Any]]:
        names = data_cfg.get("names")
        if isinstance(names, dict):
            return [
                {"id": int(k), "name": str(v), "supercategory": "object"}
                for k, v in sorted(names.items(), key=lambda item: int(item[0]))
            ]
        if isinstance(names, list):
            return [
                {"id": i, "name": str(name), "supercategory": "object"}
                for i, name in enumerate(names)
            ]
        nc = int(data_cfg.get("nc") or 1)
        return [{"id": i, "name": f"class_{i}", "supercategory": "object"} for i in range(nc)]

    def _build_flat_symlink_dir(link_dir: Path, image_files: list[Path]) -> None:
        """Create a flat directory of symlinks so DINO-DETR can find images by basename."""
        link_dir.mkdir(parents=True, exist_ok=True)
        for img in image_files:
            if not img.exists():
                continue
            dst = link_dir / img.name
            if dst.exists() or dst.is_symlink():
                continue
            try:
                dst.symlink_to(img.resolve())
            except Exception:
                pass

    def create_coco_annotation(image_files: list[Path], output_path: Path, split_name: str, images_root: Path | None = None):
        """Create COCO annotation file from YOLO labels."""
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": _categories(),
        }

        annotation_id = 0
        image_id = 0

        if not image_files:
            _log(job_id, "WARNING", f"No images found for DINO-DETR {split_name} split")

        for image_path in image_files:
            if not image_path.exists():
                continue
            label_file = _label_for_image(image_path)

            # Get image dimensions
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception:
                img_width, img_height = 640, 480  # Default dimensions

            # file_name: relative from images_root if provided, else basename only
            if images_root is not None:
                try:
                    file_name = str(image_path.resolve().relative_to(images_root.resolve()))
                except ValueError:
                    file_name = image_path.name
            else:
                file_name = image_path.name

            # Add image to COCO
            coco_output["images"].append({
                "id": image_id,
                "file_name": file_name,
                "width": img_width,
                "height": img_height,
            })

            # Parse YOLO annotations
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        class_id = int(float(parts[0]))
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
                            "category_id": class_id,
                            "bbox": [x_min, y_min, box_width, box_height],
                            "area": box_width * box_height,
                            "iscrowd": 0,
                        })
                        annotation_id += 1

            image_id += 1

        # Write COCO annotation file
        with open(output_path, 'w') as f:
            json.dump(coco_output, f, indent=2)

        _log(
            job_id,
            "INFO",
            (
                f"Created COCO annotation: {output_path} with "
                f"{len(coco_output['images'])} images and {len(coco_output['annotations'])} boxes"
            ),
        )

    # Convert train and val splits
    train_images = _resolve_split_path("train")
    val_images = _resolve_split_path("val")
    train_files = _image_files_from_split(train_images)
    val_files = _image_files_from_split(val_images)

    if not val_files and train_files:
        _log(job_id, "WARNING", "No validation split found; using train split for DINO-DETR validation")
        val_images = train_images
        val_files = train_files

    # Determine images_root: the common ancestor directory for each split's images.
    # For nested datasets (e.g. images/frontFar/.../xxx.jpg), images_root = dataset_dir/images
    # so file_name in COCO = "frontFar/.../xxx.jpg" and train2017 symlinks to images_root.
    def _find_images_root(split_path: Path, files: list[Path]) -> Path:
        """Return the deepest common ancestor that is a real directory."""
        if split_path.is_dir():
            return split_path
        # split_path is a .txt file listing absolute paths
        # Use the dataset images dir if it exists, else common parent of files
        img_dir = dataset_dir / "images"
        if img_dir.is_dir() and files:
            try:
                files[0].resolve().relative_to(img_dir.resolve())
                return img_dir
            except ValueError:
                pass
        if files:
            return files[0].parent
        return dataset_dir

    train_root = _find_images_root(train_images, train_files)
    val_root = _find_images_root(val_images, val_files)

    create_coco_annotation(train_files, train_ann, "train", images_root=train_root)
    create_coco_annotation(val_files, val_ann, "val", images_root=val_root)

    # Create symlinks for COCO directory structure (train2017, val2017)
    # Point to the images_root so relative file_name paths resolve correctly.
    train2017_link = dataset_dir / "train2017"
    val2017_link = dataset_dir / "val2017"

    def _ensure_dir_link(link: Path, target: Path) -> None:
        if link.is_symlink():
            if link.resolve() == target.resolve():
                return
            link.unlink()
        elif link.exists():
            return  # real dir already there, don't touch
        link.symlink_to(target)

    if train_root.is_dir():
        _ensure_dir_link(train2017_link, train_root)
        _log(job_id, "INFO", f"Symlink: {train2017_link} -> {train_root}")

    if val_root.is_dir():
        _ensure_dir_link(val2017_link, val_root)
        _log(job_id, "INFO", f"Symlink: {val2017_link} -> {val_root}")

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

    # Training:   Epoch: [98]  [5920/5985]  eta: ...
    # Validation: Test:  [160/5985]  eta: ...
    train_match = re.match(r"Epoch:\s*\[\s*(\d+)\s*\]\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]", line)
    test_match = re.match(r"Test:\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]", line)
    if train_match:
        metrics["phase"] = "train"
        metrics["epoch_zero_based"] = int(train_match.group(1))
        metrics["epoch"] = int(train_match.group(1)) + 1
        metrics["iteration"] = int(train_match.group(2))
        metrics["total_iterations"] = int(train_match.group(3))
    elif test_match:
        metrics["phase"] = "val"
        metrics["iteration"] = int(test_match.group(1))
        metrics["total_iterations"] = int(test_match.group(2))
    else:
        return None

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
    m = re.search(rf"class_error:\s*({_FLOAT_RE})", line)
    if m:
        metrics["class_error"] = float(m.group(1))

    # Parse every DINO-DETR meter pair: "name: current (running_avg)".
    # Keeping the full payload makes the wrapper useful for diagnostics without
    # hardcoding every decoder-layer/DN loss into the frontend fields.
    detailed: dict[str, float] = {}
    for name, current, avg in re.findall(rf"([A-Za-z_][\w]*)\s*:\s*({_FLOAT_RE})\s*\(({_FLOAT_RE})\)", line):
        current_v = float(current)
        avg_v = float(avg)
        metrics[name] = current_v
        metrics[f"{name}_avg"] = avg_v
        detailed[name] = current_v
        detailed[f"{name}_avg"] = avg_v

    if "loss" in metrics:
        metrics["train_loss"] = metrics["loss"]
        metrics["train_loss_avg"] = metrics.get("loss_avg")

    m = re.search(rf"\blr:\s*({_FLOAT_RE})", line)
    if m:
        metrics["lr"] = float(m.group(1))

    # Parse time per iteration: "time: 0.0955"
    m = re.search(rf"\btime:\s*({_FLOAT_RE})", line)
    if m:
        metrics["time_per_iter"] = float(m.group(1))

    # Parse data time: "data: 0.0017"
    m = re.search(rf"\bdata:\s*({_FLOAT_RE})", line)
    if m:
        metrics["data_time"] = float(m.group(1))

    # Parse max memory: "max mem: 3792"
    m = re.search(r"max mem:\s*(\d+)", line)
    if m:
        metrics["max_mem_mb"] = int(m.group(1))

    if detailed:
        detailed["phase"] = metrics.get("phase")
        detailed["iteration"] = metrics.get("iteration")
        detailed["total_iterations"] = metrics.get("total_iterations")
        if "epoch" in metrics:
            detailed["epoch"] = metrics["epoch"]
        if "lr" in metrics:
            detailed["lr"] = metrics["lr"]
        if "class_error" in metrics:
            detailed["class_error"] = metrics["class_error"]
        metrics["dino_detr"] = detailed

    return metrics


def _parse_dino_detr_eval_metrics(line: str) -> dict[str, Any] | None:
    """Parse DINO-DETR evaluation output for mAP metrics.

    Format: "Average Precision  (AP) @[ IoU=0.50:0.95 | area=  all | maxDets=100 ] = 0.123"
    """
    metrics: dict[str, Any] = {}
    if not re.search(r"Average Precision\s+\(AP\)", line):
        return None
    if not re.search(r"area=\s*all", line):
        return None

    # Parse AP (mAP@[0.50:0.95]): "Average Precision  (AP) @[ IoU=0.50:0.95 | area=  all | maxDets=100 ] = 0.123"
    if "IoU=0.50:0.95" in line and "] =" in line:
        m = re.search(rf"\]\s*=\s*({_FLOAT_RE})", line)
        if m:
            metrics["map"] = float(m.group(1))

    # Parse AP50: "Average Precision  (AP) @[ IoU=0.50      | area=  all | maxDets=100 ] = 0.345"
    # Match line with "IoU=0.50" followed by spaces and "|" (not ":0.95")
    # Use word boundary or check that "IoU=0.50" is not followed by ":0.95"
    if "IoU=0.50" in line and "IoU=0.50:0.95" not in line and "] =" in line:
        m = re.search(rf"\]\s*=\s*({_FLOAT_RE})", line)
        if m:
            metrics["map50"] = float(m.group(1))

    # Parse AP75: "Average Precision  (AP) @[ IoU=0.75      | area=  all | maxDets=100 ] = 0.567"
    if "IoU=0.75" in line and "] =" in line:
        m = re.search(rf"\]\s*=\s*({_FLOAT_RE})", line)
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
    configured_save_period = int(config.get("save_period", -1) or -1)
    save_period = configured_save_period if configured_save_period > 0 else epochs + 1

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
        "--options",
        f"epochs={epochs}",
        f"batch_size={batch}",
        f"lr={lr}",
        f"weight_decay={weight_decay}",
        f"save_checkpoint_interval={save_period}",
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
    current_epoch = 0
    current_train_metrics: dict[str, Any] = {}
    current_val_metrics: dict[str, Any] = {}
    accumulated_eval_metrics: dict[str, Any] = {}
    patience = int(config.get("patience", 0) or 0)
    best_map = None
    best_map50 = None
    best_epoch = None
    epochs_since_improve = 0
    stop_requested = False

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
            phase = metrics.get("phase", "train")
            iteration = metrics.get("iteration")
            total_iterations = metrics.get("total_iterations")

            if phase == "train":
                epoch = int(metrics.get("epoch") or current_epoch or 1)
                current_epoch = max(current_epoch, epoch)
                current_train_metrics = metrics
            else:
                epoch = current_epoch or 1
                metrics["epoch"] = epoch
                current_val_metrics = {
                    f"val_{k}": v
                    for k, v in metrics.items()
                    if k.startswith("loss") or k in ("time_per_iter", "data_time", "max_mem_mb")
                }
                if "time_per_iter" in metrics:
                    current_val_metrics["val_time_per_iter"] = metrics["time_per_iter"]

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

                lr_value = metrics.get("lr")
                train_snapshot = current_train_metrics if phase != "train" else metrics
                val_map50 = accumulated_eval_metrics.get("map50")
                val_map = accumulated_eval_metrics.get("map")
                val_map75 = accumulated_eval_metrics.get("map75")

                progress_data = {
                    "type": "progress",
                    "phase": "validation" if phase == "val" else "train",
                    "epoch": f"{epoch}/{epochs}",
                    "batch": f"{iteration}/{total_iterations}" if iteration and total_iterations else "0/0",
                    "percent": pct,
                    "losses": {
                        "total": train_snapshot.get("train_loss"),
                        "ce": train_snapshot.get("loss_ce"),
                        "bbox": train_snapshot.get("loss_bbox"),
                        "giou": train_snapshot.get("loss_giou"),
                        "class_error": train_snapshot.get("class_error"),
                    },
                    "val_map50": val_map50,
                    "val_map": val_map,
                    "val_map75": val_map75,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "ram_gb": system_res["ram_used_gb"],
                    "ram_total_gb": system_res["ram_total_gb"],
                    "gpu_mem_gb": None,
                    "total_elapsed_s": total_elapsed_s,
                    "epoch_elapsed_s": None,
                    "avg_epoch_s": total_elapsed_s / epoch if epoch > 0 else None,
                    "eta_s": eta_s,
                    "imgs_per_sec": imgs_per_sec,
                    "lr": lr_value or current_train_metrics.get("lr"),
                }
                job_storage.append_job_log(
                    job_id,
                    "PROGRESS",
                    f"DINO-DETR {progress_data['phase']} {epoch}/{epochs} | {pct}%",
                    progress_data,
                )

                event_bus.publish_sync(train_channel(job_id), progress_data)

            if phase == "train" and epoch is not None:
                _upsert_extended_metrics(job_id, epoch, metrics)
            elif phase == "val" and epoch is not None:
                merged_val = dict(current_val_metrics)
                merged_val["dino_detr"] = metrics.get("dino_detr")
                _upsert_extended_metrics(job_id, epoch, merged_val)

        # Accumulate evaluation metrics instead of emitting immediately
        if eval_metrics:
            accumulated_eval_metrics.update(eval_metrics)

        # Detect end of evaluation and emit accumulated metrics
        # Evaluation typically ends after all IoU thresholds are reported
        # We emit when we have all three metrics (map, map50, map75)
        if accumulated_eval_metrics and "map" in accumulated_eval_metrics and "map50" in accumulated_eval_metrics and "map75" in accumulated_eval_metrics:
            epoch_for_eval = current_epoch or 1
            eval_payload = dict(accumulated_eval_metrics)
            if current_val_metrics:
                eval_payload.update(current_val_metrics)
            _upsert_extended_metrics(job_id, epoch_for_eval, eval_payload)

            map50_95 = accumulated_eval_metrics.get("map")
            map50 = accumulated_eval_metrics.get("map50")
            if map50_95 is not None:
                improved = best_map is None or float(map50_95) > float(best_map)
                if improved:
                    best_map = float(map50_95)
                    best_map50 = float(map50) if map50 is not None else best_map50
                    best_epoch = epoch_for_eval
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1

            updates: dict[str, Any] = {
                "epoch": epoch_for_eval,
                "message": (
                    f"DINO-DETR validation epoch {epoch_for_eval}/{epochs}"
                    + (f" | mAP50={map50:.4f}" if map50 is not None else "")
                    + (f" | mAP50-95={map50_95:.4f}" if map50_95 is not None else "")
                ),
            }
            if best_map is not None:
                updates["best_mAP50_95"] = best_map
                updates["best_fitness"] = best_map
            if best_map50 is not None:
                updates["best_mAP50"] = best_map50
            _set_job(job_id, **updates)

            event_bus.publish_sync(train_channel(job_id), {
                "type": "epoch",
                "epoch": epoch_for_eval,
                "box_loss": current_train_metrics.get("loss_bbox_avg", current_train_metrics.get("loss_bbox", 0)),
                "cls_loss": current_train_metrics.get("loss_ce_avg", current_train_metrics.get("loss_ce", 0)),
                "dfl_loss": current_train_metrics.get("loss_giou_avg", current_train_metrics.get("loss_giou", 0)),
                "mAP50": map50,
                "mAP50_95": map50_95,
                "mAP75": accumulated_eval_metrics.get("map75"),
                "lr": current_train_metrics.get("lr", lr),
            })

            if patience > 0 and best_map is not None and epochs_since_improve >= patience:
                stop_requested = True
                _log_fn(
                    f"Early stopping requested at epoch {epoch_for_eval}/{epochs}: "
                    f"no mAP50-95 improvement for {patience} epochs after best epoch {best_epoch}"
                )
                if proc.poll() is None:
                    proc.terminate()
            accumulated_eval_metrics = {}

    proc.wait()

    elapsed = time.time() - started

    # Restore original vendor code
    _restore_vendor_code(root, original_slconfig)

    if proc.returncode != 0 and not stop_requested:
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

    # Save weight. Prefer upstream best checkpoints when available; fall back to
    # the rolling last checkpoint. Model Designer stores the selected file as
    # weight.pt so inference/export can resolve it uniformly across wrappers.
    run_dir = job_dir / "runs" / "dino_detr"
    candidate_checkpoints = [
        run_dir / "checkpoint_best_regular.pth",
        run_dir / "checkpoint_best_ema.pth",
        run_dir / "checkpoint.pth",
    ]
    checkpoint_path = next((p for p in candidate_checkpoints if p.exists()), run_dir / "checkpoint.pth")
    if checkpoint_path.exists():
        def _dino_key_count(path: Path) -> int | None:
            try:
                raw = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(raw, dict):
                    for key in ("model", "ema_model", "state_dict", "teacher", "student"):
                        state = raw.get(key)
                        if isinstance(state, dict):
                            return len(state)
                    return sum(1 for value in raw.values() if torch.is_tensor(value))
            except Exception:
                return None
            return None

        key_count = _dino_key_count(checkpoint_path)
        weight_id = weight_storage.save_weight_meta(
            model_id=payload.get("model_id", ""),
            model_name=payload.get("model_name", "DINO-DETR"),
            model_scale=model_scale,
            job_id=job_id,
            dataset=data_arg,
            epochs_trained=current_epoch or epochs,
            final_accuracy=best_map50,
            final_loss=None,
            weight_id=None,
            total_time=elapsed,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        dst_weight = weight_storage.weight_pt_path(weight_id)
        dst_weight.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, dst_weight)
        # Refresh metadata now that weight.pt exists and has a real file size.
        weight_id = weight_storage.save_weight_meta(
            model_id=payload.get("model_id", ""),
            model_name=payload.get("model_name", "DINO-DETR"),
            model_scale=model_scale,
            job_id=job_id,
            dataset=data_arg,
            epochs_trained=current_epoch or epochs,
            final_accuracy=best_map50,
            final_loss=None,
            weight_id=weight_id,
            total_time=elapsed,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        meta = weight_storage.load_weight_meta(weight_id)
        if meta:
            model_id = str(payload.get("model_id", ""))
            meta.update({
                "source_type": "dino",
                "benchmark_type": "detector" if model_scale == "resnet50" or model_id == "arch:dino" else "backbone",
                "task": "detect",
                "arch_plugin": str(config.get("model_arch") or f"dino_{model_scale}"),
                "model_arch": str(config.get("model_arch") or f"dino_{model_scale}"),
                "key_count": key_count,
                "train_args": {
                    "model_arch": str(config.get("model_arch") or f"dino_{model_scale}"),
                    "data": data_arg,
                    "epochs": epochs,
                    "batch": batch,
                },
            })
            weight_storage._store.save(weight_id, meta)
        _log_fn(f"Saved weight: {weight_id} from {checkpoint_path.name}")
    else:
        weight_id = None
        _log_fn("Warning: No checkpoint found")

    final_updates: dict[str, Any] = {
        "status": "completed",
        "epoch": current_epoch or epochs,
        "message": (
            "DINO-DETR detection training complete"
            if not stop_requested
            else f"DINO-DETR early stopped at epoch {current_epoch}/{epochs}"
        ),
        "weight_id": weight_id,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    if best_map is not None:
        final_updates["best_fitness"] = best_map
        final_updates["best_mAP50_95"] = best_map
    if best_map50 is not None:
        final_updates["best_mAP50"] = best_map50
    _set_job(job_id, **final_updates)
    _log_fn(f"DINO-DETR detection training completed in {elapsed:.1f}s")
