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


def _patch_vendor_code(root: Path, job_id: str) -> None:
    """Patch DINO-DETR vendor code to fix IndentationError in slconfig.py."""
    slconfig_path = root / "util" / "slconfig.py"
    if slconfig_path.exists():
        try:
            original_content = slconfig_path.read_text(encoding="utf-8")
            # Fix duplicate nested try statements and remove verify parameter
            fixed_content = re.sub(
                r'        try:\s+try:\s+try:\s+try:\s+try:\s+text, _ = FormatCode\(text, style_config=yapf_style, verify=True\)',
                '        try:\n            text, _ = FormatCode(text, style_config=yapf_style)',
                original_content,
                flags=re.DOTALL
            )
            # Also fix the single try case with verify parameter
            fixed_content = re.sub(
                r'text, _ = FormatCode\(text, style_config=yapf_style, verify=True\)',
                'text, _ = FormatCode(text, style_config=yapf_style)',
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
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch),
        "--num_workers",
        str(workers),
        "--lr",
        str(lr),
        "--weight_decay",
        str(weight_decay),
    ]
    
    if config.get("amp", True):
        cmd.append("--use_fp16")
    
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
