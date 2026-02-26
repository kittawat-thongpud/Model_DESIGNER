"""
Persistent storage for training jobs and their logs.

Directory layout (folder-per-job):
  JOBS_DIR/
    {job_id}/
      record.json      ← job data
      log.jsonl         ← per-epoch training log
      checkpoints/      ← periodic .pt checkpoints during training
      snapshots/        ← per-layer weight/activation snapshots for visualization
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import JOBS_DIR
from .base_storage import BaseJsonStorage

# ── Storage instance (folder-per-job, JSON file = record.json) ─────────────

_store = BaseJsonStorage(JOBS_DIR, folder_mode="record.json")


# ── Path helpers (still needed for log / checkpoint access) ────────────────

def _job_dir(job_id: str) -> Path:
    return _store.record_dir(job_id)


def _log_path(job_id: str) -> Path:
    return _store.record_dir(job_id) / "log.jsonl"


def _results_csv_path(job_id: str) -> Path:
    """Return path to Ultralytics results.csv file."""
    return _store.record_dir(job_id) / "runs" / "train" / "results.csv"


# ── CRUD (delegated to BaseJsonStorage) ─────────────────────────────────

def save_job(record: dict) -> None:
    """Save or update a job record to disk.
    
    Strips 'history' field to keep record.json lightweight.
    History is read from Ultralytics results.csv instead.
    """
    job_id = record["job_id"]
    
    # Create a copy to modify
    data_to_save = record.copy()
    
    # Remove history if present (we read from results.csv instead)
    data_to_save.pop("history", None)
        
    _store.save(job_id, data_to_save)


def load_job(job_id: str) -> dict | None:
    """Load a job record from disk."""
    return _store.load(job_id)


# ── History Management (Split from record.json) ─────────────────────────

def get_job_history(job_id: str) -> list[dict]:
    """
    Load job training history with comprehensive metrics.
    
    Priority:
    1. extended_metrics.jsonl (if exists) - contains ALL custom metrics
    2. results.csv (Ultralytics standard) - fallback for basic metrics
    3. record.json (legacy) - final fallback
    
    Returns:
        List of epoch metrics dicts with all available data
    """
    extended_metrics_path = _job_dir(job_id) / "extended_metrics.jsonl"
    results_path = _results_csv_path(job_id)
    
    # Try extended_metrics.jsonl first (most comprehensive)
    if extended_metrics_path.exists():
        try:
            history = []
            with open(extended_metrics_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Extended metrics already has all fields we need
                        # Just rename some keys for frontend compatibility
                        epoch_metrics = {
                            "epoch": data.get("epoch"),
                            "timestamp": data.get("timestamp"),
                            
                            # Training losses
                            "box_loss": data.get("train_box_loss"),
                            "cls_loss": data.get("train_cls_loss"),
                            "dfl_loss": data.get("train_dfl_loss"),
                            
                            # Validation losses
                            "val_box_loss": data.get("val_box_loss"),
                            "val_cls_loss": data.get("val_cls_loss"),
                            "val_dfl_loss": data.get("val_dfl_loss"),
                            
                            # Validation metrics
                            "mAP50": data.get("map50"),
                            "mAP50_95": data.get("map"),
                            "mAP75": data.get("map75"),
                            "precision": data.get("precision"),
                            "recall": data.get("recall"),
                            "fitness": data.get("fitness"),
                            
                            # Per-class metrics
                            "ap_per_class": data.get("ap_per_class"),
                            "ap50_per_class": data.get("ap50_per_class"),
                            "precision_per_class": data.get("precision_per_class"),
                            "recall_per_class": data.get("recall_per_class"),
                            "f1_per_class": data.get("f1_per_class"),
                            
                            # Latency metrics
                            "inference_latency_ms": data.get("inference_latency_ms"),
                            "preprocess_latency_ms": data.get("preprocess_latency_ms"),
                            "postprocess_latency_ms": data.get("postprocess_latency_ms"),
                            "total_latency_ms": data.get("total_latency_ms"),
                            
                            # System info
                            "device": data.get("device"),
                            "ram_gb": data.get("ram_gb"),
                            "gpu_mem_gb": data.get("gpu_mem_gb"),
                            "gpu_mem_reserved_gb": data.get("gpu_mem_reserved_gb"),
                            
                            # Learning rate
                            "lr": data.get("lr"),
                            
                            # Timing
                            "val_time_s": data.get("val_time_s"),
                        }
                        
                        # Remove None values
                        epoch_metrics = {k: v for k, v in epoch_metrics.items() if v is not None}
                        if epoch_metrics.get("epoch"):
                            history.append(epoch_metrics)
            
            if history:
                return history
        except Exception as e:
            print(f"Error reading extended_metrics.jsonl: {e}")
            # Fall through to results.csv
    
    # Fallback to results.csv (Ultralytics standard)
    if results_path.exists():
        try:
            import csv
            history = []
            with open(results_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert CSV row to metrics dict, filtering out empty values
                    epoch_data = {}
                    for key, value in row.items():
                        if value and value.strip():
                            try:
                                # Try to convert to float
                                epoch_data[key] = float(value)
                            except ValueError:
                                epoch_data[key] = value
                    
                    if epoch_data:
                        # Rename keys to match frontend expectations
                        metrics = {
                            "epoch": int(epoch_data.get("epoch", 0)),
                            "box_loss": epoch_data.get("train/box_loss"),
                            "cls_loss": epoch_data.get("train/cls_loss"),
                            "dfl_loss": epoch_data.get("train/dfl_loss"),
                            "precision": epoch_data.get("metrics/precision(B)"),
                            "recall": epoch_data.get("metrics/recall(B)"),
                            "mAP50": epoch_data.get("metrics/mAP50(B)"),
                            "mAP50_95": epoch_data.get("metrics/mAP50-95(B)"),
                            "val_box_loss": epoch_data.get("val/box_loss"),
                            "val_cls_loss": epoch_data.get("val/cls_loss"),
                            "val_dfl_loss": epoch_data.get("val/dfl_loss"),
                            "epoch_time": epoch_data.get("time"),
                            "lr": epoch_data.get("lr/pg0"),
                        }
                        # Remove None values
                        history.append({k: v for k, v in metrics.items() if v is not None})
            
            if history:
                return history
        except Exception as e:
            print(f"Error reading results.csv: {e}")
    
    # Final fallback to legacy history in record.json
    record = load_job(job_id)
    return record.get("history", []) if record else []


def append_job_history(job_id: str, epoch_metrics: dict):
    """
    No-op: Ultralytics automatically writes to results.csv.
    This function is kept for backward compatibility.
    """
    pass


def update_latest_history(job_id: str, updates: dict):
    """
    No-op: Ultralytics automatically writes to results.csv.
    This function is kept for backward compatibility.
    """
    pass



def list_jobs(status: str | None = None, model_id: str | None = None) -> list[dict]:
    """List all jobs, optionally filtered by status or model_id."""
    return _store.list_all(status=status, model_id=model_id)


def delete_job(job_id: str) -> bool:
    """Delete an entire job folder (record, logs, checkpoints, snapshots)."""
    existed = _store.delete(job_id)
    # Also remove legacy flat files if they exist
    for legacy in (JOBS_DIR / f"{job_id}.json", JOBS_DIR / f"{job_id}.log.jsonl"):
        if legacy.exists():
            legacy.unlink()
    return existed


# ── Logging ─────────────────────────────────────────────────────────────────

def append_job_log(job_id: str, level: str, message: str, data: dict | None = None) -> None:
    """Append a log line to the job-specific log file. Broadcasts to SSE subscribers."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "message": message,
        "data": data or {},
    }
    path = _log_path(job_id)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Broadcast to SSE subscribers
    try:
        from . import event_bus
        from ..constants import job_log_channel
        event_bus.publish_sync(job_log_channel(job_id), entry)
    except Exception:
        pass


def get_job_logs(job_id: str, limit: int = 200, offset: int = 0) -> list[dict]:
    """Read job-specific training logs."""
    path = _log_path(job_id)
    if not path.exists():
        return []

    entries: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Most recent first
    entries.reverse()
    return entries[offset: offset + limit]
