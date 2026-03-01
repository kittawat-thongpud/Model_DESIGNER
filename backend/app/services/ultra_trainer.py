"""
Ultralytics Training Service.

Wraps YOLO model.train() with:
  - SSE monitoring via Ultralytics callbacks
  - Job persistence (job_storage)
  - Weight saving after training completes
"""
from __future__ import annotations
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Completely disable Ultralytics CLI mode
# This is necessary because Ultralytics checks sys.argv and tries to parse CLI commands
import sys
import os

# Set environment variable to disable CLI mode
os.environ['YOLO_CLI'] = '0'

# Clear sys.argv completely and keep it empty
# DO NOT restore it - Ultralytics will try to parse it during trainer instantiation
sys.argv = []

from ..config import JOBS_DIR, WEIGHTS_DIR
from . import event_bus, job_storage
from ..constants import job_channel, train_channel
from . import dataset_yaml, dataset_registry
from ..config import JOBS_DIR


# ── Active jobs tracking ─────────────────────────────────────────────────────

_active_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def start_training(
    model_id: str,
    model_name: str,
    task: str,
    yaml_path: str,
    config: dict[str, Any],
    partition_configs: list[dict[str, Any]] | None = None,
    model_scale: str | None = None,
) -> str:
    """Start an Ultralytics training job in a background thread.
    
    Args:
        partition_configs: List of partition split configurations
                          [{'partition_id': 'p_xxx', 'train': True, 'val': False, 'test': True}, ...]
                          If empty or None, all partitions with all splits will be used.
        model_scale: Scale char ('n', 's', 'm', 'l', 'x') for model scaling.
    """
    job_id = uuid.uuid4().hex[:12]
    
    if partition_configs is None:
        partition_configs = []

    # Resolve human-readable dataset name from config.data
    _data_arg = config.get("data", "")
    _dataset_name = _data_arg if (_data_arg and dataset_registry.is_image_dataset(_data_arg)) else ""

    job = {
        "job_id": job_id,
        "model_id": model_id,
        "model_name": model_name,
        "task": task,
        "config": config,
        "dataset_name": _dataset_name,  # human-readable dataset name
        "partition_configs": partition_configs,
        "model_scale": model_scale,  # Store scale in job record
        "status": "pending",
        "epoch": 0,
        "total_epochs": config.get("epochs", 100),
        "message": "Queued",
        "history": [],
        "weight_id": None,
        "best_fitness": None,
        "best_mAP50": None,
        "best_mAP50_95": None,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "started_at": None,
        "completed_at": None,
    }
    job_storage.save_job(job)

    with _lock:
        _active_jobs[job_id] = {"thread": None, "stop": False}

    t = threading.Thread(
        target=_training_worker,
        args=(job_id, yaml_path, task, config, partition_configs, model_scale),
        daemon=False,  # Changed to non-daemon for proper cleanup
    )
    with _lock:
        _active_jobs[job_id]["thread"] = t
    t.start()

    return job_id


def stop_training(job_id: str) -> bool:
    """Signal a training job to stop and wait for cleanup."""
    thread_to_join = None
    
    with _lock:
        info = _active_jobs.get(job_id)
        if info:
            info["stop"] = True
            thread_to_join = info.get("thread")
        else:
            return False
    
    # Wait for thread to finish (with timeout) to ensure proper cleanup
    if thread_to_join and thread_to_join.is_alive():
        job_storage.append_job_log(job_id, "INFO", "Stopping worker thread...")
        thread_to_join.join(timeout=10.0)  # Wait max 10 seconds
        if thread_to_join.is_alive():
            job_storage.append_job_log(job_id, "WARNING", "Worker thread did not stop in time")
    
    return True


def resume_training(job_id: str) -> None:
    """Resume a stopped/failed job from its last checkpoint using Ultralytics resume=True.

    Reuses the same job_id — logs and history are appended to the existing record.
    Raises ValueError if job not found, already running, or no checkpoint exists.
    """
    job = job_storage.load_job(job_id)
    if not job:
        raise ValueError(f"Job not found: {job_id}")
    if job.get("status") == "running":
        raise ValueError(f"Job {job_id} is already running")

    job_dir = JOBS_DIR / job_id
    last_pt = job_dir / "runs" / "train" / "weights" / "last.pt"
    if not last_pt.exists():
        raise ValueError(f"No checkpoint (last.pt) found for job {job_id}")

    # Build resume config: same config but with resume=True pointing at last.pt
    resume_config = dict(job.get("config", {}))
    resume_config["resume"] = str(last_pt)
    resume_config.pop("pretrained", None)

    _restart_job(job_id, resume_config)


def append_training(job_id: str, additional_epochs: int) -> None:
    """Append more epochs to a completed/stopped job using last.pt as warm-start.

    Reuses the same job_id — logs and history are appended to the existing record.
    total_epochs is increased by additional_epochs.
    Raises ValueError if job not found, already running, or no checkpoint exists.
    """
    job = job_storage.load_job(job_id)
    if not job:
        raise ValueError(f"Job not found: {job_id}")
    if job.get("status") == "running":
        raise ValueError(f"Job {job_id} is already running")

    job_dir = JOBS_DIR / job_id
    last_pt = job_dir / "runs" / "train" / "weights" / "last.pt"
    best_pt = job_dir / "runs" / "train" / "weights" / "best.pt"
    checkpoint = last_pt if last_pt.exists() else (best_pt if best_pt.exists() else None)
    if checkpoint is None:
        raise ValueError(f"No checkpoint found for job {job_id}")

    # Build append config: same config, load last.pt as pretrained, set epochs = additional_epochs
    append_config = dict(job.get("config", {}))
    append_config["pretrained"] = str(checkpoint)
    append_config["epochs"] = additional_epochs
    append_config.pop("resume", None)

    # Update total_epochs in job record before starting
    job["total_epochs"] = job.get("total_epochs", 0) + additional_epochs
    job["epoch_offset"] = job.get("epoch", 0)  # remember where we left off for display
    job_storage.save_job(job)

    _restart_job(job_id, append_config)


def _restart_job(job_id: str, config: dict) -> None:
    """Reset job status to pending and launch _training_worker on the same job_id.
    
    Ensures existing thread is properly cleaned up before starting new one.
    """
    job = job_storage.load_job(job_id)
    if not job:
        raise ValueError(f"Job not found: {job_id}")

    # Check if there's an existing thread for this job_id
    existing_thread = None
    with _lock:
        info = _active_jobs.get(job_id)
        if info:
            existing_thread = info.get("thread")
            # Signal existing thread to stop
            info["stop"] = True
    
    # Wait for existing thread to finish (with timeout)
    if existing_thread and existing_thread.is_alive():
        job_storage.append_job_log(job_id, "INFO", "Waiting for previous worker to finish...")
        existing_thread.join(timeout=30.0)  # Wait max 30 seconds
        if existing_thread.is_alive():
            job_storage.append_job_log(job_id, "WARNING", "Previous worker did not finish in time, proceeding anyway")

    job["status"] = "pending"
    job["message"] = "Queued"
    job["completed_at"] = None
    job_storage.save_job(job)

    with _lock:
        _active_jobs[job_id] = {"thread": None, "stop": False}

    t = threading.Thread(
        target=_training_worker,
        args=(job_id, str(_resolve_yaml_for_job(job)), job.get("task", "detect"),
              config, job.get("partition_configs") or [], job.get("model_scale")),
        daemon=False,  # Changed to non-daemon for proper cleanup
    )
    with _lock:
        _active_jobs[job_id]["thread"] = t
    t.start()


def _resolve_yaml_for_job(job: dict) -> str:
    """Return the model YAML path for a job record."""
    from . import model_storage
    yaml_path = model_storage.load_model_yaml_path(job["model_id"])
    if yaml_path and Path(yaml_path).exists():
        return str(yaml_path)
    raise ValueError(f"YAML not found for model {job['model_id']}")


def cleanup_stale_jobs() -> None:
    """Mark any 'running' jobs as failed on startup (server restart)."""
    for job in job_storage.list_jobs(status="running"):
        job["status"] = "failed"
        job["message"] = "Server restarted during training"
        job["completed_at"] = datetime.utcnow().isoformat() + "Z"
        job_storage.save_job(job)


def cleanup_zombie_workers() -> dict[str, str]:
    """Detect and clean up zombie worker threads.
    
    Returns dict of {job_id: status} for cleaned up workers.
    """
    cleaned = {}
    
    with _lock:
        job_ids = list(_active_jobs.keys())
    
    for job_id in job_ids:
        with _lock:
            info = _active_jobs.get(job_id)
            if not info:
                continue
            
            thread = info.get("thread")
            if not thread:
                # No thread but still in active_jobs - zombie entry
                _active_jobs.pop(job_id, None)
                cleaned[job_id] = "removed_no_thread"
                continue
            
            # Check if thread is dead but job is still marked as running
            if not thread.is_alive():
                job = job_storage.load_job(job_id)
                if job and job.get("status") == "running":
                    # Thread died but job still running - mark as failed
                    job["status"] = "failed"
                    job["message"] = "Worker thread died unexpectedly"
                    job["completed_at"] = datetime.utcnow().isoformat() + "Z"
                    job_storage.save_job(job)
                    cleaned[job_id] = "marked_failed"
                
                # Remove from active jobs
                _active_jobs.pop(job_id, None)
    
    return cleaned


def get_worker_health() -> dict[str, Any]:
    """Get health status of all active workers.
    
    Returns dict with worker status information.
    """
    health = {
        "total_workers": 0,
        "alive_workers": 0,
        "dead_workers": 0,
        "workers": []
    }
    
    with _lock:
        job_ids = list(_active_jobs.keys())
    
    for job_id in job_ids:
        with _lock:
            info = _active_jobs.get(job_id)
            if not info:
                continue
            
            thread = info.get("thread")
            is_alive = thread.is_alive() if thread else False
            
            job = job_storage.load_job(job_id)
            
            worker_info = {
                "job_id": job_id,
                "thread_alive": is_alive,
                "job_status": job.get("status") if job else "unknown",
                "stop_requested": info.get("stop", False)
            }
            
            health["workers"].append(worker_info)
            health["total_workers"] += 1
            if is_alive:
                health["alive_workers"] += 1
            else:
                health["dead_workers"] += 1
    
    return health


# ── Training Worker ──────────────────────────────────────────────────────────

def _mount_fstype_for_path(path: str | Path) -> str | None:
    """Best-effort filesystem type lookup for a given path using /proc/mounts."""
    try:
        p = Path(path).resolve()
    except Exception:
        p = Path(path)
    best_match = None
    best_len = -1
    try:
        mounts = Path("/proc/mounts").read_text().splitlines()
    except Exception:
        return None
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point = parts[1]
        fstype = parts[2]
        try:
            mp = Path(mount_point)
        except Exception:
            continue
        try:
            if str(p).startswith(str(mp)) and len(str(mp)) > best_len:
                best_match = fstype
                best_len = len(str(mp))
        except Exception:
            continue
    return best_match


def _dataset_root_from_data_yaml(data_yaml_path: str | Path) -> Path | None:
    """Parse Ultralytics data.yaml to get the dataset root path (the `path:` field)."""
    try:
        txt = Path(data_yaml_path).read_text().splitlines()
    except Exception:
        return None
    for line in txt:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("path:"):
            val = s.split(":", 1)[1].strip()
            if val:
                return Path(val)
            return None
    return None


def _training_worker(
    job_id: str,
    yaml_path: str,
    task: str,
    config: dict[str, Any],
    partition_configs: list[dict[str, Any]] | None = None,
    model_scale: str | None = None,
) -> None:
    """Background thread that runs Ultralytics model.train().
    
    Args:
        partition_configs: List of partition split configurations
                          [{'partition_id': 'p_xxx', 'train': True, 'val': False, 'test': True}, ...]
        model_scale: Scale char ('n', 's', 'm', 'l', 'x') for model scaling.
    """
    job = job_storage.load_job(job_id)
    if not job:
        return
    
    if partition_configs is None:
        partition_configs = []

    job["status"] = "running"
    job["started_at"] = datetime.utcnow().isoformat() + "Z"
    job["message"] = "Initializing..."
    job_storage.save_job(job)
    _publish(job_id, job)

    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
        from . import dataset_yaml, dataset_registry
        from . import module_registry
        from .custom_trainer import CustomDetectionTrainer
        from .custom_trainer import NaNLossError

        # Register custom modules (e.g. HSG-DET) before loading model
        module_registry.register_custom_modules()

        # Check if using official YOLO model (skip YAML validation)
        # Note: Don't pop yolo_model here, it will be popped later during model loading
        yolo_model_check = config.get("yolo_model", "")
        original_yaml_path = yaml_path  # save before patching for arch detection
        if not yolo_model_check and yaml_path:
            # Custom model - validate and patch YAML
            if not Path(yaml_path).exists():
                raise FileNotFoundError(f"YAML not found: {yaml_path}")
                
            # Prepare YAML with validation and patching (shared logic)
            # This prevents "list index out of range" crashes due to cycles
            # and "RuntimeError" due to unscaled custom channels
            from ..utils.yaml_utils import prepare_model_yaml
            
            try:
                # Get path to temp patched YAML
                # We pass 'model_scale' to ensure custom modules are scaled correctly matches the backend logic
                patched_yaml_path = prepare_model_yaml(yaml_path, scale=model_scale)
                job_storage.append_job_log(job_id, "INFO", f"Validated & Patched YAML: {patched_yaml_path} (Scale: {model_scale})")
                
                # Use patched YAML for training
                yaml_path = patched_yaml_path
                
            except ValueError as e:
                # Validation failed (e.g. cycle)
                job_storage.append_job_log(job_id, "ERROR", f"Model Validation Failed: {e}")
                raise e

            job_storage.append_job_log(job_id, "INFO", f"Loading model YAML: {yaml_path}")
        elif yolo_model_check:
            job_storage.append_job_log(job_id, "INFO", f"Using official YOLO model: {yolo_model_check}")

        # ── Model Arch Plugin hook ────────────────────────────────────────────
        # If config contains 'model_arch', look up the arch plugin and:
        #   1. Register any custom nn.Modules (e.g. SparseGlobalBlock → ultralytics)
        #   2. Override yaml_path with the plugin's built-in YAML definition
        arch_plugin = None
        model_arch_key = config.pop("model_arch", None)
        if model_arch_key:
            from ..plugins.loader import get_arch_plugin, find_arch_for_yaml
            arch_plugin = get_arch_plugin(model_arch_key)
            if arch_plugin is None:
                raise ValueError(
                    f"Unknown model_arch '{model_arch_key}'. "
                    f"Make sure the arch plugin is registered via discover_plugins()."
                )
            job_storage.append_job_log(job_id, "INFO",
                f"Using arch plugin: {arch_plugin.display_name}")
            arch_plugin.register_modules()
            yaml_path = str(arch_plugin.yaml_path())
            job_storage.append_job_log(job_id, "INFO",
                f"Arch YAML: {yaml_path}")

            # yolov8_backbone warm-start is opt-in — only when user explicitly selects it in UI
            # (no auto-inject from pretrain_key() anymore)

        # Auto-detect arch plugin from yaml path if not set via model_arch key
        if arch_plugin is None and original_yaml_path:
            from ..plugins.loader import find_arch_for_yaml
            arch_plugin = find_arch_for_yaml(original_yaml_path)
            if arch_plugin:
                job_storage.append_job_log(job_id, "INFO",
                    f"Auto-detected arch plugin from YAML: {arch_plugin.display_name}")

        # Ensure config is a dict
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            config = dict(config) if hasattr(config, '__iter__') else {}
        
        # Resolve dataset: if 'data' is a registered dataset name, generate data.yaml
        data_arg = config.get("data", "")
        dataset_name = data_arg if (data_arg and dataset_registry.is_image_dataset(data_arg)) else ""
        if data_arg and dataset_registry.is_image_dataset(data_arg):
            # Generate data.yaml in job dir with selected partition split configurations
            # Generate data.yaml in job dir
            # If partition_configs are present, write_data_yaml will also create a symlinked 'dataset' dir
            data_yaml_path = job_dir / "data.yaml"
            
            partition_info = "custom partitions" if partition_configs else "all partitions"
            job_storage.append_job_log(job_id, "INFO", f"Configuring dataset '{data_arg}' with {partition_info}")

            # Auto-convert COCO JSON → YOLO .txt labels if needed (idempotent)
            try:
                from . import coco_converter
                conv_result = coco_converter.auto_convert_if_needed(data_arg)
                if conv_result:
                    if conv_result.get("status") == "success":
                        job_storage.append_job_log(job_id, "INFO",
                            f"COCO→YOLO label conversion: {conv_result['message']}")
                    elif conv_result.get("status") == "error":
                        job_storage.append_job_log(job_id, "WARNING",
                            f"COCO→YOLO conversion failed: {conv_result['error']} — training may find no labels")
            except Exception as _conv_err:
                job_storage.append_job_log(job_id, "WARNING",
                    f"COCO→YOLO auto-conversion error: {_conv_err}")

            try:
                # Delegate to service - it handles local dataset creation if needed
                dataset_yaml.write_data_yaml(
                    data_arg, 
                    data_yaml_path,
                    partition_configs=partition_configs if partition_configs else None
                )
                config["data"] = str(data_yaml_path)
                config["dataset_name"] = dataset_name  # preserve original name
                job_storage.append_job_log(job_id, "INFO", f"Using generated config: {data_yaml_path}")
            except Exception as e:
                job_storage.append_job_log(job_id, "ERROR", f"Failed to generate data.yaml: {e}")
                raise e
        
        # Update job record with complete config (including monitoring params) for visibility
        j = job_storage.load_job(job_id)
        if j:
            j["config"] = config.copy()
            job_storage.save_job(j)

        # Redirect stdout/stderr to capture all print() and tqdm output
        import io
        import contextlib
        
        class JobLogWriter(io.StringIO):
            """Custom writer that sends output to job log."""
            def __init__(self, job_id, level="INFO"):
                super().__init__()
                self.job_id = job_id
                self.level = level
                self.buffer = []
                self.last_cr_time = 0  # Track last carriage return output time
                self.cr_interval = 5  # Allow CR output every 5 seconds
            
            def write(self, text):
                if text and text.strip():
                    # Import here to avoid circular import issues
                    from . import job_storage as js
                    
                    # Filter out ANSI escape codes and unicode box chars
                    import re
                    clean_text = re.sub(r'\x1b\[[0-9;]*[mK]', '', text)
                    clean_text = re.sub(r'[\u2500-\u257F]', '', clean_text)  # Remove box drawing chars
                    clean_text = clean_text.strip()
                    
                    # Filter out uvicorn logs
                    if clean_text.startswith('INFO:') or 'HTTP/1.1' in clean_text:
                        return
                    
                    if not clean_text:
                        return
                    
                    # Check if this is a carriage return line (tqdm progress bar)
                    if '\r' in text and '\n' not in text:
                        # Rate limit carriage return lines to every 5 seconds
                        current_time = time.time()
                        if current_time - self.last_cr_time < self.cr_interval:
                            return  # Skip this update
                        self.last_cr_time = current_time
                        
                        # Try to parse tqdm format and create structured progress data
                        try:
                            parts = clean_text.split()
                            if len(parts) >= 4 and '/' in parts[0]:
                                epoch_info = parts[0]  # e.g., "2/100"
                                percent_match = re.search(r'(\d+)%', clean_text)
                                if percent_match:
                                    percent = percent_match.group(1)
                                    # batch/total — flexible: with or without it/s
                                    batch_match = re.search(r'(\d+)/(\d+)', clean_text[len(epoch_info):])
                                    batch_current = batch_match.group(1) if batch_match else '0'
                                    batch_total = batch_match.group(2) if batch_match else '0'

                                    progress_data = {
                                        'type': 'progress',
                                        'phase': 'train',
                                        'epoch': epoch_info,
                                        'batch': f"{batch_current}/{batch_total}",
                                        'percent': int(percent),
                                    }

                                    # Extract losses (parts[2..4] typically)
                                    if len(parts) >= 5:
                                        try:
                                            progress_data['losses'] = {
                                                'box': float(parts[2]),
                                                'cls': float(parts[3]),
                                                'dfl': float(parts[4]),
                                            }
                                        except Exception:
                                            pass

                                    # Timing and resource info — always enrich
                                    try:
                                        import psutil, torch as _torch
                                        now = time.time()
                                        progress_data['total_elapsed_s'] = round(now - training_start_time, 1)
                                        if hasattr(trainer, '_epoch_start_time'):
                                            progress_data['epoch_elapsed_s'] = round(now - trainer._epoch_start_time, 1)
                                        ep_num = int(epoch_info.split('/')[0])
                                        ep_total = int(epoch_info.split('/')[1])
                                        if ep_num > 0:
                                            avg_ep = progress_data['total_elapsed_s'] / ep_num
                                            batch_frac = int(percent) / 100.0
                                            remaining_in_epoch = avg_ep * (1 - batch_frac)
                                            progress_data['eta_s'] = round((ep_total - ep_num) * avg_ep + remaining_in_epoch, 0)
                                            progress_data['avg_epoch_s'] = round(avg_ep, 1)
                                        vm = psutil.virtual_memory()
                                        progress_data['ram_gb'] = round(vm.used / (1024**3), 2)
                                        progress_data['ram_total_gb'] = round(vm.total / (1024**3), 2)
                                        if _torch.cuda.is_available():
                                            dev_idx = _torch.cuda.current_device()
                                            progress_data['device'] = f'cuda:{dev_idx}'
                                            progress_data['gpu_mem_gb'] = round(_torch.cuda.memory_allocated(dev_idx) / (1024**3), 2)
                                            progress_data['gpu_mem_reserved_gb'] = round(_torch.cuda.memory_reserved(dev_idx) / (1024**3), 2)
                                        else:
                                            progress_data['device'] = 'cpu'
                                    except Exception:
                                        pass

                                    js.append_job_log(
                                        self.job_id,
                                        "PROGRESS",
                                        f"Epoch {epoch_info} | {percent}% | Batch {batch_current}/{batch_total}",
                                        progress_data
                                    )
                                    return
                        except Exception:
                            pass  # Fall through to normal logging
                    
                    self.buffer.append(clean_text)
                    
                    # Flush buffer on newline or carriage return (if allowed)
                    if '\n' in text or '\r' in text:
                        if self.buffer:
                            js.append_job_log(self.job_id, self.level, ' '.join(self.buffer))
                            self.buffer = []
            
            def flush(self):
                if self.buffer:
                    from . import job_storage as js
                    js.append_job_log(self.job_id, self.level, ' '.join(self.buffer))
                    self.buffer = []
        
        # Start redirecting output from here
        log_writer = JobLogWriter(job_id, "INFO")
        
        # Disable tqdm through Ultralytics
        try:
            from ultralytics.utils import TQDM
            # Replace TQDM class with a dummy that does nothing
            class DummyTQDM:
                def __init__(self, *args, **kwargs):
                    self.iterable = args[0] if args else []
                def __iter__(self):
                    return iter(self.iterable)
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def update(self, n=1):
                    pass
                def close(self):
                    pass
                def set_description(self, desc):
                    pass
            
            # Replace TQDM in ultralytics.utils
            import ultralytics.utils
            ultralytics.utils.TQDM = DummyTQDM
            
            # Also set disable flag
            if hasattr(TQDM, 'disable'):
                TQDM.disable = True
        except Exception as e:
            job_storage.append_job_log(job_id, "WARNING", f"Could not disable TQDM: {e}")
        
        # Patch Ultralytics LOGGER to redirect to job log
        from ultralytics.utils import LOGGER
        import logging
        
        # Create custom handler for job log
        class JobLogHandler(logging.Handler):
            def __init__(self, job_id):
                super().__init__()
                self.job_id = job_id
            
            def emit(self, record):
                try:
                    from . import job_storage as js
                    msg = self.format(record)
                    # Filter out uvicorn logs
                    if not msg.startswith('INFO:') and 'HTTP/1.1' not in msg:
                        level = record.levelname
                        js.append_job_log(self.job_id, level, msg)
                except Exception:
                    pass
        
        # Add our handler to LOGGER
        job_handler = JobLogHandler(job_id)
        job_handler.setLevel(logging.DEBUG)
        LOGGER.addHandler(job_handler)
        
        # Also disable console handler
        for handler in LOGGER.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                LOGGER.removeHandler(handler)
        
        # Load pretrained weights if specified
        pretrained = config.pop("pretrained", "")
        resume_path = config.pop("resume", "")  # set by resume_training()
        yolo_model = config.pop("yolo_model", "")  # official YOLO model (yolov8n, yolov8s, etc.)
        use_yolo_pretrained = config.pop("use_yolo_pretrained", True)
        
        # Convert boolean fields to actual boolean (handle string 'True'/'False' from config)
        if isinstance(use_yolo_pretrained, str):
            use_yolo_pretrained = use_yolo_pretrained.lower() in ('true', '1', 'yes')
        
        # resume_path might be boolean True instead of path string
        if isinstance(resume_path, bool):
            resume_path = ""  # Ignore boolean resume, only use actual path

        job_storage.append_job_log(
            job_id,
            "DEBUG",
            "Model init inputs | "
            f"resume_path={resume_path!r}, "
            f"yolo_model={yolo_model!r}, "
            f"use_yolo_pretrained={use_yolo_pretrained!r}, "
            f"pretrained={pretrained!r} (type={type(pretrained).__name__})",
        )

        pretrained_loaded = False  # set True only when explicit user weights are loaded
        with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
            if resume_path:
                # Resume mode: load directly from checkpoint — Ultralytics restores all state
                model = YOLO(resume_path)
                job_storage.append_job_log(job_id, "INFO", f"Resuming from checkpoint: {resume_path}")
            elif yolo_model:
                # Official YOLO model mode
                if use_yolo_pretrained:
                    # Load pretrained YOLO model
                    model = YOLO(f"{yolo_model}.pt")
                    job_storage.append_job_log(job_id, "INFO", f"Using official YOLO model: {yolo_model}.pt (pretrained)")
                else:
                    # Load YOLO architecture without pretrained weights (train from scratch)
                    model = YOLO(f"{yolo_model}.yaml")
                    job_storage.append_job_log(job_id, "INFO", f"Using official YOLO model: {yolo_model}.yaml (from scratch)")
            else:
                # Custom model from YAML
                model = YOLO(yaml_path, task=task)
                job_storage.append_job_log(job_id, "INFO", f"Created model from YAML: {yaml_path}")
                job_storage.append_job_log(
                    job_id,
                    "DEBUG",
                    f"Custom model pretrained candidate: {pretrained!r} (type={type(pretrained).__name__})",
                )
                
                pretrained_loaded = False
                if pretrained:
                    job_storage.append_job_log(job_id, "INFO", f"Attempting to load pretrained weights: {pretrained}")
                    weight_path = _resolve_weight(pretrained)
                    if weight_path and weight_path.exists():
                        job_storage.append_job_log(job_id, "INFO", f"Resolved weight path: {weight_path}")
                        try:
                            file_size = weight_path.stat().st_size
                            with weight_path.open("rb") as f:
                                header = f.read(8)
                            job_storage.append_job_log(
                                job_id,
                                "DEBUG",
                                f"Weight file diagnostics: size={file_size} bytes, header_hex={header.hex()}",
                            )
                        except Exception as diag_err:
                            job_storage.append_job_log(
                                job_id,
                                "WARNING",
                                f"Could not read weight file diagnostics: {diag_err}",
                            )
                        model.load(str(weight_path))
                        job_storage.append_job_log(job_id, "INFO", f"✓ Successfully loaded pretrained weights: {pretrained}")
                        pretrained_loaded = True
                    else:
                        job_storage.append_job_log(job_id, "WARNING",
                            f"✗ Pretrained weight not found: {pretrained} (resolved to: {weight_path}). "
                            f"Setting pretrained=False to prevent Ultralytics auto-download.")
                        config["pretrained"] = False
                else:
                    job_storage.append_job_log(job_id, "INFO", "No pretrained weights specified - training from scratch")
                    config["pretrained"] = False

                # ── Arch warm-start: deferred to on_pretrain_routine_end ────
                # warm_start() is registered as a callback below so it runs
                # AFTER Ultralytics setup_model() creates the real nn.Module.
                # Calling load_state_dict() before that would be overwritten.

        # Extract custom monitoring parameters before building train_kwargs
        # Save them to job config first for visibility in record.json
        record_gradients = config.get('record_gradients', False)
        gradient_interval = config.get('gradient_interval', 1)
        record_weights = config.get('record_weights', False)
        weight_interval = config.get('weight_interval', 1)
        sample_per_class = config.get('sample_per_class', 0)
        
        custom_params = {
            'job_id': job_id,
            'record_gradients': config.pop('record_gradients', False),
            'gradient_interval': config.pop('gradient_interval', 1),
            'record_weights': config.pop('record_weights', False),
            'weight_interval': config.pop('weight_interval', 1),
            'sample_per_class': config.pop('sample_per_class', 0),
        }

        # ── Pop Model Designer fields that are NOT valid Ultralytics kwargs ───
        # These are handled manually below rather than passed to model.train().
        _use_ema = config.pop('ema', True)
        _pin_memory = config.pop('pin_memory', False)
        config.pop('dataset_name', None)       # internal tracking field — not a YOLO arg
        config.pop('use_yolo_pretrained', None)  # handled above, not a YOLO arg
        config.pop('yolov8_backbone', None)      # handled above, not a YOLO arg
        config.pop('nan_retries', None)          # handled above, not a YOLO arg

        # Build train kwargs (only valid Ultralytics parameters)
        train_kwargs = {k: v for k, v in config.items() if v != ""}
        train_kwargs["project"] = str(job_dir / "runs")

        # ── Device validation & multi-GPU setup ──────────────────────────────
        # Strip GPU indices that exceed the actual device count so Ultralytics
        # doesn't raise ValueError (e.g. user set "0,1,2" but only 1 GPU exists).
        import torch as _torch
        _avail = _torch.cuda.device_count()
        _device_val = str(train_kwargs.get("device", "")).strip()

        # Log available hardware
        if _avail > 0:
            _gpu_names = [_torch.cuda.get_device_name(i) for i in range(_avail)]
            job_storage.append_job_log(job_id, "INFO",
                f"Available GPUs ({_avail}): " + ", ".join(f"[{i}] {n}" for i, n in enumerate(_gpu_names)))
        else:
            job_storage.append_job_log(job_id, "INFO", "No CUDA GPUs detected — using CPU")

        if _device_val and _device_val not in ("cpu", "mps", ""):
            _indices = []
            for _idx in _device_val.split(","):
                _idx = _idx.strip()
                if _idx.isdigit() and int(_idx) < _avail:
                    _indices.append(_idx)
            if not _indices:
                # No valid GPU indices — let Ultralytics pick automatically
                train_kwargs.pop("device", None)
                job_storage.append_job_log(job_id, "WARNING",
                    f"Requested device='{_device_val}' but only {_avail} GPU(s) available. "
                    "Falling back to automatic device selection.")
            else:
                if len(_indices) < len([x for x in _device_val.split(",") if x.strip()]):
                    train_kwargs["device"] = ",".join(_indices)
                    job_storage.append_job_log(job_id, "WARNING",
                        f"Requested device='{_device_val}' but only {_avail} GPU(s) available. "
                        f"Using device='{train_kwargs['device']}'.")
                _indices = train_kwargs.get("device", _device_val).split(",")

        # Resolve effective device indices for DDP check
        _final_device = str(train_kwargs.get("device", "")).strip()
        _is_multi_gpu = (
            _avail > 1
            and _final_device not in ("cpu", "mps", "")
            and len([x for x in _final_device.split(",") if x.strip()]) > 1
        )

        if _is_multi_gpu:
            # Ultralytics DDP spawns child processes via torch.multiprocessing.
            # In a forked uvicorn worker the default 'fork' start method can
            # cause deadlocks with CUDA.  Switch to 'spawn' before training.
            import multiprocessing as _mp
            try:
                _current_method = _mp.get_start_method(allow_none=True)
                if _current_method != "spawn":
                    _mp.set_start_method("spawn", force=True)
                    job_storage.append_job_log(job_id, "INFO",
                        f"Multi-GPU DDP: set multiprocessing start method "
                        f"'spawn' (was '{_current_method}')")
            except RuntimeError:
                # Already set — OK if already spawn; warn otherwise
                _current_method = _mp.get_start_method(allow_none=True)
                if _current_method != "spawn":
                    job_storage.append_job_log(job_id, "WARNING",
                        f"Multi-GPU DDP: could not change start method from "
                        f"'{_current_method}' to 'spawn' — DDP may deadlock")

            job_storage.append_job_log(job_id, "INFO",
                f"Multi-GPU training enabled: device='{_final_device}' "
                f"({len(_final_device.split(','))} GPUs) — using DDP")
        elif _avail > 1 and not _final_device:
            # User left device blank but multiple GPUs exist — use all of them
            all_indices = ",".join(str(i) for i in range(_avail))
            train_kwargs["device"] = all_indices
            job_storage.append_job_log(job_id, "INFO",
                f"Auto-selected all {_avail} GPUs: device='{all_indices}'")
            # Apply spawn for this case too
            import multiprocessing as _mp
            try:
                _mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass
        # ─────────────────────────────────────────────────────────────────────

        # ── ema / pin_memory note ─────────────────────────────────────────────
        # Both were already popped from config before train_kwargs was built
        # (_use_ema, _pin_memory above) because they are NOT valid Ultralytics
        # kwargs in this version and would raise SyntaxError in the validator.
        # pin_memory is left to Ultralytics default (False) intentionally to
        # avoid ConnectionResetError in Docker environments with small /dev/shm.
        job_storage.append_job_log(job_id, "DEBUG",
            f"ema={_use_ema}, pin_memory={_pin_memory} (not passed to Ultralytics — handled internally).")
        # ─────────────────────────────────────────────────────────────────────
        train_kwargs["name"] = "train"
        train_kwargs["exist_ok"] = True
        if resume_path:
            train_kwargs["resume"] = True  # tell Ultralytics to restore epoch/optimizer state

        # Detect dataset filesystem early (used to decide worker/cache strategy)
        fstype = None
        ds_root = None
        is_remote_fs = False
        try:
            data_yaml = train_kwargs.get("data")
            ds_root = _dataset_root_from_data_yaml(data_yaml) if data_yaml else None
            if ds_root:
                fstype = _mount_fstype_for_path(ds_root)
                if fstype and fstype.lower() in {"fuse", "fuseblk", "nfs", "cifs", "sshfs", "overlay"}:
                    is_remote_fs = True
                # RunPod Network Volume mounts as nfs/overlay but may appear as ext4.
                # Fallback: treat /workspace (and subdirs) as remote if fstype not caught above.
                if not is_remote_fs and ds_root:
                    _ds_str = str(ds_root.resolve())
                    if _ds_str.startswith("/workspace") or _ds_str.startswith("/runpod-volume"):
                        is_remote_fs = True
                        fstype = fstype or "runpod-network-volume"
        except Exception as _fs_err:
            job_storage.append_job_log(job_id, "DEBUG", f"Dataset filesystem detection failed: {_fs_err}")

        # Auto-tune workers: Docker CPU quota is often much lower than os.cpu_count().
        # NOTE: On remote filesystems (FUSE/NFS/etc.) we force workers=0 below.
        if not is_remote_fs:
            # Over-subscribing workers causes thrashing and slows data loading.
            if "workers" not in train_kwargs or train_kwargs.get("workers", 8) > 4:
                optimal_workers = _get_optimal_workers()
                train_kwargs["workers"] = optimal_workers
                job_storage.append_job_log(job_id, "INFO",
                    f"Auto-tuned dataloader workers: {optimal_workers}")

        # Smart cache selection (only when user has not set an explicit non-False value).
        # Priority: cache="ram"  > cache=True (disk)  > cache=False
        #
        # cache="ram":  load all images into shared memory once — zero I/O per batch.
        #               Chosen when free RAM >= dataset_image_size * 1.5.
        # cache=True:   build a .cache label file on first run, reuse after.
        #               On NFS this still saves the re-scan cost for later runs.
        # cache=False:  no caching (fallback for remote FS where .cache write is slow).
        _user_cache = train_kwargs.get("cache", None)
        if not _user_cache or str(_user_cache).lower() in ("false", "0", "none", ""):
            _chosen_cache = _select_cache_strategy(ds_root, job_id, is_remote_fs)
            train_kwargs["cache"] = _chosen_cache

        # If dataset lives on FUSE/remote mounts, Ultralytics setup can appear to hang
        # due to heavy stat()/glob across 100k+ files and/or multiprocessing overhead.
        # Mitigation: force workers=0 and disable Ultralytics dataset cache writing.
        if ds_root:
            job_storage.append_job_log(
                job_id,
                "DEBUG",
                "Dataset filesystem | "
                f"dataset_root={str(ds_root)!r}, fstype={fstype!r}",
            )
        if is_remote_fs:
            # On remote filesystems we keep cache off unless RAM cache was chosen,
            # but we should NOT force workers=0 — use a small worker count instead.
            if "workers" not in config:
                train_kwargs["workers"] = 2

            # If smart cache chose ram → keep it (best option on any FS).
            # Otherwise fall back to False — disk cache on NFS is slow.
            if str(train_kwargs.get("cache", "")).lower() not in ("ram", "true", "1", "disk"):
                train_kwargs["cache"] = False

            # Redirect Ultralytics .cache files to /tmp (fast local storage) so
            # label scan results survive the session without writing to NFS.
            if train_kwargs.get("cache") in (True, "disk") and ds_root:
                _redirect_cache_to_tmp(ds_root, job_id)

            # Ultralytics label caching scans all images/labels using a ThreadPool(NUM_THREADS).
            # On remote filesystems (FUSE/NFS/etc.) high parallelism can degrade performance
            # dramatically. Reduce to single-thread for stability.
            try:
                import ultralytics.data.dataset as _ultra_ds
                old_ds_threads = getattr(_ultra_ds, "NUM_THREADS", None)
                _ultra_ds.NUM_THREADS = 1
                try:
                    import ultralytics.utils as _ultra_utils
                    old_utils_threads = getattr(_ultra_utils, "NUM_THREADS", None)
                    _ultra_utils.NUM_THREADS = 1
                except Exception:
                    old_utils_threads = None
                job_storage.append_job_log(
                    job_id,
                    "WARNING",
                    "Remote FS mitigation: reduced Ultralytics dataset scan threads | "
                    f"ultralytics.data.dataset.NUM_THREADS: {old_ds_threads!r} -> 1, "
                    f"ultralytics.utils.NUM_THREADS: {old_utils_threads!r} -> 1",
                )
            except Exception as _thr_err:
                job_storage.append_job_log(
                    job_id,
                    "DEBUG",
                    f"Remote FS mitigation: could not patch Ultralytics NUM_THREADS: {_thr_err}",
                )
            job_storage.append_job_log(
                job_id,
                "WARNING",
                "Dataset filesystem mitigation applied | "
                f"dataset_root={str(ds_root)!r}, fstype={fstype!r}, "
                f"workers={train_kwargs.get('workers')!r}, cache={train_kwargs.get('cache')!r}",
            )

        job_storage.append_job_log(
            job_id,
            "DEBUG",
            "train_kwargs summary | "
            f"keys={sorted(train_kwargs.keys())}, "
            f"pretrained={train_kwargs.get('pretrained', '<unset>')!r}, "
            f"resume={train_kwargs.get('resume', '<unset>')!r}, "
            f"data={train_kwargs.get('data', '<unset>')!r}",
        )

        # Preflight: diagnose label cache state (writability + hash)
        _preflight_cache_diag(train_kwargs.get("data"), job_id)

        # Preflight: scan and remove corrupted Ultralytics dataset cache files (*.cache)
        cache_cleanup = _cleanup_corrupt_dataset_cache(train_kwargs.get("data"), job_id)
        if cache_cleanup["scanned"] > 0:
            job_storage.append_job_log(
                job_id,
                "DEBUG",
                "Dataset cache preflight | "
                f"dataset_root={cache_cleanup['dataset_root']!r}, "
                f"scanned={cache_cleanup['scanned']}, "
                f"removed={cache_cleanup['removed']}",
            )
            if cache_cleanup["removed"] > 0:
                removed_preview = ", ".join(cache_cleanup["removed_paths"][:5])
                if cache_cleanup["removed"] > 5:
                    removed_preview += ", ..."
                job_storage.append_job_log(
                    job_id,
                    "WARNING",
                    "Removed corrupted dataset cache files before training: "
                    f"{removed_preview}",
                )

        # Register callbacks for monitoring
        epoch_start_time = [time.time()]
        training_start_time = time.time()

        def on_train_epoch_end(trainer):
            """Called after each training epoch."""
            if _should_stop(job_id):
                raise KeyboardInterrupt("Training stopped by user")

            epoch = trainer.epoch + 1
            # Calculate time for THIS epoch only (not cumulative)
            epoch_time = time.time() - epoch_start_time[0]
            epoch_start_time[0] = time.time()
            
            # Calculate total elapsed time
            total_elapsed = time.time() - training_start_time

            # Extract loss items
            loss_items = trainer.loss_items
            if hasattr(loss_items, 'cpu'):
                loss_items = loss_items.cpu().numpy()

            box_loss = float(loss_items[0]) if len(loss_items) > 0 else 0.0
            cls_loss = float(loss_items[1]) if len(loss_items) > 1 else 0.0
            dfl_loss = float(loss_items[2]) if len(loss_items) > 2 else 0.0

            # Current LR
            lr = 0.0
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                lr = trainer.optimizer.param_groups[0].get('lr', 0.0)
            
            # Estimate time remaining
            epochs_remaining = trainer.epochs - epoch
            avg_epoch_time = total_elapsed / epoch
            eta_seconds = epochs_remaining * avg_epoch_time

            # Update job record status (Ultralytics writes metrics to results.csv automatically)
            j = job_storage.load_job(job_id)
            if j:
                j["epoch"] = epoch
                j["message"] = f"Epoch {epoch}/{j['total_epochs']}"
                j["total_time"] = total_elapsed
                job_storage.save_job(j)
                _publish(job_id, j)
                
            # Log epoch summary
            job_storage.append_job_log(job_id, "INFO",
                f"Epoch {epoch}/{trainer.epochs} completed in {epoch_time:.1f}s | "
                f"Total: {total_elapsed/60:.1f}m | ETA: {eta_seconds/60:.1f}m")

        def on_fit_epoch_end(trainer):
            """Called after validation at end of each epoch."""
            if not hasattr(trainer, 'metrics') or not trainer.metrics:
                return

            m = trainer.metrics
            
            # Update job record with best metrics (Ultralytics writes all metrics to results.csv)
            j = job_storage.load_job(job_id)
            if j:
                # Track best fitness
                fitness = getattr(trainer, 'fitness', None)
                if fitness is not None:
                    if j.get("best_fitness") is None or fitness > j["best_fitness"]:
                        j["best_fitness"] = fitness
                        j["best_mAP50"] = m.get("metrics/mAP50(B)")
                        j["best_mAP50_95"] = m.get("metrics/mAP50-95(B)")

                job_storage.save_job(j)
                _publish(job_id, j)

        def on_train_start(trainer):
            j = job_storage.load_job(job_id)
            if j:
                j["message"] = "Training started"
                j["total_epochs"] = trainer.epochs
                job_storage.save_job(j)
                _publish(job_id, j)
            job_storage.append_job_log(job_id, "INFO",
                f"Training started: {trainer.epochs} epochs, task={task}")

        # ── on_pretrain_routine_end: warm-start after setup_model() ─────────
        # setup_model() rebuilds DetectionModel from YAML before this fires,
        # so trainer.model is a live nn.Module — load_state_dict() will persist.
        _arch_plugin_ref = arch_plugin
        _pretrained_loaded_ref = pretrained_loaded
        # yolov8_backbone: explicit user choice e.g. 'yolov8n', 'yolov8m'
        # Falls back to model_scale if not set
        _yolov8_backbone_ref = config.pop("yolov8_backbone", None)
        _model_scale_ref = model_scale
        _resume_requested_ref = bool(resume_path)

        def on_pretrain_routine_end(trainer):
            trainer_resume = bool(getattr(getattr(trainer, "args", None), "resume", False))
            if _resume_requested_ref or trainer_resume:
                job_storage.append_job_log(
                    job_id,
                    "INFO",
                    "Skipping backbone warm-start because training is in resume mode",
                )
                return
            if _arch_plugin_ref is None or _pretrained_loaded_ref:
                return
            # Warm-start is opt-in ONLY: require explicit yolov8_backbone selection.
            if not _yolov8_backbone_ref:
                job_storage.append_job_log(job_id, "INFO",
                    "Backbone warm-start: no YOLOv8 backbone selected — skipping")
                return

            # Determine scale for warm-start:
            # _yolov8_backbone_ref is explicit user choice e.g. 'yolov8m' → extract scale char 'm'
            _KNOWN = {"yolov8n": "n", "yolov8s": "s", "yolov8m": "m",
                      "yolov8l": "l", "yolov8x": "x"}
            ws_scale = _KNOWN.get(_yolov8_backbone_ref.lower(), _yolov8_backbone_ref[-1:])
            try:
                ws_log = lambda msg: job_storage.append_job_log(job_id, "INFO", msg)
                # Pass trainer.model (the real nn.Module) wrapped as a shim
                # warm_start expects a YOLO-like object with .model attribute
                class _ModelShim:
                    def __init__(self, nn_module):
                        self.model = nn_module
                ws_result = _arch_plugin_ref.warm_start(
                    _ModelShim(trainer.model), log_fn=ws_log, model_scale=ws_scale
                )
                # Temp .pt is no longer needed — state_dict was applied directly
                temp_pt = ws_result.get("temp_pt")
                if temp_pt:
                    try:
                        import os as _os; _os.unlink(temp_pt)
                    except Exception:
                        pass
                if ws_result.get("transferred", 0) > 0:
                    job_storage.append_job_log(
                        job_id, "INFO",
                        f"✓ Backbone warm-start complete: "
                        f"{ws_result['transferred']} tensors transferred, "
                        f"{ws_result['skipped']} skipped, "
                        f"layers: {ws_result.get('matched_layers', [])}"
                    )
                else:
                    job_storage.append_job_log(
                        job_id, "INFO",
                        "Backbone warm-start: no tensors transferred (skipped or not available)"
                    )
            except Exception as ws_err:
                job_storage.append_job_log(
                    job_id, "WARNING",
                    f"Backbone warm-start failed ({ws_err}) — continuing without warm-start"
                )

        def _register_callbacks(_model):
            # Register callbacks
            _model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
            _model.add_callback("on_train_start", on_train_start)
            _model.add_callback("on_train_epoch_end", on_train_epoch_end)
            _model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        _register_callbacks(model)

        # Run training with custom trainer
        job_storage.append_job_log(job_id, "INFO",
            f"Starting model.train() with {len(train_kwargs)} kwargs")
        
        # Create a dynamic trainer class with custom params injected
        # This ensures Ultralytics receives a proper class, not a factory function
        class JobCustomTrainer(CustomDetectionTrainer):
            # Store custom params as class attribute
            _custom_params = custom_params
            
            def __init__(self, cfg=None, overrides=None, _callbacks=None):
                # Inject custom params into overrides
                if overrides is None:
                    overrides = {}
                # DO NOT inject into overrides, as it triggers Ultralytics validation
                # CustomDetectionTrainer will read from self._custom_params instead
                # overrides.update(self._custom_params)
                
                # Debug log
                from . import job_storage as js
                js.append_job_log(
                    self._custom_params['job_id'],
                    "INFO",
                    f"JobCustomTrainer.__init__ called with job_id: {self._custom_params['job_id']}"
                )
                
                super().__init__(cfg, overrides, _callbacks)
        
        # Continue using the same log_writer for training
        # CRITICAL: Prevent Ultralytics from entering CLI mode
        # Ultralytics checks sys.argv in multiple places and tries to parse CLI args
        import sys
        sys.argv = []
        os.environ['YOLO_CLI'] = '0'
        
        # Monkey-patch Ultralytics' entrypoint to prevent CLI parsing
        try:
            from ultralytics import cfg
            # Store original entrypoint
            original_entrypoint = getattr(cfg, 'entrypoint', None)
            # Replace with no-op function
            cfg.entrypoint = lambda: None
            
            job_storage.append_job_log(job_id, "DEBUG", "Patched Ultralytics entrypoint to prevent CLI mode")
        except Exception as e:
            job_storage.append_job_log(job_id, "WARNING", f"Could not patch entrypoint: {e}")
        
        # Retry loop: if loss becomes NaN/Inf mid-epoch, restart immediately from last.pt
        max_nan_retries = int(config.get("nan_retries", 3) or 3)
        nan_attempt = 0
        results = None

        def _fresh_model_for_retry() -> "YOLO":
            # Recreate a fresh model instance using the same initialization mode.
            # NOTE: This is used when NaN happens before last.pt exists.
            if yolo_model:
                return YOLO(f"{yolo_model}.pt") if use_yolo_pretrained else YOLO(f"{yolo_model}.yaml")
            return YOLO(yaml_path, task=task)

        while True:
            try:
                with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
                    results = model.train(trainer=JobCustomTrainer, **train_kwargs)
                break  # success
            except NaNLossError as e:
                nan_attempt += 1

                last_pt = job_dir / "runs" / "train" / "weights" / "last.pt"
                msg = (
                    f"Loss NaN/Inf detected (attempt {nan_attempt}/{max_nan_retries}), "
                    + ("recovering from last.pt..." if last_pt.exists() else "restarting from fresh init (no last.pt yet)...")
                )
                job_storage.append_job_log(job_id, "WARNING", msg)

                if nan_attempt >= max_nan_retries:
                    raise

                # Respect stop requests during recovery.
                if _should_stop(job_id):
                    raise KeyboardInterrupt("Training stopped by user")

                if last_pt.exists():
                    # Resume from last checkpoint. Ultralytics will restore optimizer/epoch state.
                    model = YOLO(str(last_pt))
                    train_kwargs["resume"] = True
                else:
                    # NaN occurred before any checkpoint was written. Best-effort restart.
                    model = _fresh_model_for_retry()
                    train_kwargs.pop("resume", None)

                # Re-register callbacks on the new YOLO object.
                _register_callbacks(model)

                continue
        
        # Restore original entrypoint if it was patched
        try:
            if original_entrypoint is not None:
                cfg.entrypoint = original_entrypoint
        except:
            pass
        
        # Flush any remaining buffered output
        log_writer.flush()

        # Training complete — save weights
        weight_id = _save_best_weight(job_id, job_dir, model_name=job["model_name"])

        j = job_storage.load_job(job_id)
        if j:
            j["status"] = "completed"
            j["message"] = "Training complete"
            j["weight_id"] = weight_id
            j["completed_at"] = datetime.utcnow().isoformat() + "Z"
            job_storage.save_job(j)
            _publish(job_id, j)

        job_storage.append_job_log(job_id, "INFO",
            f"Training completed. Weight: {weight_id}")

    except KeyboardInterrupt:
        j = job_storage.load_job(job_id)
        if j:
            j["status"] = "stopped"
            j["message"] = "Stopped by user"
            j["completed_at"] = datetime.utcnow().isoformat() + "Z"
            job_storage.save_job(j)
            _publish(job_id, j)
        job_storage.append_job_log(job_id, "WARNING", "Training stopped by user")

    except Exception as e:
        j = job_storage.load_job(job_id)
        if j:
            j["status"] = "failed"
            j["message"] = str(e)[:500]
            j["completed_at"] = datetime.utcnow().isoformat() + "Z"
            job_storage.save_job(j)
            _publish(job_id, j)
        import traceback
        failure_context = {
            "resume_path": locals().get("resume_path", "<unset>"),
            "yolo_model": locals().get("yolo_model", "<unset>"),
            "pretrained": locals().get("pretrained", "<unset>"),
            "train_kwargs_pretrained": (
                locals().get("train_kwargs", {}).get("pretrained", "<unset>")
                if isinstance(locals().get("train_kwargs", {}), dict)
                else "<unset>"
            ),
            "train_kwargs_resume": (
                locals().get("train_kwargs", {}).get("resume", "<unset>")
                if isinstance(locals().get("train_kwargs", {}), dict)
                else "<unset>"
            ),
            "yaml_path": locals().get("yaml_path", "<unset>"),
        }
        job_storage.append_job_log(job_id, "ERROR", f"Failure context: {failure_context}")
        job_storage.append_job_log(job_id, "ERROR", f"Traceback:\n{traceback.format_exc()}")
        job_storage.append_job_log(job_id, "ERROR", f"Training failed: {e}")

    finally:
        # Cleanup: ensure all resources are released
        try:
            # Kill any spawned child processes (dataloader workers) to release VRAM
            import psutil
            current_pid = os.getpid()
            try:
                current_proc = psutil.Process(current_pid)
                children = current_proc.children(recursive=True)
                for child in children:
                    try:
                        child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if children:
                    psutil.wait_procs(children, timeout=3)
                    job_storage.append_job_log(job_id, "INFO", f"Killed {len(children)} child process(es)")
            except Exception as e:
                job_storage.append_job_log(job_id, "WARNING", f"Child process cleanup warning: {e}")

            # Force garbage collection to release CUDA memory
            import gc
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            job_storage.append_job_log(job_id, "WARNING", f"Cleanup warning: {e}")
        
        with _lock:
            _active_jobs.pop(job_id, None)
        # Clear cached last progress event so future subscribers don't get stale data
        event_bus.clear_last_event(train_channel(job_id))
        
        job_storage.append_job_log(job_id, "INFO", "Worker thread cleanup completed")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _select_cache_strategy(ds_root: "Path | None", job_id: str, is_remote_fs: bool) -> "str | bool":
    """Choose the best Ultralytics cache mode given available RAM and dataset size.

    Returns:
        "ram"  — load all images into shared memory (zero I/O per batch).
                 Chosen when free RAM >= estimated_dataset_bytes * 1.5.
        True   — disk .cache file (label scan saved, images read per batch).
        False  — no caching (last resort for remote FS when RAM is tight).
    """
    import os
    try:
        import psutil
        free_ram = psutil.virtual_memory().available
    except Exception:
        free_ram = 0

    # Estimate dataset image size: count images and multiply by median size.
    dataset_bytes = 0
    if ds_root and ds_root.exists():
        try:
            _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            sizes = []
            for split in ("train", "train2017"):
                img_dir = ds_root / "images" / split
                if not img_dir.exists():
                    img_dir = ds_root / "images"
                if img_dir.exists():
                    for p in img_dir.iterdir():
                        if p.suffix.lower() in _IMG_EXTS:
                            try:
                                sizes.append(p.stat().st_size)
                            except OSError:
                                pass
                        if len(sizes) >= 200:
                            break
                if sizes:
                    break
            if sizes:
                import statistics
                median_size = statistics.median(sizes)
                # Count total images cheaply (iterdir once more)
                total_imgs = sum(
                    1 for p in img_dir.iterdir()
                    if p.suffix.lower() in _IMG_EXTS
                ) if img_dir.exists() else len(sizes)
                dataset_bytes = int(median_size * total_imgs)
        except Exception:
            dataset_bytes = 0

    required_ram = int(dataset_bytes * 1.5)
    if free_ram > 0 and dataset_bytes > 0 and free_ram >= required_ram:
        gb = dataset_bytes / (1024 ** 3)
        free_gb = free_ram / (1024 ** 3)
        job_storage.append_job_log(job_id, "INFO",
            f"Cache strategy: ram — dataset ~{gb:.1f} GB, free RAM {free_gb:.1f} GB "
            f"(>= {required_ram/(1024**3):.1f} GB threshold)")
        return "ram"

    # Not enough RAM for full image cache — fall back to disk .cache (label scan only).
    # On remote FS this is still better than False because scan only happens once.
    chosen = True  # disk .cache
    if free_ram > 0 and dataset_bytes > 0:
        gb = dataset_bytes / (1024 ** 3)
        free_gb = free_ram / (1024 ** 3)
        job_storage.append_job_log(job_id, "INFO",
            f"Cache strategy: disk — dataset ~{gb:.1f} GB > free RAM {free_gb:.1f} GB, "
            "using label .cache file only")
    else:
        job_storage.append_job_log(job_id, "INFO",
            "Cache strategy: disk — could not estimate dataset size, defaulting to disk cache")
    return chosen


def _redirect_cache_to_tmp(ds_root: "Path | None", job_id: str) -> None:
    """Symlink Ultralytics .cache files from NFS dataset dir to /tmp.

    Ultralytics writes <labels_dir>/<split>.cache next to the label .txt files.
    On NFS these writes are slow and can hang.  We create a fast local mirror
    under /tmp and symlink from the NFS path → /tmp path so Ultralytics writes
    to local disk transparently.
    """
    if not ds_root or not ds_root.exists():
        return
    import tempfile, os

    tmp_base = Path(tempfile.gettempdir()) / "ultralytics_cache" / ds_root.name
    try:
        tmp_base.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    redirected = []
    for split in ("train", "val", "train2017", "val2017", "test"):
        labels_dir = ds_root / "labels" / split
        if not labels_dir.exists():
            continue
        cache_nfs = labels_dir / f"{split}.cache"
        cache_tmp = tmp_base / f"{split}.cache"
        # Only redirect if NFS cache doesn't exist yet (avoid clobbering valid cache)
        if cache_nfs.exists() and not cache_nfs.is_symlink():
            continue
        # Remove stale symlink
        if cache_nfs.is_symlink():
            try:
                cache_nfs.unlink()
            except Exception:
                continue
        # Create symlink NFS path → /tmp path
        try:
            cache_nfs.symlink_to(cache_tmp)
            redirected.append(f"{cache_nfs} → {cache_tmp}")
        except Exception:
            pass

    if redirected:
        job_storage.append_job_log(job_id, "INFO",
            f"Redirected {len(redirected)} .cache path(s) to /tmp: " +
            ", ".join(redirected))


def _get_optimal_workers() -> int:
    """Return a safe dataloader worker count for the current environment.

    In Docker containers the CPU quota (cpu.cfs_quota_us / cpu.cfs_period_us)
    is often much lower than ``os.cpu_count()``.  Spawning more workers than
    available CPU shares causes context-switch thrashing and *slows* data
    loading.  We read the cgroup v1/v2 quota and clamp accordingly.

    Safe defaults:
      - Docker with CPU quota  → min(quota_cpus, 4)
      - No quota / bare-metal  → min(os.cpu_count() // 2, 8), minimum 2
    """
    import os

    def _read_int(path: str) -> int | None:
        try:
            return int(open(path).read().strip())
        except Exception:
            return None

    # ── cgroup v2 ────────────────────────────────────────────────────────────
    v2_max = "/sys/fs/cgroup/cpu.max"
    try:
        parts = open(v2_max).read().strip().split()
        if len(parts) == 2 and parts[0] != "max":
            quota, period = int(parts[0]), int(parts[1])
            capped = max(1, quota // period)
            return min(capped, 4)
    except Exception:
        pass

    # ── cgroup v1 ────────────────────────────────────────────────────────────
    quota = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and quota > 0:
        capped = max(1, quota // period)
        return min(capped, 4)

    # ── Bare-metal / no quota ────────────────────────────────────────────────
    cpu_count = os.cpu_count() or 4
    return max(2, min(cpu_count // 2, 8))


def _should_stop(job_id: str) -> bool:
    with _lock:
        info = _active_jobs.get(job_id)
        return info["stop"] if info else False


def _publish(job_id: str, job: dict) -> None:
    """Publish job update to SSE."""
    event_bus.publish_sync(job_channel(job_id), {
        "type": "job_update",
        "job_id": job_id,
        "status": job["status"],
        "epoch": job.get("epoch", 0),
        "total_epochs": job.get("total_epochs", 0),
        "message": job.get("message", ""),
        "best_fitness": job.get("best_fitness"),
    })


def _resolve_weight(pretrained: str) -> Path | None:
    """Resolve a weight_id or path to an actual .pt file."""
    # Direct path
    p = Path(pretrained)
    if p.exists():
        return p
    # Weight ID → look in weights dir
    from . import weight_storage
    wp = weight_storage.weight_pt_path(pretrained)
    if wp.exists():
        return wp
    return None


def _read_dataset_root_from_data_yaml(data_yaml_path: Path) -> Path | None:
    """Extract dataset root from Ultralytics data.yaml `path:` key."""
    try:
        for raw in data_yaml_path.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or not line.startswith("path:"):
                continue
            value = line.split(":", 1)[1].strip().strip('"').strip("'")
            if not value:
                continue
            ds_root = Path(value).expanduser()
            if not ds_root.is_absolute():
                ds_root = (data_yaml_path.parent / ds_root).resolve()
            return ds_root
    except Exception:
        return None
    return None


def _preflight_cache_diag(data_arg: Any, job_id: str) -> None:
    """Log label cache writability and hash validity for each split before training."""
    if not data_arg:
        return
    data_path = Path(str(data_arg))
    if not (data_path.exists() and data_path.suffix.lower() in {".yaml", ".yml"}):
        return

    dataset_root = _read_dataset_root_from_data_yaml(data_path)
    if not dataset_root:
        return

    labels_dir = dataset_root / "labels"
    if not labels_dir.exists():
        job_storage.append_job_log(job_id, "DEBUG",
            f"Cache diag: labels dir not found at {labels_dir}")
        return

    import os
    import numpy as np
    from ultralytics.data.utils import get_hash, img2label_paths

    writeable = os.access(str(labels_dir), os.W_OK)

    # Read im_paths from data.yaml the same way Ultralytics does
    # Ultralytics resolves: path + train/val → list of image files, then img2label_paths()
    try:
        import yaml as _yaml
        data_cfg = _yaml.safe_load(data_path.read_text())
    except Exception:
        data_cfg = {}

    ds_root_str = data_cfg.get("path", str(dataset_root))
    ds_root_p = Path(ds_root_str)

    def _resolve_im_files(split_key: str) -> list[str]:
        """Resolve image paths for a split the same way Ultralytics does."""
        split_val = data_cfg.get(split_key, "")
        if not split_val:
            return []
        split_val = str(split_val).strip()
        p = Path(split_val)
        if not p.is_absolute():
            p = ds_root_p / p
        if p.suffix in {".txt"}:
            # txt file list mode
            if not p.exists():
                return []
            return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        elif p.is_dir():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
            return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in exts)
        return []

    for split_name in ("train", "val"):
        # Compute current hash and cache path the same way Ultralytics does:
        # cache_path = Path(label_files[0]).parent.with_suffix(".cache")
        try:
            im_files = _resolve_im_files(split_name)
            if not im_files:
                job_storage.append_job_log(job_id, "DEBUG",
                    f"Cache diag: split={split_name} — no images resolved from data.yaml")
                continue
            label_files = img2label_paths(im_files)
            current_hash = get_hash(label_files + im_files)
            # Cache path = first label file's parent dir with .cache suffix
            cache_file = Path(label_files[0]).parent.with_suffix(".cache")
        except Exception as _he:
            job_storage.append_job_log(job_id, "DEBUG",
                f"Cache diag: split={split_name} hash error: {_he}")
            continue

        if cache_file.exists():
            try:
                cached = np.load(str(cache_file), allow_pickle=True).item()
                ver = cached.get("version", "?")
                stored_hash = cached.get("hash", "?")
                match = "✅ MATCH" if stored_hash == current_hash else "❌ MISMATCH"
                job_storage.append_job_log(job_id, "DEBUG",
                    f"Cache diag: {cache_file.name} ({split_name}) | version={ver} | {match} | "
                    f"stored={str(stored_hash)[:12]} current={current_hash[:12]} | writable={writeable}")
            except Exception as e:
                job_storage.append_job_log(job_id, "WARNING",
                    f"Cache diag: {cache_file.name} UNREADABLE ({e}) — will rescan")
        else:
            job_storage.append_job_log(job_id, "DEBUG",
                f"Cache diag: {cache_file.name} ({split_name}) NOT FOUND | "
                f"writable={writeable} — {'will save after scan' if writeable else 'NOT saving (read-only!)'}")


def _cleanup_corrupt_dataset_cache(data_arg: Any, job_id: str) -> dict[str, Any]:
    """Scan dataset label caches and remove corrupted *.cache files.

    Returns summary: {dataset_root, scanned, removed, removed_paths}.
    """
    result = {
        "dataset_root": None,
        "scanned": 0,
        "removed": 0,
        "removed_paths": [],
    }
    if not data_arg:
        return result

    data_path = Path(str(data_arg))
    dataset_root: Path | None = None

    # If data arg is a YAML file, resolve dataset root from `path:` key.
    if data_path.exists() and data_path.is_file() and data_path.suffix.lower() in {".yaml", ".yml"}:
        dataset_root = _read_dataset_root_from_data_yaml(data_path)
        if dataset_root is None:
            job_storage.append_job_log(
                job_id,
                "DEBUG",
                f"Dataset cache preflight skipped: could not resolve dataset root from {data_path}",
            )
            return result
    elif data_path.exists() and data_path.is_dir():
        dataset_root = data_path
    else:
        # `data` may be a registry key or non-path string; nothing to scan here.
        return result

    result["dataset_root"] = str(dataset_root)
    labels_dir = dataset_root / "labels"
    if not labels_dir.exists():
        return result

    cache_files = sorted(labels_dir.rglob("*.cache"))
    if not cache_files:
        return result

    import numpy as np

    job_storage.append_job_log(job_id, "DEBUG",
        f"Dataset cache files found: {[str(p.relative_to(dataset_root)) for p in cache_files]}")

    for cache_path in cache_files:
        result["scanned"] += 1
        try:
            np.load(str(cache_path), allow_pickle=True).item()
        except Exception as cache_err:
            try:
                cache_path.unlink(missing_ok=True)
                result["removed"] += 1
                result["removed_paths"].append(str(cache_path))
                job_storage.append_job_log(
                    job_id,
                    "WARNING",
                    f"Removed corrupted dataset cache: {cache_path} ({type(cache_err).__name__}: {cache_err})",
                )
            except Exception as unlink_err:
                job_storage.append_job_log(
                    job_id,
                    "WARNING",
                    f"Detected corrupted dataset cache but failed to delete: {cache_path} ({unlink_err})",
                )

    return result


def _save_best_weight(job_id: str, job_dir: Path, model_name: str = "") -> str | None:
    """Find best.pt from training run and save to weight storage with lineage."""
    best_pt = job_dir / "runs" / "train" / "weights" / "best.pt"
    is_best = True
    if not best_pt.exists():
        # Try last.pt as fallback
        best_pt = job_dir / "runs" / "train" / "weights" / "last.pt"
        is_best = False
    if not best_pt.exists():
        return None

    import shutil
    from . import weight_storage
    
    # Load job to get training details
    job = job_storage.load_job(job_id)
    if not job:
        return None
    
    # Generate weight_id
    weight_id = uuid.uuid4().hex[:12]
    dest_dir = WEIGHTS_DIR / weight_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy weight file
    shutil.copy2(best_pt, dest_dir / "weight.pt")
    
    # Extract metrics from job history
    final_accuracy = None
    final_loss = None
    if job.get("history"):
        last_epoch = job["history"][-1]
        final_accuracy = last_epoch.get("mAP50_95") or last_epoch.get("fitness")
        final_loss = last_epoch.get("box_loss")
    
    # Get parent weight from config
    parent_weight_id = job.get("config", {}).get("pretrained", "")
    if parent_weight_id and not Path(parent_weight_id).exists():
        # It's a weight_id, not a path
        pass
    else:
        parent_weight_id = None
    
    # Calculate total training time
    total_time = None
    if job.get("started_at") and job.get("completed_at"):
        try:
            from datetime import datetime as dt
            start = dt.fromisoformat(job["started_at"].replace("Z", "+00:00"))
            end = dt.fromisoformat(job["completed_at"].replace("Z", "+00:00"))
            total_time = (end - start).total_seconds()
        except Exception:
            pass
    
    # Save weight metadata with lineage
    # Prefer dataset_name (original registered name) over full path
    _cfg = job.get("config", {})
    _dataset = _cfg.get("dataset_name") or _cfg.get("data", "")
    weight_storage.save_weight_meta(
        model_id=job.get("model_id", ""),
        model_name=model_name,
        model_scale=job.get("model_scale", "n"),
        job_id=job_id,
        dataset=_dataset,
        epochs_trained=job.get("total_epochs", 0),
        final_accuracy=final_accuracy,
        final_loss=final_loss,
        weight_id=weight_id,
        parent_weight_id=parent_weight_id,
        total_time=total_time,
        device=job.get("config", {}).get("device", ""),
    )
    
    return weight_id
