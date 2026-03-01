"""
Custom Ultralytics Trainer with enhanced monitoring and logging.

Extends DetectionTrainer to provide:
- Custom logging (no tqdm conflicts)
- Extended metrics collection
- Gradient and weight recording
- Enhanced checkpoint management
- Plot generation
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

import threading
import torch
import numpy as np
from copy import copy
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import ModelEMA

from . import job_storage
from ..config import JOBS_DIR


class NaNLossError(RuntimeError):
    pass


class CustomValidator(DetectionValidator):
    """Custom validator that saves test samples per class."""
    
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, sample_per_class=0):
        # Ultralytics 8.3+ removed pbar from Validator __init__
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        # Accept sample_per_class as direct parameter (not from args to avoid Ultralytics validation)
        self.sample_per_class = sample_per_class
        self.saved_counts = {}  # class_idx -> count
        self.class_names = args.names if hasattr(args, 'names') else {}
        
        # Latency tracking
        self.inference_times = []  # Track per-batch inference times
        self.preprocess_times = []
        self.postprocess_times = []

    def __call__(self, trainer=None, model=None):
        """Override validation call to track latency."""
        import time
        import torch
        
        # Reset latency tracking
        self.inference_times = []
        self.preprocess_times = []
        self.postprocess_times = []
        
        # Store original model forward to wrap it
        import torch.nn as nn
        is_nn_model = model is not None and isinstance(model, nn.Module)
        if is_nn_model:
            original_forward = model.forward
            
            def timed_forward(*args, **kwargs):
                """Wrapper to track inference time."""
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                result = original_forward(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.inference_times.append(time.time() - start)
                return result
            
            # Temporarily replace forward
            model.forward = timed_forward
        
        try:
            # Run parent validation
            result = super().__call__(trainer=trainer, model=model)
        finally:
            # Restore original forward
            if is_nn_model:
                model.forward = original_forward
        
        # Calculate average latencies
        if self.inference_times:
            self.avg_inference_ms = sum(self.inference_times) / len(self.inference_times) * 1000
            self.avg_preprocess_ms = sum(self.preprocess_times) / len(self.preprocess_times) * 1000 if self.preprocess_times else 0
            self.avg_postprocess_ms = sum(self.postprocess_times) / len(self.postprocess_times) * 1000 if self.postprocess_times else 0
            self.total_latency_ms = self.avg_preprocess_ms + self.avg_inference_ms + self.avg_postprocess_ms
        else:
            self.avg_inference_ms = 0
            self.avg_preprocess_ms = 0
            self.avg_postprocess_ms = 0
            self.total_latency_ms = 0
        
        return result
    
    def update_metrics(self, preds, batch):
        """Override to save samples per class."""
        # Call parent method first
        super().update_metrics(preds, batch)
        
        # Save samples if enabled
        if self.sample_per_class > 0:
            try:
                self._save_class_samples(preds, batch)
            except Exception as e:
                LOGGER.warning(f"Failed to save class samples: {e}")

    def _save_class_samples(self, preds, batch):
        """Save images containing specific classes."""
        if not hasattr(self, 'saved_counts'):
            self.saved_counts = {}
            
        # Import plotting utils
        from ultralytics.utils.plotting import Annotator, colors
        import cv2
        
        # batch['img'] is [B, 3, H, W] (normalized usually or not)
        images = batch['img']
        if images.is_cuda:
            images = images.cpu()
        images = images.float()
        
        # Get image filenames if available
        im_files = batch.get('im_file', [])
        
        batch_size = len(images)
        
        for i in range(batch_size):
            # Identify GT classes in this image
            mask = batch['batch_idx'] == i
            if not mask.any():
                continue
                
            gt_classes = batch['cls'][mask].int().tolist()
            unique_classes = set(gt_classes)
            
            # Check if we need to save for any of these classes
            save_for_classes = []
            for c in unique_classes:
                count = self.saved_counts.get(c, 0)
                if count < self.sample_per_class:
                    save_for_classes.append(c)
            
            if not save_for_classes:
                continue
                
            # Prepare image for saving
            # Assuming images are 0-1 float or 0-255 float. YOLOv8 normalize puts them 0-1.
            im = images[i].numpy().transpose((1, 2, 0))
            if im.max() <= 1.0:
                im = im * 255
            im = im.astype(np.uint8)
            im = np.ascontiguousarray(im)
            
            # Create annotator
            fname = Path(im_files[i]).name if i < len(im_files) else f"val_{self.seen}_{i}.jpg"
            annotator = Annotator(im, line_width=2, example=str(self.class_names))
            
            # Draw predictions
            if len(preds) > i and preds[i] is not None:
                det = preds[i]
                if len(det) > 0:
                    det = det.clone()
                    # det is [xyxy, conf, cls]
                    for *box, conf, cls in det:
                        c = int(cls)
                        # Only draw boxes if they match one of the classes we care about?
                        # Or draw all boxes to show context? Draw all is better.
                        label = f'{self.class_names[c]} {conf:.2f}'
                        annotator.box_label(box, label, color=colors(c, True))
            
            # Also draw GT boxes? Usually validator plots predictions. 
            # If we want to evaluate "sample", seeing predictions is more useful.
            
            im_with_plots = annotator.result()
            
            # Save to class-specific folders
            for c in save_for_classes:
                class_name = self.class_names[c] if isinstance(self.class_names, dict) else str(c)
                # Sanitize filename
                class_name = "".join([x if x.isalnum() else "_" for x in str(class_name)])
                
                class_dir = self.save_dir / 'samples' / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = class_dir / fname
                cv2.imwrite(str(save_path), im_with_plots)
                
                self.saved_counts[c] = self.saved_counts.get(c, 0) + 1
    
    def preprocess(self, batch):
        """Track preprocessing time."""
        import time
        start = time.time()
        result = super().preprocess(batch)
        self.preprocess_times.append(time.time() - start)
        return result
    
    def postprocess(self, preds):
        """Track postprocessing time."""
        import time
        start = time.time()
        result = super().postprocess(preds)
        self.postprocess_times.append(time.time() - start)
        return result


class CustomDetectionTrainer(DetectionTrainer):
    """Custom trainer with enhanced monitoring for Model Designer."""
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """Initialize custom trainer.
        
        Args:
            cfg: Configuration dict or path to config file
            overrides: Override parameters
            _callbacks: Callback dict
        """
        # CRITICAL: Ensure sys.argv is empty to prevent Ultralytics CLI parsing
        import sys
        import os
        os.environ['YOLO_CLI'] = '0'
        sys.argv = []
        
        # Ensure overrides is a dict
        if overrides is None:
            overrides = {}
        
        # 1. Create a copy of overrides to modify
        clean_overrides = overrides.copy()
        
        # Helper to get extraction source (overrides priority, then _custom_params)
        custom_source = getattr(self, '_custom_params', {})
        
        # 2. Extract and remove custom params
        self.job_id = clean_overrides.pop('job_id', custom_source.get('job_id'))
        self.record_gradients = clean_overrides.pop('record_gradients', custom_source.get('record_gradients', False))
        self.record_weights = clean_overrides.pop('record_weights', custom_source.get('record_weights', False))
        self.gradient_interval = clean_overrides.pop('gradient_interval', custom_source.get('gradient_interval', 1))
        self.weight_interval = clean_overrides.pop('weight_interval', custom_source.get('weight_interval', 1))
        self.sample_per_class = clean_overrides.pop('sample_per_class', custom_source.get('sample_per_class', 0))
        
        # _partition_configs / _dataset_name no longer needed — TXT splits in data.yaml handle partition filtering
        clean_overrides.pop('_partition_configs', None)
        clean_overrides.pop('_dataset_name', None)

        # Debug logs
        if self.job_id:
             job_storage.append_job_log(self.job_id, "DEBUG", f"Cleaned overrides keys: {list(clean_overrides.keys())}")
        
        # CRITICAL: Filter out invalid YOLO arguments before passing to parent
        # Ultralytics validates all config keys and rejects unknown ones
        from ultralytics.cfg import get_cfg, DEFAULT_CFG_DICT, check_dict_alignment
        
        # List of keys to remove (invalid for YOLO training)
        INVALID_KEYS = {'session', 'sample_per_class', 'record_gradients', 'gradient_interval', 
                        'record_weights', 'weight_interval', '_partition_configs', '_dataset_name'}
        
        # Build complete config by merging defaults with our overrides
        if cfg is None:
            cfg = DEFAULT_CFG_DICT.copy()
        elif isinstance(cfg, str):
            # If cfg is a path, let get_cfg handle it
            pass
        elif isinstance(cfg, dict):
            # Merge with defaults
            cfg = {**DEFAULT_CFG_DICT, **cfg}
        
        # Merge clean_overrides into cfg
        if isinstance(cfg, dict):
            cfg.update(clean_overrides)
            
            # CRITICAL: Remove invalid keys that would trigger validation errors
            for key in INVALID_KEYS:
                cfg.pop(key, None)
            
            clean_overrides = {}  # Already merged
        
        super().__init__(cfg, clean_overrides, _callbacks)
        
        # DO NOT inject custom params into self.args - Ultralytics validates all args keys
        # Keep custom params as instance variables only (self.sample_per_class, etc.)
        # They will be accessed directly in get_validator() and other methods
        
        # Disable tqdm progress bars completely
        self.args.verbose = False
        
        # Also disable progress bar at the class level
        from ultralytics.utils import TQDM
        TQDM.disable = True
        
        # Time tracking for progress events
        import time as _time
        self._train_start_time: float = _time.time()
        self._epoch_start_time: float = _time.time()
        self._epoch_completed: int = 0  # number of fully completed epochs
        self._batch_start_time: float = _time.time()
        self._imgs_per_sec: float | None = None
        self._batch_counter: int = 0
        self._last_batch_time: float = _time.time()

        def _on_train_batch_end_cb(trainer):
            # Detect NaN/Inf early (every batch) before emitting rate-limited progress logs.
            trainer._check_nan_loss_items()
            trainer._on_batch_end()

        # Register batch-end callback (Ultralytics calls with trainer as arg)
        self.add_callback("on_train_batch_end", _on_train_batch_end_cb)

        # Verify job_id is set
        if self.job_id:
            job_storage.append_job_log(self.job_id, "INFO", f"CustomDetectionTrainer initialized with job_id: {self.job_id}")
        else:
            LOGGER.warning("CustomDetectionTrainer initialized WITHOUT job_id - logs will go to console!")

    def get_validator(self):
        """Return custom validator."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return CustomValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args), 
            _callbacks=self.callbacks,
            sample_per_class=self.sample_per_class  # Pass as direct parameter
        )

        
    def log(self, text: str, level: str = "INFO") -> None:
        """Custom logging that goes to job storage instead of console.
        
        Args:
            text: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        """
        if self.job_id:
            job_storage.append_job_log(self.job_id, level, text)
        else:
            LOGGER.info(f"[{level}] {text}")
    
    def _do_train(self, world_size=1):
        """Override training loop to add custom logging."""
        self.log(f"Starting training for {self.epochs} epochs", "INFO")
        
        # Call parent training loop
        result = super()._do_train()
        
        self.log("Training loop completed", "INFO")
        return result
    
    def _setup_train(self):
        """Override setup to add logging."""
        self.log("Setting up training...", "INFO")

        import threading
        import time as _time
        import sys
        import traceback

        done = threading.Event()
        start_t = _time.time()
        timeout_s = 600  # 10 min — large datasets (IDD ~47k images) need time to scan+cache
        heartbeat_interval = 30  # log progress every 30s so user knows it is not hung

        def _watchdog():
            """Emit heartbeat logs every 30s; dump stacks on timeout."""
            while True:
                triggered = not done.wait(heartbeat_interval)
                if done.is_set():
                    break
                elapsed = _time.time() - start_t
                if elapsed >= timeout_s:
                    # Timeout — dump stacks
                    try:
                        self.log(
                            f"Training setup watchdog triggered after {elapsed:.1f}s - dumping thread stacks",
                            "WARNING",
                        )
                        frames = sys._current_frames()
                        for th in threading.enumerate():
                            try:
                                frame = frames.get(th.ident)
                                if frame is None:
                                    continue
                                stack = "".join(traceback.format_stack(frame))
                                self.log(f"Thread stack | name={th.name} ident={th.ident}\n{stack}", "WARNING")
                            except Exception:
                                continue
                    except Exception as e:
                        self.log(f"Training setup watchdog failed: {e}", "WARNING")
                    break
                else:
                    # Heartbeat — user sees setup is still running
                    self.log(
                        f"Dataset setup still running ({elapsed:.0f}s elapsed) — "
                        "scanning labels / building .cache file...",
                        "INFO",
                    )

        threading.Thread(target=_watchdog, daemon=True, name="setup_train_watchdog").start()
        try:
            result = super()._setup_train()
        finally:
            done.set()

        self.log(
            f"Training setup complete - {self.train_loader.dataset.ni} train images, "
            f"{self.test_loader.dataset.ni} val images",
            "INFO",
        )
        return result

    def _load_checkpoint_state(self, ckpt):
        """Load resume state with backward-compatible EMA state_dict handling."""
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])

        if self.ema and ckpt.get("ema"):
            # Keep upstream behavior of rebuilding EMA wrapper first.
            self.ema = ModelEMA(self.model)
            ema_sd = ckpt["ema"].float().state_dict()

            try:
                self.ema.ema.load_state_dict(ema_sd)
            except RuntimeError as e:
                msg = str(e)
                profile_key_mismatch = (
                    "total_ops" in msg
                    or "total_params" in msg
                    or "Missing key(s) in state_dict" in msg
                )
                if not profile_key_mismatch:
                    raise

                self.log(
                    "Resume EMA strict load failed due to checkpoint/model key mismatch "
                    "(likely profiling buffers such as total_ops/total_params). "
                    "Retrying with strict=False.",
                    "WARNING",
                )
                incompatible = self.ema.ema.load_state_dict(ema_sd, strict=False)
                missing_n = len(getattr(incompatible, "missing_keys", []))
                unexpected_n = len(getattr(incompatible, "unexpected_keys", []))
                self.log(
                    f"Resume EMA non-strict load completed: missing={missing_n}, unexpected={unexpected_n}",
                    "INFO",
                )

            self.ema.updates = ckpt["updates"]

        self.best_fitness = ckpt.get("best_fitness", 0.0)
    
    
    def progress_string(self):
        """Override to provide clean progress string without tqdm formatting."""
        # Return empty string to disable default progress bar
        return ""
    
    def get_pbar(self, desc, total):
        """Override to disable tqdm progress bar completely."""
        # Return a dummy object that does nothing
        class DummyPbar:
            def __init__(self):
                pass
            def update(self, n=1):
                pass
            def close(self):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyPbar()
    
    def on_train_epoch_start(self):
        """Track epoch start time."""
        import time as _time
        self._epoch_start_time = _time.time()
        super_method = getattr(super(), 'on_train_epoch_start', None)
        if super_method:
            super_method()

    def optimizer_step(self):
        """Override optimizer step."""
        super().optimizer_step()

    def _on_batch_end(self):
        """Called after every batch via callback. Tracks ni internally."""
        import time as _time
        import torch
        import psutil

        # Track per-batch timing for speed calculation
        now = _time.time()
        if not hasattr(self, '_last_batch_time'):
            self._last_batch_time = now

        # Increment internal batch counter
        self._batch_counter = getattr(self, '_batch_counter', 0) + 1

        nb = len(self.train_loader)  # total batches per epoch
        batch = (self._batch_counter - 1) % nb + 1  # 1-indexed within epoch
        total_batches = nb

        # Rate-limit: emit every 50 batches, first batch, and last batch
        if batch != 1 and batch % 50 != 0 and batch != total_batches:
            self._last_batch_time = now
            return

        epoch = self.epoch + 1
        batch_pct = round((batch / total_batches) * 100, 1)

        # ── Time calculations ──────────────────────────────────────────────
        total_elapsed_s = round(now - self._train_start_time, 1)
        epoch_elapsed_s = round(now - self._epoch_start_time, 1)

        # Speed: images per second
        batch_size = getattr(self.args, 'batch', 1) or 1
        dt = now - self._last_batch_time
        imgs_per_sec: float | None = round(batch_size * 50 / dt, 1) if dt > 0 else None
        self._imgs_per_sec = imgs_per_sec
        self._last_batch_time = now

        # ETA
        completed_epochs = self._epoch_completed
        avg_epoch_s: float | None = None
        if completed_epochs > 0:
            avg_epoch_s = round(total_elapsed_s / completed_epochs, 1)
        elif epoch_elapsed_s > 0 and batch > 0:
            avg_epoch_s = round(epoch_elapsed_s / (batch / total_batches), 1)

        eta_s: float | None = None
        if avg_epoch_s is not None:
            remaining_epochs = self.epochs - epoch
            remaining_in_epoch = epoch_elapsed_s * (1 - batch / total_batches)
            eta_s = round(remaining_epochs * avg_epoch_s + remaining_in_epoch, 0)

        # ── Loss values ────────────────────────────────────────────────────
        box_loss = cls_loss = dfl_loss = None
        if hasattr(self, 'loss_items') and self.loss_items is not None:
            loss_items = self.loss_items.cpu().numpy() if hasattr(self.loss_items, 'cpu') else self.loss_items
            box_loss = round(float(loss_items[0]), 4) if len(loss_items) > 0 else None
            cls_loss = round(float(loss_items[1]), 4) if len(loss_items) > 1 else None
            dfl_loss = round(float(loss_items[2]), 4) if len(loss_items) > 2 else None

        # ── Device / resource info ─────────────────────────────────────────
        device_str = 'cpu'
        gpu_mem_gb = None
        gpu_mem_reserved_gb = None
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                # DDP: sum memory across all GPUs
                device_str = ",".join(f"cuda:{i}" for i in range(n_gpus))
                gpu_mem_gb = round(
                    sum(torch.cuda.memory_allocated(i) for i in range(n_gpus)) / (1024**3), 2)
                gpu_mem_reserved_gb = round(
                    sum(torch.cuda.memory_reserved(i) for i in range(n_gpus)) / (1024**3), 2)
            else:
                dev_idx = torch.cuda.current_device()
                device_str = f'cuda:{dev_idx}'
                gpu_mem_gb = round(torch.cuda.memory_allocated(dev_idx) / (1024**3), 2)
                gpu_mem_reserved_gb = round(torch.cuda.memory_reserved(dev_idx) / (1024**3), 2)
        vm = psutil.virtual_memory()
        ram_used_gb = round(vm.used / (1024**3), 2)
        ram_total_gb = round(vm.total / (1024**3), 2)

        if not self.job_id or box_loss is None:
            return

        progress_data = {
            "type": "progress",
            "phase": "train",
            "epoch": f"{epoch}/{self.epochs}",
            "total_epochs": self.epochs,
            "batch": f"{batch}/{total_batches}",
            "total_batches": total_batches,
            "percent": batch_pct,
            "losses": {"box": box_loss, "cls": cls_loss, "dfl": dfl_loss},
            "device": device_str,
            "ram_gb": ram_used_gb,
            "ram_total_gb": ram_total_gb,
            "gpu_mem_gb": gpu_mem_gb,
            "gpu_mem_reserved_gb": gpu_mem_reserved_gb,
            "epoch_elapsed_s": epoch_elapsed_s,
            "total_elapsed_s": total_elapsed_s,
            "avg_epoch_s": avg_epoch_s,
            "eta_s": eta_s,
            "imgs_per_sec": imgs_per_sec,
        }

        # Log to job logs (SSE tails this file)
        job_storage.append_job_log(self.job_id, "PROGRESS",
            f"Epoch {epoch}/{self.epochs} | Batch {batch}/{total_batches} ({batch_pct}%)",
            progress_data
        )

    def _check_nan_loss_items(self) -> None:
        li = getattr(self, 'loss_items', None)
        if li is None:
            return

        if isinstance(li, torch.Tensor):
            t = li.detach()
        else:
            t = torch.as_tensor(li)

        if not torch.isfinite(t).all():
            epoch = int(getattr(self, 'epoch', -1)) + 1 if hasattr(self, 'epoch') else None
            batch_i = getattr(self, 'batch_i', None)
            if self.job_id:
                job_storage.append_job_log(
                    self.job_id,
                    "WARNING",
                    "Loss NaN/Inf detected mid-epoch, triggering recovery...",
                    {
                        "type": "nan_detected",
                        "phase": "train",
                        "epoch": epoch,
                        "batch_i": int(batch_i) if isinstance(batch_i, (int, np.integer)) else batch_i,
                    },
                )
            raise NaNLossError("NaN/Inf loss detected")
    
    def plot_training_samples(self, batch, ni):
        """Override to add logging when plotting samples."""
        self.log(f"Plotting training samples (batch {ni})", "DEBUG")
        return super().plot_training_samples(batch, ni)
    
    def plot_metrics(self):
        """Override to add logging when plotting metrics."""
        self.log("Plotting training metrics", "INFO")
        return super().plot_metrics()
    
    def plot_training_labels(self):
        """Override to add logging when plotting labels."""
        self.log("Plotting training labels", "INFO")
        return super().plot_training_labels()
    
    def save_model(self):
        """Override to add logging when saving model."""
        self.log(f"Saving model checkpoint at epoch {self.epoch + 1}", "INFO")
        return super().save_model()
    
    def final_eval(self):
        """Override to add logging for final evaluation."""
        self.log("Running final evaluation", "INFO")
        result = super().final_eval()
        self.log("Final evaluation complete", "INFO")
        return result
    
    def _setup_ddp(self, world_size):
        """Override DDP setup to add logging."""
        self.log(f"Setting up DDP with world_size={world_size}", "INFO")
        return super()._setup_ddp(world_size)
    
    def validate(self):
        """Run validation and collect extended metrics.
        
        Returns:
            Validation metrics dict
        """
        import psutil
        import torch
        
        # Log validation start with structured PROGRESS data
        if self.job_id:
            job_storage.append_job_log(self.job_id, "PROGRESS",
                f"Running validation for epoch {self.epoch + 1}...",
                {
                    'type': 'progress',
                    'phase': 'validation',
                    'epoch': f"{self.epoch + 1}/{self.epochs}",
                    'batch': '0/0',
                    'percent': 100,
                    'losses': {},
                }
            )
        else:
            self.log(f"Running validation for epoch {self.epoch + 1}...", "PROGRESS")
        
        # Emit SSE progress event for validation phase to train_channel (frontend listens here)
        if self.job_id:
            from . import event_bus
            from ..constants import train_channel
            event_bus.publish_sync(train_channel(self.job_id), {
                "type": "progress",
                "phase": "validation",
                "epoch": self.epoch + 1,
                "total_epochs": self.epochs,
                "batch": 0,
                "total_batches": 0,
                "percent": 100.0,
                "losses": {},
                "message": f"Validating epoch {self.epoch + 1}...",
            })
        
        # Run parent validation
        val_start = time.time()
        metrics = super().validate()
        val_time = time.time() - val_start
        
        # Collect device and resource info
        device_info = {}
        if torch.cuda.is_available():
            device_info['device'] = f"cuda:{torch.cuda.current_device()}"
            device_info['gpu_mem_gb'] = torch.cuda.memory_allocated() / (1024**3)
        else:
            device_info['device'] = 'cpu'
        device_info['ram_gb'] = psutil.virtual_memory().used / (1024**3)
        
        # Collect extended metrics from validator
        if hasattr(self, 'validator') and hasattr(self.validator, 'metrics'):
            box_metrics = self.validator.metrics.box
            
            # Extract all available metrics
            extended_metrics = self._extract_box_metrics(box_metrics)
            
            # Add latency metrics from validator
            if hasattr(self.validator, 'avg_inference_ms'):
                extended_metrics['inference_latency_ms'] = round(self.validator.avg_inference_ms, 2)
                extended_metrics['preprocess_latency_ms'] = round(self.validator.avg_preprocess_ms, 2)
                extended_metrics['postprocess_latency_ms'] = round(self.validator.avg_postprocess_ms, 2)
                extended_metrics['total_latency_ms'] = round(self.validator.total_latency_ms, 2)
            
            # Add train losses to extended metrics
            if hasattr(self, 'loss_items') and self.loss_items is not None:
                li = self.loss_items.cpu().numpy() if hasattr(self.loss_items, 'cpu') else self.loss_items
                extended_metrics['train_box_loss'] = round(float(li[0]), 6) if len(li) > 0 else None
                extended_metrics['train_cls_loss'] = round(float(li[1]), 6) if len(li) > 1 else None
                extended_metrics['train_dfl_loss'] = round(float(li[2]), 6) if len(li) > 2 else None
            
            # Add validation losses from metrics
            if hasattr(self, 'metrics') and self.metrics:
                m = self.metrics
                extended_metrics['val_box_loss'] = m.get('val/box_loss')
                extended_metrics['val_cls_loss'] = m.get('val/cls_loss')
                extended_metrics['val_dfl_loss'] = m.get('val/dfl_loss')
            
            # Add learning rate
            if hasattr(self, 'optimizer') and self.optimizer:
                extended_metrics['lr'] = self.optimizer.param_groups[0].get('lr', 0.0)
            
            # Add system info
            extended_metrics['device'] = device_info.get('device')
            extended_metrics['ram_gb'] = round(device_info.get('ram_gb', 0), 2)
            if torch.cuda.is_available():
                dev_idx = torch.cuda.current_device()
                extended_metrics['gpu_mem_gb'] = round(torch.cuda.memory_allocated(dev_idx) / (1024**3), 2)
                extended_metrics['gpu_mem_reserved_gb'] = round(torch.cuda.memory_reserved(dev_idx) / (1024**3), 2)
            
            # Add validation time
            extended_metrics['val_time_s'] = round(val_time, 2)

            # Emit structured PROGRESS log with all val metrics + resources
            if self.job_id and extended_metrics.get('map50') is not None:
                train_losses = {}
                if hasattr(self, 'loss_items') and self.loss_items is not None:
                    li = self.loss_items.cpu().numpy() if hasattr(self.loss_items, 'cpu') else self.loss_items
                    train_losses = {
                        'box': round(float(li[0]), 4) if len(li) > 0 else None,
                        'cls': round(float(li[1]), 4) if len(li) > 1 else None,
                        'dfl': round(float(li[2]), 4) if len(li) > 2 else None,
                    }
                vm = psutil.virtual_memory()
                gpu_mem = None
                gpu_reserved = None
                if torch.cuda.is_available():
                    dev_idx = torch.cuda.current_device()
                    gpu_mem = round(torch.cuda.memory_allocated(dev_idx) / (1024**3), 2)
                    gpu_reserved = round(torch.cuda.memory_reserved(dev_idx) / (1024**3), 2)
                # Build time/speed fields
                now = time.time()
                total_elapsed_s = round(now - self._train_start_time, 1) if hasattr(self, '_train_start_time') else None
                epoch_elapsed_s = round(now - self._epoch_start_time, 1) if hasattr(self, '_epoch_start_time') else None
                completed = max(self._epoch_completed, 1)
                avg_epoch_s = round(total_elapsed_s / completed, 1) if total_elapsed_s else None
                eta_s = round(avg_epoch_s * (self.epochs - (self.epoch + 1)), 0) if avg_epoch_s is not None else None
                imgs_per_sec = getattr(self, '_imgs_per_sec', None)

                # Build complete progress data with all metrics
                progress_data = {
                    'type': 'progress',
                    'phase': 'validation_done',
                    'epoch': f"{self.epoch + 1}/{self.epochs}",
                    'total_epochs': self.epochs,
                    'batch': '0/0',
                    'percent': 100,
                    'losses': train_losses,
                    'val_map50': round(extended_metrics['map50'], 4),
                    'val_map': round(extended_metrics['map'], 4),
                    'val_map75': round(extended_metrics.get('map75', 0) or 0, 4),
                    'val_precision': round(extended_metrics.get('mp', 0) or 0, 4),
                    'val_recall': round(extended_metrics.get('mr', 0) or 0, 4),
                    'val_time_s': round(val_time, 1),
                    'device': device_info.get('device'),
                    'ram_gb': round(device_info.get('ram_gb', 0), 2),
                    'ram_total_gb': round(vm.total / (1024**3), 2),
                    'gpu_mem_gb': gpu_mem,
                    'gpu_mem_reserved_gb': gpu_reserved,
                    # Time/speed fields
                    'epoch_elapsed_s': epoch_elapsed_s,
                    'total_elapsed_s': total_elapsed_s,
                    'avg_epoch_s': avg_epoch_s,
                    'eta_s': eta_s,
                    'imgs_per_sec': imgs_per_sec,
                }
                
                # Add latency metrics if available
                if 'inference_latency_ms' in extended_metrics:
                    progress_data['inference_latency_ms'] = extended_metrics['inference_latency_ms']
                    progress_data['preprocess_latency_ms'] = extended_metrics['preprocess_latency_ms']
                    progress_data['postprocess_latency_ms'] = extended_metrics['postprocess_latency_ms']
                    progress_data['total_latency_ms'] = extended_metrics['total_latency_ms']
                
                job_storage.append_job_log(self.job_id, "PROGRESS",
                    f"Validation done epoch {self.epoch + 1}/{self.epochs} | "
                    f"mAP50={extended_metrics['map50']:.4f} | mAP50-95={extended_metrics['map']:.4f}" +
                    (f" | Latency={extended_metrics['total_latency_ms']:.1f}ms" if 'total_latency_ms' in extended_metrics else ""),
                    progress_data
                )
                
                # Also publish to SSE train_channel for real-time frontend updates
                from . import event_bus
                from ..constants import train_channel
                event_bus.publish_sync(train_channel(self.job_id), progress_data)

            # Log comprehensive validation results with all info
            if extended_metrics.get('map50') is not None:
                self.log(
                    f"Epoch {self.epoch + 1}/{self.epochs} Summary:",
                    "INFO"
                )
                self.log(
                    f"  Train Metrics: box_loss={self.loss_items[0]:.4f} cls_loss={self.loss_items[1]:.4f} dfl_loss={self.loss_items[2]:.4f}" if hasattr(self, 'loss_items') else "  Train Metrics: N/A",
                    "INFO"
                )
                self.log(
                    f"  Validation Metrics:",
                    "INFO"
                )
                self.log(
                    f"    Precision: {extended_metrics.get('mp', 0):.4f} | Recall: {extended_metrics.get('mr', 0):.4f}",
                    "INFO"
                )
                self.log(
                    f"    mAP@0.5: {extended_metrics['map50']:.4f} | "
                    f"mAP@0.5:0.95: {extended_metrics['map']:.4f} | "
                    f"mAP@0.75: {extended_metrics.get('map75', 0):.4f}",
                    "INFO"
                )
                if 'inference_latency_ms' in extended_metrics:
                    self.log(
                        f"  Inference Latency: "
                        f"preprocess={extended_metrics['preprocess_latency_ms']:.1f}ms | "
                        f"inference={extended_metrics['inference_latency_ms']:.1f}ms | "
                        f"postprocess={extended_metrics['postprocess_latency_ms']:.1f}ms | "
                        f"total={extended_metrics['total_latency_ms']:.1f}ms",
                        "INFO"
                    )
                self.log(
                    f"  Device: {device_info.get('device', 'N/A')} | "
                    f"RAM: {device_info.get('ram_gb', 0):.1f} GB" +
                    (f" | GPU: {device_info['gpu_mem_gb']:.1f} GB" if 'gpu_mem_gb' in device_info else ""),
                    "INFO"
                )
                self.log(
                    f"  Validation time: {val_time:.1f}s",
                    "INFO"
                )
            
            # Save extended metrics to job
            if self.job_id:
                self._save_extended_metrics(extended_metrics)
        
        return metrics
    
    def _extract_box_metrics(self, box_metrics) -> dict[str, Any]:
        """Extract all metrics from box metrics object.
        
        Args:
            box_metrics: Ultralytics box metrics object
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Helper to safely convert tensors/arrays to lists
        def to_list(val):
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, torch.Tensor):
                if val.numel() == 0:
                    return []
                return val.tolist() if val.numel() > 1 else float(val.item())
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    return []
                if val.ndim == 0:
                    return float(val)
                return val.tolist() if val.size > 1 else float(val.flat[0])
            return val
        
        # Extract all available metrics
        metric_names = [
            'all_ap', 'ap', 'ap50', 'ap_class_index', 'class_result',
            'f1', 'f1_curve', 'fitness', 'map', 'map50', 'map75', 'maps',
            'mean_results', 'mp', 'mr', 'p', 'p_curve', 'prec_values',
            'px', 'r', 'r_curve', 'precision', 'recall'
        ]
        
        for name in metric_names:
            if hasattr(box_metrics, name):
                val = getattr(box_metrics, name)
                metrics[name] = to_list(val)
        
        return metrics
    
    def _save_extended_metrics(self, metrics: dict[str, Any]) -> None:
        """Save comprehensive extended metrics to JSONL file.
        
        This captures ALL custom train/val metrics that we want to track,
        including latency, system info, and detailed metrics not in results.csv.
        
        Args:
            metrics: Extended metrics dictionary from validation
        """
        if not self.job_id:
            return
        
        from ..config import JOBS_DIR
        import json
        from pathlib import Path
        
        job_dir = JOBS_DIR / self.job_id
        extended_metrics_file = job_dir / "extended_metrics.jsonl"
        
        # Build comprehensive epoch data with all metrics
        epoch_data = {
            "epoch": self.epoch + 1,
            "timestamp": time.time(),
            
            # Training losses (from trainer.loss_items)
            "train_box_loss": metrics.get('train_box_loss'),
            "train_cls_loss": metrics.get('train_cls_loss'),
            "train_dfl_loss": metrics.get('train_dfl_loss'),
            
            # Validation losses (from metrics dict)
            "val_box_loss": metrics.get('val_box_loss'),
            "val_cls_loss": metrics.get('val_cls_loss'),
            "val_dfl_loss": metrics.get('val_dfl_loss'),
            
            # Validation metrics (mAP, precision, recall)
            "map50": metrics.get('map50'),
            "map": metrics.get('map'),
            "map75": metrics.get('map75'),
            "precision": metrics.get('mp'),  # mean precision
            "recall": metrics.get('mr'),     # mean recall
            "fitness": metrics.get('fitness'),
            
            # Per-class metrics (if available)
            "ap_per_class": metrics.get('ap'),
            "ap50_per_class": metrics.get('ap50'),
            "precision_per_class": metrics.get('p'),
            "recall_per_class": metrics.get('r'),
            "f1_per_class": metrics.get('f1'),
            
            # Inference latency metrics
            "inference_latency_ms": metrics.get('inference_latency_ms'),
            "preprocess_latency_ms": metrics.get('preprocess_latency_ms'),
            "postprocess_latency_ms": metrics.get('postprocess_latency_ms'),
            "total_latency_ms": metrics.get('total_latency_ms'),
            
            # System info
            "device": metrics.get('device'),
            "ram_gb": metrics.get('ram_gb'),
            "gpu_mem_gb": metrics.get('gpu_mem_gb'),
            "gpu_mem_reserved_gb": metrics.get('gpu_mem_reserved_gb'),
            
            # Learning rate
            "lr": metrics.get('lr'),
            
            # Validation time
            "val_time_s": metrics.get('val_time_s'),
        }
        
        # Remove None values to keep file clean
        epoch_data = {k: v for k, v in epoch_data.items() if v is not None}
        
        # Convert non-JSON-serializable types (numpy arrays, tensors, methods)
        def to_serializable(v):
            import numpy as np
            import torch
            if callable(v) and not isinstance(v, (int, float, str, bool)):
                return None  # skip methods/callables
            if isinstance(v, torch.Tensor):
                return v.tolist()
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v
        
        epoch_data = {k: to_serializable(v) for k, v in epoch_data.items()}
        epoch_data = {k: v for k, v in epoch_data.items() if v is not None}
        
        try:
            with open(extended_metrics_file, "a") as f:
                f.write(json.dumps(epoch_data) + "\n")
            self.log(f"Extended metrics saved: {len(epoch_data)} fields", "DEBUG")
        except Exception as e:
            self.log(f"Failed to save extended metrics: {e}", "WARNING")
    
    def save_model(self):
        """Save model checkpoint with enhanced metadata."""
        # Track completed epochs for ETA calculation
        self._epoch_completed += 1

        # Record gradients if enabled
        if self.record_gradients and self.epoch % self.gradient_interval == 0:
            self._record_gradients()
        
        # Record weights if enabled
        if self.record_weights and self.epoch % self.weight_interval == 0:
            self._record_weights()
        
        # Call parent save
        super().save_model()
        
        # Log checkpoint save
        ckpt_file = self.wdir / f"epoch{self.epoch}.pt" if self.epoch else self.wdir / "last.pt"
        self.log(f"Checkpoint saved: {ckpt_file.name}", "INFO")
    
    def _record_gradients(self) -> None:
        """Record gradient statistics for current epoch."""
        if not self.job_id or not hasattr(self, 'model'):
            return
        
        grad_stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                grad_stats[name] = {
                    "mean": float(grad.mean()),
                    "std": float(grad.std()),
                    "min": float(grad.min()),
                    "max": float(grad.max()),
                    "norm": float(grad.norm()),
                }
        
        # Save to job directory
        job_dir = Path(job_storage._jobs_dir) / self.job_id
        grad_dir = job_dir / "gradients"
        grad_dir.mkdir(exist_ok=True)
        
        grad_file = grad_dir / f"epoch_{self.epoch}.json"
        grad_file.write_text(json.dumps(grad_stats, indent=2))
        
        self.log(f"Gradient statistics recorded: {len(grad_stats)} parameters", "DEBUG")
    
    def _record_weights(self) -> None:
        """Record weight statistics for current epoch."""
        if not self.job_id or not hasattr(self, 'model'):
            return
        
        weight_stats = {}
        for name, param in self.model.named_parameters():
            if param.data is not None:
                weight = param.data.detach()
                weight_stats[name] = {
                    "mean": float(weight.mean()),
                    "std": float(weight.std()),
                    "min": float(weight.min()),
                    "max": float(weight.max()),
                    "norm": float(weight.norm()),
                }
        
        # Save to job directory
        job_dir = JOBS_DIR / self.job_id
        weight_dir = job_dir / "weights_stats"
        weight_dir.mkdir(exist_ok=True)
        
        weight_file = weight_dir / f"epoch_{self.epoch}.json"
        weight_file.write_text(json.dumps(weight_stats, indent=2))
        
        self.log(f"Weight statistics recorded: {len(weight_stats)} parameters", "DEBUG")
    
    def plot_metrics(self):
        """Generate training plots including confusion matrix and curves."""
        # Call parent plot generation
        super().plot_metrics()
        
        # Generate additional custom plots
        self._generate_custom_plots()
    
    def _generate_custom_plots(self) -> None:
        """Generate custom plots for Model Designer."""
        if not self.job_id:
            return
        
        try:
            # Plots are generated by Ultralytics in save_dir/
            # We just log that they're available
            plots_dir = self.save_dir
            
            plot_files = [
                "confusion_matrix.png",
                "confusion_matrix_normalized.png", 
                "F1_curve.png",
                "P_curve.png",
                "R_curve.png",
                "PR_curve.png",
                "results.png",
            ]
            
            available_plots = []
            for plot_file in plot_files:
                plot_path = plots_dir / plot_file
                if plot_path.exists():
                    available_plots.append(plot_file)
            
            if available_plots:
                self.log(f"Generated plots: {', '.join(available_plots)}", "INFO")
        
        except Exception as e:
            self.log(f"Error generating custom plots: {e}", "WARNING")
    
    def on_train_batch_end(self):
        """Called after each training batch."""
        super().on_train_batch_end()
        # NaN detection is handled via add_callback('on_train_batch_end', ...) in __init__.
    
    def progress_string(self):
        """Override progress string to use custom format."""
        # Return custom progress format without tqdm
        if hasattr(self, 'epoch') and hasattr(self, 'epochs'):
            return f"Epoch {self.epoch + 1}/{self.epochs}"
        return "Training..."


# ── Top-level importable trainer (required for Ultralytics DDP) ───────────────
# Ultralytics DDP spawns a subprocess via torch.distributed.run and imports the
# trainer class by its fully-qualified module path.  Inner classes (closures)
# inside functions are not importable and cause CalledProcessError exit 1.
#
# JobCustomTrainer must be a top-level class here.  custom_params are injected
# via the class-level registry _params_registry (keyed by job_id) before
# model.train() is called, and read back inside __init__.

class JobCustomTrainer(CustomDetectionTrainer):
    """Top-level trainer class used by model.train(trainer=JobCustomTrainer).

    custom_params must be registered before calling model.train() via:
        JobCustomTrainer.set_params(custom_params)

    Uses threading.local() so parallel jobs (each in their own thread) never
    read each other's params even when both call set_params() concurrently.
    """

    _registry_lock = threading.Lock()
    _params_registry: "dict[str, dict]" = {}   # shared: job_id → params
    _thread_local = threading.local()           # per-thread: active_job_id

    @classmethod
    def set_params(cls, custom_params: dict) -> None:
        """Register custom_params for this thread's training run."""
        job_id = custom_params.get("job_id")
        with cls._registry_lock:
            cls._params_registry[job_id] = custom_params
        # Store active job_id per-thread so parallel jobs don't overwrite each other
        cls._thread_local.active_job_id = job_id

    @classmethod
    def _get_params(cls) -> dict:
        # Prefer thread-local job_id (set by the calling training thread)
        job_id = getattr(cls._thread_local, "active_job_id", None)
        with cls._registry_lock:
            if job_id and job_id in cls._params_registry:
                return cls._params_registry[job_id]
            # Fallback for DDP subprocess (different thread/process, no thread-local):
            # return the only registered params if exactly one job is running
            if len(cls._params_registry) == 1:
                return next(iter(cls._params_registry.values()))
        return {}

    @classmethod
    def cleanup_params(cls, job_id: str) -> None:
        """Remove params after training completes to avoid stale state."""
        with cls._registry_lock:
            cls._params_registry.pop(job_id, None)

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        # Read params from per-thread registry — safe for parallel jobs.
        # In DDP subprocess (fresh spawn process) the registry is empty;
        # params will be read from overrides instead (injected below).
        params = self._get_params()
        self._custom_params = params or {}

        # Inject custom_params into overrides so DDP subprocess receives them
        # via the Ultralytics-generated temp file (the only channel available).
        # CustomDetectionTrainer.__init__ pops these keys from clean_overrides.
        if self._custom_params:
            for k, v in self._custom_params.items():
                if k not in overrides:
                    overrides[k] = v

        from . import job_storage as js
        _job_id = self._custom_params.get("job_id") or overrides.get("job_id", "unknown")
        js.append_job_log(_job_id, "INFO",
            f"JobCustomTrainer.__init__ called with job_id: {_job_id}")
        super().__init__(cfg, overrides, _callbacks)
