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
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model

from . import job_storage
from .config_service import get_monitoring_config
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
        """Override validation call to track latency.

        Uses CUDA events for non-blocking timing instead of torch.cuda.synchronize()
        around every forward — the previous implementation forced a full GPU pipeline
        flush twice per batch, making val 2–5× slower than training.
        """
        import time
        import torch

        # Reset latency tracking
        self.inference_times = []
        self.preprocess_times = []
        self.postprocess_times = []
        self._pending_cuda_events = []  # list[(start_event, end_event)]

        # Store original model forward to wrap it
        import torch.nn as nn
        is_nn_model = model is not None and isinstance(model, nn.Module)
        use_cuda_events = is_nn_model and torch.cuda.is_available()
        if is_nn_model:
            original_forward = model.forward

            if use_cuda_events:
                def timed_forward(*args, **kwargs):
                    """Non-blocking GPU timing via CUDA events."""
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    start_evt.record()
                    result = original_forward(*args, **kwargs)
                    end_evt.record()
                    self._pending_cuda_events.append((start_evt, end_evt))
                    return result
            else:
                def timed_forward(*args, **kwargs):
                    """CPU timing (no CUDA available)."""
                    start = time.time()
                    result = original_forward(*args, **kwargs)
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

        # Convert CUDA events → seconds (single sync at the end, not per-batch)
        if use_cuda_events and self._pending_cuda_events:
            torch.cuda.synchronize()  # ensure all events recorded
            for start_evt, end_evt in self._pending_cuda_events:
                self.inference_times.append(start_evt.elapsed_time(end_evt) / 1000.0)
            self._pending_cuda_events = []

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
        """Track preprocessing time + emit heartbeat every 32 batches to detect val hangs."""
        import time
        start = time.time()
        result = super().preprocess(batch)
        elapsed = time.time() - start
        self.preprocess_times.append(elapsed)

        # Heartbeat: log every 32 batches OR if this single preprocess took >5s (NFS stall signal)
        bi = getattr(self, "batch_i", None) or len(self.preprocess_times) - 1
        job_id = getattr(self, "_heartbeat_job_id", None)
        total = len(self.dataloader) if self.dataloader is not None else None
        if job_id and (bi % 32 == 0 or elapsed > 5.0):
            try:
                job_storage.append_job_log(
                    job_id, "DEBUG",
                    f"[val] batch {bi}/{total or '?'} preprocess={elapsed*1000:.0f}ms",
                )
            except Exception:
                pass
        return result

    def postprocess(self, preds):
        """Track postprocessing time + log slow NMS / postprocess batches."""
        import time
        start = time.time()
        result = super().postprocess(preds)
        elapsed = time.time() - start
        self.postprocess_times.append(elapsed)

        # Warn if postprocess takes >5s (indicates NaN-in-NMS hang or extreme pred count)
        job_id = getattr(self, "_heartbeat_job_id", None)
        if job_id and elapsed > 5.0:
            bi = getattr(self, "batch_i", None) or len(self.postprocess_times) - 1
            try:
                job_storage.append_job_log(
                    job_id, "WARNING",
                    f"[val] batch {bi} SLOW postprocess={elapsed:.1f}s (possible NMS stall)",
                )
            except Exception:
                pass
        return result


class CustomDetectionTrainer(DetectionTrainer):
    """Custom trainer with enhanced monitoring for Model Designer."""

    # Scheduled Saliency-Guided Query Selection (alpha schedule)
    # alpha: 0 -> HSG_ALPHA_TARGET over HSG_ALPHA_WARMUP_EPOCHS epochs
    HSG_ALPHA_TARGET = 0.50          # max saliency weight for query selection
    HSG_ALPHA_START_EPOCH = 0        # start warming from epoch 0
    HSG_ALPHA_WARMUP_EPOCHS = 15     # linear warmup duration
    HSG_ALPHA_RESUME_RAMP_EPOCHS = 20  # ramp after resume to avoid jumps
    
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

        # In DDP subprocess Ultralytics passes trainer config as cfg dict (not overrides).
        # We must read custom params from cfg first, then overrides, then _custom_params.
        cfg_source = cfg if isinstance(cfg, dict) else {}
        custom_source = getattr(self, '_custom_params', {})

        # 1. Create a copy of overrides to modify
        clean_overrides = overrides.copy()

        # 2. Extract and remove custom params (cfg_source → overrides → _custom_params)
        import os as _os

        def _get(key, default=None):
            if key in clean_overrides:
                return clean_overrides.pop(key)
            if key in cfg_source:
                return cfg_source[key]
            return custom_source.get(key, default)

        # Fallback chain: overrides → _custom_params → MD_JOB_ID env var
        # The env var is set by ultra_trainer before model.train() and is inherited
        # by DDP child subprocesses (Ultralytics spawns them via subprocess.Popen,
        # so all os.environ entries are available in the child).
        self.job_id = _get('job_id') or _os.environ.get('MD_JOB_ID')

        # DDP rank: -1 = single GPU, 0 = main DDP rank, >0 = worker rank.
        # Only rank -1 and 0 should write progress to job_storage to avoid
        # duplicate log entries from every GPU worker.
        _local_rank = int(_os.environ.get('LOCAL_RANK', -1))
        self._is_logging_rank = _local_rank in (-1, 0)

        self.record_gradients = _get('record_gradients', False)
        self.record_weights = _get('record_weights', False)
        self.gradient_interval = _get('gradient_interval', 1)
        self.weight_interval = _get('weight_interval', 1)
        self.sample_per_class = _get('sample_per_class', 0)
        
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
        # Must include all custom params injected by JobCustomTrainer.set_params()
        INVALID_KEYS = {
            'session', 'job_id',
            'sample_per_class', 'record_gradients', 'gradient_interval',
            'record_weights', 'weight_interval',
            '_partition_configs', '_dataset_name',
        }
        
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
            
            # Cache strategy: force disk cache (True) for HSG-DETR.
            # Direct read (False) causes JPEG decode overhead every epoch.
            # Only allow 'ram' if explicitly set and RAM is sufficient.
            cache_val = cfg.get('cache')
            if cache_val is None or cache_val == 'auto' or cache_val == False:
                cfg['cache'] = True
                if self.job_id:
                    job_storage.append_job_log(
                        self.job_id, "INFO",
                        f"Cache strategy: disk (HSG-DETR forced, was {cache_val})"
                    )
        
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
        self._nonfinite_grad_steps: int = 0
        self._max_nonfinite_grad_skips: int = 8
        self._amp_initial_scale: float = 512.0
        self._amp_growth_interval: int = 2000
        self._amp_backoff_factor: float = 0.25
        self._target_sanitize_logs: int = 0
        self._last_train_batch_summary: dict[str, Any] | None = None
        self._hsg_alpha_last: float | None = None
        self._hsg_alpha_resume_base: float | None = None
        self._hsg_alpha_resume_epoch: int | None = None
        
        # HSG-DETR sparse metrics caches for extended_metrics.jsonl persistence
        self._last_hsg_metrics: dict[str, float] | None = None
        self._last_grad_norms: dict[str, float] = {}

        def _on_train_batch_end_cb(trainer):
            # Detect NaN/Inf early (every batch) before emitting rate-limited progress logs.
            trainer._check_nan_loss_items()
            trainer._on_batch_end()

        def _on_train_epoch_start_cb(trainer):
            trainer._on_train_epoch_start()

        def _on_train_epoch_end_cb(trainer):
            trainer._on_train_epoch_end()

        # Register batch-end callback (Ultralytics calls with trainer as arg)
        self.add_callback("on_train_batch_end", _on_train_batch_end_cb)
        self.add_callback("on_train_epoch_start", _on_train_epoch_start_cb)
        self.add_callback("on_train_epoch_end", _on_train_epoch_end_cb)

        # Verify job_id is set
        if self.job_id:
            job_storage.append_job_log(self.job_id, "INFO", f"CustomDetectionTrainer initialized with job_id: {self.job_id}")
        else:
            LOGGER.warning("CustomDetectionTrainer initialized WITHOUT job_id - logs will go to console!")

    def get_validator(self):
        """Return custom validator."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        v = CustomValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            sample_per_class=self.sample_per_class  # Pass as direct parameter
        )
        # Enable heartbeat logging in CustomValidator.preprocess/postprocess
        v._heartbeat_job_id = getattr(self, "job_id", None)
        return v

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess batch and guard HSG-DETR targets before RT-DETR loss."""
        batch = super().preprocess_batch(batch)
        if self._is_hsg_detr_model():
            batch = self._sanitize_hsg_detr_targets(batch)
        self._last_train_batch_summary = self._summarize_training_batch(batch)
        return batch

    def _is_hsg_detr_model(self) -> bool:
        try:
            model = unwrap_model(self.model)
            return any(m.__class__.__name__ in {"RTDETRDecoderSGB", "RTDETRDecoderV2"} for m in model.modules())
        except Exception:
            return False

    def _sanitize_hsg_detr_targets(self, batch: dict) -> dict:
        """Drop non-finite boxes and keep xywh targets in a valid numeric range."""
        bboxes = batch.get("bboxes")
        cls = batch.get("cls")
        batch_idx = batch.get("batch_idx")
        if not torch.is_tensor(bboxes) or bboxes.numel() == 0:
            return batch

        b = bboxes.float()
        finite = torch.isfinite(b).all(dim=1)
        wh = b[:, 2:4]
        valid_wh = torch.isfinite(wh).all(dim=1) & (wh > 1e-6).all(dim=1)
        keep = finite & valid_wh
        dropped = int((~keep).sum().item())
        clipped = int(((b < 0.0) | (b > 1.0)).any(dim=1).sum().item())

        if not keep.all():
            batch["bboxes"] = bboxes[keep]
            if torch.is_tensor(cls) and cls.shape[0] == keep.shape[0]:
                batch["cls"] = cls[keep]
            if torch.is_tensor(batch_idx) and batch_idx.shape[0] == keep.shape[0]:
                batch["batch_idx"] = batch_idx[keep]
            b = batch["bboxes"].float()

        if b.numel():
            b = b.clone()
            b[:, 0:2] = b[:, 0:2].clamp(0.0, 1.0)
            b[:, 2:4] = b[:, 2:4].clamp(1e-4, 1.0)
            batch["bboxes"] = b.to(dtype=bboxes.dtype)

        if self.job_id and (dropped or clipped) and self._target_sanitize_logs < 20:
            self._target_sanitize_logs += 1
            job_storage.append_job_log(
                self.job_id,
                "WARNING",
                f"HSG-DETR sanitized target boxes: dropped={dropped}, clipped={clipped}",
                {"type": "target_sanitized", "dropped": dropped, "clipped": clipped},
            )
        return batch

    def _summarize_training_batch(self, batch: dict) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        bboxes = batch.get("bboxes")
        if torch.is_tensor(bboxes):
            b = bboxes.detach().float()
            summary["bbox_count"] = int(b.shape[0])
            if b.numel():
                finite = torch.isfinite(b)
                summary["bbox_nonfinite"] = int((~finite).sum().item())
                summary["bbox_min"] = float(b[finite].min().item()) if finite.any() else None
                summary["bbox_max"] = float(b[finite].max().item()) if finite.any() else None
                wh = b[:, 2:4]
                wh_finite = torch.isfinite(wh)
                summary["wh_min"] = float(wh[wh_finite].min().item()) if wh_finite.any() else None
                summary["wh_max"] = float(wh[wh_finite].max().item()) if wh_finite.any() else None
                invalid_wh = (~torch.isfinite(wh).all(dim=1)) | (wh <= 0).any(dim=1)
                summary["invalid_wh"] = int(invalid_wh.sum().item())
                summary["out_of_range_boxes"] = int(((b < 0.0) | (b > 1.0)).any(dim=1).sum().item())
        cls = batch.get("cls")
        if torch.is_tensor(cls):
            c = cls.detach().float()
            summary["cls_count"] = int(c.numel())
            if c.numel():
                finite = torch.isfinite(c)
                summary["cls_nonfinite"] = int((~finite).sum().item())
                summary["cls_min"] = float(c[finite].min().item()) if finite.any() else None
                summary["cls_max"] = float(c[finite].max().item()) if finite.any() else None
        img = batch.get("img")
        if torch.is_tensor(img):
            summary["img_shape"] = list(img.shape)
            summary["img_finite"] = bool(torch.isfinite(img).all().item())
        return summary

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Override Ultralytics default that doubles workers for val.

        Ultralytics sets `workers = self.args.workers * 2` for val mode, which causes
        dataloader thrashing on shared / NFS storage and makes val slower than train.
        We use the same worker count for both modes to keep val fast.
        """
        if mode == "val":
            original_workers = self.args.workers
            try:
                # Multiply by 0.5 so the base impl's `*2` brings it back to original
                # (We can't directly override since build_dataloader() is called by super().)
                # Simpler: temporarily halve self.args.workers.
                self.args.workers = max(1, original_workers // 2) if original_workers > 1 else original_workers
                # Base impl does workers*2, so with workers=original//2 we get original.
                # If workers=1, base gives 2 (acceptable minimum).
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            finally:
                self.args.workers = original_workers
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

        
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
        """Override training loop to add custom logging and DDP cleanup."""
        self.log(f"Starting training for {self.epochs} epochs", "INFO")
        
        try:
            # BaseTrainer._do_train() takes no positional args in Ultralytics 8.4.x
            result = super()._do_train()
            self.log("Training loop completed", "INFO")
            return result
        except Exception as e:
            self.log(f"Training failed with error: {e}", "ERROR")
            # Cleanup orphaned DDP processes on failure
            self._cleanup_ddp_processes()
            raise
        finally:
            # Always cleanup DDP processes after training ends
            self._cleanup_ddp_processes()
    
    def _cleanup_ddp_processes(self):
        """Cleanup orphaned DDP/torchrun processes."""
        try:
            import psutil
            import os
            current_pid = os.getpid()
            killed = []
            for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
                try:
                    info = proc.info
                    cmdline = ' '.join(info.get('cmdline', [])) if info.get('cmdline') else ''
                    name = info.get('name', '')
                    if any(x in cmdline or x in name for x in ['torchrun', 'torch.distributed.run']):
                        pid = info.get('pid', 0)
                        if pid != current_pid and pid != os.getppid():
                            try:
                                proc.terminate()
                                proc.wait(timeout=1)
                                killed.append(pid)
                            except psutil.TimeoutExpired:
                                proc.kill()
                                killed.append(pid)
                            except (psutil.NoSuchProcess, PermissionError):
                                pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if killed:
                self.log(f"Cleaned up DDP processes: {killed}", "WARNING")
        except ImportError:
            pass
    
    def build_optimizer(self, model, name='auto', lr=None, momentum=0.937, decay=1e-3, iterations=1e5):
        """Build optimizer with selected-token SGB-aware param groups.

        Param groups:
          - base model:             lr=lr0,        wd=decay
          - SGB sparse projections: lr=lr0 x 2.0,  wd=decay
          - SGB gamma:              lr=lr0 x 5.0,  wd=0
          - norm / bias:            lr=lr0,        wd=0
          - decoder:                lr=lr0 x 1.5,  wd=decay
        """
        lr0 = lr if lr is not None else self.args.lr0
        wd = decay

        # Map parameters by owning module instance so grouping does not depend
        # on fragile string paths emitted by Ultralytics parse_model.
        sgb_roles: dict[int, str] = {}
        decoder_ids: set[int] = set()

        for module in model.modules():
            cls_name = module.__class__.__name__
            if cls_name in {"SGTokenBlock", "SGTokenBlockV2"}:
                for local_name, param in module.named_parameters(recurse=True):
                    if not param.requires_grad:
                        continue
                    if local_name == "gamma":
                        sgb_roles[id(param)] = "sgb_gamma"
                    elif local_name.startswith((
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "out_proj",
                        "se_fc",
                    )):
                        sgb_roles[id(param)] = "sgb_sparse"
                    else:
                        sgb_roles[id(param)] = "norm_bias"
            elif cls_name in {"RTDETRDecoderSGB", "RTDETRDecoderV2"}:
                for _, param in module.named_parameters(recurse=True):
                    if param.requires_grad:
                        decoder_ids.add(id(param))

        # ── Partition parameters ──────────────────────────────────────────
        pg_base, pg_sgb_sparse, pg_sgb_gamma = [], [], []
        pg_norm_bias, pg_decoder = [], []
        assigned: set[int] = set()

        for n, p in model.named_parameters():
            pid = id(p)
            if not p.requires_grad or pid in assigned:
                continue

            role = sgb_roles.get(pid)
            if role == "sgb_gamma":
                pg_sgb_gamma.append(p)
                assigned.add(pid)
                continue
            if role == "sgb_sparse":
                pg_sgb_sparse.append(p)
                assigned.add(pid)
                continue
            if role == "norm_bias":
                pg_norm_bias.append(p)
                assigned.add(pid)
                continue

            if pid in decoder_ids or 'decoder' in n.lower():
                pg_decoder.append(p)
                assigned.add(pid)
                continue

            # Norm / bias (weight_decay = 0)
            if 'norm' in n.lower() or 'bias' in n.lower() or n.endswith('.bias'):
                pg_norm_bias.append(p)
                assigned.add(pid)
                continue

            # Everything else → base
            pg_base.append(p)
            assigned.add(pid)

        groups = []
        for name, params, lr_mult, use_wd in [
            ('base',       pg_base,       1.0, True),
            ('sgb_sparse', pg_sgb_sparse, 2.0, True),
            ('sgb_gamma',  pg_sgb_gamma,  5.0, False),
            ('norm_bias',  pg_norm_bias,  1.0, False),
            ('decoder',    pg_decoder,    1.5, True),
        ]:
            if not params:
                continue
            g = {
                'params': params,
                'lr': lr0 * lr_mult,
                'weight_decay': wd if use_wd else 0.0,
                'name': name,
            }
            groups.append(g)

        # ── Build optimizer ────────────────────────────────────────────────
        opt_name = name if name and name != 'auto' else 'AdamW'
        if opt_name.lower() == 'adamw':
            from torch.optim import AdamW
            optimizer = AdamW(groups, lr=lr0, weight_decay=wd, betas=(0.9, 0.999))
        elif opt_name.lower() == 'adam':
            from torch.optim import Adam
            optimizer = Adam(groups, lr=lr0, weight_decay=wd, betas=(0.9, 0.999))
        elif opt_name.lower() == 'sgd':
            from torch.optim import SGD
            optimizer = SGD(groups, lr=lr0, momentum=momentum, weight_decay=wd)
        else:
            from torch.optim import AdamW
            optimizer = AdamW(groups, lr=lr0, weight_decay=wd, betas=(0.9, 0.999))

        # Log group sizes
        if self.job_id:
            sizes = {g.get('name', '?'): len(g['params']) for g in groups}
            group_cfg = {
                g.get('name', '?'): {
                    'params': len(g['params']),
                    'lr': float(g.get('lr', 0.0)),
                    'weight_decay': float(g.get('weight_decay', 0.0)),
                }
                for g in groups
            }
            job_storage.append_job_log(
                self.job_id, 'INFO',
                f'Optimizer param groups: {sizes} | lr0={lr0}, wd={wd}',
                {'type': 'optimizer_param_groups', 'groups': group_cfg}
            )

        return optimizer

    def _setup_train(self):
        """Override setup to add logging."""
        self.log("Setting up training...", "INFO")

        import threading
        import time as _time
        import sys
        import traceback

        done = threading.Event()
        start_t = _time.time()
        monitoring_config = get_monitoring_config()
        timeout_s = int(monitoring_config.get("training_setup_watchdog_timeout_s", 600))
        heartbeat_interval = int(monitoring_config.get("training_setup_heartbeat_s", 30))

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

        # Clear pretrained on resume — don't load external weights when resuming
        if getattr(self.args, 'resume', False) and getattr(self.args, 'pretrained', None):
            self.args.pretrained = None
            self.log("Cleared pretrained on resume", "DEBUG")

        threading.Thread(target=_watchdog, daemon=True, name="setup_train_watchdog").start()
        try:
            result = super()._setup_train()
        finally:
            done.set()

        # Auto-disable removed - AMP enabled by default for HSG-DETR
        self._disable_deterministic_for_hsg_detr()
        self._lower_amp_initial_scale_for_hsg_detr()
        self.log(
            f"Training setup complete - {self.train_loader.dataset.ni} train images, "
            f"{self.test_loader.dataset.ni} val images",
            "INFO",
        )
        return result

    def _disable_deterministic_for_hsg_detr(self) -> None:
        """Avoid deterministic CUDA paths that RT-DETR/HSG-DETR cannot fully support."""
        model = unwrap_model(self.model)
        has_hsg_decoder = any(m.__class__.__name__ in {"RTDETRDecoderSGB", "RTDETRDecoderV2"} for m in model.modules())
        if not has_hsg_decoder:
            return

        self.args.deterministic = False
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        self.log("HSG-DETR deterministic CUDA mode disabled for training stability", "WARNING")

    def _lower_amp_initial_scale_for_hsg_detr(self) -> None:
        """Use a smaller AMP GradScaler scale for HSG-DETR to avoid early overflows."""
        if not bool(getattr(self, "amp", False)):
            return
        try:
            model = unwrap_model(self.model)
            has_hsg_decoder = any(m.__class__.__name__ in {"RTDETRDecoderSGB", "RTDETRDecoderV2"} for m in model.modules())
            if not has_hsg_decoder:
                return
            self.scaler = self._new_grad_scaler(
                enabled=True,
                init_scale=self._amp_initial_scale,
                growth_interval=self._amp_growth_interval,
                backoff_factor=self._amp_backoff_factor,
            )
            self.log(
                "HSG-DETR AMP pre-check: "
                f"GradScaler init_scale={self._amp_initial_scale:g}, "
                f"growth_interval={self._amp_growth_interval}, "
                f"backoff_factor={self._amp_backoff_factor:g}",
                "WARNING",
            )
        except Exception as e:
            self.log(f"HSG-DETR AMP scaler pre-check failed: {e}", "WARNING")

    def _new_grad_scaler(
        self,
        enabled: bool,
        init_scale: float | None = None,
        growth_interval: int | None = None,
        backoff_factor: float | None = None,
    ):
        kwargs = {"enabled": enabled}
        if init_scale is not None:
            kwargs["init_scale"] = float(init_scale)
        if growth_interval is not None:
            kwargs["growth_interval"] = int(growth_interval)
        if backoff_factor is not None:
            kwargs["backoff_factor"] = float(backoff_factor)
        try:
            return torch.amp.GradScaler("cuda", **kwargs)
        except Exception:
            return torch.cuda.amp.GradScaler(**kwargs)

    def _load_checkpoint_state(self, ckpt):
        """Load resume state with backward-compatible EMA state_dict handling."""
        if ckpt.get("optimizer") is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as e:
                if "different number of parameter groups" in str(e):
                    self.log(
                        f"Resume optimizer state incompatible (architecture changed). "
                        f"Starting optimizer fresh from epoch {self.start_epoch}.",
                        "WARNING",
                    )
                else:
                    raise
        if ckpt.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except RuntimeError:
                self.log("Skipping empty scaler state (AMP was disabled)", "DEBUG")

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
    
    def _on_train_epoch_start(self):
        """Track epoch start time and toggle lightweight SGB debug capture."""
        import time as _time
        self._epoch_start_time = _time.time()
        self._update_hsg_detr_alpha()
        # Full selector tensors are expensive; only capture them when gradient
        # recording is explicitly enabled. Scalar SGB metrics are always stored
        # by the block itself.
        try:
            model = unwrap_model(self.model)
            capture_sparse_debug = bool(getattr(self, "record_gradients", False))
            for m in model.modules():
                if m.__class__.__name__ in {'SGTokenBlock', 'SGTokenBlockV2'} and hasattr(m, 'set_debug'):
                    m.set_debug(capture_sparse_debug, cpu=True)
        except Exception:
            pass

    def _on_train_epoch_end(self):
        """Log HSG-DETR metrics at epoch end when gradients are fresh."""
        self._log_hsg_detr_metrics()

    def _update_hsg_detr_alpha(self) -> None:
        """Warm up RTDETRDecoderSGB saliency selection without resume-time jumps."""
        try:
            model = unwrap_model(self.model)
        except Exception:
            return

        decoders = [
            m for m in model.modules()
            if m.__class__.__name__ in {"RTDETRDecoderSGB", "RTDETRDecoderV2"} and hasattr(m, "set_alpha")
        ]
        if not decoders:
            return

        epoch = int(getattr(self, "epoch", 0))
        progress = max(
            0.0,
            min(
                (epoch - self.HSG_ALPHA_START_EPOCH) / self.HSG_ALPHA_WARMUP_EPOCHS,
                1.0,
            ),
        )
        scheduled_alpha = self.HSG_ALPHA_TARGET * progress

        checkpoint_alpha = 0.0
        for decoder in decoders:
            alpha_tensor = getattr(decoder, "alpha", None)
            if isinstance(alpha_tensor, torch.Tensor):
                try:
                    checkpoint_alpha = max(
                        checkpoint_alpha,
                        float(alpha_tensor.detach().float().reshape(-1)[0]),
                    )
                except Exception:
                    pass

        # Only trigger the resume ramp on actual training resumes (start_epoch > 0).
        # Without this guard, the ramp fires at epoch 2 of every fresh training run
        # because the model's alpha (set at epoch 1) is non-zero but less than
        # the scheduled value at epoch 2 — producing a false-positive resume signal.
        is_actual_resume = int(getattr(self, "start_epoch", 0)) > 0
        if (
            is_actual_resume
            and self._hsg_alpha_resume_base is None
            and checkpoint_alpha > 0.0
            and checkpoint_alpha < scheduled_alpha
        ):
            self._hsg_alpha_resume_base = checkpoint_alpha
            self._hsg_alpha_resume_epoch = epoch

        alpha = scheduled_alpha
        if (
            self._hsg_alpha_resume_base is not None
            and self._hsg_alpha_resume_epoch is not None
            and self._hsg_alpha_resume_base < scheduled_alpha
        ):
            resume_progress = max(
                0.0,
                min(
                    (epoch - self._hsg_alpha_resume_epoch)
                    / self.HSG_ALPHA_RESUME_RAMP_EPOCHS,
                    1.0,
                ),
            )
            alpha = self._hsg_alpha_resume_base + (
                scheduled_alpha - self._hsg_alpha_resume_base
            ) * resume_progress

        alpha = max(0.0, min(float(alpha), float(self.HSG_ALPHA_TARGET)))

        for decoder in decoders:
            decoder.set_alpha(alpha)

        self._sync_hsg_detr_alpha_to_ema()

        self._hsg_alpha_last = alpha
        if epoch <= self.HSG_ALPHA_START_EPOCH:
            phase_text = (
                f"hold_until_epoch={self.HSG_ALPHA_START_EPOCH}, "
                "alpha updates start after this epoch"
            )
        elif progress < 1.0:
            phase_text = (
                f"warming_up over {self.HSG_ALPHA_WARMUP_EPOCHS} epochs "
                f"from epoch {self.HSG_ALPHA_START_EPOCH}"
            )
        else:
            phase_text = "target_reached"

        resume_text = ""
        if (
            self._hsg_alpha_resume_base is not None
            and self._hsg_alpha_resume_epoch is not None
            and self._hsg_alpha_resume_base < scheduled_alpha
        ):
            resume_text = (
                f", resume_base={self._hsg_alpha_resume_base:.6f}, "
                f"resume_epoch={self._hsg_alpha_resume_epoch}"
            )
        self.log(
            f"HSG-DETR query saliency alpha set to {alpha:.6f} "
            f"(target={self.HSG_ALPHA_TARGET:.6f}, "
            f"scheduled={scheduled_alpha:.6f}, progress={progress:.3f}"
            f", phase={phase_text}{resume_text})",
            "INFO",
        )

    def _sync_hsg_detr_alpha_to_ema(self) -> None:
        """Keep the EMA decoder's scheduled alpha identical to the live model."""
        ema_model = getattr(getattr(self, "ema", None), "ema", None)
        if ema_model is None:
            return
        try:
            model = unwrap_model(self.model)
            ema_model = unwrap_model(ema_model)
        except Exception:
            return

        model_decoders = [
            m for m in model.modules()
            if m.__class__.__name__ in {"RTDETRDecoderSGB", "RTDETRDecoderV2"} and hasattr(m, "alpha")
        ]
        ema_decoders = [
            m for m in ema_model.modules()
            if m.__class__.__name__ in {"RTDETRDecoderSGB", "RTDETRDecoderV2"} and hasattr(m, "set_alpha")
        ]
        for src, dst in zip(model_decoders, ema_decoders):
            try:
                dst.set_alpha(float(src.alpha.detach().reshape(-1)[0]))
            except Exception:
                pass

    def _log_hsg_detr_metrics(self) -> None:
        """Log selected-token SGB metrics."""
        if not self.job_id:
            return
        try:
            model = unwrap_model(self.model)
        except Exception:
            return

        sgb_blocks = [
            m for m in model.modules()
            if m.__class__.__name__ in {'SGTokenBlock', 'SGTokenBlockV2'}
        ]
        decoder_modules = [m for m in model.modules() if m.__class__.__name__ in {'RTDETRDecoderSGB', 'RTDETRDecoderV2'}]

        metrics: dict[str, float] = {}

        # ── Selected-token SGB metrics ───────────────────────────────────
        # HSG-DETR YAML emits SGB blocks in P5 -> P4 -> P3 order. Tag by
        # spatial token count so Job Detail and analysis do not invert levels.
        def _sgb_level(block, fallback_idx: int) -> str:
            N = getattr(block, 'last_N', None)
            if N is not None:
                try:
                    n_val = int(N)
                    known = sorted(
                        {int(getattr(b, 'last_N', 0) or 0) for b in sgb_blocks},
                        reverse=True,
                    )
                    known = [v for v in known if v > 0]
                    if n_val in known:
                        rank = known.index(n_val)
                        if rank == 0:
                            return 'P3'
                        if rank == 1:
                            return 'P4'
                        if rank == 2:
                            return 'P5'
                except Exception:
                    pass
            return ('P5', 'P4', 'P3')[fallback_idx] if fallback_idx < 3 else f'P{fallback_idx + 3}'

        for i, blk in enumerate(sgb_blocks):
            tag = f'sgb/{_sgb_level(blk, i)}'
            metrics[f'{tag}_ratio'] = float(getattr(blk, 'ratio', 0))
            N = getattr(blk, 'last_N', None)
            k = getattr(blk, 'last_k', None)
            if N is not None:
                metrics[f'{tag}_N'] = float(N)
            if k is not None:
                metrics[f'{tag}_k'] = float(k)
            if N is not None and k is not None and N > 0:
                metrics[f'{tag}_k_over_N'] = float(k) / float(N)

            metrics[f'{tag}_reference_guided'] = 0.0

            # Handle different attribute names between v1 (SGTokenBlock) and v2 (SGTokenBlockV2)
            is_v2 = blk.__class__.__name__ == 'SGTokenBlockV2'

            if is_v2:
                # V2 attributes
                last_gate = getattr(blk, 'last_gate', None)
                if last_gate is not None:
                    metrics[f'{tag}_gamma_abs_mean'] = float(last_gate)
                last_score_std = getattr(blk, 'last_score_std', None)
                if last_score_std is not None:
                    metrics[f'{tag}_score_std'] = float(last_score_std)
            else:
                # V1/Legacy attributes
                selected_ratio = getattr(blk, 'last_selected_ratio', None)
                if selected_ratio is not None:
                    metrics[f'{tag}_selected_ratio'] = float(selected_ratio)
                gamma_raw_abs = getattr(blk, 'last_gamma_raw_abs_mean', None)
                if gamma_raw_abs is not None:
                    metrics[f'{tag}_gamma_raw_abs_mean'] = float(gamma_raw_abs)
                gamma_abs = getattr(blk, 'last_gamma_abs_mean', None)
                if gamma_abs is not None:
                    metrics[f'{tag}_gamma_abs_mean'] = float(gamma_abs)
                gamma_floor = getattr(blk, 'last_gamma_floor', None)
                if gamma_floor is not None:
                    metrics[f'{tag}_gamma_floor'] = float(gamma_floor)
                delta_selected = getattr(blk, 'last_delta_norm_selected', None)
                if delta_selected is not None:
                    metrics[f'{tag}_delta_norm_selected'] = float(delta_selected)
                delta_nonselected = getattr(blk, 'last_delta_norm_nonselected', None)
                if delta_nonselected is not None:
                    metrics[f'{tag}_delta_norm_nonselected'] = float(delta_nonselected)
                delta_scaled = getattr(blk, 'last_delta_scaled_norm_selected', None)
                if delta_scaled is not None:
                    metrics[f'{tag}_delta_scaled_norm_selected'] = float(delta_scaled)
                selected_grad = getattr(blk, 'last_selected_grad_norm', None)
                if selected_grad is not None:
                    metrics[f'{tag}_selected_grad_norm'] = float(selected_grad)
                nonselected_sparse_grad = getattr(blk, 'last_nonselected_sparse_grad', None)
                if nonselected_sparse_grad is not None:
                    metrics[f'{tag}_nonselected_sparse_grad'] = float(nonselected_sparse_grad)
                finite_guard_count = getattr(blk, 'last_finite_guard_count', None)
                if finite_guard_count is not None:
                    metrics[f'{tag}_finite_guard_count'] = float(finite_guard_count)
                score_std = getattr(blk, 'last_score_std', None)
                if score_std is not None:
                    metrics[f'{tag}_score_std'] = float(score_std)

            saliency = getattr(blk, 'last_saliency', None)
            if saliency is not None and saliency.numel():
                metrics[f'{tag}_saliency_mean'] = float(saliency.detach().float().mean())

        # ── Decoder metrics ──────────────────────────────────────────────
        for dec in decoder_modules:
            alpha = getattr(dec, 'alpha', None)
            if alpha is not None:
                metrics['decoder/alpha'] = float(alpha.detach())
            metrics['decoder/num_queries'] = int(getattr(dec, 'num_queries', 0))

        # ── Gradient norms (from cached _last_grad_norms) ───────────────
        # Cached in optimizer_step() before zero_grad clears gradients
        for k, v in getattr(self, '_last_grad_norms', {}).items():
            metrics[f'grad/{k}_norm'] = v

        # ── NaN/Inf flags ────────────────────────────────────────────────
        has_nan = False
        has_inf = False
        for n, p in model.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    has_nan = True
                if torch.isinf(p.grad).any():
                    has_inf = True
        metrics['grad/has_nan'] = float(has_nan)
        metrics['grad/has_inf'] = float(has_inf)

        if metrics:
            epoch = getattr(self, 'epoch', 0) + 1
            job_storage.append_job_log(
                self.job_id, 'METRICS',
                f'HSG-DETR metrics epoch {epoch}',
                {'type': 'hsg_detr_metrics', 'epoch': epoch, **metrics}
            )
            
            # Cache for _save_extended_metrics to include in epoch record
            # (prevents duplicate writes to extended_metrics.jsonl)
            self._last_hsg_metrics = metrics

    def optimizer_step(self):
        """Optimizer step with pre-EMA finite guards for BN buffers and gradients."""
        # Unscale before finite checks. With AMP, scaled gradients may overflow
        # transiently; diagnostics after unscale are much more actionable.
        amp_scale_before = None
        try:
            amp_scale_before = float(self.scaler.get_scale())
        except Exception:
            pass
        try:
            self.scaler.unscale_(self.optimizer)
        except Exception as e:
            self._handle_nonfinite_gradients(f"GradScaler unscale failed: {e}", amp_scale_before=amp_scale_before)
            return

        # Pre-check: if any unscaled gradient is NaN/Inf, skip this step entirely
        has_nan_grad = False
        for p in self.model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    has_nan_grad = True
                    break
        if has_nan_grad:
            self._handle_nonfinite_gradients(
                "Pre-check: NaN/Inf detected in unscaled gradients",
                amp_scale_before=amp_scale_before,
            )
            self.optimizer.zero_grad(set_to_none=True)
            return
        
        self._assert_batchnorm_buffers_finite("before optimizer step")
        try:
            clip_max_norm = 5.0
            try:
                model_unwrapped = unwrap_model(self.model)
                if any(m.__class__.__name__ == "RTDETRDecoderV2" for m in model_unwrapped.modules()):
                    # V2's Look-Forward-Twice decoder keeps gradients chained
                    # across refinement layers, so it needs a tighter clip.
                    clip_max_norm = 0.1
            except Exception:
                pass

            # Keep clipping conservative for the sparse CNN/DETR hybrid.
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=clip_max_norm,
                error_if_nonfinite=True,
            )
        except TypeError:
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_max_norm)
            if not torch.isfinite(total_norm):
                self._handle_nonfinite_gradients(
                    f"NaN/Inf gradient norm detected: {float(total_norm)}",
                    amp_scale_before=amp_scale_before,
                )
                return
        except RuntimeError as e:
            self._handle_nonfinite_gradients(
                f"NaN/Inf gradient norm detected: {e}",
                amp_scale_before=amp_scale_before,
            )
            return

        self.scaler.step(self.optimizer)
        self.scaler.update()
        # Cache grad norms BEFORE zero_grad clears them (for _log_hsg_detr_metrics)
        self._cache_last_step_grad_norms()
        self.optimizer.zero_grad()
        self._nonfinite_grad_steps = 0
        self._assert_batchnorm_buffers_finite("before EMA update")
        if self.ema:
            self.ema.update(self.model)
            self._sync_hsg_detr_alpha_to_ema()

    def _handle_nonfinite_gradients(self, reason: str, amp_scale_before: float | None = None) -> None:
        """Log offending gradients and skip a bounded number of bad optimizer steps."""
        self._nonfinite_grad_steps += 1
        diagnostics = self._collect_nonfinite_grad_diagnostics(limit=24)
        epoch = int(getattr(self, 'epoch', -1)) + 1 if hasattr(self, 'epoch') else None
        batch_i = getattr(self, 'batch_i', None)
        amp_scale = None
        try:
            amp_scale = float(self.scaler.get_scale())
        except Exception:
            pass
        amp_scale_after_update = None
        try:
            self.scaler.update()
            amp_scale_after_update = float(self.scaler.get_scale())
        except Exception:
            pass
        payload = {
            "type": "nonfinite_gradients",
            "skip_count": self._nonfinite_grad_steps,
            "max_skip_count": self._max_nonfinite_grad_skips,
            "epoch": epoch,
            "batch_i": int(batch_i) if isinstance(batch_i, (int, np.integer)) else batch_i,
            "amp_enabled": bool(getattr(getattr(self, "args", None), "amp", False)),
            "amp_scale_before_unscale": amp_scale_before,
            "amp_scale": amp_scale,
            "amp_scale_after_update": amp_scale_after_update,
            "amp_kept_enabled": bool(getattr(self, "amp", False)),
            "loss_items": self._safe_loss_items(),
            "batch_summary": getattr(self, "_last_train_batch_summary", None),
            "diagnostics": diagnostics,
        }
        diagnostics_path = self._write_nonfinite_diagnostics_file(payload)
        if diagnostics_path:
            payload["diagnostics_file"] = diagnostics_path
        if self.job_id:
            job_storage.append_job_log(
                self.job_id,
                "ERROR",
                reason,
                payload,
            )

        self.optimizer.zero_grad(set_to_none=True)

        if self._nonfinite_grad_steps <= self._max_nonfinite_grad_skips:
            if self.job_id:
                job_storage.append_job_log(
                    self.job_id,
                    "WARNING",
                    "Skipped optimizer step due to non-finite gradients "
                    f"({self._nonfinite_grad_steps}/{self._max_nonfinite_grad_skips}); "
                    "AMP remains enabled and GradScaler was backed off",
                )
            return

        raise NaNLossError(reason)

    def _cache_last_step_grad_norms(self) -> None:
        """Cache gradient norms before optimizer.zero_grad clears them.
        
        Called in optimizer_step() before zero_grad.
        These cached values are used by _log_hsg_detr_metrics at epoch end.
        """
        try:
            from ultralytics.utils.torch_utils import unwrap_model
            model = unwrap_model(self.model)
        except Exception:
            return

        sgb_roles: dict[int, str] = {}
        decoder_ids: set[int] = set()
        region_roles: dict[int, str] = {}

        # HSG-DETR parsed modules are named model.0, model.1, ... rather than
        # backbone.* / neck.*. Use YAML layer boundaries for persistent debug
        # metrics so Job Detail can show backbone and neck gradient health.
        try:
            backbone_count = len(getattr(model, "yaml", {}).get("backbone", []))
        except Exception:
            backbone_count = 0
        parsed_layers = getattr(model, "model", None)
        if isinstance(parsed_layers, torch.nn.Sequential) or isinstance(parsed_layers, torch.nn.ModuleList):
            for layer_idx, module in enumerate(parsed_layers):
                cls_name = module.__class__.__name__
                if cls_name in {"RTDETRDecoderSGB", "RTDETRDecoderV2"}:
                    region = "decoder"
                elif backbone_count and layer_idx < backbone_count:
                    region = "backbone"
                else:
                    region = "neck"
                for _, param in module.named_parameters(recurse=True):
                    region_roles.setdefault(id(param), region)

        for module in model.modules():
            cls_name = module.__class__.__name__
            if cls_name in {"SGTokenBlock", "SGTokenBlockV2"}:
                for local_name, param in module.named_parameters(recurse=True):
                    if local_name == "gamma":
                        sgb_roles[id(param)] = "sgb_gamma"
                    elif local_name.startswith((
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "out_proj",
                        "se_fc",
                    )):
                        sgb_roles[id(param)] = "sgb_sparse"
                    elif local_name.startswith("norm"):
                        sgb_roles[id(param)] = "sgb_norm"
                    else:
                        sgb_roles[id(param)] = "sgb"
            elif cls_name in {"RTDETRDecoderSGB", "RTDETRDecoderV2"}:
                for _, param in module.named_parameters(recurse=True):
                    decoder_ids.add(id(param))

        grad_norms: dict[str, float] = {}
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            gn = float(p.grad.norm())
            role = sgb_roles.get(id(p))
            if role:
                grad_norms[role] = max(grad_norms.get(role, 0), gn)
            elif id(p) in decoder_ids or 'decoder' in n.lower():
                grad_norms['decoder'] = max(grad_norms.get('decoder', 0), gn)
            elif region_roles.get(id(p)) == 'backbone':
                grad_norms['backbone'] = max(grad_norms.get('backbone', 0), gn)
            elif region_roles.get(id(p)) == 'neck':
                grad_norms['neck'] = max(grad_norms.get('neck', 0), gn)
            elif 'backbone' in n.lower():
                grad_norms['backbone'] = max(grad_norms.get('backbone', 0), gn)
            elif 'neck' in n.lower() or 'head' in n.lower():
                grad_norms['neck'] = max(grad_norms.get('neck', 0), gn)
        
        self._last_grad_norms = grad_norms

    def _collect_nonfinite_grad_diagnostics(self, limit: int = 12) -> list[dict[str, Any]]:
        """Summarize the first parameters whose gradients contain NaN/Inf."""
        issues: list[dict[str, Any]] = []
        optimizer_groups = self._optimizer_group_lookup()
        module_lookup = dict(self.model.named_modules())
        for name, param in self.model.named_parameters():
            grad = param.grad
            if grad is None or not torch.is_tensor(grad):
                continue
            grad_summary = self._tensor_nonfinite_summary(grad.detach())
            if grad_summary is None:
                continue
            summary: dict[str, Any] = {
                "name": name,
                "grad": grad_summary,
            }
            module_name = name.rsplit(".", 1)[0] if "." in name else ""
            module = module_lookup.get(module_name)
            summary["module"] = module_name
            summary["module_type"] = module.__class__.__name__ if module is not None else "<root>"
            group = optimizer_groups.get(id(param))
            if group:
                summary["optimizer_group"] = group
            if torch.is_tensor(param):
                pdata = param.detach()
                summary["param_norm"] = self._safe_tensor_norm(pdata)
                summary["param_abs_max"] = self._safe_tensor_abs_max(pdata)
                psummary = self._tensor_nonfinite_summary(pdata)
                if psummary is not None:
                    summary["param_bad"] = {
                        "bad": psummary["bad"],
                        "nan": psummary["nan"],
                        "posinf": psummary["posinf"],
                        "neginf": psummary["neginf"],
                    }
            issues.append(summary)
            if len(issues) >= limit:
                break
        return issues

    def _optimizer_group_lookup(self) -> dict[int, dict[str, Any]]:
        lookup: dict[int, dict[str, Any]] = {}
        opt = getattr(self, "optimizer", None)
        if opt is None:
            return lookup
        for idx, group in enumerate(getattr(opt, "param_groups", [])):
            info = {
                "index": idx,
                "name": group.get("name", f"group_{idx}"),
                "lr": float(group.get("lr", 0.0)),
                "weight_decay": float(group.get("weight_decay", 0.0)),
            }
            for param in group.get("params", []):
                lookup[id(param)] = info
        return lookup

    def _safe_loss_items(self) -> Any:
        li = getattr(self, 'loss_items', None)
        if li is None:
            return None
        try:
            t = li.detach() if isinstance(li, torch.Tensor) else torch.as_tensor(li)
            return t.float().cpu().tolist()
        except Exception:
            return str(li)

    def _safe_tensor_norm(self, t: torch.Tensor) -> float | None:
        try:
            finite = torch.isfinite(t)
            if not finite.any():
                return None
            return float(t[finite].detach().float().norm().item())
        except Exception:
            return None

    def _safe_tensor_abs_max(self, t: torch.Tensor) -> float | None:
        try:
            finite = torch.isfinite(t)
            if not finite.any():
                return None
            return float(t[finite].detach().float().abs().max().item())
        except Exception:
            return None

    def _write_nonfinite_diagnostics_file(self, payload: dict[str, Any]) -> str | None:
        if not self.job_id:
            return None
        try:
            epoch = payload.get("epoch", "unknown")
            batch_i = payload.get("batch_i", "unknown")
            out_dir = JOBS_DIR / str(self.job_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"nonfinite_gradients_e{epoch}_b{batch_i}_s{self._nonfinite_grad_steps}.json"
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            return str(path)
        except Exception:
            return None

    def _assert_batchnorm_buffers_finite(self, phase: str) -> None:
        """Fail before EMA/save if BatchNorm running buffers become non-finite."""
        issues = []
        for name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue
            for buffer_name in ("running_mean", "running_var"):
                tensor = getattr(module, buffer_name, None)
                if tensor is not None and not torch.isfinite(tensor).all():
                    finite = torch.isfinite(tensor)
                    bad = int((~finite).sum().item())
                    issues.append(f"{name}.{buffer_name}: bad={bad}/{tensor.numel()}")
                    break
            if len(issues) >= 8:
                break

        if not issues:
            return

        message = f"NaN/Inf BatchNorm buffer detected {phase}: " + "; ".join(issues)
        if self.job_id:
            job_storage.append_job_log(
                self.job_id,
                "ERROR",
                message,
                {"type": "nonfinite_bn_buffer", "phase": phase, "issues": issues},
            )
        raise NaNLossError(message)

    def _on_batch_end(self):
        """Called after every batch via callback. Tracks ni internally."""
        import time as _time
        import torch
        import psutil

        # DDP: only rank -1 (single GPU) or rank 0 (main DDP rank) should write
        # progress logs. Worker ranks (LOCAL_RANK > 0) skip logging entirely.
        if not getattr(self, '_is_logging_rank', True):
            return

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
            loss_value = None
            if torch.is_tensor(getattr(self, "loss", None)):
                try:
                    loss_value = float(self.loss.detach().float().item())
                except Exception:
                    loss_value = None
            try:
                amp_scale = float(self.scaler.get_scale())
            except Exception:
                amp_scale = None
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
                        "loss": loss_value,
                        "loss_items": t.detach().float().cpu().tolist(),
                        "amp_scale": amp_scale,
                        "batch_summary": getattr(self, "_last_train_batch_summary", None),
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
    
    def _setup_ddp(self, world_size=None, *args, **kwargs):
        """Override DDP setup to add logging — forward-compatible signature."""
        self.log(f"Setting up DDP with world_size={world_size}", "INFO")
        if world_size is not None:
            return super()._setup_ddp(world_size, *args, **kwargs)
        return super()._setup_ddp(*args, **kwargs)
    
    def validate(self):
        """Run validation and collect extended metrics.
        
        Returns:
            Validation metrics dict
        """
        import psutil
        import torch

        _logging = getattr(self, '_is_logging_rank', True)

        # Log validation start with structured PROGRESS data (rank 0 / single-GPU only)
        if _logging:
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
        
        # Clean GPU memory before validation (free training tensors/graphs)
        if torch.cuda.is_available():
            # Clear cached memory and run garbage collection
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            if hasattr(self, 'loss_items') and self.loss_items is not None:
                # Move loss tensor to CPU to free GPU memory but keep for validator
                self.loss_items = self.loss_items.cpu()

        # Run parent validation (all ranks must execute this)
        val_start = time.time()
        metrics = super().validate()
        val_time = time.time() - val_start

        # Worker DDP ranks: skip all logging / extended metrics writing
        if not _logging:
            return metrics

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
            
            # Add validation losses from metrics. HSG-DETR/RT-DETR validators use
            # giou/l1 names, while YOLO validators use box/dfl names.
            extended_metrics.update(self._extract_validation_losses(metrics))
            
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
                    'val_box_loss': extended_metrics.get('val_box_loss'),
                    'val_cls_loss': extended_metrics.get('val_cls_loss'),
                    'val_dfl_loss': extended_metrics.get('val_dfl_loss'),
                    'val_giou_loss': extended_metrics.get('val_giou_loss'),
                    'val_l1_loss': extended_metrics.get('val_l1_loss'),
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

    def _extract_validation_losses(self, metrics: Any) -> dict[str, float]:
        """Normalize validation loss names for chart/history consumers."""

        def as_mapping(value: Any) -> dict[str, Any] | None:
            if isinstance(value, dict):
                return value
            results_dict = getattr(value, 'results_dict', None)
            if isinstance(results_dict, dict):
                return results_dict
            return None

        sources: list[dict[str, Any]] = []
        for source in (
            metrics,
            getattr(self, 'metrics', None),
            getattr(getattr(self, 'validator', None), 'metrics', None),
        ):
            mapping = as_mapping(source)
            if mapping:
                sources.append(mapping)

        def to_float(value: Any) -> float | None:
            if value is None:
                return None
            try:
                if isinstance(value, torch.Tensor):
                    if value.numel() == 0:
                        return None
                    return float(value.detach().float().reshape(-1)[0].cpu().item())
                if isinstance(value, np.ndarray):
                    if value.size == 0:
                        return None
                    return float(value.reshape(-1)[0])
                if isinstance(value, (list, tuple)):
                    if not value:
                        return None
                    return to_float(value[0])
                return float(value)
            except (TypeError, ValueError, OverflowError):
                return None

        def pick(*keys: str) -> float | None:
            for source in sources:
                for key in keys:
                    if key in source:
                        parsed = to_float(source.get(key))
                        if parsed is not None:
                            return round(parsed, 6)
            return None

        val_giou = pick('val/giou_loss', 'val_giou_loss', 'giou_loss', 'val/loss_giou', 'loss_giou')
        val_l1 = pick('val/l1_loss', 'val_l1_loss', 'l1_loss', 'val/loss_bbox', 'loss_bbox')
        val_box = pick(
            'val/box_loss',
            'val_box_loss',
            'box_loss',
            'val/bbox_loss',
            'val_bbox_loss',
            'bbox_loss',
        )
        val_dfl = pick('val/dfl_loss', 'val_dfl_loss', 'dfl_loss')

        losses = {
            # Keep the UI's existing chart keys stable.
            'val_box_loss': val_box if val_box is not None else val_giou,
            'val_cls_loss': pick('val/cls_loss', 'val_cls_loss', 'cls_loss'),
            'val_dfl_loss': val_dfl if val_dfl is not None else val_l1,
            # Preserve DETR-native names for debugging and future UI labels.
            'val_giou_loss': val_giou,
            'val_l1_loss': val_l1,
        }
        return {key: value for key, value in losses.items() if value is not None}
    
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
        
        # Prepare HSG-DETR sparse metrics from cache
        _hsg = getattr(self, '_last_hsg_metrics', None) or {}
        # Filter grad/ prefixed keys and strip prefix for cleaner JSON
        _grad_norms = {
            k.replace('grad/', '').replace('_norm', ''): v
            for k, v in _hsg.items()
            if k.startswith('grad/') and k.endswith('_norm')
        } or None  # None so it gets filtered if empty
        
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
            "val_giou_loss": metrics.get('val_giou_loss'),
            "val_l1_loss": metrics.get('val_l1_loss'),
            
            # HSG-DETR metrics (cached from _log_hsg_detr_metrics)
            "hsg_detr": _hsg or None,
            "gradient_norms": _grad_norms,
            
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

        self._sync_hsg_detr_alpha_to_ema()
        self._log_ema_nonfinite_diagnostics()
        
        # Call parent save
        saved = super().save_model()
        
        # Log checkpoint save
        ckpt_file = self.wdir / f"epoch{self.epoch}.pt" if self.epoch else self.wdir / "last.pt"
        if saved is False:
            self.log(f"Checkpoint save skipped: {ckpt_file.name}", "WARNING")
        else:
            self.log(f"Checkpoint saved: {ckpt_file.name}", "INFO")

    def _tensor_nonfinite_summary(self, t: torch.Tensor) -> dict[str, Any] | None:
        if not t.dtype.is_floating_point:
            return None
        finite = torch.isfinite(t)
        bad = int((~finite).sum().item())
        if bad == 0:
            return None
        nan = int(torch.isnan(t).sum().item())
        posinf = int(torch.isposinf(t).sum().item())
        neginf = int(torch.isneginf(t).sum().item())
        summary: dict[str, Any] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "bad": bad,
            "nan": nan,
            "posinf": posinf,
            "neginf": neginf,
        }
        good = t[finite]
        if good.numel():
            gf = good.detach().float()
            summary.update({
                "finite_min": float(gf.min().item()),
                "finite_max": float(gf.max().item()),
                "finite_mean": float(gf.mean().item()),
            })
        return summary

    def _log_ema_nonfinite_diagnostics(self, limit: int = 12) -> None:
        if not self.job_id or not getattr(self, "ema", None) or not getattr(self.ema, "ema", None):
            return

        try:
            ema_sd = self.ema.ema.state_dict()
            model_sd = self.model.state_dict() if hasattr(self, "model") else {}

            issues = []
            for name, tensor in ema_sd.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                summary = self._tensor_nonfinite_summary(tensor)
                if summary is None:
                    continue

                model_summary = None
                model_tensor = model_sd.get(name)
                if isinstance(model_tensor, torch.Tensor):
                    model_summary = self._tensor_nonfinite_summary(model_tensor)
                summary["model_has_nonfinite"] = model_summary is not None
                if model_summary is not None:
                    summary["model_bad"] = {
                        "bad": model_summary["bad"],
                        "nan": model_summary["nan"],
                        "posinf": model_summary["posinf"],
                        "neginf": model_summary["neginf"],
                    }
                issues.append({"name": name, **summary})
                if len(issues) >= limit:
                    break

            if not issues:
                return

            self.log(
                "EMA non-finite diagnostics before checkpoint save: "
                + json.dumps(
                    {
                        "epoch": int(getattr(self, "epoch", -1)),
                        "shown": len(issues),
                        "issues": issues,
                    },
                    ensure_ascii=False,
                ),
                "ERROR",
            )
        except Exception as e:
            self.log(f"EMA non-finite diagnostic failed: {e}", "WARNING")
    
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

from ultralytics.models.rtdetr.val import RTDETRValidator as _RTDETRValidator


class DeferredRTDETRValidator(_RTDETRValidator):
    """RT-DETR validator that defers CPU-heavy metric matching until after the val loop.

    Problem: RT-DETR outputs 300 queries/image (no NMS). DetectionValidator.update_metrics
    runs CPU match_predictions across 10 IoU thresholds × 300 preds × N_gt per image,
    stalling the GPU between every batch.

    Fix: During the val loop, stash (preds, batch) to CPU memory (img tensor dropped —
    only shape is needed). After the loop, process all stashed items in get_stats().
    GPU forward passes run back-to-back; metric matching happens once at the end.
    """

    def init_metrics(self, model):
        super().init_metrics(model)
        self._deferred = []  # list of (preds_cpu, batch_cpu)
        self.inference_times = []
        self.preprocess_times = []
        self.postprocess_times = []
        # batch["img"] is a zero-memory proxy; disable visualize to avoid indexing it.
        if hasattr(self, "args") and self.args:
            self.args.visualize = False

    def preprocess(self, batch):
        import time
        start = time.time()
        result = super().preprocess(batch)
        elapsed = time.time() - start
        self.preprocess_times.append(elapsed)
        bi = getattr(self, "batch_i", None) or len(self.preprocess_times) - 1
        job_id = getattr(self, "_heartbeat_job_id", None)
        total = len(self.dataloader) if self.dataloader is not None else None
        if job_id and (bi % 32 == 0 or elapsed > 5.0):
            try:
                job_storage.append_job_log(
                    job_id, "DEBUG",
                    f"[val] batch {bi}/{total or '?'} preprocess={elapsed*1000:.0f}ms",
                )
            except Exception:
                pass
        return result

    def postprocess(self, preds):
        import time
        start = time.time()
        result = super().postprocess(preds)
        elapsed = time.time() - start
        self.postprocess_times.append(elapsed)
        job_id = getattr(self, "_heartbeat_job_id", None)
        if job_id and elapsed > 5.0:
            bi = getattr(self, "batch_i", None) or len(self.postprocess_times) - 1
            try:
                job_storage.append_job_log(
                    job_id, "WARNING",
                    f"[val] batch {bi} SLOW postprocess={elapsed:.1f}s (possible NMS stall)",
                )
            except Exception:
                pass
        return result

    def update_metrics(self, preds, batch):
        # Stash preds + minimal batch on CPU. Skip img pixels (only shape is needed).
        preds_cpu = [
            {k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in p.items()}
            for p in preds
        ]
        img_shape = batch["img"].shape
        batch_cpu = {
            "cls": batch["cls"].detach().cpu(),
            "bboxes": batch["bboxes"].detach().cpu(),
            "batch_idx": batch["batch_idx"].detach().cpu(),
            "ori_shape": batch["ori_shape"],
            "ratio_pad": batch.get("ratio_pad"),
            "im_file": batch.get("im_file"),
                # Shape-only proxy — _prepare_batch only reads img.shape[2:].
                # A full tensor would leak 48 GB (640 batches × 16×3×640×640).
                # Use a zero-batch slice so shape=(0, C, H, W) → shape[2:] = (H, W) without memory.
            "img": batch["img"][:0].cpu(),  # zero-memory shape proxy
        }
        self._deferred.append((preds_cpu, batch_cpu))

    def get_stats(self):
        # Process all deferred metric updates now (after the GPU val loop finished).
        # Temporarily switch self.device to CPU so _prepare_batch builds tensors on CPU.
        prev_device = self.device
        self.device = torch.device("cpu")
        try:
            for preds_cpu, batch_cpu in self._deferred:
                super().update_metrics(preds_cpu, batch_cpu)
        finally:
            self.device = prev_device
        self._deferred = []
        return super().get_stats()


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
        self.job_id = _job_id

        # Set NUM_THREADS for parallel image loading
        try:
            from .config_service import get_cache_config
            _cache_cfg = get_cache_config()
            _num_threads = int(_cache_cfg.get("num_threads", 16))
            import ultralytics.data.dataset as _ultra_ds
            import ultralytics.utils as _ultra_utils
            _ultra_ds.NUM_THREADS = _num_threads
            _ultra_utils.NUM_THREADS = _num_threads
            js.append_job_log(_job_id, "INFO", f"NUM_THREADS set to {_num_threads}")
        except Exception as _nt_err:
            js.append_job_log(_job_id, "DEBUG", f"Could not set NUM_THREADS: {_nt_err}")

        js.append_job_log(_job_id, "INFO",
            f"JobCustomTrainer.__init__ called with job_id: {_job_id}")
        super().__init__(cfg, overrides, _callbacks)

    @staticmethod
    def _is_rtdetr(cfg) -> bool:
        """Detect whether a model config contains RTDETRDecoder."""
        try:
            if isinstance(cfg, (str, Path)):
                text = Path(cfg).read_text()
                return "RTDETRDecoder" in text
            if isinstance(cfg, dict):
                head = cfg.get("head", [])
                return any("RTDETRDecoder" in str(layer) for layer in head)
        except Exception:
            pass
        return False

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return model — RTDETRDetectionModel for RT-DETR, otherwise DetectionModel."""
        if self._is_rtdetr(cfg):
            from ultralytics.nn.tasks import RTDETRDetectionModel
            from ultralytics.utils import RANK as _RANK
            model = RTDETRDetectionModel(
                cfg, nc=self.data["nc"], ch=self.data["channels"],
                verbose=verbose and _RANK == -1
            )
            if weights:
                model.load(weights)
            return model
        return super().get_model(cfg, weights, verbose)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build dataset — RTDETRDataset for RT-DETR, otherwise standard YOLO dataset."""
        # DetectionTrainer stores model as string path during init; after setup_model it's an nn.Module.
        # We detect RTDETR from the original args.model path or the loaded model's yaml.
        cfg = getattr(self.model, "yaml", None) or self.args.model
        if self._is_rtdetr(cfg):
            from copy import copy as _copy
            from ultralytics.models.rtdetr.train import RTDETRDataset
            from ultralytics.utils import colorstr
            return RTDETRDataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=self.args,
                rect=False,
                cache=self.args.cache or None,
                single_cls=self.args.single_cls or False,
                prefix=colorstr(f"{mode}: "),
                classes=self.args.classes,
                data=self.data,
                fraction=self.args.fraction if mode == "train" else 1.0,
            )
        return super().build_dataset(img_path, mode, batch)

    def get_validator(self):
        """Return validator — RTDETRValidator for RT-DETR, otherwise standard DetectionValidator."""
        cfg = getattr(self.model, "yaml", None) or self.args.model
        if self._is_rtdetr(cfg):
            from copy import copy as _copy
            self.loss_names = "giou_loss", "cls_loss", "l1_loss"
            # RT-DETR emits 300 queries/image; low conf passes ~all → metric matching is O(300*N_gt)/image.
            # Raise conf to 0.05 to cut metric-matching cost ~6× with negligible mAP impact.
            val_args = _copy(self.args)
            if getattr(val_args, "conf", None) in (None, 0.001):
                val_args.conf = 0.05
            v = DeferredRTDETRValidator(self.test_loader, save_dir=self.save_dir, args=val_args)
            v._heartbeat_job_id = getattr(self, "job_id", None)
            return v
        return super().get_validator()
