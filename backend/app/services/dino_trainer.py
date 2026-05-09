"""Isolated facebookresearch/DINO trainer integration.

This runs upstream DINO in a subprocess while preserving Model Designer job
records, logs, SSE events, queue ownership, and dataset selection behavior.
"""
from __future__ import annotations

import json
import os
import psutil
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..config import DATASETS_DIR, JOBS_DIR, WEIGHTS_DIR
from ..constants import job_channel
from . import dataset_registry, dataset_yaml, event_bus, job_storage


_SCALE_TO_SPEC = {
    "vits16": {
        "arch": "vit_small",
        "patch_size": 16,
        "label": "ViT-S/16",
        "checkpoint": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
    },
    "vits8": {
        "arch": "vit_small",
        "patch_size": 8,
        "label": "ViT-S/8",
        "checkpoint": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
    },
    "vitb16": {
        "arch": "vit_base",
        "patch_size": 16,
        "label": "ViT-B/16",
        "checkpoint": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
    },
    "vitb8": {
        "arch": "vit_base",
        "patch_size": 8,
        "label": "ViT-B/8",
        "checkpoint": "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
    },
    "resnet50": {
        "arch": "resnet50",
        "patch_size": 16,
        "label": "ResNet-50",
        "checkpoint": "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
    },
}


def _publish(job_id: str, job: dict) -> None:
    event_bus.publish_sync(job_channel(job_id), {
        "type": "job_update",
        "job_id": job_id,
        "status": job.get("status"),
        "epoch": job.get("epoch", 0),
        "total_epochs": job.get("total_epochs", 0),
        "message": job.get("message", ""),
        "best_fitness": job.get("best_fitness"),
    })


def _set_job(job_id: str, **updates: Any) -> dict | None:
    job = job_storage.load_job(job_id)
    if not job:
        return None
    job.update(updates)
    job_storage.save_job(job)
    _publish(job_id, job)
    return job


def _read_data_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_split_paths(data: dict[str, Any], split: str) -> list[Path]:
    root = Path(str(data.get("path") or "."))
    raw = data.get(split) or data.get("train")
    if raw is None:
        return []
    if isinstance(raw, list):
        paths: list[Path] = []
        for item in raw:
            clone = dict(data)
            clone[split] = item
            paths.extend(_resolve_split_paths(clone, split))
        return paths
    raw_path = Path(str(raw))
    if raw_path.suffix == ".txt":
        txt = raw_path if raw_path.is_absolute() else root / raw_path
        return [
            Path(line.strip())
            for line in txt.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    base = raw_path if raw_path.is_absolute() else root / raw_path
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_imagefolder(
    job_id: str,
    data_yaml_path: Path,
    out_dir: Path,
    max_images: int | None = None,
) -> tuple[Path, int]:
    data = _read_data_yaml(data_yaml_path)
    image_paths = _resolve_split_paths(data, "train")
    if not image_paths:
        raise RuntimeError(f"No train images found from {data_yaml_path}")
    if max_images and max_images > 0:
        image_paths = image_paths[:max_images]

    class_dir = out_dir / "all"
    class_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for idx, image_path in enumerate(image_paths):
        suffix = image_path.suffix.lower() or ".jpg"
        name = f"{idx:08d}_{image_path.stem}{suffix}"
        while name in seen:
            name = f"{idx:08d}_{uuid.uuid4().hex[:8]}{suffix}"
        seen.add(name)
        _safe_link_or_copy(image_path.resolve(), class_dir / name)

    job_storage.append_job_log(
        job_id,
        "INFO",
        f"DINO ImageFolder export: train={len(image_paths)} images, class_folder=all",
        {"imagefolder": str(out_dir), "source_data": str(data_yaml_path)},
    )
    return out_dir, len(image_paths)


def _checkpoint_candidates(out_dir: Path) -> list[Path]:
    return [
        out_dir / "checkpoint.pth",
        *sorted(out_dir.glob("checkpoint*.pth"), key=lambda p: p.stat().st_mtime, reverse=True),
        *sorted(out_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True),
    ]


def _read_last_log_stats(out_dir: Path) -> dict[str, Any]:
    log_path = out_dir / "log.txt"
    if not log_path.exists():
        return {}
    last: dict[str, Any] = {}
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        try:
            last = json.loads(raw)
        except json.JSONDecodeError:
            continue
    return last


def _save_weight(job_id: str, out_dir: Path, total_time: float | None) -> str | None:
    checkpoint = next((p for p in _checkpoint_candidates(out_dir) if p.exists()), None)
    if checkpoint is None:
        return None
    from . import weight_storage

    job = job_storage.load_job(job_id) or {}
    cfg = job.get("config", {})
    stats = _read_last_log_stats(out_dir)
    final_loss = stats.get("train_loss")
    weight_id = uuid.uuid4().hex[:12]
    dest_dir = WEIGHTS_DIR / weight_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, dest_dir / "weight.pt")
    weight_storage.save_weight_meta(
        model_id=job.get("model_id", "arch:dino"),
        model_name=job.get("model_name", "DINO"),
        model_scale=job.get("model_scale", ""),
        job_id=job_id,
        dataset=cfg.get("dataset_name") or cfg.get("data", ""),
        epochs_trained=job.get("total_epochs", 0),
        final_accuracy=None,
        final_loss=float(final_loss) if isinstance(final_loss, (int, float)) else None,
        weight_id=weight_id,
        parent_weight_id=None,
        total_time=total_time,
        device=str(cfg.get("device", "")),
    )
    meta = weight_storage.load_weight_meta(weight_id)
    if meta:
        meta.update(
            {
                "arch_plugin": cfg.get("model_arch"),
                "model_arch": cfg.get("model_arch"),
                "source_type": "dino",
                "train_args": {
                    "model_arch": cfg.get("model_arch"),
                    "data": cfg.get("data"),
                    "epochs": cfg.get("epochs"),
                    "batch": cfg.get("batch"),
                },
            }
        )
        weight_storage._store.save(weight_id, meta)
    return weight_id


def _resolve_checkpoint(value: str) -> Path | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"false", "0", "none"}:
        return None
    p = Path(raw)
    if p.exists():
        return p
    try:
        from . import weight_storage

        candidate = weight_storage.weight_pt_path(raw)
        if candidate.exists():
            return candidate
    except Exception:
        return None
    return None


def _stage_dino_backbone_pretrained(url: str, out_path: Path) -> None:
    """Convert official DINO backbone weights into a restart checkpoint shape."""
    import torch

    raw = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unexpected DINO checkpoint type from {url}: {type(raw).__name__}")

    student = {f"module.backbone.{k}": v for k, v in raw.items()}
    teacher = {f"backbone.{k}": v for k, v in raw.items()}
    torch.save({"student": student, "teacher": teacher, "epoch": 0}, out_path)


def _run_knn_evaluation(job_id: str, root: Path, imagefolder: Path, checkpoint: Path, spec: dict, batch: int, workers: int, out_dir: Path) -> None:
    """Run k-NN evaluation on DINO checkpoint to provide validation metrics."""
    try:
        job_storage.append_job_log(job_id, "INFO", "Starting k-NN evaluation for validation metrics...")
        
        # Check if dataset has train/val split (required for k-NN evaluation)
        train_path = imagefolder / "train"
        val_path = imagefolder / "val"
        
        if not (train_path.exists() and val_path.exists()):
            job_storage.append_job_log(job_id, "INFO", "Dataset does not have train/val split with labels, skipping k-NN evaluation")
            job_storage.append_job_log(job_id, "INFO", "k-NN evaluation requires labeled dataset with train/val split")
            return
        
        # Build eval_knn.py command
        cmd = [
            sys.executable,
            "eval_knn.py",
            "--arch",
            str(spec["arch"]),
            "--patch_size",
            str(spec["patch_size"]),
            "--pretrained_weights",
            str(checkpoint),
            "--batch_size_per_gpu",
            str(min(batch, 128)),  # k-NN evaluation uses smaller batch
            "--num_workers",
            str(workers),
            "--data_path",
            str(imagefolder),
        ]
        
        env = os.environ.copy()
        backend_root = str(Path(__file__).resolve().parents[2])
        pythonpath = [str(root), backend_root]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        env.setdefault("RANK", "0")
        env.setdefault("WORLD_SIZE", "1")
        env.setdefault("LOCAL_RANK", "0")
        
        # Run k-NN evaluation
        job_storage.append_job_log(job_id, "INFO", f"Running k-NN evaluation: {' '.join(cmd)}")
        
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        knn_accuracy = None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            job_storage.append_job_log(job_id, "INFO", f"k-NN eval: {line}")
            # Parse k-NN accuracy from output
            if "k-NN classifier" in line and "%" in line:
                try:
                    # Expected format: "k-NN classifier @k=200: 75.23%"
                    accuracy_str = line.split(":")[1].strip().replace("%", "")
                    knn_accuracy = float(accuracy_str) / 100.0
                except (IndexError, ValueError):
                    pass
        
        proc.wait()
        
        if proc.returncode != 0:
            job_storage.append_job_log(job_id, "WARNING", f"k-NN evaluation failed with code {proc.returncode}")
        else:
            job_storage.append_job_log(job_id, "INFO", "k-NN evaluation completed")
            
            # Update job with k-NN accuracy
            if knn_accuracy is not None:
                job = job_storage.load_job(job_id)
                if job:
                    history = list(job.get("history") or [])
                    if history:
                        history[-1]["knn_accuracy"] = knn_accuracy
                    job["knn_accuracy"] = knn_accuracy
                    job["history"] = history
                    job_storage.save_job(job)
                    job_storage.append_job_log(job_id, "INFO", f"k-NN accuracy: {knn_accuracy:.4f}")
                    
                    # Publish update
                    event_bus.publish_sync(job_channel(job_id), {"type": "job_update", "job_id": job_id, "knn_accuracy": knn_accuracy})
        
    except Exception as e:
        job_storage.append_job_log(job_id, "WARNING", f"k-NN evaluation error: {str(e)}")


def _checkpoint_candidates(out_dir: Path) -> list[Path]:
    """Return list of checkpoint file paths to try, in priority order."""
    return [
        out_dir / "checkpoint.pth",
        *sorted(out_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True),
    ]


def _save_weight(job_id: str, out_dir: Path) -> str | None:
    """Find and save DINO checkpoint to weight storage."""
    checkpoint = next((p for p in _checkpoint_candidates(out_dir) if p.exists()), None)
    if checkpoint is None:
        return None
    
    from . import weight_storage
    
    job = job_storage.load_job(job_id) or {}
    weight_id = uuid.uuid4().hex[:12]
    dest_dir = WEIGHTS_DIR / weight_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(checkpoint, dest_dir / "weight.pt")
    
    cfg = job.get("config", {})
    weight_storage.save_weight_meta(
        model_id=job.get("model_id", ""),
        model_name=job.get("model_name", "DINO"),
        model_scale=job.get("model_scale", ""),
        job_id=job_id,
        dataset=cfg.get("dataset_name") or cfg.get("data", ""),
        epochs_trained=job.get("total_epochs", 0),
        final_accuracy=job.get("knn_accuracy"),
        checkpoint_path=str(checkpoint),
        weight_id=weight_id,
    )
    
    job_storage.append_job_log(job_id, "INFO", f"Saved weight: {weight_id}")
    return weight_id


def _append_history_from_log(job_id: str, out_dir: Path) -> None:
    log_path = out_dir / "log.txt"
    if not log_path.exists():
        return
    job = job_storage.load_job(job_id)
    if not job:
        return
    history = list(job.get("history") or [])
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        epoch = int(row.get("epoch", len(history)))
        history.append(
            {
                "epoch": epoch,
                "loss": row.get("train_loss"),
                "lr": row.get("train_lr"),
                "weight_decay": row.get("train_wd"),
                "timestamp": time.time(),
                "metrics": row,
            }
        )
    job["history"] = history
    job_storage.save_job(job)
    _publish(job_id, job)


def _append_output_log(job_id: str, line: str) -> None:
    text = line.strip()
    if not text:
        return
    job_storage.append_job_log(job_id, "INFO", text)
    m = re.search(r"Epoch:\s*\[(\d+)/(\d+)\]", text)
    if m:
        epoch = int(m.group(1))
        total = int(m.group(2))
        _set_job(job_id, epoch=epoch, total_epochs=total, message=f"DINO epoch {epoch}/{total}")


def _get_system_resources() -> dict[str, Any]:
    """Get current system resource usage."""
    try:
        mem = psutil.virtual_memory()
        ram_used_gb = mem.used / (1024**3)
        ram_total_gb = mem.total / (1024**3)
        return {
            "ram_used_gb": round(ram_used_gb, 1),
            "ram_total_gb": round(ram_total_gb, 1),
        }
    except Exception:
        return {"ram_used_gb": 0, "ram_total_gb": 0}

def _parse_dino_metrics(line: str) -> dict[str, Any] | None:
    """Parse DINO log line for metrics."""
    try:
        # Try JSON format first (DINO log.txt outputs JSON)
        data = json.loads(line.strip())
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    
    # Parse text format from stdout (e.g., "Epoch: [8/300]  [372/373]  eta: 0:00:00  loss: 4.870012 (5.043951)  lr: 0.000006 (0.000006)  wd: 0.050772 (0.050693)  time: 0.086712  data: 0.000046  max mem: 4157")
    metrics = {}
    
    # Parse epoch
    m = re.search(r"Epoch:\s*\[(\d+)/(\d+)\]", line)
    if m:
        metrics["epoch"] = int(m.group(1))
        metrics["total_epochs"] = int(m.group(2))
    
    # Parse loss (current and average)
    m = re.search(r"loss:\s*([\d.]+)\s*\(([\d.]+)\)", line)
    if m:
        metrics["train_loss"] = float(m.group(1))
        metrics["train_loss_avg"] = float(m.group(2))
    
    # Parse learning rate
    m = re.search(r"lr:\s*([\d.e-]+)\s*\(([\d.e-]+)\)", line)
    if m:
        metrics["train_lr"] = float(m.group(1))
        metrics["train_lr_avg"] = float(m.group(2))
    
    # Parse weight decay
    m = re.search(r"wd:\s*([\d.]+)\s*\(([\d.]+)\)", line)
    if m:
        metrics["weight_decay"] = float(m.group(1))
        metrics["weight_decay_avg"] = float(m.group(2))
    
    # Parse iterations (for epoch time calculation)
    m = re.search(r"\[(\d+)/(\d+)\]", line)
    if m:
        metrics["iteration"] = int(m.group(1))
        metrics["total_iterations"] = int(m.group(2))
    
    # Parse time per iteration
    m = re.search(r"time:\s*([\d.]+)", line)
    if m:
        metrics["time_per_iter"] = float(m.group(1))
    
    # Parse data loading time
    m = re.search(r"data:\s*([\d.]+)", line)
    if m:
        metrics["data_time"] = float(m.group(1))
    
    # Parse max memory
    m = re.search(r"max mem:\s*(\d+)", line)
    if m:
        metrics["max_mem_mb"] = int(m.group(1))
    
    # Parse ETA
    m = re.search(r"eta:\s*([\d:]+)", line)
    if m:
        metrics["eta"] = m.group(1)
    
    # Return metrics if we found at least epoch
    return metrics if "epoch" in metrics else None

def _update_job_metrics(job_id: str, metrics: dict[str, Any], batch_size: int, start_time: float) -> None:
    """Update job with training metrics."""
    job = job_storage.load_job(job_id)
    if not job:
        return
    
    # Get system resources
    system_res = _get_system_resources()
    
    # Parse metrics
    epoch = metrics.get("epoch", job.get("epoch", 0))
    train_loss = metrics.get("train_loss")
    train_lr = metrics.get("train_lr")
    weight_decay = metrics.get("weight_decay")
    time_per_iter = metrics.get("time_per_iter")
    data_time = metrics.get("data_time")
    max_mem_mb = metrics.get("max_mem_mb")
    eta = metrics.get("eta")
    
    # Calculate derived metrics
    elapsed = time.time() - start_time
    total_minutes = elapsed / 60
    
    # Calculate speed (img/s) from batch size and time per iteration
    speed_img_s = 0
    if time_per_iter and time_per_iter > 0:
        speed_img_s = batch_size / time_per_iter
    
    # Calculate epoch time (if we have iterations per epoch)
    epoch_time = None
    if time_per_iter:
        # Use parsed iterations if available, otherwise default
        iterations = metrics.get("total_iterations", 373)  # Default from DINO log
        epoch_time = time_per_iter * iterations
    
    # Update average epoch time
    avg_epoch_time = job.get("avg_epoch_time")
    if epoch_time:
        if avg_epoch_time is None:
            avg_epoch_time = epoch_time
        else:
            # Running average
            avg_epoch_time = (avg_epoch_time * 0.9) + (epoch_time * 0.1)
    
    updates = {
        "epoch": epoch,
        "ram_used_gb": system_res["ram_used_gb"],
        "ram_total_gb": system_res["ram_total_gb"],
        "total_minutes": round(total_minutes, 1),
    }
    if train_loss is not None:
        updates["loss"] = float(train_loss)
    if train_lr is not None:
        updates["lr"] = float(train_lr)
    if weight_decay is not None:
        updates["weight_decay"] = float(weight_decay)
    if time_per_iter is not None:
        updates["time_per_iter"] = float(time_per_iter)
    if data_time is not None:
        updates["data_time"] = float(data_time)
    if max_mem_mb is not None:
        updates["max_mem_mb"] = int(max_mem_mb)
    if eta is not None:
        updates["eta"] = eta
    if speed_img_s > 0:
        updates["speed_img_s"] = round(speed_img_s)
    if epoch_time is not None:
        updates["epoch_time"] = round(epoch_time)
    if avg_epoch_time is not None:
        updates["avg_epoch_time"] = round(avg_epoch_time)
    
    # Update history with metrics matching EpochMetrics interface (frontend compatibility)
    # Map DINO metrics to Ultralytics-style field names
    history = list(job.get("history") or [])
    history.append({
        "epoch": epoch,
        # Loss: map DINO train_loss → Ultralytics box_loss (self-supervised loss)
        "box_loss": train_loss,
        "cls_loss": None,  # DINO doesn't have classification loss
        "dfl_loss": None,  # DINO doesn't have distribution focal loss
        # Learning rate
        "lr": train_lr,
        # Epoch time
        "epoch_time": epoch_time,
        # GPU memory: map max_mem_mb → gpu_memory_mb
        "gpu_memory_mb": max_mem_mb,
        # Validation metrics: DINO is self-supervised, no mAP/Precision/Recall
        "precision": None,
        "recall": None,
        "mAP50": None,
        "mAP50_95": None,
        "fitness": None,
        # Additional DINO-specific metrics (frontend will ignore if not needed)
        "weight_decay": weight_decay,
        "time_per_iter": time_per_iter,
        "data_time": data_time,
        "max_mem_mb": max_mem_mb,
        "eta": eta,
        "ram_used_gb": system_res["ram_used_gb"],
        "ram_total_gb": system_res["ram_total_gb"],
        "speed_img_s": speed_img_s if speed_img_s > 0 else None,
        "avg_epoch_time": avg_epoch_time,
        "timestamp": time.time(),
        "metrics": metrics,
    })
    updates["history"] = history
    
    _set_job(job_id, **updates)

def _cleanup_old_checkpoints(out_dir: Path, keep_last: int = 3) -> None:
    """Keep only the last N checkpoints to prevent disk filling."""
    checkpoints = sorted(
        out_dir.glob("checkpoint*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    # Keep the most recent ones
    for ckpt in checkpoints[keep_last:]:
        try:
            ckpt.unlink()
        except Exception:
            pass


def run_worker(payload: dict[str, Any]) -> None:
    """Run one DINO training job inside the existing training child."""
    from dino.installer import ensure_installed

    job_id = str(payload["job_id"])
    config = dict(payload.get("config") or {})
    partition_configs = list(payload.get("partition_configs") or [])
    model_scale = str(payload.get("model_scale") or "").lower() or "vits16"
    if model_scale not in _SCALE_TO_SPEC:
        model_arch = str(config.get("model_arch") or "")
        prefix = "dino_"
        if model_arch.startswith(prefix) and model_arch[len(prefix):] in _SCALE_TO_SPEC:
            model_scale = model_arch[len(prefix):]
        else:
            model_scale = "vits16"

    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _set_job(
        job_id,
        status="running",
        started_at=datetime.utcnow().isoformat() + "Z",
        message="Preparing DINO training...",
    )

    def _log(msg: str) -> None:
        job_storage.append_job_log(job_id, "INFO", msg)

    root = ensure_installed(log_fn=_log)
    if not (root / "main_dino.py").exists():
        raise RuntimeError(f"DINO upstream trainer not found: {root / 'main_dino.py'}")

    data_arg = str(config.get("data") or "")
    if not dataset_registry.is_image_dataset(data_arg):
        raise RuntimeError(f"DINO requires a registered image dataset, got: {data_arg!r}")

    data_yaml_path = job_dir / "data.yaml"
    dataset_yaml.write_data_yaml(
        data_arg,
        data_yaml_path,
        partition_configs=partition_configs if partition_configs else None,
    )
    config["data"] = str(data_yaml_path)
    config["dataset_name"] = data_arg
    job = job_storage.load_job(job_id)
    if job:
        job["config"] = config.copy()
        job_storage.save_job(job)

    max_images = int(config.get("dino_max_images") or config.get("max_images") or 0) or None
    imagefolder, image_count = _prepare_imagefolder(
        job_id,
        data_yaml_path,
        job_dir / "dino_imagefolder",
        max_images=max_images,
    )
    if image_count < int(config.get("batch", 1)):
        job_storage.append_job_log(
            job_id,
            "WARNING",
            f"DINO train image count ({image_count}) is smaller than batch ({config.get('batch')}); upstream uses drop_last=True.",
        )

    out_dir = job_dir / "runs" / "dino"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = _SCALE_TO_SPEC[model_scale]
    batch = int(config["batch"] if config.get("batch") is not None else config.get("batch_size", 64))
    workers = int(config["workers"]) if config.get("workers") is not None else 4
    epochs = int(config["epochs"] if config.get("epochs") is not None else 100)
    use_fp16 = bool(config.get("amp", True))
    save_period = int(config["save_period"]) if config.get("save_period") is not None else 20
    warmup_epochs = int(float(config.get("warmup_epochs", 10)))
    lr = float(config.get("lr0", config.get("lr", 0.0005)))
    min_lr = float(config.get("min_lr", 1e-6))
    weight_decay = float(config.get("weight_decay", 0.04))
    seed = int(config.get("seed", 0))
    local_crops_number = int(config.get("local_crops_number", config.get("dino_local_crops_number", 8)))

    # Resume logic: prioritize last.pth, then best.pth, then checkpoint.pth
    resume_ckpt = _resolve_checkpoint(str(config.get("resume") or ""))
    if resume_ckpt is None:
        # Try last.pth from previous training
        last_ckpt = out_dir / "last.pth"
        if last_ckpt.exists():
            resume_ckpt = last_ckpt
            job_storage.append_job_log(job_id, "INFO", f"DINO resuming from last.pth: {last_ckpt}")
    if resume_ckpt is None:
        resume_ckpt = _resolve_checkpoint(str(config.get("pretrained") or ""))
    if resume_ckpt is not None:
        shutil.copy2(resume_ckpt, out_dir / "checkpoint.pth")
        job_storage.append_job_log(job_id, "INFO", f"DINO resume checkpoint staged: {resume_ckpt}")
    elif bool(config.get("use_yolo_pretrained", True)):
        pretrained_url = str(spec.get("checkpoint") or "")
        if pretrained_url:
            staged = out_dir / "checkpoint.pth"
            _stage_dino_backbone_pretrained(pretrained_url, staged)
            job_storage.append_job_log(
                job_id,
                "INFO",
                f"DINO official pretrained backbone staged: {pretrained_url}",
            )

    cmd = [
        sys.executable,
        "main_dino.py",
        "--arch",
        str(spec["arch"]),
        "--patch_size",
        str(spec["patch_size"]),
        "--data_path",
        str(imagefolder),
        "--output_dir",
        str(out_dir),
        "--epochs",
        str(epochs),
        "--batch_size_per_gpu",
        str(batch),
        "--num_workers",
        str(workers),
        "--seed",
        str(seed),
        "--use_fp16",
        "true" if use_fp16 else "false",
        "--saveckp_freq",
        str(save_period),
        "--warmup_epochs",
        str(max(0, min(warmup_epochs, epochs))),
        "--lr",
        str(lr),
        "--min_lr",
        str(min_lr),
        "--weight_decay",
        str(weight_decay),
        "--local_crops_number",
        str(local_crops_number),
    ]

    env = os.environ.copy()
    backend_root = str(Path(__file__).resolve().parents[2])
    pythonpath = [str(root), backend_root]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["MD_JOB_ID"] = job_id
    env["PYTHONFAULTHANDLER"] = "1"
    env.setdefault("RANK", "0")
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("LOCAL_RANK", "0")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", str(29500 + (int(job_id[:4], 16) % 1000)))
    device = str(config.get("device") or "").strip()
    if device and device.lower() != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = device.split(",")[0]

    job_storage.append_job_log(
        job_id,
        "INFO",
        f"Launching DINO training: {' '.join(cmd)}",
        {"cwd": str(root), "output_dir": str(out_dir), "variant": spec["label"]},
    )

    started = time.time()
    best_loss = float('inf')
    best_checkpoint: Path | None = None
    seen_epochs = set()  # Track seen epochs for dedup
    
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Track log.txt for metrics parsing
    log_path = out_dir / "log.txt"
    ext_metrics_path = job_dir / "extended_metrics.jsonl"
    last_log_size = 0
    
    for line in proc.stdout or []:
        _append_output_log(job_id, line)
        
        # Parse metrics from stdout directly (like RT-DETRv2)
        metrics = _parse_dino_metrics(line)
        if metrics:
            _update_job_metrics(job_id, metrics, batch, started)
            
            # Publish SSE progress event (like Ultralytics)
            epoch = metrics.get("epoch")
            iteration = metrics.get("iteration")
            total_iterations = metrics.get("total_iterations")
            if epoch is not None:
                system_res = _get_system_resources()
                progress_data = {
                    'type': 'progress',
                    'phase': 'train',
                    'epoch': f"{epoch}/{epochs}",
                    'batch': f"{iteration}/{total_iterations}" if iteration and total_iterations else "0/0",
                    'percent': int((iteration / total_iterations) * 100) if iteration and total_iterations else 0,
                    'losses': {
                        'box': metrics.get("train_loss"),  # DINO uses single loss, map to box loss
                    },
                    'device': device if device else 'cuda:0' if config.get("device") != 'cpu' else 'cpu',
                    'ram_gb': system_res["ram_used_gb"],
                    'ram_total_gb': system_res["ram_total_gb"],
                    'gpu_mem_gb': metrics.get("max_mem_mb", 0) / 1024 if metrics.get("max_mem_mb") else None,
                    'total_elapsed_s': time.time() - started,
                    'epoch_elapsed_s': None,  # DINO doesn't track epoch start time
                    'avg_epoch_s': job.get("avg_epoch_time"),
                    'eta_s': None,  # DINO provides eta as string, not seconds
                    'imgs_per_sec': batch / metrics.get("time_per_iter", 1) if metrics.get("time_per_iter") else None,
                }
                event_bus.publish_sync(job_channel(job_id), progress_data)
            
            # Write to extended_metrics.jsonl (like RT-DETRv2)
            # Field names MUST match Ultralytics-style names for frontend compatibility
            epoch = metrics.get("epoch")
            if epoch is not None and epoch not in seen_epochs:
                seen_epochs.add(epoch)
                epoch_data = {
                    "epoch": epoch,
                    "timestamp": time.time(),
                    # Training loss (map DINO → Ultralytics-style)
                    "train_box_loss": metrics.get("train_loss"),
                    # Learning rate
                    "lr": metrics.get("train_lr"),
                    # Other metrics
                    "weight_decay": metrics.get("weight_decay"),
                    "time_per_iter": metrics.get("time_per_iter"),
                    "data_time": metrics.get("data_time"),
                    "max_mem_mb": metrics.get("max_mem_mb"),
                    "eta": metrics.get("eta"),
                }
                # Remove None values
                epoch_data = {k: v for k, v in epoch_data.items() if v is not None}
                try:
                    with ext_metrics_path.open("a") as mf:
                        mf.write(json.dumps(epoch_data) + "\n")
                except Exception:
                    pass
            
            # Track best checkpoint
            train_loss = metrics.get("train_loss")
            if train_loss is not None and isinstance(train_loss, (int, float)):
                if train_loss < best_loss:
                    best_loss = train_loss
                    # Copy current checkpoint to best.pth
                    current_ckpt = out_dir / "checkpoint.pth"
                    if current_ckpt.exists():
                        best_ckpt = out_dir / "best.pth"
                        shutil.copy2(current_ckpt, best_ckpt)
    
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"DINO training failed with exit code {returncode}")

    elapsed = time.time() - started
    
    # Cleanup old checkpoints (keep only last 3 + best)
    _cleanup_old_checkpoints(out_dir, keep_last=3)
    
    # Save best checkpoint if available, otherwise use last
    best_ckpt = out_dir / "best.pth"
    last_ckpt = out_dir / "checkpoint.pth"
    final_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
    
    if final_ckpt.exists():
        # Copy to last.pth for resume functionality
        last_path = out_dir / "last.pth"
        shutil.copy2(final_ckpt, last_path)
        job_storage.append_job_log(job_id, "INFO", f"DINO final checkpoint: {final_ckpt.name}")
        
        # Run k-NN evaluation if dataset has train/val split with labels
        # This provides validation metrics similar to mAP for self-supervised learning
        _run_knn_evaluation(job_id, root, imagefolder, final_ckpt, spec, batch, workers, out_dir)
    
    _append_history_from_log(job_id, out_dir)
    weight_id = _save_weight(job_id, out_dir)
    _set_job(
        job_id,
        status="completed",
        epoch=epochs,
        message="Training complete",
        weight_id=weight_id,
        completed_at=datetime.utcnow().isoformat() + "Z",
    )
    job_storage.append_job_log(
        job_id,
        "INFO",
        f"DINO training completed in {elapsed:.1f}s. Weight: {weight_id}",
    )
