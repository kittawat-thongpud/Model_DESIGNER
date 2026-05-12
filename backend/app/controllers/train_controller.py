"""
Train Controller — Start/stop/monitor Ultralytics training jobs.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..schemas.job_schema import TrainRequest
from ..services import ultra_trainer, job_storage, model_storage, system_metrics
from ..services.config_service import get_model_config, get_training_config
from ..services.preflight import validate_train_request
from .. import logging_service as logger

router = APIRouter(prefix="/api/train", tags=["Training"])
_TRAINING_API_DEFAULTS = get_training_config().get("api_defaults", {})
_MODEL_DEFAULTS = get_model_config().get("defaults", {})


@router.post("/start", summary="Start a training job")
async def start_training(req: TrainRequest):
    """Start an Ultralytics training job."""
    # Check if official YOLO model
    if req.model_id.startswith("yolo:"):
        # Official YOLO model - no database lookup needed
        model_name = req.model_id.split(":")[1].upper()  # e.g., "yolo:yolov8" -> "YOLOV8"
        task = str(_MODEL_DEFAULTS.get("task", "detect"))
        yaml_path = ""  # Not used for official YOLO models
    elif req.model_id.startswith("arch:"):
        # Arch plugin model — resolve family + scale to specific plugin
        # model_id = "arch:mamba_yolo", model_scale = "t"  → plugin "mamba_yolo_t"
        # model_id = "arch:rtdetr",      model_scale = "l"  → plugin "rtdetr_l"
        # model_id = "arch:hsg_det"      (no scale)         → plugin "hsg_det"
        family_key = req.model_id[len("arch:"):]
        scale = (req.model_scale or "").strip().lower()
        arch_key = f"{family_key}_{scale}" if scale else family_key

        from ..plugins.loader import get_arch_plugin
        arch_plugin = get_arch_plugin(arch_key)
        # Fallback: try family key directly (single-variant plugins)
        if arch_plugin is None and scale:
            arch_plugin = get_arch_plugin(family_key)
        if arch_plugin is None:
            raise HTTPException(404, f"Arch plugin not found: {arch_key!r}")
        # Run preflight check — e.g. selective_scan install state for Mamba-YOLO
        preflight_err = arch_plugin.preflight_check()
        if preflight_err:
            raise HTTPException(400, preflight_err)
        # Job name: "Mamba-YOLO T", "RT-DETR L", "HSG-DET M" etc.
        _scale_upper = (arch_plugin.scale or "").upper()
        model_name = (
            f"{arch_plugin.family_display_name} {_scale_upper}"
            if _scale_upper else arch_plugin.family_display_name
        )
        task = arch_plugin.task_type
        yaml_path = ""  # arch plugin sets yaml_path in training worker
        # Inject model_arch key so the training worker picks up the plugin
        config = req.config.to_train_kwargs()
        config["model_arch"] = arch_plugin.name
    else:
        # Custom model - load from database
        record = model_storage.load_model(req.model_id)
        if not record:
            raise HTTPException(404, f"Model not found: {req.model_id}")

        yaml_path = model_storage.load_model_yaml_path(req.model_id)
        if not yaml_path:
            raise HTTPException(400, "No YAML file for this model")

        task = record.get("task", str(_MODEL_DEFAULTS.get("task", "detect")))
        model_name = record.get("name", "Untitled")

    if not req.model_id.startswith("arch:"):
        config = req.config.to_train_kwargs()

    # Convert PartitionSplitConfig objects to dict format
    partition_configs = [p.dict() for p in req.partitions] if req.partitions else []

    # Preflight validation — check dataset, YAML, config, and disk space before
    # admitting the job to avoid failures that surface mid-training.
    dataset_name = config.get("data", "")
    preflight = validate_train_request(
        model_id=req.model_id,
        yaml_path=str(yaml_path) if yaml_path else "",
        dataset_name=dataset_name,
        config=config,
        weight_id=config.get("pretrained") or None,
        partition_configs=partition_configs,
    )
    if preflight.warnings:
        for w in preflight.warnings:
            logger.log("training", "WARNING", f"Preflight warning for {dataset_name}: {w}",
                       {"model_id": req.model_id, "dataset": dataset_name})
    if not preflight.ok:
        raise HTTPException(400, {
            "message": "Preflight validation failed",
            "errors": preflight.errors,
            "warnings": preflight.warnings,
            "info": preflight.info,
        })

    job_id = ultra_trainer.start_training(
        model_id=req.model_id,
        model_name=model_name,
        task=task,
        yaml_path=str(yaml_path) if yaml_path else "",
        config=config,
        partition_configs=partition_configs,
        model_scale=req.model_scale,
    )

    return {"job_id": job_id, "message": "Training started"}


@router.post("/{job_id}/stop", summary="Stop a training job")
async def stop_training(job_id: str):
    if not ultra_trainer.stop_training(job_id):
        raise HTTPException(404, f"Job not found or not running: {job_id}")
    return {"message": "Stop signal sent"}


@router.post("/{job_id}/resume", summary="Resume training from last checkpoint")
async def resume_training(job_id: str):
    """Resume a stopped/failed job from its last.pt checkpoint (same job_id)."""
    try:
        ultra_trainer.resume_training(job_id)
        return {"job_id": job_id, "message": "Resuming training from last checkpoint"}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/queue", summary="Get training queue status")
async def get_queue_status():
    """Return current training queue status including pending jobs and slot limits."""
    from ..services.task_queue import queue_status, TaskType
    return queue_status(TaskType.TRAINING)


@router.get("/workers/health", summary="Get worker health status")
async def get_worker_health():
    """Get health status of all active worker threads."""
    return ultra_trainer.get_worker_health()


@router.post("/workers/cleanup", summary="Clean up zombie workers")
async def cleanup_zombie_workers():
    """Detect and clean up zombie worker threads."""
    cleaned = ultra_trainer.cleanup_zombie_workers()
    return {
        "message": f"Cleaned up {len(cleaned)} zombie workers",
        "cleaned": cleaned
    }


class AppendRequest(BaseModel):
    additional_epochs: int = int(_TRAINING_API_DEFAULTS.get("append_additional_epochs", 50))


@router.post("/{job_id}/append", summary="Append more epochs to a completed/stopped job")
async def append_training(job_id: str, req: AppendRequest):
    """Append additional epochs to a job using last.pt as warm-start (same job_id)."""
    try:
        ultra_trainer.append_training(job_id, req.additional_epochs)
        return {"job_id": job_id, "message": f"Appending {req.additional_epochs} more epochs"}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/{job_id}", summary="Get training job status")
async def get_job(job_id: str):
    job = job_storage.load_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


@router.get("/{job_id}/logs", summary="Get training logs")
async def get_job_logs(job_id: str, limit: int = int(_TRAINING_API_DEFAULTS.get("job_log_limit", 200)), offset: int = 0):
    return job_storage.get_job_logs(job_id, limit=limit, offset=offset)


@router.get("/{job_id}/plots", summary="List available training plots")
async def list_plots(job_id: str):
    """List all available plots for a training job."""
    from pathlib import Path
    from ..config import JOBS_DIR
    
    job_dir = JOBS_DIR / job_id
    plots_dir = job_dir / "runs" / "train"
    
    if not plots_dir.exists():
        return {"plots": []}
    
    plot_files = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "PR_curve.png",
        "results.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]
    
    available = []
    for plot_file in plot_files:
        plot_path = plots_dir / plot_file
        if plot_path.exists():
            available.append({
                "name": plot_file,
                "path": str(plot_path.relative_to(JOBS_DIR)),
                "size": plot_path.stat().st_size,
            })
    
    return {"plots": available}


@router.get("/{job_id}/plots/{plot_name}", summary="Get a specific plot image")
async def get_plot(job_id: str, plot_name: str):
    """Serve a plot image file."""
    from pathlib import Path
    from fastapi.responses import FileResponse
    from ..config import JOBS_DIR
    
    job_dir = JOBS_DIR / job_id
    plot_path = job_dir / "runs" / "train" / plot_name
    
    if not plot_path.exists() or not plot_path.is_file():
        raise HTTPException(404, f"Plot not found: {plot_name}")
    
    return FileResponse(plot_path, media_type="image/png")


@router.get("/{job_id}/gradients", summary="List gradient statistics files")
async def list_gradients(job_id: str):
    """List all gradient statistics files for a job."""
    from pathlib import Path
    from ..config import JOBS_DIR
    
    job_dir = JOBS_DIR / job_id
    grad_dir = job_dir / "gradients"
    
    if not grad_dir.exists():
        return {"gradients": []}
    
    gradient_files = sorted(grad_dir.glob("epoch_*.json"))
    
    return {
        "gradients": [
            {
                "epoch": int(f.stem.split("_")[1]),
                "file": f.name,
                "size": f.stat().st_size,
            }
            for f in gradient_files
        ]
    }


@router.get("/{job_id}/gradients/{epoch}", summary="Get gradient statistics for an epoch")
async def get_gradient_stats(job_id: str, epoch: int):
    """Get gradient statistics for a specific epoch."""
    from pathlib import Path
    import json
    from ..config import JOBS_DIR
    
    job_dir = JOBS_DIR / job_id
    grad_file = job_dir / "gradients" / f"epoch_{epoch}.json"
    
    if not grad_file.exists():
        raise HTTPException(404, f"Gradient stats not found for epoch {epoch}")
    
    return json.loads(grad_file.read_text())


@router.get("/{job_id}/weights_stats", summary="List weight statistics files")
async def list_weight_stats(job_id: str):
    """List all weight statistics files for a job."""
    from pathlib import Path
    from ..config import JOBS_DIR
    
    job_dir = JOBS_DIR / job_id
    weight_dir = job_dir / "weights_stats"
    
    if not weight_dir.exists():
        return {"weights": []}
    
    weight_files = sorted(weight_dir.glob("epoch_*.json"))
    
    return {
        "weights": [
            {
                "epoch": int(f.stem.split("_")[1]),
                "file": f.name,
                "size": f.stat().st_size,
            }
            for f in weight_files
        ]
    }


@router.get("/{job_id}/weights_stats/{epoch}", summary="Get weight statistics for an epoch")
async def get_weight_stats(job_id: str, epoch: int):
    """Get weight statistics for a specific epoch."""
    from pathlib import Path
    import json
    from ..config import JOBS_DIR
    
    job_dir = JOBS_DIR / job_id
    weight_file = job_dir / "weights_stats" / f"epoch_{epoch}.json"
    
    if not weight_file.exists():
        raise HTTPException(404, f"Weight stats not found for epoch {epoch}")
    
    return json.loads(weight_file.read_text())


@router.get("/{job_id}/samples", summary="List validation class samples")
async def list_class_samples(job_id: str):
    """List available validation samples grouped by class."""
    from pathlib import Path
    from ..config import JOBS_DIR
    
    job_dir = JOBS_DIR / job_id
    samples_dir = job_dir / "runs" / "train" / "samples"
    
    if not samples_dir.exists():
        return {"classes": []}
    
    classes = []
    for class_dir in sorted(samples_dir.iterdir()):
        if class_dir.is_dir():
            images = sorted([f.name for f in class_dir.glob("*.jpg")])
            if images:
                classes.append({
                    "name": class_dir.name,
                    "count": len(images),
                    "images": images
                })
    
    return {"classes": classes}


@router.get("/{job_id}/samples/{class_name}/{filename}", summary="Get a specific sample image")
async def get_class_sample(job_id: str, class_name: str, filename: str):
    """Serve a specific validation sample image."""
    from pathlib import Path
    from fastapi.responses import FileResponse
    
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, f"Job not found: {job_id}")
    
    sample_path = job_dir / "runs" / "train" / "samples" / class_name / filename
    if not sample_path.exists():
        raise HTTPException(404, f"Sample not found: {class_name}/{filename}")
    
    return FileResponse(sample_path, media_type="image/jpeg")


@router.get("/{job_id}/checkpoints", summary="List available checkpoints for a job")
async def list_checkpoints(job_id: str):
    """List checkpoints available for export or profile creation."""
    from pathlib import Path
    from ..config import JOBS_DIR

    job = job_storage.load_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    job_dir = JOBS_DIR / job_id
    family = _job_arch_family(job)

    checkpoints = []
    if family == "rtdetrv2":
        run_dir = job_dir / "runs" / "rtdetrv2"
        candidates = [run_dir / "best.pth", run_dir / "last.pth", run_dir / "checkpoint.pth"]
    elif family == "dino":
        candidates = []
        for run_name in ("dino_detr", "dino"):
            run_dir = job_dir / "runs" / run_name
            candidates.extend([run_dir / "best.pth", run_dir / "last.pth", run_dir / "checkpoint.pth"])
            candidates.extend(sorted(run_dir.glob("checkpoint*.pth"), key=lambda p: p.stat().st_mtime, reverse=True) if run_dir.exists() else [])
    else:
        weights_dir = job_dir / "runs" / "train" / "weights"
        candidates = [weights_dir / "best.pt", weights_dir / "last.pt"]

    seen = set()
    for p in candidates:
        if p.exists():
            if p.name in seen:
                continue
            seen.add(p.name)
            checkpoints.append({
                "name": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "modified_at": p.stat().st_mtime,
            })

    return {"checkpoints": checkpoints}


def _job_arch_family(job: dict | None) -> str:
    if not job:
        return ""
    model_id = str(job.get("model_id") or "").lower()
    if model_id.startswith("arch:"):
        model_id = model_id[len("arch:"):]
    model_arch = str((job.get("config") or {}).get("model_arch") or "").lower()
    if model_id == "rtdetrv2" or model_arch.startswith("rtdetrv2"):
        return "rtdetrv2"
    if model_id == "dino" or model_arch.startswith("dino"):
        return "dino"
    return ""


def _checkpoint_path_for_job(job_id: str, checkpoint_name: str) -> Path:
    from urllib.parse import unquote
    from ..config import JOBS_DIR

    job = job_storage.load_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    name = Path(unquote(checkpoint_name)).name
    job_dir = JOBS_DIR / job_id
    family = _job_arch_family(job)
    if family == "rtdetrv2":
        candidates = [job_dir / "runs" / "rtdetrv2" / name]
    elif family == "dino":
        candidates = [job_dir / "runs" / run_name / name for run_name in ("dino_detr", "dino")]
    else:
        candidates = [job_dir / "runs" / "train" / "weights" / name]
    src = next((p for p in candidates if p.exists() and p.is_file()), None)
    if src is None:
        raise HTTPException(404, f"Checkpoint '{name}' not found for job {job_id}")
    return src


@router.post("/{job_id}/checkpoints/{checkpoint_name}/create-weight-profile",
             summary="Create a weight profile from a job checkpoint")
async def create_weight_from_checkpoint(job_id: str, checkpoint_name: str):
    """Save a job checkpoint as a named weight profile in the weight library."""
    import torch
    from ..services import weight_storage

    job = job_storage.load_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    src = _checkpoint_path_for_job(job_id, checkpoint_name)
    family = _job_arch_family(job)

    label = "best" if "best" in src.name else "last" if "last" in src.name else src.stem
    profile_name = f"{job.get('model_name', job_id)} ({label})"

    weight_id = weight_storage.save_weight_meta(
        model_id=job.get("model_id", ""),
        model_name=profile_name,
        job_id=job_id,
        dataset=job.get("dataset_name") or job.get("config", {}).get("data", ""),
        epochs_trained=job.get("epoch", 0),
        final_accuracy=job.get("best_mAP50"),
        final_loss=None,
        model_scale=job.get("model_scale"),
        total_time=job.get("total_time"),
        device=job.get("config", {}).get("device", ""),
    )

    import shutil
    dest = weight_storage.weight_pt_path(weight_id)
    shutil.copy2(str(src), str(dest))

    meta = weight_storage.load_weight_meta(weight_id)
    if meta:
        try:
            raw = torch.load(str(dest), map_location="cpu", weights_only=False)
            sd = None
            if isinstance(raw, dict):
                for key in ("ema", "model", "state_dict", "model_state_dict"):
                    value = raw.get(key)
                    if isinstance(value, dict):
                        sd = value
                        break
                if sd is None and any(torch.is_tensor(v) for v in raw.values()):
                    sd = raw
            if isinstance(sd, dict):
                tensors = [v for v in sd.values() if hasattr(v, "numel")]
                meta["key_count"] = len(sd)
                meta["param_count"] = sum(v.numel() for v in tensors)
        except Exception:
            pass
        meta["file_size_bytes"] = dest.stat().st_size
        if family in {"rtdetrv2", "dino"}:
            cfg = job.get("config", {}) or {}
            meta.update({
                "source_type": family,
                "arch_plugin": cfg.get("model_arch"),
                "model_arch": cfg.get("model_arch"),
                "train_args": {
                    "model_arch": cfg.get("model_arch"),
                    "model_scale": job.get("model_scale"),
                    "checkpoint": src.name,
                    "job_id": job_id,
                },
            })
        weight_storage._store.save(weight_id, meta)

    return {
        "weight_id": weight_id,
        "model_name": profile_name,
        "checkpoint": checkpoint_name,
        "job_id": job_id,
        "message": f"Weight profile created from {checkpoint_name}",
    }


@router.get("/{job_id}/metrics", summary="Get system metrics for a job")
async def get_job_metrics(job_id: str):
    """Get current system metrics (GPU, CPU, RAM) for monitoring."""
    # Verify job exists
    job = job_storage.load_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    
    return system_metrics.get_job_metrics(job_id)
