"""
Train Controller — Start/stop/monitor Ultralytics training jobs.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..schemas.job_schema import TrainRequest
from ..services import ultra_trainer, job_storage, model_storage, system_metrics

router = APIRouter(prefix="/api/train", tags=["Training"])


@router.post("/start", summary="Start a training job")
async def start_training(req: TrainRequest):
    """Start an Ultralytics training job."""
    # Check if official YOLO model
    if req.model_id.startswith("yolo:"):
        # Official YOLO model - no database lookup needed
        model_name = req.model_id.split(":")[1].upper()  # e.g., "yolo:yolov8" -> "YOLOV8"
        task = "detect"
        yaml_path = ""  # Not used for official YOLO models
    else:
        # Custom model - load from database
        record = model_storage.load_model(req.model_id)
        if not record:
            raise HTTPException(404, f"Model not found: {req.model_id}")

        yaml_path = model_storage.load_model_yaml_path(req.model_id)
        if not yaml_path:
            raise HTTPException(400, "No YAML file for this model")

        task = record.get("task", "detect")
        model_name = record.get("name", "Untitled")

    config = req.config.to_train_kwargs()

    # Convert PartitionSplitConfig objects to dict format
    partition_configs = [p.dict() for p in req.partitions] if req.partitions else []

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
    additional_epochs: int = 50


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
async def get_job_logs(job_id: str, limit: int = 200, offset: int = 0):
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


@router.get("/{job_id}/metrics", summary="Get system metrics for a job")
async def get_job_metrics(job_id: str):
    """Get current system metrics (GPU, CPU, RAM) for monitoring."""
    # Verify job exists
    job = job_storage.load_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    
    return system_metrics.get_job_metrics(job_id)
