"""
Job Controller — List and retrieve training jobs.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..services import job_storage, system_metrics
from ..services.config_service import get_training_config
from .. import logging_service as logger

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])
_TRAINING_API_DEFAULTS = get_training_config().get("api_defaults", {})


@router.get("", include_in_schema=False)
@router.get("/", summary="List all training jobs")
async def list_jobs(status: str | None = None, model_id: str | None = None):
    """Return all training jobs with optional filters."""
    return job_storage.list_jobs(status=status, model_id=model_id)


def _resolve_partition_datasets(partition_configs: list[dict], dataset_name: str) -> list[dict]:
    """Enrich partition configs with dataset names from partition cache."""
    if not partition_configs or not dataset_name:
        return partition_configs
    
    try:
        from pathlib import Path
        from ..config import SPLITS_DIR
        
        partition_file = Path(SPLITS_DIR) / f"{dataset_name.lower()}_partitions.json"
        if not partition_file.exists():
            return partition_configs
        
        import json
        cache = json.loads(partition_file.read_text())
        cache_partitions = {p["id"]: p for p in cache.get("partitions", [])}
        
        enriched = []
        for pc in partition_configs:
            partition_id = pc.get("partition_id", "")
            if partition_id in cache_partitions:
                enriched.append({
                    **pc,
                    "dataset_name": dataset_name,
                    "partition_name": cache_partitions[partition_id].get("name", partition_id)
                })
            else:
                enriched.append({**pc, "dataset_name": dataset_name})
        return enriched
    except Exception:
        return partition_configs


@router.get("/{job_id}", summary="Get job details")
async def get_job(job_id: str, include_history: bool = True):
    """Return full training job record, optionally with history."""
    record = job_storage.load_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    # Load history separately if requested
    if include_history:
        record["history"] = job_storage.get_job_history(job_id)
    
    # Enrich partition_configs with dataset names
    partition_configs = record.get("partition_configs")
    dataset_name = record.get("dataset_name")
    if partition_configs and dataset_name:
        record["partition_configs"] = _resolve_partition_datasets(partition_configs, dataset_name)
    
    return record


@router.get("/{job_id}/history", summary="Get job training history")
async def get_job_history(job_id: str):
    """Return epoch-by-epoch metrics history for a specific job."""
    record = job_storage.load_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    return {"history": job_storage.get_job_history(job_id)}


@router.get("/{job_id}/logs", summary="Get job training logs")
async def get_job_logs(job_id: str, limit: int = int(_TRAINING_API_DEFAULTS.get("job_log_limit", 200)), offset: int = 0):
    """Return per-epoch training log entries for a specific job."""
    return job_storage.get_job_logs(job_id, limit=limit, offset=offset)


@router.delete("/{job_id}", summary="Delete a training job")
async def delete_job(job_id: str):
    """Delete a job record and its training logs."""
    if job_storage.delete_job(job_id):
        logger.log("system", "INFO", f"Job deleted", {"job_id": job_id})
        return {"message": f"Job '{job_id}' deleted"}
    raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")


@router.get("/{job_id}/metrics", summary="Get system metrics for a job")
async def get_job_metrics(job_id: str):
    """Get current system metrics (GPU, CPU, RAM) for monitoring."""
    # Verify job exists
    job = job_storage.load_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    
    return system_metrics.get_job_metrics(job_id)
