"""
Job Controller — List and retrieve training jobs.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..services import job_storage, system_metrics
from .. import logging_service as logger

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


@router.get("/", summary="List all training jobs")
async def list_jobs(status: str | None = None, model_id: str | None = None):
    """Return all training jobs with optional filters."""
    return job_storage.list_jobs(status=status, model_id=model_id)


@router.get("/{job_id}", summary="Get job details")
async def get_job(job_id: str, include_history: bool = True):
    """Return full training job record, optionally with history."""
    record = job_storage.load_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    # Load history separately if requested
    if include_history:
        record["history"] = job_storage.get_job_history(job_id)
    
    return record


@router.get("/{job_id}/history", summary="Get job training history")
async def get_job_history(job_id: str):
    """Return epoch-by-epoch metrics history for a specific job."""
    record = job_storage.load_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    return {"history": job_storage.get_job_history(job_id)}


@router.get("/{job_id}/logs", summary="Get job training logs")
async def get_job_logs(job_id: str, limit: int = 200, offset: int = 0):
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
