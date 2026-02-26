"""
Log Controller — system-wide structured application logs.
Tagged as "Logs" for ReDoc grouping.

Enhanced with time-range filtering, job/model_id filtering,
log stats, cleanup, and runtime level configuration.
"""
from __future__ import annotations
from fastapi import APIRouter

from .. import logging_service as logger

router = APIRouter(prefix="/api/logs", tags=["Logs"])


@router.get("/", summary="Fetch application logs")
async def get_logs(
    category: str | None = None,
    level: str | None = None,
    limit: int = 100,
    offset: int = 0,
    job_id: str | None = None,
    model_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    days: int = 7,
):
    """
    Retrieve structured log entries with optional filters.

    - **category**: system, model, training, dataset
    - **level**: DEBUG, INFO, WARNING, ERROR
    - **job_id**: filter by training job ID
    - **model_id**: filter by model ID
    - **since/until**: ISO timestamp bounds
    - **days**: how many days of logs to scan (default 7)
    """
    return logger.get_logs(
        category=category, level=level, limit=limit, offset=offset,
        job_id=job_id, model_id=model_id, since=since, until=until, days=days,
    )


@router.delete("/", summary="Clear all logs")
async def clear_logs():
    """Delete all log entries across all categories."""
    count = logger.clear_logs()
    return {"message": f"Cleared {count} log entries"}


@router.get("/stats", summary="Get log storage stats")
async def get_log_stats():
    """Return summary of log files, sizes, and configuration."""
    return logger.get_log_stats()


@router.post("/cleanup", summary="Clean up old log files")
async def cleanup_old_logs(retention_days: int | None = None):
    """Delete log files older than retention_days (default: LOG_RETENTION_DAYS env var)."""
    deleted = logger.cleanup_old_logs(retention_days)
    return {"message": f"Deleted {deleted} old log files", "deleted": deleted}


@router.get("/level", summary="Get current minimum log level")
async def get_log_level():
    """Return the current minimum log level."""
    return {"level": logger.get_min_level()}


@router.put("/level", summary="Set minimum log level")
async def set_log_level(level: str):
    """Set the minimum log level at runtime (DEBUG, INFO, WARNING, ERROR)."""
    level = level.upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Invalid level: {level}")
    logger.set_min_level(level)
    return {"message": f"Minimum log level set to {level}", "level": level}
