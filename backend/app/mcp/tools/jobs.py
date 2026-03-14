"""
MCP tools — Training job records.
Wraps job_storage and system_metrics services.
"""
from __future__ import annotations
from typing import Any

from ...services import job_storage, system_metrics
from ..filters import apply_view, apply_list_view, paginate
from ..serializers import ok, err, safe_list


def list_jobs(
    status: str | None = None,
    model_id: str | None = None,
    view: str = "summary",
    limit: int | None = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List training jobs with optional filters.

    Args:
        status: Filter by status (pending, running, completed, failed, stopped).
        model_id: Filter by model ID.
        view: "summary" (default) or "detail".
        limit: Max items to return (default 50).
        offset: Items to skip.
    """
    try:
        records = job_storage.list_jobs(status=status, model_id=model_id)
        records = paginate(records, limit=limit, offset=offset)
        items = apply_list_view(records, "job", view=view)
        return {"ok": True, "count": len(items), "items": items}
    except Exception as e:
        return err(str(e), "list_jobs_failed")


def get_job(
    job_id: str,
    include_history: bool = False,
    view: str = "summary",
) -> dict[str, Any]:
    """Get full training job record.

    Args:
        job_id: Target job ID.
        include_history: If True, include epoch-by-epoch history array.
        view: "summary" (default) or "detail".
    """
    try:
        record = job_storage.load_job(job_id)
        if not record:
            return err(f"Job not found: {job_id}", "not_found")

        if include_history:
            record["history"] = job_storage.get_job_history(job_id)

        filtered = apply_view(
            record, "job", view=view,
            extra_fields=["history"] if include_history else None,
        )
        return ok(filtered)
    except Exception as e:
        return err(str(e), "get_job_failed")


def get_job_logs(
    job_id: str,
    limit: int = 50,
    offset: int = 0,
    level: str | None = None,
) -> dict[str, Any]:
    """Get training logs for a job (most recent first).

    Args:
        job_id: Target job ID.
        limit: Max log lines to return (default 50).
        offset: Lines to skip.
        level: Optional filter by log level (INFO, WARNING, ERROR, DEBUG, PROGRESS).
    """
    try:
        record = job_storage.load_job(job_id)
        if not record:
            return err(f"Job not found: {job_id}", "not_found")

        logs = job_storage.get_job_logs(job_id, limit=500, offset=0)
        if level:
            logs = [l for l in logs if l.get("level") == level.upper()]
        logs = paginate(logs, limit=limit, offset=offset)

        slim = [
            {"timestamp": l.get("timestamp"), "level": l.get("level"), "message": l.get("message")}
            for l in logs
        ]
        return {"ok": True, "job_id": job_id, "count": len(slim), "logs": slim}
    except Exception as e:
        return err(str(e), "get_job_logs_failed")


def get_job_metrics(job_id: str) -> dict[str, Any]:
    """Get current system metrics (GPU, CPU, RAM) for a running job.

    Args:
        job_id: Target job ID.
    """
    try:
        job = job_storage.load_job(job_id)
        if not job:
            return err(f"Job not found: {job_id}", "not_found")

        metrics = system_metrics.get_job_metrics(job_id)
        return ok(metrics)
    except Exception as e:
        return err(str(e), "get_job_metrics_failed")


def get_job_history(
    job_id: str,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Get epoch-by-epoch training metrics history.

    Args:
        job_id: Target job ID.
        limit: Max epochs to return.
        offset: Epochs to skip.
    """
    try:
        job = job_storage.load_job(job_id)
        if not job:
            return err(f"Job not found: {job_id}", "not_found")

        history = job_storage.get_job_history(job_id)
        history = paginate(history, limit=limit, offset=offset)
        return {"ok": True, "job_id": job_id, "count": len(history), "history": history}
    except Exception as e:
        return err(str(e), "get_job_history_failed")
