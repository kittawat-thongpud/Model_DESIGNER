"""
Stats Controller — dashboard statistics.
Tagged as "Logs" for ReDoc grouping (alongside system info).
"""
from __future__ import annotations
from fastapi import APIRouter

from ..services import model_storage, module_storage, job_storage, weight_storage

router = APIRouter(tags=["Logs"])


@router.get("/api/stats", summary="Dashboard statistics")
async def get_stats():
    """Return aggregated stats for the dashboard."""
    models = model_storage.list_models()
    modules = module_storage.list_modules()
    jobs = job_storage.list_jobs()
    weights = weight_storage.list_weights()

    active_jobs = sum(1 for j in jobs if j.get("status") in ("pending", "running"))

    return {
        "total_models": len(models),
        "total_modules": len(modules),
        "total_jobs": len(jobs),
        "active_jobs": active_jobs,
        "total_weights": len(weights),
    }
