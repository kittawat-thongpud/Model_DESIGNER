"""
System Controller — Global system metrics endpoint.
"""
from __future__ import annotations
from fastapi import APIRouter

from ..services import system_metrics
from ..services.config_service import get_config_path, get_effective_config
from ..config import DATA_DIR, JOBS_DIR, WEIGHTS_DIR, DATASETS_DIR
from ..services import job_storage, weight_storage
from ..services.dataset_registry import get_all_datasets

router = APIRouter(prefix="/api/system", tags=["System"])


@router.get("/metrics", summary="Get global system metrics")
async def get_system_metrics():
    """Get current system metrics (GPU, CPU, RAM) for dashboard monitoring."""
    return system_metrics.get_system_metrics()


@router.get("/config/effective", summary="Get effective runtime configuration")
async def get_effective_runtime_config():
    return {
        "config_path": str(get_config_path()),
        "config": get_effective_config(),
    }


@router.get("/storage/diagnostics", summary="Diagnose runtime storage visibility")
async def get_storage_diagnostics():
    """Return what this backend process can actually see on disk."""
    dataset_dirs = []
    try:
        dataset_dirs = sorted(
            p.name for p in DATASETS_DIR.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )
    except Exception:
        dataset_dirs = []

    datasets_error = None
    datasets_count = None
    try:
        datasets_count = len(get_all_datasets())
    except Exception as exc:
        datasets_error = str(exc)

    return {
        "data_dir": str(DATA_DIR),
        "jobs_dir": str(JOBS_DIR),
        "weights_dir": str(WEIGHTS_DIR),
        "datasets_dir": str(DATASETS_DIR),
        "jobs": job_storage._store.diagnose(),
        "weights": weight_storage._store.diagnose(),
        "datasets": {
            "directory": str(DATASETS_DIR),
            "directory_exists": DATASETS_DIR.exists(),
            "disk_count": len(dataset_dirs),
            "disk_ids_sample": dataset_dirs[:20],
            "registry_count": datasets_count,
            "registry_error": datasets_error,
        },
    }


@router.post("/storage/rebuild-indexes", summary="Force rebuild jobs and weights storage indexes")
async def rebuild_storage_indexes():
    """Rebuild jobs/weights indexes from disk and return diagnostics."""
    return {
        "jobs": job_storage._store.rebuild_index(),
        "weights": weight_storage._store.rebuild_index(),
    }
