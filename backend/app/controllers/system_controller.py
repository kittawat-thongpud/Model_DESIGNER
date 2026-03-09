"""
System Controller — Global system metrics endpoint.
"""
from __future__ import annotations
from fastapi import APIRouter

from ..services import system_metrics
from ..services.config_service import get_config_path, get_effective_config

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
