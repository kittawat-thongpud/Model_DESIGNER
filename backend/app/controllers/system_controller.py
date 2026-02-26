"""
System Controller — Global system metrics endpoint.
"""
from __future__ import annotations
from fastapi import APIRouter

from ..services import system_metrics

router = APIRouter(prefix="/api/system", tags=["System"])


@router.get("/metrics", summary="Get global system metrics")
async def get_system_metrics():
    """Get current system metrics (GPU, CPU, RAM) for dashboard monitoring."""
    return system_metrics.get_system_metrics()
