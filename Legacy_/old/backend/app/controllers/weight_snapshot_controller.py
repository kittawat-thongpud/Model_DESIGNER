"""
Weight Snapshot Controller — API endpoints for weight progression data.
Tagged as "Weight Snapshots" in ReDoc.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..services import weight_snapshots

router = APIRouter(prefix="/api/jobs", tags=["Weight Snapshots"])


@router.get("/{job_id}/snapshots", summary="List weight snapshots")
async def list_snapshots(job_id: str):
    """Return all snapshot metadata for a training job."""
    return weight_snapshots.list_snapshots(job_id)


@router.get("/{job_id}/snapshots/stats", summary="Get snapshot stats timeline")
async def get_snapshot_stats(job_id: str):
    """Return organized stats timeline grouped by layer."""
    return weight_snapshots.get_snapshot_stats(job_id)


@router.get("/{job_id}/snapshots/layers", summary="Get recorded layers")
async def get_recorded_layers(job_id: str):
    """Return list of layer names that have been recorded."""
    return weight_snapshots.get_recorded_layers(job_id)


@router.get("/{job_id}/snapshots/epochs", summary="Get recorded epochs")
async def get_recorded_epochs(job_id: str):
    """Return list of epochs that have snapshot data."""
    return weight_snapshots.get_recorded_epochs(job_id)


@router.get("/{job_id}/snapshots/{epoch}/{layer_name}", summary="Get snapshot data")
async def get_snapshot_data(job_id: str, epoch: int, layer_name: str):
    """Return snapshot tensor data reshaped for heatmap visualization."""
    data = weight_snapshots.load_snapshot_data(job_id, epoch, layer_name)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Snapshot not found: job={job_id} epoch={epoch} layer={layer_name}"
        )
    return data
