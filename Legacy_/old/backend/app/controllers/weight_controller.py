"""
Weight Controller — API endpoints for trained weight management.
Tagged as "Weights" in ReDoc.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..services import weight_storage
from .. import logging_service as logger

router = APIRouter(prefix="/api/weights", tags=["Weights"])


@router.get("/", summary="List all saved weights")
async def list_weights(model_id: str | None = None):
    """Return all weight metadata, optionally filtered by parent model."""
    return weight_storage.list_weights(model_id=model_id)


@router.get("/{weight_id}", summary="Get weight details")
async def get_weight(weight_id: str):
    """Return full weight metadata including parent model and job info."""
    record = weight_storage.load_weight_meta(weight_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Weight '{weight_id}' not found")
    return record


@router.delete("/{weight_id}", summary="Delete a weight")
async def delete_weight(weight_id: str):
    """Delete weight .pt file and its metadata."""
    if weight_storage.delete_weight(weight_id):
        logger.log("system", "INFO", f"Weight deleted", {"weight_id": weight_id})
        return {"message": f"Weight '{weight_id}' deleted"}
    raise HTTPException(status_code=404, detail=f"Weight '{weight_id}' not found")
