"""
Build Controller — CRUD for built (code-generated) model records.
Tagged as "Builds" for ReDoc grouping.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..schemas.build_schema import BuildRecord, BuildSummary
from ..services import build_storage
from .. import logging_service as logger

router = APIRouter(prefix="/api/builds", tags=["Builds"])


@router.get("/", response_model=list[BuildSummary], summary="List all builds")
async def list_builds():
    """Return summary of all built models."""
    builds = build_storage.list_builds()
    return [
        BuildSummary(
            build_id=b["build_id"],
            model_id=b.get("model_id", ""),
            model_name=b.get("model_name", "Untitled"),
            class_name=b.get("class_name", ""),
            layer_count=len(b.get("layers", [])),
            layer_types=[l.get("layer_type", "") for l in b.get("layers", [])],
            created_at=b.get("created_at", ""),
        )
        for b in builds
    ]


@router.get("/check", summary="Check if build name exists")
async def check_build_name(name: str):
    """Check if a build with the given model name already exists."""
    existing = build_storage.find_build_by_name(name)
    if existing:
        return {"exists": True, "build_id": existing["build_id"], "model_name": existing.get("model_name", "")}
    return {"exists": False}


@router.get("/{build_id}", response_model=BuildRecord, summary="Get build details")
async def get_build(build_id: str):
    """Load a specific build record by ID."""
    record = build_storage.load_build(build_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Build '{build_id}' not found")
    return BuildRecord(**record)


@router.delete("/{build_id}", summary="Delete a build")
async def delete_build(build_id: str):
    """Delete a built model record."""
    if build_storage.delete_build(build_id):
        logger.log("model", "INFO", "Build deleted", {"build_id": build_id})
        return {"message": "Build deleted"}
    raise HTTPException(status_code=404, detail=f"Build '{build_id}' not found")
