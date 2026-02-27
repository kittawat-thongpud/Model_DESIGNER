"""
Package Controller — export / import .mdpkg bundles.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Form, HTTPException, Query, UploadFile, File
from fastapi.responses import Response

from ..services import package_service
from .. import logging_service as logger

router = APIRouter(prefix="/api/packages", tags=["Packages"])


# ── Export ────────────────────────────────────────────────────────────────────

@router.get("/weights/{weight_id}/export", summary="Export weight package (.mdpkg)")
async def export_weight_package(
    weight_id: str,
    include_jobs: bool = Query(False, description="Include training job records in package"),
):
    try:
        data, filename = package_service.build_weight_package(weight_id, include_jobs=include_jobs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.log("system", "INFO", "Weight package exported",
               {"weight_id": weight_id, "include_jobs": include_jobs, "size_bytes": len(data)})
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/jobs/{job_id}/export", summary="Export job package (.mdpkg)")
async def export_job_package(
    job_id: str,
    include_jobs: bool = Query(False, description="Include training job records in package"),
):
    try:
        data, filename = package_service.build_job_package(job_id, include_jobs=include_jobs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.log("system", "INFO", "Job package exported",
               {"job_id": job_id, "include_jobs": include_jobs, "size_bytes": len(data)})
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Peek ──────────────────────────────────────────────────────────────────────

@router.post("/peek", summary="Preview contents of a .mdpkg without importing")
async def peek_package(file: UploadFile = File(...)):
    """
    Returns manifest info (weight list with names/datasets) without writing anything.
    Used by the UI to show a rename form before committing the import.
    """
    data = await file.read()
    result = package_service.peek_package(data)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ── Import ────────────────────────────────────────────────────────────────────

@router.post("/import", summary="Import a .mdpkg package")
async def import_package(
    file: UploadFile = File(...),
    rename_map: str = Form(default="{}", description="JSON object {old_weight_id: new_display_name}"),
    include_jobs: bool = Form(default=False, description="Also import job records"),
):
    """
    Import a .mdpkg archive. Always assigns NEW IDs — never clashes with existing data.
    rename_map: JSON string mapping old weight IDs to new display names.
    include_jobs: if True, also import job records bundled in the package.
    """
    data = await file.read()
    try:
        rmap: dict[str, str] = json.loads(rename_map) if rename_map else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="rename_map must be valid JSON")

    result = package_service.import_package(data, rename_map=rmap, include_jobs=include_jobs)

    if result.errors and not result.weights_imported and not result.jobs_imported:
        raise HTTPException(status_code=400, detail=result.errors)

    logger.log("system", "INFO", "Package imported", result.to_dict())
    return result.to_dict()
