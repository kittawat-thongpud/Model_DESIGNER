"""
Package Controller — export / import .mdpkg bundles.

Supports both single-request upload and chunked upload for large packages
that exceed reverse-proxy body-size limits (e.g. RunPod ≈ 100 MB).

Chunked upload flow:
  1. POST /upload/init         → { upload_id }
  2. POST /upload/{id}/chunk   → (index + blob)  × N
  3. POST /upload/{id}/finalize → { upload_id, size, chunks }
  4. POST /peek?upload_id=…    or  POST /import?upload_id=…
"""
from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Query, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field
import requests

from ..services.config_service import get_package_config
from ..services import package_service
from .. import logging_service as logger

router = APIRouter(prefix="/api/packages", tags=["Packages"])
_PACKAGE_DEFAULTS = get_package_config().get("defaults", {})

# In-memory registry of active chunked uploads: upload_id → temp dir path
_active_uploads: dict[str, Path] = {}


# ── Export ────────────────────────────────────────────────────────────────────

@router.get("/weights/{weight_id}/export", summary="Export weight package (.mdpkg)")
async def export_weight_package(
    weight_id: str,
    include_jobs: bool = Query(bool(_PACKAGE_DEFAULTS.get("include_jobs", False)), description="Include training job records in package"),
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
    include_jobs: bool = Query(bool(_PACKAGE_DEFAULTS.get("include_jobs", False)), description="Include training job records in package"),
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


# ── Chunked Upload ────────────────────────────────────────────────────────────

def _get_upload_dir(upload_id: str) -> Path:
    """Return the temp directory for an active chunked upload, or raise 404."""
    d = _active_uploads.get(upload_id)
    if d is None or not d.exists():
        raise HTTPException(status_code=404, detail=f"Upload session not found: {upload_id}")
    return d


def _read_upload_data(upload_id: str) -> bytes:
    """Read the assembled file from a finalized chunked upload."""
    d = _get_upload_dir(upload_id)
    assembled = d / "assembled.mdpkg"
    if not assembled.exists():
        raise HTTPException(status_code=400, detail="Upload not finalized yet — call /upload/{id}/finalize first")
    return assembled.read_bytes()


def _cleanup_upload(upload_id: str) -> None:
    """Remove temp dir for a completed upload."""
    d = _active_uploads.pop(upload_id, None)
    if d and d.exists():
        shutil.rmtree(d, ignore_errors=True)


@router.post("/upload/init", summary="Start a chunked upload session")
async def upload_init(
    filename: str = Form("package.mdpkg"),
    total_chunks: int = Form(1),
    total_size: int = Form(0),
):
    upload_id = uuid.uuid4().hex[:12]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"mdpkg_{upload_id}_"))
    _active_uploads[upload_id] = tmp_dir
    # Write metadata
    meta = {"filename": filename, "total_chunks": total_chunks, "total_size": total_size}
    (tmp_dir / "meta.json").write_text(json.dumps(meta))
    logger.log("system", "INFO", "Chunked upload started",
               {"upload_id": upload_id, "filename": filename, "total_chunks": total_chunks})
    return {"upload_id": upload_id}


@router.post("/upload/{upload_id}/chunk", summary="Upload one chunk")
async def upload_chunk(
    upload_id: str,
    index: int = Form(...),
    data: UploadFile = File(...),
):
    d = _get_upload_dir(upload_id)
    chunk_path = d / f"chunk_{index:06d}"
    content = await data.read()
    chunk_path.write_bytes(content)
    return {"upload_id": upload_id, "index": index, "size": len(content)}


@router.post("/upload/{upload_id}/finalize", summary="Assemble chunks into final file")
async def upload_finalize(upload_id: str):
    d = _get_upload_dir(upload_id)
    meta = json.loads((d / "meta.json").read_text())
    total_chunks = meta["total_chunks"]

    assembled = d / "assembled.mdpkg"
    with open(assembled, "wb") as out:
        for i in range(total_chunks):
            chunk_path = d / f"chunk_{i:06d}"
            if not chunk_path.exists():
                raise HTTPException(status_code=400,
                                    detail=f"Missing chunk {i}/{total_chunks}")
            out.write(chunk_path.read_bytes())

    size = assembled.stat().st_size
    logger.log("system", "INFO", "Chunked upload finalized",
               {"upload_id": upload_id, "size": size, "chunks": total_chunks})
    return {"upload_id": upload_id, "size": size, "chunks": total_chunks}


# ── Peek ──────────────────────────────────────────────────────────────────────

@router.post("/peek", summary="Preview contents of a .mdpkg without importing")
async def peek_package(
    file: Optional[UploadFile] = File(None),
    upload_id: str = Form(default=""),
):
    """
    Returns manifest info (weight list with names/datasets) without writing anything.
    Accepts either a direct file upload OR an upload_id from chunked upload.
    """
    if upload_id:
        data = _read_upload_data(upload_id)
    elif file:
        data = await file.read()
    else:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'upload_id'")

    result = package_service.peek_package(data)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ── Import ────────────────────────────────────────────────────────────────────

@router.post("/import", summary="Import a .mdpkg package")
async def import_package(
    file: Optional[UploadFile] = File(None),
    upload_id: str = Form(default=""),
    rename_map: str = Form(default=str(_PACKAGE_DEFAULTS.get("rename_map", "{}")), description="JSON object {old_weight_id: new_display_name}"),
    include_jobs: bool = Form(default=bool(_PACKAGE_DEFAULTS.get("include_jobs", False)), description="Also import job records"),
):
    """
    Import a .mdpkg archive. Always assigns NEW IDs — never clashes with existing data.
    Accepts either a direct file upload OR an upload_id from chunked upload.
    """
    if upload_id:
        data = _read_upload_data(upload_id)
    elif file:
        data = await file.read()
    else:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'upload_id'")

    try:
        rmap: dict[str, str] = json.loads(rename_map) if rename_map else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="rename_map must be valid JSON")

    result = package_service.import_package(data, rename_map=rmap, include_jobs=include_jobs)

    # Clean up temp upload after successful import
    if upload_id:
        _cleanup_upload(upload_id)

    if result.errors and not result.weights_imported and not result.jobs_imported:
        raise HTTPException(status_code=400, detail=result.errors)

    logger.log("system", "INFO", "Package imported", result.to_dict())
    return result.to_dict()


class LocalPackageImportRequest(BaseModel):
    local_path: str = Field(..., description="Absolute path to .mdpkg file on server")
    rename_map: dict[str, str] = Field(default_factory=dict, description="JSON object {old_weight_id: new_display_name}")
    include_jobs: bool = Field(default=False, description="Also import job records")


@router.post("/import/local", summary="Import package from local server path")
async def import_package_local(body: LocalPackageImportRequest):
    """Import a .mdpkg package already present on the server's local filesystem.

    Bypasses proxy upload limits by reading directly from disk.
    """
    src_path = Path(body.local_path)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {body.local_path}")
    if not src_path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {body.local_path}")

    try:
        data = src_path.read_bytes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    result = package_service.import_package(data, rename_map=body.rename_map, include_jobs=body.include_jobs)

    if result.errors and not result.weights_imported and not result.jobs_imported:
        raise HTTPException(status_code=400, detail=result.errors)

    logger.log("system", "INFO", "Package imported from local", {**result.to_dict(), "source_path": str(src_path)})
    return result.to_dict()


class UrlPackageImportRequest(BaseModel):
    url: str = Field(..., description="HTTP(S) URL to download .mdpkg file from")
    rename_map: dict[str, str] = Field(default_factory=dict, description="JSON object {old_weight_id: new_display_name}")
    include_jobs: bool = Field(default=False, description="Also import job records")
    timeout: int = Field(300, ge=10, le=3600, description="Download timeout in seconds")


@router.post("/import/url", summary="Import package from download URL")
async def import_package_url(body: UrlPackageImportRequest):
    """Download and import a .mdpkg package from an external URL.

    Bypasses proxy upload limits by downloading directly on the server.
    """
    try:
        logger.log("system", "INFO", "Starting package download", {
            "url": body.url[:100] + "..." if len(body.url) > 100 else body.url,
            "timeout": body.timeout,
        })

        with requests.get(body.url, stream=True, timeout=body.timeout) as resp:
            resp.raise_for_status()
            data = b""
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    data += chunk

        logger.log("system", "INFO", "Package download complete", {"bytes": len(data)})

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail=f"Download timeout after {body.timeout}s")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")

    result = package_service.import_package(data, rename_map=body.rename_map, include_jobs=body.include_jobs)

    if result.errors and not result.weights_imported and not result.jobs_imported:
        raise HTTPException(status_code=400, detail=result.errors)

    logger.log("system", "INFO", "Package imported from URL", {**result.to_dict(), "source_url": body.url[:100] + "..."})
    return result.to_dict()
