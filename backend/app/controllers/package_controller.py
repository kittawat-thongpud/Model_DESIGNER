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
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Literal

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
_url_tasks: dict[str, dict] = {}
_url_tasks_lock = threading.Lock()


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


class LocalPackagePeekRequest(BaseModel):
    local_path: str = Field(..., description="Absolute path to .mdpkg file on server")


@router.post("/peek/local", summary="Preview package from local path")
async def peek_package_local(body: LocalPackagePeekRequest):
    """Preview a .mdpkg package on the server's local filesystem without importing."""
    src_path = Path(body.local_path)
    logger.log("system", "INFO", "Package peek local", {"path": str(src_path), "exists": src_path.exists()})

    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {body.local_path}")
    if not src_path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {body.local_path}")

    try:
        data = src_path.read_bytes()
        logger.log("system", "INFO", "Package peek local read", {"bytes": len(data)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    result = package_service.peek_package(data)
    if "error" in result:
        logger.log("system", "WARNING", "Package peek local failed", {"error": result["error"], "bytes": len(data)})
        raise HTTPException(status_code=400, detail=result["error"])
    return result


class UrlPackagePeekRequest(BaseModel):
    url: str = Field(..., description="HTTP(S) URL to download .mdpkg from")
    timeout: int = Field(300, ge=10, le=3600, description="Download timeout in seconds")


class UrlPackageTaskRequest(BaseModel):
    url: str = Field(..., description="HTTP(S) URL to download .mdpkg from")
    rename_map: dict[str, str] = Field(default_factory=dict, description="JSON object {old_weight_id: new_display_name}")
    include_jobs: bool = Field(default=False, description="Also import job records")
    timeout: int = Field(300, ge=10, le=3600, description="Download timeout in seconds")
    action: Literal["peek", "import"] = "peek"


def _convert_google_drive_url(url: str) -> str:
    """Convert Google Drive share/download URLs to direct download URLs."""
    import re

    # Pattern 1: drive.google.com/file/d/FILE_ID/view
    file_match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', url)
    if file_match:
        file_id = file_match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    # Pattern 2: drive.google.com/open?id=FILE_ID
    open_match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
    if open_match and 'drive.google.com' in url:
        file_id = open_match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    # Pattern 3: googleusercontent.com/download - already direct but need confirm
    if 'googleusercontent.com' in url and 'download' in url:
        # Keep as-is but will need to handle confirm token
        return url

    return url


def _set_url_task(task_id: str, **updates) -> None:
    with _url_tasks_lock:
        state = _url_tasks.setdefault(task_id, {})
        state["task_id"] = task_id
        state.update(updates)
        state["updated_at"] = time.time()


def _get_url_task(task_id: str) -> dict:
    with _url_tasks_lock:
        state = _url_tasks.get(task_id)
        if not state:
            raise HTTPException(status_code=404, detail=f"Package URL task not found: {task_id}")
        public = {k: v for k, v in state.items() if k not in {"file_path"}}
    return public


def _download_package_url_to_file(task_id: str, url: str, timeout: int) -> Path:
    """Download URL to a temp file while updating task progress.

    This intentionally runs outside the request lifecycle. A reverse-proxy 524
    can no longer kill the server route or hide progress from the UI.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    session = requests.Session()
    session.headers.update(headers)

    def _stream_to_file(download_url: str, suffix: str = ".mdpkg") -> tuple[Path, bytes]:
        resp = session.get(download_url, stream=True, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        content_type = resp.headers.get("content-type", "")
        fd, tmp_name = tempfile.mkstemp(prefix=f"mdpkg_url_{task_id}_", suffix=suffix)
        tmp_path = Path(tmp_name)
        first = bytearray()
        downloaded = 0
        _set_url_task(
            task_id,
            status="downloading",
            message="Downloading package...",
            bytes_total=total,
            bytes_downloaded=0,
            content_type=content_type,
            progress=5,
        )
        with open(fd, "wb") as out:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                if len(first) < 8192:
                    first.extend(chunk[: 8192 - len(first)])
                out.write(chunk)
                downloaded += len(chunk)
                progress = 5 + int((downloaded / total) * 70) if total > 0 else min(70, 5 + downloaded // (1024 * 1024))
                _set_url_task(
                    task_id,
                    bytes_downloaded=downloaded,
                    bytes_total=total,
                    progress=min(progress, 75),
                    message=f"Downloaded {downloaded / (1024 * 1024):.1f} MB"
                    + (f" / {total / (1024 * 1024):.1f} MB" if total else ""),
                )
        return tmp_path, bytes(first)

    tmp_path, head = _stream_to_file(url)

    # Google Drive can return a small HTML confirmation page first.
    if b"google.com" in head[:5000] and (
        b"virus" in head[:5000].lower()
        or b"confirm" in head[:5000].lower()
        or b"download_warning" in head[:5000]
    ):
        import re

        preview = tmp_path.read_bytes()[:20000].decode("utf-8", errors="ignore")
        confirm_match = re.search(r"confirm=([a-zA-Z0-9_-]+)", preview)
        if confirm_match:
            tmp_path.unlink(missing_ok=True)
            confirm_url = f"{url}&confirm={confirm_match.group(1)}"
            _set_url_task(task_id, message="Google Drive confirmation accepted; retrying download...", progress=3)
            tmp_path, head = _stream_to_file(confirm_url)

    if head.startswith(b"<!DOCTYPE") or head.startswith(b"<html") or b"<!DOCTYPE html" in head[:1000]:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("URL returned HTML page instead of a package file. The file may require authentication or confirmation.")

    _set_url_task(task_id, file_path=str(tmp_path), progress=78, message="Download complete; reading package...")
    return tmp_path


def _run_url_package_task(task_id: str, body: UrlPackageTaskRequest) -> None:
    url = _convert_google_drive_url(body.url)
    tmp_path: Path | None = None
    try:
        _set_url_task(
            task_id,
            status="queued",
            action=body.action,
            source_url=body.url[:160],
            progress=0,
            message="Queued",
            bytes_downloaded=0,
            bytes_total=0,
            started_at=time.time(),
        )
        tmp_path = _download_package_url_to_file(task_id, url, body.timeout)
        data = tmp_path.read_bytes()
        _set_url_task(task_id, progress=82, message="Parsing package...")

        if body.action == "peek":
            result = package_service.peek_package(data)
            if "error" in result:
                raise RuntimeError(str(result["error"]))
            _set_url_task(task_id, status="completed", progress=100, message="Preview ready", result=result)
            return

        _set_url_task(task_id, progress=88, message="Importing package...")
        result = package_service.import_package(data, rename_map=body.rename_map, include_jobs=body.include_jobs)
        if result.errors and not result.weights_imported and not result.jobs_imported:
            raise RuntimeError(str(result.errors))
        result_dict = result.to_dict()
        logger.log("system", "INFO", "Package imported from URL task", {**result_dict, "task_id": task_id, "source_url": url[:100] + "..."})
        _set_url_task(task_id, status="completed", progress=100, message="Import complete", result=result_dict)
    except Exception as e:
        logger.log("system", "ERROR", "Package URL task failed", {"task_id": task_id, "error": str(e)})
        _set_url_task(task_id, status="failed", progress=100, message=str(e), error=str(e))
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


@router.post("/url-task", summary="Start async package URL peek/import")
async def start_package_url_task(body: UrlPackageTaskRequest):
    """Start URL package download in the background and return immediately."""
    task_id = uuid.uuid4().hex[:12]
    _set_url_task(task_id, status="queued", action=body.action, progress=0, message="Queued")
    thread = threading.Thread(target=_run_url_package_task, args=(task_id, body), daemon=True)
    thread.start()
    return {"task_id": task_id, "status": "queued", "progress": 0, "message": "Queued"}


@router.get("/url-task/{task_id}", summary="Get package URL task progress")
async def get_package_url_task(task_id: str):
    return _get_url_task(task_id)


@router.post("/peek/url", summary="Preview package from download URL")
async def peek_package_url(body: UrlPackagePeekRequest):
    """Download and preview a .mdpkg package from URL without importing."""
    url = _convert_google_drive_url(body.url)

    # Google Drive special handling
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        logger.log("system", "INFO", "Package peek URL download start", {"url": url[:80]})

        session = requests.Session()
        session.headers.update(headers)

        # First request - may get virus scan warning page for large files
        resp = session.get(url, stream=True, timeout=body.timeout, allow_redirects=True)
        resp.raise_for_status()

        content_type = resp.headers.get('content-type', '')
        logger.log("system", "INFO", "Package peek URL response", {"content_type": content_type, "status": resp.status_code})

        data = b""
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                data += chunk

        logger.log("system", "INFO", "Package peek URL download complete", {"bytes": len(data)})

        # Check if we got Google Drive virus scan confirmation page
        if b'google.com' in data[:5000] and (b'virus' in data[:5000].lower() or b'confirm' in data[:5000].lower() or b'download_warning' in data[:5000]):
            # Try to extract confirm token and retry
            import re
            confirm_match = re.search(r'confirm=([a-zA-Z0-9_-]+)', data.decode('utf-8', errors='ignore'))
            if confirm_match:
                confirm_token = confirm_match.group(1)
                logger.log("system", "INFO", "Google Drive confirm token found", {"token": confirm_token[:20]})

                # Retry with confirm token
                confirm_url = f"{url}&confirm={confirm_token}"
                resp2 = session.get(confirm_url, stream=True, timeout=body.timeout)
                resp2.raise_for_status()

                data = b""
                for chunk in resp2.iter_content(chunk_size=8192):
                    if chunk:
                        data += chunk

                logger.log("system", "INFO", "Google Drive retry download complete", {"bytes": len(data)})

        # Check if still got HTML
        if data.startswith(b'<!DOCTYPE') or data.startswith(b'<html') or b'<!DOCTYPE html' in data[:1000]:
            logger.log("system", "WARNING", "URL returned HTML", {"preview": data[:200].decode('utf-8', errors='ignore')})
            raise HTTPException(status_code=400, detail="URL returned HTML page instead of file. The file may require authentication or confirmation.")

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail=f"Download timeout after {body.timeout}s")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")

    result = package_service.peek_package(data)
    if "error" in result:
        logger.log("system", "WARNING", "Package peek URL failed", {"error": result["error"], "bytes": len(data)})
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
    url = _convert_google_drive_url(body.url)

    # Google Drive special handling
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        logger.log("system", "INFO", "Starting package download", {
            "url": url[:100] + "..." if len(url) > 100 else url,
            "timeout": body.timeout,
        })

        session = requests.Session()
        session.headers.update(headers)

        resp = session.get(url, stream=True, timeout=body.timeout, allow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get('content-type', '')
        logger.log("system", "INFO", "Package download response", {"content_type": content_type, "status": resp.status_code})

        data = b""
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                data += chunk

        logger.log("system", "INFO", "Package download complete", {"bytes": len(data)})

        # Handle Google Drive virus scan confirmation
        if b'google.com' in data[:5000] and (b'virus' in data[:5000].lower() or b'confirm' in data[:5000].lower()):
            import re
            confirm_match = re.search(r'confirm=([a-zA-Z0-9_-]+)', data.decode('utf-8', errors='ignore'))
            if confirm_match:
                confirm_token = confirm_match.group(1)
                confirm_url = f"{url}&confirm={confirm_token}"
                resp2 = session.get(confirm_url, stream=True, timeout=body.timeout)
                resp2.raise_for_status()
                data = b""
                for chunk in resp2.iter_content(chunk_size=8192):
                    if chunk:
                        data += chunk
                logger.log("system", "INFO", "Google Drive retry complete", {"bytes": len(data)})

        # Check if still got HTML
        if data.startswith(b'<!DOCTYPE') or data.startswith(b'<html') or b'<!DOCTYPE html' in data[:1000]:
            raise HTTPException(status_code=400, detail="URL returned HTML page instead of file.")

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail=f"Download timeout after {body.timeout}s")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")

    result = package_service.import_package(data, rename_map=body.rename_map, include_jobs=body.include_jobs)

    if result.errors and not result.weights_imported and not result.jobs_imported:
        raise HTTPException(status_code=400, detail=result.errors)

    logger.log("system", "INFO", "Package imported from URL", {**result.to_dict(), "source_url": url[:100] + "..."})
    return result.to_dict()
