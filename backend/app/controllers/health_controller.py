"""
Health Controller — liveness and readiness endpoints for production monitoring.

GET /api/health         — liveness probe (always fast, no heavy checks)
GET /api/health/ready   — readiness probe (checks storage dirs, plugin counts)
"""
from __future__ import annotations
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..config import APP_VERSION, DATA_DIR, JOBS_DIR, DATASETS_DIR, WEIGHTS_DIR, LOGS_DIR

router = APIRouter(prefix="/api/health", tags=["Health"])

_START_TIME = time.time()


@router.get("", summary="Liveness probe")
async def liveness():
    """Minimal liveness check — confirms the process is alive and responding."""
    return {
        "status": "ok",
        "version": APP_VERSION,
        "uptime_s": round(time.time() - _START_TIME, 1),
    }


@router.get("/ready", summary="Readiness probe")
async def readiness():
    """
    Readiness check — verifies that storage directories are accessible and
    plugin discovery completed at startup.

    Returns HTTP 200 if ready, HTTP 503 if not ready.
    """
    checks: dict[str, dict] = {}
    all_ok = True

    # ── Storage directories ──────────────────────────────────────────────────
    for name, path in [
        ("data_dir", DATA_DIR),
        ("jobs_dir", JOBS_DIR),
        ("datasets_dir", DATASETS_DIR),
        ("weights_dir", WEIGHTS_DIR),
        ("logs_dir", LOGS_DIR),
    ]:
        try:
            exists = path.exists()
            writable = False
            if exists:
                test_file = path / ".health_check"
                try:
                    test_file.touch()
                    test_file.unlink()
                    writable = True
                except Exception:
                    pass
            checks[name] = {"ok": exists and writable, "path": str(path)}
            if not (exists and writable):
                all_ok = False
        except Exception as e:
            checks[name] = {"ok": False, "error": str(e)}
            all_ok = False

    # ── Plugin registry ──────────────────────────────────────────────────────
    try:
        from ..plugins.loader import all_dataset_plugins, all_arch_plugins, all_weight_source_plugins
        checks["plugins"] = {
            "ok": True,
            "datasets": len(all_dataset_plugins()),
            "archs": len(all_arch_plugins()),
            "weight_sources": len(all_weight_source_plugins()),
        }
    except Exception as e:
        checks["plugins"] = {"ok": False, "error": str(e)}
        all_ok = False

    # ── Active workers ───────────────────────────────────────────────────────
    try:
        from ..services.ultra_trainer import get_worker_health
        worker_health = get_worker_health()
        checks["workers"] = {
            "ok": True,
            "alive": worker_health.get("alive_workers", 0),
            "total": worker_health.get("total_workers", 0),
        }
    except Exception as e:
        checks["workers"] = {"ok": False, "error": str(e)}

    # ── Platform capabilities ────────────────────────────────────────────────
    try:
        from ..services.platform_caps import get_platform_info
        checks["platform"] = {"ok": True, **get_platform_info()}
    except Exception as e:
        checks["platform"] = {"ok": False, "error": str(e)}

    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ok else "not_ready",
            "version": APP_VERSION,
            "uptime_s": round(time.time() - _START_TIME, 1),
            "checks": checks,
        },
    )
