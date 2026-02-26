"""
Model DESIGNER — FastAPI Backend
=================================
Ultralytics-native model designer, module builder, and training platform.

Three designers:
  - Module Designer: custom nn.Module blocks
  - Model Designer: Ultralytics YAML [from, repeats, module, args]
  - Train Designer: model.train() config + live monitoring

Uses the create_app() factory pattern for clean initialization.
All paths and settings are centralized in config.py.
"""
from __future__ import annotations
import json
import math
import time
import warnings
from typing import Any
try:
    from numpy import VisibleDeprecationWarning
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from . import logging_service as logger
from .config import APP_NAME, APP_VERSION, CORS_ORIGINS


import sys
# Keep sys.argv empty to prevent Ultralytics CLI parsing
sys.argv = []
# ─── Safe JSON Response (replaces NaN/Inf with None) ─────────────────────────

def _sanitize(obj: Any) -> Any:
    """Recursively replace inf/nan floats with None so JSON never fails."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    """JSONResponse that silently converts NaN/Inf to None."""
    def render(self, content: Any) -> bytes:
        return json.dumps(
            _sanitize(content),
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")


# ─── System Logging Middleware ────────────────────────────────────────────────

class SystemLogMiddleware(BaseHTTPMiddleware):
    """Auto-logs every HTTP request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = round((time.time() - start) * 1000, 1)

        path = request.url.path
        if path in ("/docs", "/redoc", "/openapi.json", "/favicon.ico"):
            return response

        logger.log("system", "INFO", f"{request.method} {path}", {
            "status": response.status_code,
            "duration_ms": duration_ms,
            "client": request.client.host if request.client else "unknown",
        })
        return response


# ─── App Factory ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    application = FastAPI(
        title=APP_NAME,
        description="Ultralytics-native model designer and training platform. "
                    "Design model architectures visually, create custom modules, "
                    "and train with full monitoring.",
        version=APP_VERSION,
        default_response_class=SafeJSONResponse,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {"name": "Models", "description": "Model YAML architecture CRUD"},
            {"name": "Modules", "description": "Custom nn.Module block designer + catalog"},
            {"name": "Training", "description": "Ultralytics training jobs + monitoring"},
            {"name": "Datasets", "description": "Dataset management and preview"},
            {"name": "Jobs", "description": "Training job listing and logs"},
            {"name": "Weights", "description": "Trained weight management"},
            {"name": "Logs", "description": "System-wide structured logs"},
            {"name": "Streaming", "description": "Server-Sent Events for real-time updates"},
        ],
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    application.add_middleware(SystemLogMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────────────────────
    from .controllers.model_controller import router as model_router
    from .controllers.module_controller import router as module_router
    from .controllers.train_controller import router as train_router
    from .controllers.dataset_controller import router as dataset_router
    from .controllers.dataset_samples_controller import router as dataset_samples_router
    from .controllers.job_controller import router as job_router
    from .controllers.weight_controller import router as weight_router
    from .controllers.weight_snapshot_controller import router as snapshot_router
    from .controllers.log_controller import router as log_router
    from .controllers.stats_controller import router as stats_router
    from .controllers.stream_controller import router as stream_router
    from .controllers.system_controller import router as system_router

    for router in (
        model_router, module_router, train_router,
        dataset_router, dataset_samples_router, job_router,
        weight_router, snapshot_router, log_router, stats_router, stream_router, system_router
    ):
        application.include_router(router)

    # ── Startup: discover plugins ───────────────────────────────────────────
    from .plugins.loader import discover_plugins
    counts = discover_plugins()
    logger.log("system", "INFO", f"Plugins discovered: {counts}")

    # ── Startup: clean up stale running jobs ────────────────────────────────
    try:
        from .services.ultra_trainer import cleanup_stale_jobs
        cleanup_stale_jobs()
    except Exception as e:
        logger.log("system", "WARNING", f"Stale job cleanup failed: {e}")
    
    # ── Startup: start worker monitor ───────────────────────────────────────
    try:
        from .services.worker_monitor import start_monitor
        monitor = start_monitor(check_interval=60)  # Check every 60 seconds
        
        # Add logging callback
        def on_zombie_cleanup(result):
            logger.log("system", "WARNING", 
                      f"Zombie workers detected and cleaned: {result['cleaned']}")
        
        monitor.add_callback(on_zombie_cleanup)
        logger.log("system", "INFO", "Worker monitor started (check_interval=60s)")
    except Exception as e:
        logger.log("system", "WARNING", f"Worker monitor startup failed: {e}")
    
    # ── Shutdown: stop worker monitor ────────────────────────────────────────
    @application.on_event("shutdown")
    async def shutdown_event():
        try:
            from .services.worker_monitor import stop_monitor
            stop_monitor(timeout=5.0)
            logger.log("system", "INFO", "Worker monitor stopped")
        except Exception as e:
            logger.log("system", "WARNING", f"Worker monitor shutdown failed: {e}")

    # ── Frontend static files (production) ───────────────────────────────────
    from pathlib import Path as _Path
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    _dist = _Path(__file__).parent.parent.parent / "frontend" / "dist"
    if _dist.exists():
        # Serve static assets (JS/CSS/images) under /assets
        application.mount("/assets", StaticFiles(directory=str(_dist / "assets")), name="assets")

        # SPA fallback — any non-API route returns index.html
        @application.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):
            index = _dist / "index.html"
            return FileResponse(str(index))
    else:
        # No dist — return JSON info at root
        @application.get("/", include_in_schema=False)
        async def root():
            return {
                "app": APP_NAME,
                "version": APP_VERSION,
                "docs": "/docs",
                "redoc": "/redoc",
            }

    return application


# ─── Module-level app instance (used by uvicorn) ────────────────────────────

app = create_app()
