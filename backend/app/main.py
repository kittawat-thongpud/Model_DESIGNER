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
from starlette.routing import Route

from . import logging_service as logger
from .config import APP_NAME, APP_VERSION, CORS_ORIGINS
from .services.config_service import get_monitoring_config


import sys
import logging as _logging
# Keep sys.argv empty to prevent Ultralytics CLI parsing
sys.argv = []

# ─── Suppress noisy AssertionError stack traces from POST /mcp/sse ───────────
class _SuppressMcpAssertion(_logging.Filter):
    """Drop the AssertionError log record that Starlette emits when a client
    POSTs to /mcp/sse (GET-only endpoint).  The 405 response is still sent;
    only the server-side traceback is suppressed to keep logs clean."""
    def filter(self, record: _logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("AssertionError" in msg and "mcp" in msg.lower())

_logging.getLogger("starlette.middleware.base").addFilter(_SuppressMcpAssertion())
_logging.getLogger("uvicorn.error").addFilter(_SuppressMcpAssertion())
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


# ─── Disable Conditional Request Middleware ─────────────────────────────────────

class DisableConditionalRequestMiddleware(BaseHTTPMiddleware):
    """Strip If-None-Match and If-Modified-Since headers to prevent 304 responses."""
    
    async def dispatch(self, request: Request, call_next):
        # Remove conditional request headers to force full response
        headers = dict(request.headers)
        headers.pop('if-none-match', None)
        headers.pop('if-modified-since', None)
        headers.pop('if-range', None)
        
        # Create new request with modified headers
        scope = request.scope
        scope['headers'] = [
            (k.lower().encode(), v.encode())
            for k, v in headers.items()
        ]
        
        return await call_next(request)


# ─── System Logging Middleware ────────────────────────────────────────────────

class SystemLogMiddleware(BaseHTTPMiddleware):
    """Auto-logs every HTTP request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/mcp/") or path == "/mcp-http":
            return await call_next(request)

        start = time.time()
        response = await call_next(request)
        duration_ms = round((time.time() - start) * 1000, 1)

        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            if "etag" in response.headers:
                del response.headers["etag"]

        if path in ("/docs", "/redoc", "/openapi.json", "/favicon.ico"):
            return response

        logger.log("system", "INFO", f"{request.method} {path}", {
            "status": response.status_code,
            "duration_ms": duration_ms,
            "client": request.client.host if request.client else "unknown",
        })
        return response


class _ExactPathASGIProxy:
    def __init__(self, app, mount_path: str):
        self.app = app
        self.mount_path = mount_path.rstrip("/")

    async def __call__(self, scope, receive, send):
        child_scope = dict(scope)
        child_scope["path"] = "/"
        child_scope["raw_path"] = b"/"
        child_scope["root_path"] = scope.get("root_path", "") + self.mount_path
        await self.app(child_scope, receive, send)


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
            {"name": "Health", "description": "Liveness and readiness probes"},
        ],
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    application.add_middleware(DisableConditionalRequestMiddleware)
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
    from .controllers.inference_controller import router as inference_router
    from .controllers.benchmark_controller import router as benchmark_router
    from .controllers.package_controller import router as package_router
    from .controllers.health_controller import router as health_router
    from .controllers.plugin_controller import router as plugin_router

    for router in (
        model_router, module_router, train_router,
        dataset_router, dataset_samples_router, job_router,
        weight_router, snapshot_router, log_router, stats_router, stream_router, system_router,
        inference_router, benchmark_router, package_router, health_router,
        plugin_router,
    ):
        application.include_router(router)

    # ── MCP (Model Context Protocol) interface ───────────────────────────────
    try:
        from .mcp.server import create_mcp_app, create_mcp_http_app, mcp as _mcp_instance
        mcp_http_app = create_mcp_http_app()
        application.mount("/mcp", create_mcp_app())
        application.mount("/mcp-http", mcp_http_app)
        application.router.routes.append(
            Route("/mcp-http", endpoint=_ExactPathASGIProxy(mcp_http_app, "/mcp-http"))
        )

        # FastAPI ignores sub-app lifespans on .mount(); we must drive the
        # StreamableHTTPSessionManager task-group ourselves via startup/shutdown.
        import asyncio as _asyncio

        _mcp_state: dict = {"event": None, "task": None}

        @application.on_event("startup")
        async def _start_mcp_http_session_manager():
            shutdown_event = _asyncio.Event()
            _mcp_state["event"] = shutdown_event

            async def _lifespan_task():
                async with _mcp_instance.session_manager.run():
                    await shutdown_event.wait()

            task = _asyncio.get_event_loop().create_task(_lifespan_task())
            _mcp_state["task"] = task
            # brief yield so session_manager._task_group is set before first request
            await _asyncio.sleep(0.1)

        @application.on_event("shutdown")
        async def _stop_mcp_http_session_manager():
            ev = _mcp_state.get("event")
            if ev is not None:
                ev.set()
            task = _mcp_state.get("task")
            if task is not None:
                try:
                    await _asyncio.wait_for(task, timeout=5.0)
                except Exception:
                    pass

        logger.log("system", "INFO", "MCP server mounted successfully", {
            "mount_path": "/mcp",
            "sse_endpoint": "/mcp/sse",
            "message_endpoint": "/mcp/messages/",
            "http_endpoint": "/mcp-http",
            "notes": "Connect MCP clients to the SSE endpoint",
        })
    except Exception as e:
        logger.log("system", "WARNING", "MCP server mount failed", {
            "mount_path": "/mcp",
            "sse_endpoint": "/mcp/sse",
            "message_endpoint": "/mcp/messages/",
            "http_endpoint": "/mcp-http",
            "error": str(e),
        })

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
        monitoring_config = get_monitoring_config()
        check_interval = int(monitoring_config.get("worker_check_interval_s", 60))
        monitor = start_monitor(check_interval=check_interval)

        # Add logging callback
        def on_zombie_cleanup(result):
            logger.log("system", "WARNING", 
                      f"Zombie workers detected and cleaned: {result['cleaned']}")

        monitor.add_callback(on_zombie_cleanup)
        logger.log("system", "INFO", f"Worker monitor started (check_interval={check_interval}s)")
    except Exception as e:
        logger.log("system", "WARNING", f"Worker monitor startup failed: {e}")

    # ── Shutdown: stop worker monitor ────────────────────────────────────────
    @application.on_event("shutdown")
    async def shutdown_event():
        monitoring_config = get_monitoring_config()
        stop_timeout = float(monitoring_config.get("worker_stop_timeout_s", 5.0))
        try:
            from .services.worker_monitor import stop_monitor
            stop_monitor(timeout=stop_timeout)
            logger.log("system", "INFO", "Worker monitor stopped")
        except Exception as e:
            logger.log("system", "WARNING", f"Worker monitor shutdown failed: {e}")
        try:
            from .services.ultra_trainer import shutdown_training_workers
            stopped = shutdown_training_workers(timeout=stop_timeout)
            if stopped:
                logger.log("system", "WARNING", "Training workers stopped for shutdown", stopped)
        except Exception as e:
            logger.log("system", "WARNING", f"Training worker shutdown failed: {e}")

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
