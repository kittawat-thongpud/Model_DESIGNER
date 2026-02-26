"""
Model DESIGNER — FastAPI Backend
=================================
Visual PyTorch model builder API with ReDoc documentation.

Modules:
  - Models     (/api/models)   — CRUD + code generation
  - Datasets   (/api/datasets) — Dataset info + preview
  - Jobs       (/api/jobs)     — Training job queue + logs
  - Weights    (/api/weights)  — Trained weight registry
  - Training   (/api/train)    — Start training + validation
  - Logs       (/api/logs)     — Structured logging
"""
from __future__ import annotations
import time
import warnings
try:
    from numpy import VisibleDeprecationWarning
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .controllers.model_controller import router as model_router
from .controllers.dataset_controller import router as dataset_router
from .controllers.job_controller import router as job_router
from .controllers.weight_controller import router as weight_router
from .controllers.build_controller import router as build_router
from .controllers.weight_snapshot_controller import router as snapshot_router
from .controllers.package_controller import router as package_router
from .schemas.model_schema import TrainRequest, TrainStatusResponse, ValidateRequest, ValidateResponse, PredictRequest, PredictResponse
from .services import trainer
from .services.validator import validate_model
from . import logging_service as logger
from . import storage


# ─── System Logging Middleware ────────────────────────────────────────────────

class SystemLogMiddleware(BaseHTTPMiddleware):
    """Auto-logs every HTTP request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = round((time.time() - start) * 1000, 1)

        # Skip logging for docs/static/favicon
        path = request.url.path
        if path in ("/docs", "/redoc", "/openapi.json", "/favicon.ico"):
            return response

        logger.log("system", "INFO", f"{request.method} {path}", {
            "status": response.status_code,
            "duration_ms": duration_ms,
            "client": request.client.host if request.client else "unknown",
        })
        return response


# ─── App creation ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Model DESIGNER API",
    description="Backend API for the visual PyTorch model builder. "
                "Design neural networks visually, generate code, train, and validate.",
    version="2.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc
    openapi_tags=[
        {"name": "Models", "description": "Model graph CRUD and PyTorch code generation"},
        {"name": "Builds", "description": "Built (compiled) model registry with generated code"},
        {"name": "Datasets", "description": "Available datasets, metadata, and preview"},
        {"name": "Jobs", "description": "Training job queue, history, and per-job logs"},
        {"name": "Weights", "description": "Trained weight registry with parent model tracking"},
        {"name": "Training", "description": "Start training, validation, and status polling"},
        {"name": "Logs", "description": "System-wide structured application logs"},
    ],
)

# ─── Middleware ───────────────────────────────────────────────────────────────

app.add_middleware(SystemLogMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register module routers ─────────────────────────────────────────────────

app.include_router(model_router)
app.include_router(build_router)
app.include_router(dataset_router)
app.include_router(job_router)
app.include_router(weight_router)
app.include_router(weight_router)
app.include_router(snapshot_router)
app.include_router(package_router)



# ─── Training routes (tagged "Training") ─────────────────────────────────────

@app.post("/api/train", tags=["Training"], summary="Start a training job")
async def start_training(req: TrainRequest):
    """Launch a background training job for a saved model."""
    # Look up model name for job record
    try:
        _, graph = storage.load_model(req.model_id)
        model_name = graph.meta.name
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found")

    config = req.model_dump()
    # Map legacy alias
    if "learning_rate" in config and "lr0" not in config:
        config["lr0"] = config["learning_rate"]
    job_id = trainer.start_training(req.model_id, model_name, config)
    logger.log("training", "INFO", f"Training job created", {"job_id": job_id, "model_id": req.model_id})
    return {"job_id": job_id, "status": "started"}


@app.get("/api/train/{job_id}/status", tags=["Training"], response_model=TrainStatusResponse, summary="Get training status")
async def get_train_status(job_id: str):
    """Poll the status and metrics of a running training job."""
    status = trainer.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return TrainStatusResponse(**{k: v for k, v in status.items() if k in TrainStatusResponse.model_fields})


@app.post("/api/train/{job_id}/stop", tags=["Training"], summary="Stop a training job")
async def stop_training(job_id: str):
    """Request a running training job to stop."""
    if trainer.stop_training(job_id):
        return {"message": "Stop signal sent"}
    raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")


@app.post("/api/predict", tags=["Training"], response_model=PredictResponse, summary="Run model inference")
async def predict(req: PredictRequest):
    """Run an interactive prediction on a single image."""
    from .services import inference
    try:
        return inference.run_prediction(req.model_id, req.image_base64, req.weight_id)
    except Exception as e:
        import traceback
        logger.log("training", "ERROR", f"Prediction failed: {e}", {"traceback": traceback.format_exc()})
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/validate", tags=["Training"], response_model=ValidateResponse, summary="Validate a trained model")
async def validate(req: ValidateRequest):
    """Run validation on a trained model and return metrics."""
    try:
        return validate_model(req.model_id, req.dataset, req.batch_size)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── Log routes (tagged "Logs") ──────────────────────────────────────────────

@app.get("/api/logs", tags=["Logs"], summary="Fetch application logs")
async def get_logs(
    category: str | None = None,
    level: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """Retrieve structured log entries with optional filters."""
    return logger.get_logs(category=category, level=level, limit=limit, offset=offset)


@app.delete("/api/logs", tags=["Logs"], summary="Clear all logs")
async def clear_logs():
    """Delete all log entries."""
    count = logger.clear_logs()
    return {"message": f"Cleared {count} log entries"}


# ─── Dashboard stats ─────────────────────────────────────────────────────────

@app.get("/api/stats", tags=["Logs"], summary="Dashboard statistics")
async def get_stats():
    """Return aggregated stats for the dashboard."""
    from .services import job_storage, weight_storage

    models = storage.list_models()
    jobs = job_storage.list_jobs()
    weights = weight_storage.list_weights()

    active_jobs = sum(1 for j in jobs if j.get("status") in ("pending", "running"))

    return {
        "total_models": len(models),
        "total_jobs": len(jobs),
        "active_jobs": active_jobs,
        "total_weights": len(weights),
    }


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "app": "Model DESIGNER API",
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }
