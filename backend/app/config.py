"""
Centralized configuration — all paths, env vars, and settings in one place.

All data directories are consolidated under DATA_DIR (backend/data/).
Environment variables:
  - DATA_DIR:           override data root (default: backend/data/)
  - LOG_LEVEL:          minimum log level (default: INFO)
  - LOG_RETENTION_DAYS: auto-cleanup threshold (default: 30)
"""
from __future__ import annotations
import os
from pathlib import Path

# ─── Root directories ────────────────────────────────────────────────────────

# Project root: Model_DESIGNER/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Backend root: Model_DESIGNER/backend/
BACKEND_ROOT = Path(__file__).parent.parent

# Data root: all persistent data lives here
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BACKEND_ROOT / "data")))

# ─── Data sub-directories ────────────────────────────────────────────────────

MODELS_DIR   = DATA_DIR / "models"
MODULES_DIR  = DATA_DIR / "modules"        # custom nn.Module blocks
JOBS_DIR     = DATA_DIR / "jobs"
WEIGHTS_DIR  = DATA_DIR / "weights"
LOGS_DIR     = DATA_DIR / "logs"
EXPORTS_DIR  = DATA_DIR / "exports"
DATASETS_DIR     = DATA_DIR / "datasets"
SPLITS_DIR       = DATA_DIR / "splits"

# Ensure all directories exist
for _d in (MODELS_DIR, MODULES_DIR, JOBS_DIR, WEIGHTS_DIR,
           LOGS_DIR, EXPORTS_DIR, DATASETS_DIR, SPLITS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Logging ─────────────────────────────────────────────────────────────────

LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
LOG_RETENTION_DAYS = int(os.environ.get("LOG_RETENTION_DAYS", "30"))

# ─── CORS ─────────────────────────────────────────────────────────────────────

_DEFAULT_CORS = "http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:5174"
CORS_ORIGINS: list[str] = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", _DEFAULT_CORS).split(",") if o.strip()
]

# ─── App metadata ────────────────────────────────────────────────────────────

APP_NAME = "Model DESIGNER API"
APP_VERSION = "2.0.0"
