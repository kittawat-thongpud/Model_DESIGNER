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

from .services.config_service import get_effective_config

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

_EFFECTIVE_CONFIG = get_effective_config()
LOG_LEVEL = str(_EFFECTIVE_CONFIG.get("logging", {}).get("level", "DEBUG")).upper()
LOG_RETENTION_DAYS = int(_EFFECTIVE_CONFIG.get("logging", {}).get("retention_days", 30))

# ─── CORS ─────────────────────────────────────────────────────────────────────

CORS_ORIGINS: list[str] = [
    str(origin).strip()
    for origin in _EFFECTIVE_CONFIG.get("app", {}).get("cors_origins", [])
    if str(origin).strip()
]

# ─── App metadata ────────────────────────────────────────────────────────────

APP_NAME = str(_EFFECTIVE_CONFIG.get("app", {}).get("name", "Model DESIGNER API"))
APP_VERSION = str(_EFFECTIVE_CONFIG.get("app", {}).get("version", "2.0.0"))
