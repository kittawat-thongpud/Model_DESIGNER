"""
Structured logging service — per-category log directories, daily rotation,
configurable minimum level, structured metadata, auto-cleanup, SSE broadcasting.

Directory layout:
  backend/data/logs/
    system/system-YYYY-MM-DD.jsonl
    model/model-YYYY-MM-DD.jsonl
    training/training-YYYY-MM-DD.jsonl   (lifecycle only: start/complete/fail)
    dataset/dataset-YYYY-MM-DD.jsonl

  Per-job training logs are managed by job_storage as {job_id}.log.jsonl
"""
from __future__ import annotations
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from .config import LOGS_DIR as LOG_DIR
from .config import LOG_LEVEL, LOG_RETENTION_DAYS

# Legacy alias — kept for clear_logs cleanup of stale files
LOG_FILE = LOG_DIR / "app.jsonl"

# ─── Configuration ──────────────────────────────────────────────────────────

Category = Literal["model", "dataset", "training", "system"]
Level = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}

# Runtime-configurable minimum level
_min_level: str = LOG_LEVEL
_lock = threading.Lock()


def set_min_level(level: str) -> None:
    """Set the minimum log level at runtime."""
    global _min_level
    _min_level = level.upper()


def get_min_level() -> str:
    return _min_level


# ─── Category directories ───────────────────────────────────────────────────

_CATEGORIES = ("system", "model", "training", "dataset")

for _cat in _CATEGORIES:
    (LOG_DIR / _cat).mkdir(parents=True, exist_ok=True)


def _category_dir(category: str) -> Path:
    d = LOG_DIR / category
    d.mkdir(parents=True, exist_ok=True)
    return d


def _daily_file(category: str, date: datetime | None = None) -> Path:
    """Return the daily log file for a category."""
    dt = date or datetime.utcnow()
    day_str = dt.strftime("%Y-%m-%d")
    return _category_dir(category) / f"{category}-{day_str}.jsonl"


# ─── Core log function ──────────────────────────────────────────────────────

def log(
    category: Category,
    level: Level,
    message: str,
    data: dict | None = None,
    *,
    job_id: str | None = None,
    model_id: str | None = None,
    component: str | None = None,
) -> dict:
    """
    Write a structured log entry.

    Backward-compatible: existing callers pass (category, level, message, data).
    New callers can also pass job_id, model_id, component as keyword args.
    """
    # Level gate
    if _LEVEL_ORDER.get(level, 1) < _LEVEL_ORDER.get(_min_level, 1):
        return {}

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "category": category,
        "level": level,
        "message": message,
        "data": data or {},
    }

    # Structured metadata — only include if provided
    if job_id:
        entry["job_id"] = job_id
    if model_id:
        entry["model_id"] = model_id
    if component:
        entry["component"] = component

    # Also extract job_id/model_id from data dict (backward compat)
    if not job_id and isinstance(data, dict):
        if "job_id" in data:
            entry["job_id"] = data["job_id"]
        if "model_id" in data:
            entry["model_id"] = data["model_id"]

    line = json.dumps(entry, default=str) + "\n"

    # Write to category-specific daily file
    with _lock:
        cat_file = _daily_file(category)
        with open(cat_file, "a") as f:
            f.write(line)

    # Broadcast to SSE subscribers (lazy import to avoid circular)
    try:
        from .services.event_bus import publish_sync
        from .constants import SYSTEM_LOG_CHANNEL
        publish_sync(SYSTEM_LOG_CHANNEL, entry)
    except Exception:
        pass

    return entry


# ─── Query / Read ────────────────────────────────────────────────────────────

def get_logs(
    category: str | None = None,
    level: str | None = None,
    limit: int = 100,
    offset: int = 0,
    *,
    job_id: str | None = None,
    model_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    days: int = 7,
) -> list[dict]:
    """
    Read structured log entries with filters.

    Params:
        category: filter by category (None = all)
        level: filter by exact level
        limit: max entries to return
        offset: skip first N matching entries
        job_id: filter by job_id field
        model_id: filter by model_id field
        since: ISO timestamp lower bound (inclusive)
        until: ISO timestamp upper bound (inclusive)
        days: how many days of log files to scan (default 7)
    """
    # Collect daily files within range
    now = datetime.utcnow()
    files_to_scan: list[Path] = []
    cats_to_scan = [category] if (category and category in _CATEGORIES) else list(_CATEGORIES)
    for cat in cats_to_scan:
        for d in range(days):
            dt = now - timedelta(days=d)
            f = _daily_file(cat, dt)
            if f.exists():
                files_to_scan.append(f)

    entries: list[dict] = []
    for fp in files_to_scan:
        try:
            with open(fp) as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        entry = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    # Apply filters
                    if category and entry.get("category") != category:
                        continue
                    if level and entry.get("level") != level:
                        continue
                    if job_id and entry.get("job_id") != job_id and entry.get("data", {}).get("job_id") != job_id:
                        continue
                    if model_id and entry.get("model_id") != model_id and entry.get("data", {}).get("model_id") != model_id:
                        continue
                    if since and entry.get("timestamp", "") < since:
                        continue
                    if until and entry.get("timestamp", "") > until:
                        continue

                    entries.append(entry)
        except Exception:
            continue

    # Most recent first
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    return entries[offset: offset + limit]


# ─── Clear / Cleanup ────────────────────────────────────────────────────────

def clear_logs() -> int:
    """Clear all logs across all categories. Returns total entries deleted."""
    count = 0

    # Clear legacy file
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            count += sum(1 for _ in f)
        LOG_FILE.unlink()

    # Clear daily files in all category dirs
    for cat in _CATEGORIES:
        cat_dir = LOG_DIR / cat
        if cat_dir.is_dir():
            for fp in cat_dir.glob("*.jsonl"):
                try:
                    with open(fp) as f:
                        count += sum(1 for _ in f)
                    fp.unlink()
                except Exception:
                    continue

    return count


def cleanup_old_logs(retention_days: int | None = None) -> int:
    """Delete log files older than retention_days. Returns number of files deleted."""
    days = retention_days if retention_days is not None else LOG_RETENTION_DAYS
    if days <= 0:
        return 0

    cutoff = datetime.utcnow() - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    deleted = 0

    for cat in _CATEGORIES:
        cat_dir = LOG_DIR / cat
        if not cat_dir.is_dir():
            continue
        for fp in cat_dir.glob("*.jsonl"):
            # Extract date from filename: e.g. "system-2026-02-12.jsonl"
            stem = fp.stem  # "system-2026-02-12"
            parts = stem.rsplit("-", 3)
            if len(parts) >= 4:
                file_date = f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
                if file_date < cutoff_str:
                    try:
                        fp.unlink()
                        deleted += 1
                    except Exception:
                        continue

    return deleted


def get_log_stats() -> dict:
    """Return summary stats about log storage."""
    stats: dict = {
        "min_level": _min_level,
        "retention_days": LOG_RETENTION_DAYS,
        "categories": {},
        "total_files": 0,
        "total_size_bytes": 0,
    }

    for cat in _CATEGORIES:
        cat_dir = LOG_DIR / cat
        if not cat_dir.is_dir():
            continue
        files = list(cat_dir.glob("*.jsonl"))
        size = sum(f.stat().st_size for f in files if f.exists())
        stats["categories"][cat] = {
            "file_count": len(files),
            "size_bytes": size,
        }
        stats["total_files"] += len(files)
        stats["total_size_bytes"] += size

    return stats
