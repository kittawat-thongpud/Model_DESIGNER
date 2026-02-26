"""
Structured logging service.
Logs are JSON lines stored in the logs/ directory.
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "app.jsonl"

Category = Literal["model", "dataset", "training", "system"]
Level = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def log(
    category: Category,
    level: Level,
    message: str,
    data: dict | None = None,
) -> dict:
    """Write a structured log entry."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "category": category,
        "level": level,
        "message": message,
        "data": data or {},
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def get_logs(
    category: str | None = None,
    level: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Read structured log entries with optional filters."""
    if not LOG_FILE.exists():
        return []

    entries: list[dict] = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if category and entry.get("category") != category:
                continue
            if level and entry.get("level") != level:
                continue
            entries.append(entry)

    # Most recent first
    entries.reverse()
    return entries[offset: offset + limit]


def clear_logs() -> int:
    """Clear all logs. Returns number of entries deleted."""
    count = 0
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            count = sum(1 for _ in f)
        LOG_FILE.unlink()
    return count
