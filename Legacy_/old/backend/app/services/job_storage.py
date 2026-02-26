"""
Persistent storage for training jobs and their logs.
Each job is stored as a JSON file in the jobs/ directory.
Training logs are stored as JSONL files alongside the job JSON.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..schemas.job_schema import JobRecord, JobSummary

JOBS_DIR = Path(__file__).parent.parent.parent / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _job_log_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.log.jsonl"


def save_job(record: dict) -> None:
    """Save or update a job record to disk."""
    job_id = record["job_id"]
    path = _job_path(job_id)
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)


def load_job(job_id: str) -> dict | None:
    """Load a job record from disk."""
    path = _job_path(job_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def list_jobs(status: str | None = None, model_id: str | None = None) -> list[dict]:
    """List all jobs, optionally filtered by status or model_id."""
    results: list[dict] = []
    for p in sorted(JOBS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p) as f:
                data = json.load(f)
            if status and data.get("status") != status:
                continue
            if model_id and data.get("model_id") != model_id:
                continue
            results.append(data)
        except Exception:
            continue
    return results


def delete_job(job_id: str) -> bool:
    """Delete a job record and its log file."""
    deleted = False
    path = _job_path(job_id)
    if path.exists():
        path.unlink()
        deleted = True
    log_path = _job_log_path(job_id)
    if log_path.exists():
        log_path.unlink()
    return deleted


def append_job_log(job_id: str, level: str, message: str, data: dict | None = None) -> None:
    """Append a log line to the job-specific log file."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "message": message,
        "data": data or {},
    }
    path = _job_log_path(job_id)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_job_logs(job_id: str, limit: int = 200, offset: int = 0) -> list[dict]:
    """Read job-specific training logs."""
    path = _job_log_path(job_id)
    if not path.exists():
        return []

    entries: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Most recent first
    entries.reverse()
    return entries[offset: offset + limit]
