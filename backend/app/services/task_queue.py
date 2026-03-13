"""
Task Queue Service — SQLite-backed admission control for heavy tasks.

Option A (conservative shared-server) classification:

LIGHTWEIGHT (not queued, always allowed):
  - log reads / job history / metrics reads
  - metadata reads / dataset info / model info / weight info
  - status polling / SSE subscription
  - inference on small/explicitly bounded inputs
  - validation within configurable resource thresholds

HEAVY (queued, admission-controlled):
  - training           → max 1 GPU training job running at a time
  - benchmark          → max 1 concurrent
  - dataset extraction / conversion / repartition
  - package import/export
  - large plot generation / report generation
  - weight transfer on large files

Queue behavior:
  - If a slot is available → task is admitted immediately (status = running)
  - If no slot available → task is queued (status = pending)
  - Queue is polled on task completion to admit the next pending task
  - State is persisted in SQLite so pending jobs survive restarts
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ..config import DATA_DIR
from .config_service import get_queue_config

_QUEUE_CONFIG = get_queue_config()
_DB_PATH = Path(str(_QUEUE_CONFIG.get("sqlite_path", "task_queue.db")))
if not _DB_PATH.is_absolute():
    _DB_PATH = DATA_DIR / _DB_PATH
_SQLITE_TIMEOUT_S = float(_QUEUE_CONFIG.get("sqlite_timeout_s", 10.0))
_QUEUE_CLEANUP_MAX_AGE_S = float(_QUEUE_CONFIG.get("cleanup_max_age_s", 86400 * 7))

# ── Task classification ───────────────────────────────────────────────────────

class TaskType(str, Enum):
    TRAINING = "training"
    BENCHMARK = "benchmark"
    DATASET_CONVERSION = "dataset_conversion"
    DATASET_EXTRACTION = "dataset_extraction"
    EXPORT = "export"
    PACKAGE_IMPORT = "package_import"
    PACKAGE_EXPORT = "package_export"
    PLOT_GENERATION = "plot_generation"
    WEIGHT_TRANSFER = "weight_transfer"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Option A: max concurrent slots per task class
# training → 1 GPU job at a time; all other heavy tasks → 2 concurrent (non-GPU)
_QUEUE_LIMITS_CONFIG = _QUEUE_CONFIG.get("concurrency_limits", {})
_CONCURRENCY_LIMITS: dict[str, int] = {
    TaskType.TRAINING: int(_QUEUE_LIMITS_CONFIG.get("training", 1)),
    TaskType.BENCHMARK: int(_QUEUE_LIMITS_CONFIG.get("benchmark", 1)),
    TaskType.DATASET_CONVERSION: int(_QUEUE_LIMITS_CONFIG.get("dataset_conversion", 2)),
    TaskType.DATASET_EXTRACTION: int(_QUEUE_LIMITS_CONFIG.get("dataset_extraction", 2)),
    TaskType.EXPORT: int(_QUEUE_LIMITS_CONFIG.get("export", 2)),
    TaskType.PACKAGE_IMPORT: int(_QUEUE_LIMITS_CONFIG.get("package_import", 1)),
    TaskType.PACKAGE_EXPORT: int(_QUEUE_LIMITS_CONFIG.get("package_export", 1)),
    TaskType.PLOT_GENERATION: int(_QUEUE_LIMITS_CONFIG.get("plot_generation", 2)),
    TaskType.WEIGHT_TRANSFER: int(_QUEUE_LIMITS_CONFIG.get("weight_transfer", 2)),
}

# Lightweight task types that bypass the queue entirely (no admission check)
LIGHTWEIGHT_TASK_TYPES: frozenset[str] = frozenset({
    "log_read",
    "metadata_read",
    "status_poll",
    "sse_subscription",
    "inference",       # treated as lightweight under Option A threshold
    "validation",      # treated as lightweight under Option A threshold
})


# ── DB helpers ────────────────────────────────────────────────────────────────

_db_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=_SQLITE_TIMEOUT_S, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _db():
    with _db_lock:
        conn = _get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _init_db() -> None:
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queue_tasks (
                task_id     TEXT PRIMARY KEY,
                task_type   TEXT NOT NULL,
                ref_id      TEXT,           -- e.g. job_id, benchmark_id
                status      TEXT NOT NULL DEFAULT 'pending',
                priority    INTEGER NOT NULL DEFAULT 0,
                payload     TEXT,           -- JSON blob for task metadata
                error       TEXT,
                created_at  REAL NOT NULL,
                started_at  REAL,
                completed_at REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_queue_status_type
            ON queue_tasks(task_type, status, priority DESC, created_at ASC)
        """)


# ── Public API ────────────────────────────────────────────────────────────────

_admission_callbacks: list[Callable[[str, str], None]] = []


def on_task_admitted(callback: Callable[[str, str], None]) -> None:
    """Register a callback called when a pending task is admitted.
    Signature: callback(task_id, task_type)
    """
    _admission_callbacks.append(callback)


def _fire_admission_callbacks(admitted: list[tuple[str, str]]) -> None:
    for task_id, task_type in admitted:
        for cb in _admission_callbacks:
            try:
                cb(task_id, task_type)
            except Exception:
                pass


def _admit_pending_locked(conn: sqlite3.Connection, task_type: str, now: float) -> list[tuple[str, str]]:
    admitted: list[tuple[str, str]] = []
    limit = _CONCURRENCY_LIMITS.get(task_type, 1)
    running_count = conn.execute(
        "SELECT COUNT(*) FROM queue_tasks WHERE task_type=? AND status='running'",
        (task_type,),
    ).fetchone()[0]
    while running_count < limit:
        next_task = conn.execute(
            """SELECT task_id FROM queue_tasks
               WHERE task_type=? AND status='pending'
               ORDER BY priority DESC, created_at ASC
               LIMIT 1""",
            (task_type,),
        ).fetchone()
        if not next_task:
            break
        next_id = next_task["task_id"]
        conn.execute(
            "UPDATE queue_tasks SET status='running', started_at=? WHERE task_id=?",
            (now, next_id),
        )
        admitted.append((next_id, task_type))
        running_count += 1
    return admitted


def enqueue(
    task_type: str,
    ref_id: str | None = None,
    payload: dict | None = None,
    priority: int = 0,
) -> tuple[str, bool]:
    """
    Attempt to admit a heavy task.

    Returns (task_id, admitted):
      - admitted=True  → task is running immediately
      - admitted=False → task is pending (queued), will be admitted when a slot opens
    """
    import json as _json

    task_id = uuid.uuid4().hex[:16]
    now = time.time()
    limit = _CONCURRENCY_LIMITS.get(task_type, 1)

    admitted_callbacks: list[tuple[str, str]] = []
    with _db() as conn:
        running_count = conn.execute(
            "SELECT COUNT(*) FROM queue_tasks WHERE task_type=? AND status='running'",
            (task_type,),
        ).fetchone()[0]

        admitted = running_count < limit
        status = TaskStatus.RUNNING if admitted else TaskStatus.PENDING

        conn.execute(
            """INSERT INTO queue_tasks
               (task_id, task_type, ref_id, status, priority, payload, created_at, started_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                task_type,
                ref_id,
                status.value,
                priority,
                _json.dumps(payload or {}),
                now,
                now if admitted else None,
            ),
        )
        if admitted:
            admitted_callbacks.append((task_id, task_type))
    if admitted_callbacks:
        _fire_admission_callbacks(admitted_callbacks)
    return task_id, admitted


def complete(task_id: str, error: str | None = None) -> None:
    """Mark a task as completed (or failed) and try to admit the next pending task."""
    final_status = TaskStatus.FAILED if error else TaskStatus.COMPLETED
    finalize_task(task_id, final_status, error=error, admit_pending=True)


def cancel(task_id: str) -> bool:
    """Cancel a pending task. Returns True if cancelled, False if not pending."""
    with _db() as conn:
        row = conn.execute(
            "SELECT status FROM queue_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if not row or row["status"] != TaskStatus.PENDING:
            return False
        conn.execute(
            "UPDATE queue_tasks SET status='cancelled', completed_at=? WHERE task_id=?",
            (time.time(), task_id),
        )
    return True


def get_task(task_id: str) -> dict | None:
    import json as _json
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM queue_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["payload"] = _json.loads(d.get("payload") or "{}")
        except Exception:
            pass
        return d


def list_tasks(
    task_type: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[dict]:
    import json as _json
    clauses = []
    params: list[Any] = []
    if task_type:
        clauses.append("task_type=?")
        params.append(task_type)
    if status:
        clauses.append("status=?")
        params.append(status)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = (
        "SELECT task_id, task_type, ref_id, status, priority, payload, error, "
        "created_at, started_at, completed_at "
        f"FROM queue_tasks {where_sql} "
        "ORDER BY created_at DESC LIMIT ?"
    )
    params.append(limit)
    with _db() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    results: list[dict] = []
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = _json.loads(item.get("payload") or "{}")
        except Exception:
            pass
        results.append(item)
    return results


def finalize_task(
    task_id: str,
    final_status: TaskStatus | str,
    error: str | None = None,
    admit_pending: bool = True,
) -> bool:
    now = time.time()
    if isinstance(final_status, str):
        final_status = TaskStatus(final_status)
    admitted_callbacks: list[tuple[str, str]] = []
    with _db() as conn:
        row = conn.execute(
            "SELECT task_type, status FROM queue_tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()
        if not row:
            return False
        current_status = row["status"]
        if current_status in (
            TaskStatus.COMPLETED.value,
            TaskStatus.FAILED.value,
            TaskStatus.CANCELLED.value,
        ):
            return False
        task_type = row["task_type"]
        conn.execute(
            "UPDATE queue_tasks SET status=?, completed_at=?, error=? WHERE task_id=?",
            (final_status.value, now, error, task_id),
        )
        if admit_pending and current_status == TaskStatus.RUNNING.value:
            admitted_callbacks = _admit_pending_locked(conn, task_type, now)
    if admitted_callbacks:
        _fire_admission_callbacks(admitted_callbacks)
    return True


def reconcile_tasks(
    task_type: str,
    valid_ref_ids: set[str],
    running_ref_ids: set[str] | None = None,
    admit_pending: bool = True,
) -> dict[str, int]:
    now = time.time()
    admitted_callbacks: list[tuple[str, str]] = []
    stale_running = 0
    orphan_pending = 0
    with _db() as conn:
        rows = conn.execute(
            """SELECT task_id, ref_id, status
               FROM queue_tasks
               WHERE task_type=? AND status IN ('running', 'pending')""",
            (task_type,),
        ).fetchall()
        for row in rows:
            ref_id = row["ref_id"]
            status = row["status"]
            if status == TaskStatus.RUNNING:
                if ref_id not in valid_ref_ids or (running_ref_ids is not None and ref_id not in running_ref_ids):
                    conn.execute(
                        "UPDATE queue_tasks SET status='failed', completed_at=?, error=? WHERE task_id=?",
                        (now, "stale queue task reconciled", row["task_id"]),
                    )
                    stale_running += 1
            elif status == TaskStatus.PENDING:
                if ref_id not in valid_ref_ids:
                    conn.execute(
                        "UPDATE queue_tasks SET status='cancelled', completed_at=? WHERE task_id=?",
                        (now, row["task_id"]),
                    )
                    orphan_pending += 1
        if admit_pending:
            admitted_callbacks = _admit_pending_locked(conn, task_type, now)
    if admitted_callbacks:
        _fire_admission_callbacks(admitted_callbacks)
    return {
        "stale_running": stale_running,
        "orphan_pending": orphan_pending,
        "admitted": len(admitted_callbacks),
    }


def queue_status(task_type: str | None = None) -> dict:
    """Return current queue status, optionally filtered by task_type."""
    import json as _json
    with _db() as conn:
        if task_type:
            rows = conn.execute(
                """SELECT task_type, status, COUNT(*) as count
                   FROM queue_tasks
                   WHERE task_type=?
                   GROUP BY task_type, status""",
                (task_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT task_type, status, COUNT(*) as count
                   FROM queue_tasks
                   GROUP BY task_type, status""",
            ).fetchall()

        summary: dict[str, dict[str, int]] = {}
        for row in rows:
            tt = row["task_type"]
            st = row["status"]
            if tt not in summary:
                summary[tt] = {}
            summary[tt][st] = row["count"]

        pending_tasks = []
        q = conn.execute(
            """SELECT task_id, task_type, ref_id, priority, created_at
               FROM queue_tasks WHERE status='pending'
               ORDER BY priority DESC, created_at ASC LIMIT 50"""
        ).fetchall()
        for r in q:
            pending_tasks.append(dict(r))

        return {
            "summary": summary,
            "pending": pending_tasks,
            "concurrency_limits": _CONCURRENCY_LIMITS,
        }


def cleanup_old_tasks(max_age_seconds: float | None = None) -> int:
    """Delete completed/failed/cancelled tasks older than max_age_seconds."""
    if max_age_seconds is None:
        max_age_seconds = _QUEUE_CLEANUP_MAX_AGE_S
    cutoff = time.time() - max_age_seconds
    with _db() as conn:
        result = conn.execute(
            """DELETE FROM queue_tasks
               WHERE status IN ('completed', 'failed', 'cancelled')
               AND completed_at IS NOT NULL
               AND completed_at < ?""",
            (cutoff,),
        )
        return result.rowcount


# ── Initialize on import ──────────────────────────────────────────────────────
_init_db()
