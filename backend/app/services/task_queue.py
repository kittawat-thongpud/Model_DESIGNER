"""
Task Queue Service — SQLite-backed admission control for heavy tasks.

Option A (conservative shared-server) classification:

LIGHTWEIGHT (not queued, always allowed):
  - log reads / job history / metrics reads
  - metadata reads / dataset info / model info / weight info
  - status polling / SSE subscription
  - inference on small/explicitly bounded inputs

HEAVY (queued, admission-controlled):
  - training           → max 1 GPU training job running at a time (or VRAM-limited)
  - benchmark          → max 1 concurrent (GPU) (or VRAM-limited)
  - validation         → max 1 concurrent (GPU, high-priority) (or VRAM-limited)
  - dataset extraction / conversion / repartition
  - package import/export
  - large plot generation / report generation
  - weight transfer on large files

VRAM-Aware Queue (when enabled):
  - Tracks VRAM usage per GPU device
  - Admits jobs only if sufficient VRAM available (with safety buffer)
  - Prevents OOM by checking free VRAM before admission
  - Monitors VRAM during execution and kills jobs exceeding limits
  - Safety buffer: 15% + 2GB minimum to prevent system instability

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

# ── VRAM Config ────────────────────────────────────────────────────────────────
_VRAM_CONFIG = _QUEUE_CONFIG.get("vram", {})
_VRAM_ENABLED = bool(_VRAM_CONFIG.get("enabled", False))
_VRAM_SAFETY_BUFFER_GB = float(_VRAM_CONFIG.get("safety_buffer_gb", 2.0))
_VRAM_SAFETY_BUFFER_PERCENT = float(_VRAM_CONFIG.get("safety_buffer_percent", 0.15))
_VRAM_MONITOR_INTERVAL_S = float(_VRAM_CONFIG.get("monitor_interval_s", 10.0))
_VRAM_OOM_KILL_THRESHOLD = float(_VRAM_CONFIG.get("oom_kill_threshold_percent", 0.95))
_MODEL_SCALE_VRAM_GB = _VRAM_CONFIG.get("model_scale_vram_gb", {})
_TASK_TYPE_VRAM_GB = _VRAM_CONFIG.get("task_type_vram_gb", {})

# ── Task classification ───────────────────────────────────────────────────────

class TaskType(str, Enum):
    TRAINING = "training"
    BENCHMARK = "benchmark"
    VALIDATION = "validation"
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
    TaskType.VALIDATION: int(_QUEUE_LIMITS_CONFIG.get("validation", 1)),
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
    # "validation" removed — GPU validation runs through GPU_TASK_TYPES
})

# GPU tasks compete for a single shared GPU execution slot.
# When any GPU task is running, no other GPU task can be admitted.
GPU_TASK_TYPES: frozenset[str] = frozenset({
    TaskType.TRAINING, TaskType.BENCHMARK, TaskType.VALIDATION,
})
_GPU_CONCURRENCY_LIMIT: int = 1


# ── DB helpers ────────────────────────────────────────────────────────────────

_db_lock = threading.Lock()


def _gpu_running_count(conn: sqlite3.Connection) -> int:
    """Count running tasks across all GPU task types (training + benchmark)."""
    placeholders = ",".join("?" * len(GPU_TASK_TYPES))
    return conn.execute(
        f"SELECT COUNT(*) FROM queue_tasks WHERE task_type IN ({placeholders}) AND status='running'",
        tuple(GPU_TASK_TYPES),
    ).fetchone()[0]


def _get_gpu_vram_info() -> dict:
    """Get VRAM info for all available GPUs."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}

        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            device = f"cuda:{i}"
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            free_gb = torch.cuda.mem_get_info(i)[0] / (1024**3)
            gpu_info[device] = {
                "total_gb": total_gb,
                "free_gb": free_gb,
                "used_gb": total_gb - free_gb,
            }
        gpu_info["available"] = True
        return gpu_info
    except Exception:
        return {"available": False}


def _get_allocated_vram_on_device(conn: sqlite3.Connection, gpu_device: str) -> float:
    """Sum of VRAM allocated for running tasks on a specific GPU device."""
    row = conn.execute(
        "SELECT COALESCE(SUM(vram_allocated_gb), 0.0) as total FROM queue_tasks WHERE gpu_device=? AND status='running'",
        (gpu_device,),
    ).fetchone()
    return float(row["total"]) if row else 0.0


def _estimate_vram_requirement(task_type: str, model_scale: str | None = None) -> float:
    """Estimate VRAM requirement based on task type and model scale."""
    base_vram = float(_TASK_TYPE_VRAM_GB.get(task_type, 0.0))
    if model_scale:
        base_vram += float(_MODEL_SCALE_VRAM_GB.get(model_scale, 0.0))
    return max(base_vram, 0.0)


def _check_vram_available(gpu_device: str, required_gb: float) -> tuple[bool, str]:
    """Check if sufficient VRAM is available on the device with safety buffer."""
    gpu_info = _get_gpu_vram_info()
    if not gpu_info.get("available"):
        return True, "VRAM monitoring not available, allowing admission"

    device_info = gpu_info.get(gpu_device)
    if not device_info:
        return False, f"GPU device {gpu_device} not found"

    free_gb = device_info["free_gb"]
    total_gb = device_info["total_gb"]

    # Calculate safety buffer (15% + 2GB minimum)
    buffer_gb = max(
        _VRAM_SAFETY_BUFFER_GB,
        total_gb * _VRAM_SAFETY_BUFFER_PERCENT,
    )

    # Check if free VRAM (minus buffer) is sufficient
    available_after_buffer = free_gb - buffer_gb
    if available_after_buffer < required_gb:
        return False, f"Insufficient VRAM: need {required_gb:.1f}GB, have {available_after_buffer:.1f}GB available (free: {free_gb:.1f}GB, buffer: {buffer_gb:.1f}GB)"

    return True, f"VRAM OK: need {required_gb:.1f}GB, have {available_after_buffer:.1f}GB available"


def update_vram_usage(task_id: str, vram_used_gb: float) -> bool:
    """Update actual VRAM usage for a running task."""
    with _db() as conn:
        conn.execute(
            "UPDATE queue_tasks SET vram_used_gb=? WHERE task_id=?",
            (vram_used_gb, task_id),
        )
        return True


def check_oom_violations() -> list[dict]:
    """Check all running GPU tasks for OOM violations and return violations."""
    if not _VRAM_ENABLED:
        return []

    violations: list[dict] = []
    gpu_info = _get_gpu_vram_info()

    if not gpu_info.get("available"):
        return violations

    with _db() as conn:
        placeholders = ",".join("?" * len(GPU_TASK_TYPES))
        rows = conn.execute(
            f"""SELECT task_id, task_type, ref_id, gpu_device, vram_allocated_gb, vram_used_gb
               FROM queue_tasks
               WHERE task_type IN ({placeholders}) AND status='running'""",
            tuple(GPU_TASK_TYPES),
        ).fetchall()

        for row in rows:
            task_id = row["task_id"]
            task_type = row["task_type"]
            ref_id = row["ref_id"]
            gpu_device = row["gpu_device"] or "cuda:0"
            vram_allocated = float(row["vram_allocated_gb"] or 0.0)
            vram_used = float(row["vram_used_gb"] or 0.0)

            if vram_allocated <= 0:
                continue

            device_info = gpu_info.get(gpu_device)
            if not device_info:
                continue

            total_gb = device_info["total_gb"]
            used_ratio = vram_used / total_gb if total_gb > 0 else 0.0

            # Check if exceeding allocated limit significantly
            if vram_used > vram_allocated * 1.5:  # 50% over allocation
                violations.append({
                    "task_id": task_id,
                    "task_type": task_type,
                    "ref_id": ref_id,
                    "gpu_device": gpu_device,
                    "vram_allocated_gb": vram_allocated,
                    "vram_used_gb": vram_used,
                    "reason": f"VRAM usage {vram_used:.1f}GB exceeds allocation {vram_allocated:.1f}GB by 50%",
                    "severity": "high",
                })

            # Check if approaching total GPU limit (95%)
            if used_ratio > _VRAM_OOM_KILL_THRESHOLD:
                violations.append({
                    "task_id": task_id,
                    "task_type": task_type,
                    "ref_id": ref_id,
                    "gpu_device": gpu_device,
                    "vram_allocated_gb": vram_allocated,
                    "vram_used_gb": vram_used,
                    "reason": f"VRAM usage {vram_used:.1f}GB ({used_ratio*100:.1f}%) exceeds OOM threshold {_VRAM_OOM_KILL_THRESHOLD*100:.1f}%",
                    "severity": "critical",
                })

    return violations


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
                completed_at REAL,
                gpu_device  TEXT,           -- GPU device ID (e.g., "cuda:0")
                vram_allocated_gb REAL,    -- VRAM allocated for this task
                vram_used_gb REAL,          -- Actual VRAM usage (updated during execution)
                model_scale TEXT            -- Model scale (n/s/m/l/x) for VRAM estimation
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_queue_status_type
            ON queue_tasks(task_type, status, priority DESC, created_at ASC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_queue_gpu_device
            ON queue_tasks(gpu_device, status)
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


def _admit_pending_gpu_locked(conn: sqlite3.Connection, now: float) -> list[tuple[str, str]]:
    """Admit the next pending GPU task (training or benchmark) respecting VRAM and concurrency."""
    admitted: list[tuple[str, str]] = []
    placeholders = ",".join("?" * len(GPU_TASK_TYPES))

    # Get all pending GPU tasks ordered by priority
    pending_rows = conn.execute(
        f"""SELECT task_id, task_type, model_scale, gpu_device
           FROM queue_tasks
           WHERE task_type IN ({placeholders}) AND status='pending'
           ORDER BY priority DESC, created_at ASC""",
        tuple(GPU_TASK_TYPES),
    ).fetchall()

    for row in pending_rows:
        task_id = row["task_id"]
        task_type = row["task_type"]
        model_scale = row["model_scale"]
        gpu_device = row["gpu_device"] or "cuda:0"

        # Check concurrency limit
        if _gpu_running_count(conn) >= _GPU_CONCURRENCY_LIMIT:
            break

        # VRAM-aware admission
        if _VRAM_ENABLED:
            vram_required = _estimate_vram_requirement(task_type, model_scale)
            vram_available, vram_msg = _check_vram_available(gpu_device, vram_required)

            if not vram_available:
                # Skip this task, try next one
                continue

            # VRAM OK, admit the task
            conn.execute(
                "UPDATE queue_tasks SET status='running', started_at=?, gpu_device=?, vram_allocated_gb=? WHERE task_id=?",
                (now, gpu_device, vram_required, task_id),
            )
            admitted.append((task_id, task_type))
        else:
            # Fallback: simple concurrency check
            conn.execute(
                "UPDATE queue_tasks SET status='running', started_at=?, gpu_device=? WHERE task_id=?",
                (now, gpu_device, task_id),
            )
            admitted.append((task_id, task_type))

    return admitted


def enqueue(
    task_type: str,
    ref_id: str | None = None,
    payload: dict | None = None,
    priority: int = 0,
    gpu_device: str | None = None,
    model_scale: str | None = None,
) -> tuple[str, bool, str]:
    """
    Attempt to admit a heavy task.

    Returns (task_id, admitted, message):
      - admitted=True  → task is running immediately
      - admitted=False → task is pending (queued), will be admitted when a slot opens
      - message        → status message explaining admission decision
    """
    import json as _json

    task_id = uuid.uuid4().hex[:16]
    now = time.time()
    limit = _CONCURRENCY_LIMITS.get(task_type, 1)

    # Determine GPU device if not specified
    if gpu_device is None and task_type in GPU_TASK_TYPES:
        gpu_info = _get_gpu_vram_info()
        if gpu_info.get("available") and len(gpu_info) > 1:
            # Use first available GPU
            gpu_device = list(gpu_info.keys())[0]
        else:
            gpu_device = "cuda:0"

    # Estimate VRAM requirement
    vram_required = _estimate_vram_requirement(task_type, model_scale) if _VRAM_ENABLED else 0.0

    admitted_callbacks: list[tuple[str, str]] = []
    admission_message = ""

    with _db() as conn:
        if task_type in GPU_TASK_TYPES:
            # VRAM-aware admission for GPU tasks
            if _VRAM_ENABLED and gpu_device:
                # Check VRAM availability
                vram_available, vram_msg = _check_vram_available(gpu_device, vram_required)
                admission_message = vram_msg

                if not vram_available:
                    admitted = False
                    status = TaskStatus.PENDING
                else:
                    # VRAM OK, check concurrency limit
                    running_count = _gpu_running_count(conn)
                    admitted = running_count < _GPU_CONCURRENCY_LIMIT
                    status = TaskStatus.RUNNING if admitted else TaskStatus.PENDING
                    if not admitted:
                        admission_message = f"VRAM OK but concurrency limit reached ({running_count}/{_GPU_CONCURRENCY_LIMIT})"
            else:
                # Fallback to simple concurrency limit
                running_count = _gpu_running_count(conn)
                admitted = running_count < _GPU_CONCURRENCY_LIMIT
                status = TaskStatus.RUNNING if admitted else TaskStatus.PENDING
                admission_message = f"Concurrency check: {running_count}/{_GPU_CONCURRENCY_LIMIT}"
        else:
            # Non-GPU tasks use simple concurrency limit
            running_count = conn.execute(
                "SELECT COUNT(*) FROM queue_tasks WHERE task_type=? AND status='running'",
                (task_type,),
            ).fetchone()[0]
            admitted = running_count < limit
            status = TaskStatus.RUNNING if admitted else TaskStatus.PENDING
            admission_message = f"Concurrency check: {running_count}/{limit}"

        conn.execute(
            """INSERT INTO queue_tasks
               (task_id, task_type, ref_id, status, priority, payload, created_at, started_at, gpu_device, vram_allocated_gb, model_scale)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                task_type,
                ref_id,
                status.value,
                priority,
                _json.dumps(payload or {}),
                now,
                now if admitted else None,
                gpu_device,
                vram_required if admitted else None,
                model_scale,
            ),
        )
        if admitted:
            admitted_callbacks.append((task_id, task_type))
    if admitted_callbacks:
        _fire_admission_callbacks(admitted_callbacks)
    return task_id, admitted, admission_message


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


def get_active_tasks_for_ref(ref_id: str) -> list[dict]:
    """Return pending/running tasks whose ref_id matches (e.g. dataset name, job_id)."""
    import json as _json
    with _db() as conn:
        rows = conn.execute(
            "SELECT task_id, task_type, ref_id, status, priority, payload, error, "
            "created_at, started_at, completed_at "
            "FROM queue_tasks WHERE ref_id=? AND status IN ('pending','running') "
            "ORDER BY created_at DESC",
            (ref_id,),
        ).fetchall()
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
            if task_type in GPU_TASK_TYPES:
                admitted_callbacks = _admit_pending_gpu_locked(conn, now)
            else:
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
            """SELECT task_id, task_type, ref_id, priority, created_at, model_scale
               FROM queue_tasks WHERE status='pending'
               ORDER BY priority DESC, created_at ASC LIMIT 50"""
        ).fetchall()
        for r in q:
            pending_tasks.append(dict(r))

        gpu_running = _gpu_running_count(conn)

        # Get VRAM info if enabled
        vram_info = {}
        if _VRAM_ENABLED:
            gpu_info = _get_gpu_vram_info()
            if gpu_info.get("available"):
                vram_info["gpus"] = gpu_info
                # Get running tasks with VRAM allocation
                running_vram = conn.execute(
                    """SELECT task_id, task_type, gpu_device, vram_allocated_gb, vram_used_gb, model_scale
                       FROM queue_tasks WHERE status='running' AND task_type IN ('training','benchmark','validation')"""
                ).fetchall()
                vram_info["running_tasks"] = [
                    {
                        "task_id": r["task_id"],
                        "task_type": r["task_type"],
                        "gpu_device": r["gpu_device"],
                        "vram_allocated_gb": float(r["vram_allocated_gb"] or 0.0),
                        "vram_used_gb": float(r["vram_used_gb"] or 0.0),
                        "model_scale": r["model_scale"],
                    }
                    for r in running_vram
                ]
                vram_info["config"] = {
                    "enabled": _VRAM_ENABLED,
                    "safety_buffer_gb": _VRAM_SAFETY_BUFFER_GB,
                    "safety_buffer_percent": _VRAM_SAFETY_BUFFER_PERCENT,
                    "model_scale_vram_gb": _MODEL_SCALE_VRAM_GB,
                }

        return {
            "summary": summary,
            "pending": pending_tasks,
            "concurrency_limits": _CONCURRENCY_LIMITS,
            "gpu_busy": gpu_running >= _GPU_CONCURRENCY_LIMIT,
            "gpu_running": gpu_running,
            "vram": vram_info,
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
