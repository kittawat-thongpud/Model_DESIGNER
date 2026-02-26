"""
Shared constants used across backend modules.
Consolidates hardcoded string literals for SSE event types, channel names, etc.
"""
from __future__ import annotations


# ── SSE Event Types ───────────────────────────────────────────────────────────
# Used by trainer.py (publish), stream_controller.py (terminal check), etc.

class SSEEvent:
    """Event type strings sent via Server-Sent Events."""
    ITERATION = "iteration"
    EPOCH = "epoch"
    COMPLETE = "complete"
    STOPPED = "stopped"
    ERROR = "error"
    TEST_EVAL = "test_eval"
    MESSAGE = "message"

    # Terminal events — stream_controller closes after these
    TERMINAL = frozenset({COMPLETE, ERROR, STOPPED})


# ── SSE Channel Name Helpers ──────────────────────────────────────────────────

def train_channel(job_id: str) -> str:
    """Channel for training events: ``train:{job_id}``."""
    return f"train:{job_id}"


def job_log_channel(job_id: str) -> str:
    """Channel for job-specific log entries: ``logs:job:{job_id}``."""
    return f"logs:job:{job_id}"


def job_channel(job_id: str) -> str:
    """Channel for job status updates: ``job:{job_id}``."""
    return f"job:{job_id}"


SYSTEM_LOG_CHANNEL = "logs:system"
