"""
Stream Controller — Server-Sent Events (SSE) endpoints for real-time updates.

Endpoints:
  - GET /api/train/{job_id}/stream  — live training metrics
  - GET /api/logs/stream            — live system log entries
  - GET /api/jobs/{job_id}/logs/stream — live job-specific log entries
"""
from __future__ import annotations
import asyncio
import json
import math
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..services import event_bus
from ..constants import SSEEvent, train_channel, job_log_channel, SYSTEM_LOG_CHANNEL
from ..config import JOBS_DIR

router = APIRouter(tags=["Streaming"])


def _sanitize(obj: Any) -> Any:
    """Recursively replace inf/nan floats with None so JSON serialization never fails."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


async def _sse_generator(channel: str, timeout: float = 30.0):
    """
    Generic SSE generator. Subscribes to a channel, yields events as SSE format.
    Sends a heartbeat comment every `timeout` seconds to keep the connection alive.
    Terminates when a terminal event is received (type=complete/error/stopped).
    """
    queue = event_bus.subscribe(channel)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=timeout)
                data = json.dumps(_sanitize(event), default=str)
                event_type = event.get("type", "message")
                yield f"event: {event_type}\ndata: {data}\n\n"

                # Terminal events — close stream after sending
                if event_type in SSEEvent.TERMINAL:
                    return
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                yield ": heartbeat\n\n"
    finally:
        event_bus.unsubscribe(channel, queue)


async def _tail_log_generator(job_id: str, poll_interval: float = 0.5):
    """
    Tail log.jsonl for a job and stream PROGRESS entries as SSE events.
    Works across multiple workers since it reads from disk, not in-memory event_bus.
    Terminates when job status becomes non-running (complete/failed/stopped).
    """
    log_path = Path(JOBS_DIR) / job_id / "log.jsonl"
    record_path = Path(JOBS_DIR) / job_id / "record.json"

    # Wait up to 5s for log file to appear
    for _ in range(10):
        if log_path.exists():
            break
        await asyncio.sleep(0.5)

    # Seek to end so we only stream new entries
    offset = log_path.stat().st_size if log_path.exists() else 0

    # Send heartbeat immediately so browser doesn't timeout
    yield ": connected\n\n"

    while True:
        await asyncio.sleep(poll_interval)

        # Read new lines since last offset
        if log_path.exists():
            size = log_path.stat().st_size
            if size > offset:
                with open(log_path, "r") as f:
                    f.seek(offset)
                    new_data = f.read(size - offset)
                offset = size

                for line in new_data.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Only stream PROGRESS entries that have a data payload
                    if entry.get("level") == "PROGRESS" and entry.get("data"):
                        data = _sanitize(entry["data"])
                        event_type = data.get("type", "progress")
                        yield f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"

        # Check if job is still running
        if record_path.exists():
            try:
                rec = json.loads(record_path.read_text())
                status = rec.get("status", "running")
                if status not in ("running", "pending"):
                    # Drain any remaining new lines one more time
                    if log_path.exists():
                        size = log_path.stat().st_size
                        if size > offset:
                            with open(log_path, "r") as f:
                                f.seek(offset)
                                for line in f.read().splitlines():
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        entry = json.loads(line)
                                    except json.JSONDecodeError:
                                        continue
                                    if entry.get("level") == "PROGRESS" and entry.get("data"):
                                        data = _sanitize(entry["data"])
                                        event_type = data.get("type", "progress")
                                        yield f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"
                    # Send terminal event
                    terminal_map = {"completed": "complete", "failed": "error", "stopped": "stopped"}
                    term_type = terminal_map.get(status, "complete")
                    yield f"event: {term_type}\ndata: {{\"type\": \"{term_type}\", \"status\": \"{status}\"}}\n\n"
                    return
            except Exception:
                pass

        else:
            # No record file yet — keep waiting
            yield ": heartbeat\n\n"


@router.get("/api/train/{job_id}/stream", summary="SSE stream of training metrics")
async def stream_training(job_id: str):
    """
    Subscribe to live training updates for a job.
    Tails log.jsonl directly — works with any number of uvicorn workers.
    Events: progress (batch/validation), complete, stopped, error.
    """
    return StreamingResponse(
        _tail_log_generator(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/logs/stream", summary="SSE stream of system logs")
async def stream_system_logs():
    """
    Subscribe to live system log entries.
    Each event contains a full log entry dict.
    """
    return StreamingResponse(
        _sse_generator(SYSTEM_LOG_CHANNEL, timeout=60.0),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/jobs/{job_id}/logs/stream", summary="SSE stream of job logs")
async def stream_job_logs(job_id: str):
    """
    Subscribe to live log entries for a specific training job.
    """
    return StreamingResponse(
        _sse_generator(job_log_channel(job_id), timeout=60.0),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
