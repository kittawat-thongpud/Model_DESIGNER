"""
Event Bus — asyncio-based pub/sub for SSE streaming.

Supports multiple named channels (e.g. "train:{job_id}", "logs:system").
Subscribers get an asyncio.Queue that receives events as dicts.
Thread-safe: training runs in a background thread and publishes via publish_sync().
"""
from __future__ import annotations
import asyncio
import threading
from collections import defaultdict
from typing import Any


_lock = threading.Lock()
_subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
_loops: dict[str, list[asyncio.AbstractEventLoop]] = defaultdict(list)
_last_event: dict[str, dict] = {}  # last event per channel for replay on subscribe


def subscribe(channel: str) -> asyncio.Queue:
    """Subscribe to a channel. Returns a Queue that will receive events.
    
    If a previous event was published on this channel, it is replayed
    immediately into the queue so late subscribers don't miss the current state.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    loop = asyncio.get_running_loop()
    with _lock:
        _subscribers[channel].append(queue)
        _loops[channel].append(loop)
        last = _last_event.get(channel)
    # Replay last event outside the lock
    if last is not None:
        try:
            queue.put_nowait(last)
        except asyncio.QueueFull:
            pass
    return queue


def unsubscribe(channel: str, queue: asyncio.Queue) -> None:
    """Remove a subscriber from a channel."""
    with _lock:
        subs = _subscribers.get(channel, [])
        loops = _loops.get(channel, [])
        for i, q in enumerate(subs):
            if q is queue:
                subs.pop(i)
                if i < len(loops):
                    loops.pop(i)
                break
        if not subs:
            _subscribers.pop(channel, None)
            _loops.pop(channel, None)


def publish_sync(channel: str, event: dict[str, Any]) -> None:
    """
    Publish an event from a synchronous context (e.g. training thread).
    Thread-safe. Uses call_soon_threadsafe to push into asyncio queues.
    Also stores the event as the last event for replay on new subscribers.
    """
    with _lock:
        _last_event[channel] = event
        subs = list(_subscribers.get(channel, []))
        loops = list(_loops.get(channel, []))

    for queue, loop in zip(subs, loops):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except (asyncio.QueueFull, RuntimeError):
            pass  # Drop if queue full or loop closed


async def publish(channel: str, event: dict[str, Any]) -> None:
    """Publish an event from an async context."""
    with _lock:
        _last_event[channel] = event
        subs = list(_subscribers.get(channel, []))

    for queue in subs:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop if subscriber is slow


def clear_last_event(channel: str) -> None:
    """Clear the stored last event for a channel (call when job completes/stops)."""
    with _lock:
        _last_event.pop(channel, None)


def channel_has_subscribers(channel: str) -> bool:
    """Check if anyone is listening on a channel."""
    with _lock:
        return len(_subscribers.get(channel, [])) > 0
