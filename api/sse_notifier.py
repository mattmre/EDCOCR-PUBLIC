"""In-process asyncio.Event notification for SSE streams.

Eliminates per-poll SQLite queries by signaling waiting SSE generators
directly when job status changes.  Falls back to 30-second timeout polling
for safety (missed signals, process restart, etc.).
"""

import asyncio
import logging
import threading
from typing import Dict, Set, Tuple

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# job_id -> set of (event, loop) tuples; we store the loop so that
# notify_job_update can use call_soon_threadsafe from sync threads.
_events: Dict[str, Set[Tuple[asyncio.Event, asyncio.AbstractEventLoop]]] = {}


def create_job_event(job_id: str) -> asyncio.Event:
    """Register a new asyncio.Event for a job SSE stream.

    Must be called from an async context (running event loop).
    """
    evt = asyncio.Event()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    with _lock:
        if job_id not in _events:
            _events[job_id] = set()
        _events[job_id].add((evt, loop))
    return evt


def remove_job_event(job_id: str, evt: asyncio.Event) -> None:
    """Unregister an asyncio.Event when the SSE stream closes."""
    with _lock:
        if job_id in _events:
            # Find and discard the tuple matching this event
            to_discard = None
            for pair in _events[job_id]:
                if pair[0] is evt:
                    to_discard = pair
                    break
            if to_discard is not None:
                _events[job_id].discard(to_discard)
            if not _events[job_id]:
                del _events[job_id]


def notify_job_update(job_id: str) -> None:
    """Signal all SSE streams waiting on *job_id* (called from sync code).

    Uses ``loop.call_soon_threadsafe`` so that the event-loop thread
    receives the wakeup even when this function runs in a worker thread.
    """
    with _lock:
        pairs = list(_events.get(job_id, []))
    for evt, loop in pairs:
        try:
            loop.call_soon_threadsafe(evt.set)
        except RuntimeError:
            # Loop already closed -- stream is gone; ignore.
            pass
