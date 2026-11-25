"""Live bridge for chat-model tokens.

The engine collects a node's stream writes and hands them to the consumer when the node finishes, so
`run.messages` would show tokens only after the model returns. LangChain already reports tokens
through the callback handler (`on_llm_new_token`), which runs *on the engine thread while the model
generates*; this module gives that handler a direct line to the consumer.

`langgraph_rs.stream._events` activates a bridge for the duration of a `stream_events` run; the
callback handler pushes each token into it, and the run stream drains it as it pumps, so deltas reach
`handle.text` while the model is still generating.

Only one bridge is active at a time (the handler cannot tell which run a token belongs to), which is
what a single threaded consumer needs; concurrent `stream_events` runs share the newest bridge.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

__all__ = ["LiveBridge", "active_bridge", "activate", "deactivate"]

_LOCK = threading.Lock()
# Bridges by `thread_id`; a run without one uses the `None` bucket. Keying by thread keeps
# concurrent runs from stealing each other's tokens.
_ACTIVE: dict[str | None, "LiveBridge"] = {}


class LiveBridge:
    """Thread-safe queue of live `messages` parts for one run."""

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[dict[str, Any], dict[str, Any]] | None] = queue.Queue()
        self._closed = threading.Event()
        self.count = 0

    def push(self, payload: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Called from the engine thread for every streamed token."""
        if self._closed.is_set():
            return
        self.count += 1
        self._queue.put((payload, metadata))

    def drain(self, timeout: float = 0.05) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Return the items queued so far (blocking up to `timeout` for the first one)."""
        items: list[tuple[dict[str, Any], dict[str, Any]]] = []
        try:
            first = self._queue.get(timeout=timeout)
        except queue.Empty:
            return items
        if first is None:
            self._closed.set()
            return items
        items.append(first)
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._closed.set()
                break
            items.append(item)
        return items

    def close(self) -> None:
        """Signal that no more tokens will arrive."""
        self._closed.set()
        self._queue.put(None)


def active_bridge(thread_id: str | None = None) -> LiveBridge | None:
    """The bridge for `thread_id`, if a run is streaming tokens for it."""
    return _ACTIVE.get(thread_id)


def activate(bridge: LiveBridge, thread_id: str | None = None) -> None:
    """Register `bridge` for `thread_id` (replacing any previous one)."""
    with _LOCK:
        _ACTIVE[thread_id] = bridge


def deactivate(bridge: LiveBridge, thread_id: str | None = None) -> None:
    """Remove `bridge` if it is still registered for `thread_id`."""
    with _LOCK:
        if _ACTIVE.get(thread_id) is bridge:
            del _ACTIVE[thread_id]
    bridge.close()