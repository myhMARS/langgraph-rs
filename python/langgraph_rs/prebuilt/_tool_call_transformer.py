"""Transformer that projects `tools` channel events into `ToolCallStream` handles.

Mirrors ``langgraph.prebuilt._tool_call_transformer.ToolCallTransformer``: each `tool-started` event
spawns a handle pushed onto the `tool_calls` projection, deltas append to it, and
`tool-finished` / `tool-error` close it. Events are passed through untouched, so consumers can also
subscribe to the raw `tools` channel.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ..stream._types import ProtocolEvent, StreamTransformer
from ..stream.stream_channel import StreamChannel
from ._tool_call_stream import ToolCallStream


def _is_serialized_tool_message(value: Any) -> bool:
    """Detect a serialized LangChain `ToolMessage` payload."""
    return (
        isinstance(value, dict)
        and value.get("type") == "constructor"
        and isinstance(value.get("id"), list)
        and bool(value["id"])
        and value["id"][-1] == "ToolMessage"
    )


def _normalize_tool_output(output: Any) -> Any:
    """Unwrap a tool message / its serialized form down to the content payload."""
    try:  # a real `ToolMessage` when langchain-core is installed
        from langchain_core.messages import ToolMessage

        if isinstance(output, ToolMessage):
            return output.content
    except Exception:  # pragma: no cover - langchain-core absent
        pass
    if _is_serialized_tool_message(output):
        kwargs = output.get("kwargs")
        if isinstance(kwargs, dict):
            return kwargs.get("content")
    return output


class ToolCallTransformer(StreamTransformer):
    """Project `tools` channel events into `ToolCallStream` handles.

    Registered explicitly at compile time or through `stream_events(transformers=[...])`. The
    `tool_calls` projection is native, so it is also a direct attribute on the run stream.
    """

    _native = True
    required_stream_modes = ("tools",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[ToolCallStream] = StreamChannel()
        self._active: dict[str, ToolCallStream] = {}
        self._is_async = False
        self._pump_fn: Callable[[], bool] | None = None
        self._apump_fn: Callable[[], Awaitable[bool]] | None = None

    def init(self) -> dict[str, Any]:
        return {"tool_calls": self._log}

    def _bind_pump(self, fn: Callable[[], bool]) -> None:
        """Wire the sync pull callback onto this transformer."""
        self._pump_fn = fn
        self._is_async = False

    def _bind_apump(self, fn: Callable[[], Awaitable[bool]]) -> None:
        """Async counterpart to `_bind_pump`."""
        self._apump_fn = fn
        self._is_async = True

    def _new_stream(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_input: dict[str, Any] | None,
    ) -> ToolCallStream:
        stream = ToolCallStream(tool_call_id, tool_name, tool_input)
        stream._bind(is_async=self._is_async)
        if self._apump_fn is not None:
            stream._output_deltas._arequest_more = self._apump_fn
        if self._pump_fn is not None:
            stream._output_deltas._request_more = self._pump_fn
        return stream

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "tools":
            return True
        # Only project events emitted at this transformer's scope.
        if tuple(event["params"]["namespace"]) != self.scope:
            return True

        data = event["params"]["data"]
        tool_call_id = data.get("tool_call_id")
        if tool_call_id is None:
            return True
        event_type = data.get("event")

        stream: ToolCallStream | None
        if event_type == "tool-started":
            stream = self._new_stream(
                tool_call_id,
                data.get("tool_name", ""),
                data.get("input"),
            )
            self._active[tool_call_id] = stream
            self._log.push(stream)
        elif event_type == "tool-output-delta":
            stream = self._active.get(tool_call_id)
            if stream is not None:
                stream._push_delta(data.get("delta"))
        elif event_type == "tool-finished":
            stream = self._active.pop(tool_call_id, None)
            if stream is not None:
                stream._finish(_normalize_tool_output(data.get("output")))
        elif event_type == "tool-error":
            stream = self._active.pop(tool_call_id, None)
            if stream is not None:
                stream._fail(data.get("message", ""))

        # Pass-through: wire consumers subscribe to the `tools` channel directly.
        return True

    def finalize(self) -> None:
        """Close any still-active tool streams left open at run end."""
        for stream in self._active.values():
            if not stream.completed:
                stream._finish(None)
        self._active.clear()

    def fail(self, err: BaseException) -> None:
        """Fail any still-active tool streams when the run errors."""
        message = str(err)
        for stream in self._active.values():
            if not stream.completed:
                stream._fail(message)
        self._active.clear()