"""In-process handle for a single tool call's streaming execution.

Mirrors ``langgraph.prebuilt._tool_call_stream.ToolCallStream``: one output channel per call, filled
by `ToolCallTransformer` as `tool-started` / `tool-output-delta` / `tool-finished` / `tool-error`
events arrive on the `tools` channel.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from ..stream.stream_channel import StreamChannel


class ToolCallStream:
    """Scoped view of a single tool call's lifecycle.

    Yielded once per `tool-started` event. `output_deltas` streams partial output; `output` and
    `error` hold the terminal payloads; `completed` flips once either terminal event is seen.
    """

    def __init__(
        self,
        tool_call_id: str,
        tool_name: str,
        input: dict[str, Any] | None = None,
    ) -> None:
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.input = input
        self._output_deltas: StreamChannel[Any] = StreamChannel()
        self.output: Any = None
        self.error: str | None = None
        self.completed = False

    @property
    def output_deltas(self) -> StreamChannel[Any]:
        """The channel of streamed `tool-output-delta` payloads."""
        return self._output_deltas

    def _bind(self, *, is_async: bool) -> None:
        """Bind the deltas log to sync or async iteration."""
        self._output_deltas._bind(is_async=is_async)

    def _push_delta(self, delta: Any) -> None:
        self._output_deltas.push(delta)

    def _finish(self, output: Any) -> None:
        self.output = output
        self.completed = True
        self._output_deltas.close()

    def _fail(self, message: str) -> None:
        self.error = message
        self.completed = True
        self._output_deltas.close()

    def __iter__(self) -> Iterator[Any]:
        """Iterate delta chunks synchronously."""
        return iter(self._output_deltas)

    def __aiter__(self) -> AsyncIterator[Any]:
        """Iterate delta chunks asynchronously."""
        return self._output_deltas.__aiter__()

    def __repr__(self) -> str:
        status = (
            "completed"
            if self.completed and self.error is None
            else "failed"
            if self.completed
            else "running"
        )
        return (
            f"ToolCallStream(tool_call_id={self.tool_call_id!r}, "
            f"tool_name={self.tool_name!r}, status={status})"
        )