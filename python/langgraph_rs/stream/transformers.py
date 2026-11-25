"""Built-in stream transformers, mirroring ``langgraph.stream.transformers``.

Each transformer captures one protocol method at its own scope and exposes it as a
`StreamChannel`. `ValuesTransformer` additionally tracks the latest snapshot and interrupts.

Implemented here: `ValuesTransformer`, `CustomTransformer`, `UpdatesTransformer`,
`CheckpointsTransformer`, `DebugTransformer`, `TasksTransformer`.
Still pending (they need the run-stream handle types): `MessagesTransformer`,
`LifecycleTransformer`, `SubgraphTransformer`.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, NotRequired, TypedDict

from .._compat import fill_unsupported
from ._types import ProtocolEvent, StreamTransformer
from .stream_channel import StreamChannel



# ---------------------------------------------------------------------------
# Lifecycle / subgraph discovery
# ---------------------------------------------------------------------------


def _parse_ns_segment(segment: str) -> tuple[str, str | None]:
    """Split a namespace segment into `(graph_name, trigger_call_id)`."""
    name, sep, task_id = segment.partition(":")
    return name, task_id if sep else None


def _status_from_exception(err: BaseException) -> tuple[SubgraphStatus, str | None]:
    """Map a run exception to a terminal subgraph status."""
    kind = type(err).__name__
    if kind == "GraphDrained":
        return "drained", None
    if kind == "GraphInterrupt":
        return "interrupted", None
    return "failed", str(err)


def _terminal_from_result(payload: dict[str, Any]) -> tuple[SubgraphStatus, str | None]:
    """Map a task result payload to `(status, error)`."""
    if payload.get("interrupts"):
        return "interrupted", None
    error = payload.get("error")
    if error:
        return "failed", str(error)
    return "completed", None


class _TasksLifecycleBase(StreamTransformer):
    """Bookkeeping shared by the lifecycle and subgraph transformers.

    Subgraphs are discovered by watching `tasks` events: the first event at a tracked namespace
    opens it, and the parent's task result closes it. Subclasses implement `_should_track`,
    `_on_started` and `_on_terminal`.
    """

    required_stream_modes = ("tasks",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._seen: set[tuple[str, ...]] = set()
        self._open: dict[tuple[str, ...], str] = {}
        self._pending_tool_calls: dict[str, str] = {}

    def _should_track(self, ns: tuple[str, ...]) -> bool:
        raise NotImplementedError

    def _on_started(
        self,
        ns: tuple[str, ...],
        graph_name: str | None,
        trigger_call_id: str | None,
    ) -> None:
        raise NotImplementedError

    def _on_terminal(
        self,
        ns: tuple[str, ...],
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        raise NotImplementedError

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "tasks":
            return True
        ns = tuple(event["params"]["namespace"])
        data = event["params"]["data"]
        if "result" in data:
            self._handle_task_result(ns, data)
        else:
            self._record_pending_tool_calls(data)
            self._handle_task_start(ns, data)
        # Tasks are folded into the synthesized projection, not the raw log.
        return False

    def _record_pending_tool_calls(self, data: dict[str, Any]) -> None:
        """Harvest a task's triggering `tool_call_id`, keyed by its task id."""
        task_id = data.get("id")
        if not isinstance(task_id, str):
            return
        payload = data.get("input")
        tool_call_id: str | None = None
        if isinstance(payload, dict) and isinstance(payload.get("tool_call"), dict):
            candidate = payload["tool_call"].get("id")
            if isinstance(candidate, str):
                tool_call_id = candidate
        elif isinstance(payload, list):
            for call in payload:
                if isinstance(call, dict) and isinstance(call.get("id"), str):
                    tool_call_id = call["id"]
                    break
        if tool_call_id is not None:
            self._pending_tool_calls[task_id] = tool_call_id

    def _handle_task_start(self, ns: tuple[str, ...], data: dict[str, Any]) -> None:
        if not self._should_track(ns) or ns in self._seen:
            return
        self._seen.add(ns)
        parsed_name, trigger_call_id = _parse_ns_segment(ns[-1])
        metadata = data.get("metadata") or {}
        child_lc = metadata.get("lc_agent_name")
        graph_name = child_lc if child_lc is not None else (parsed_name or None)
        self._on_started(ns, graph_name, trigger_call_id)
        if trigger_call_id is not None:
            self._open[ns] = trigger_call_id

    def _pop_terminal_transitions(
        self, ns: tuple[str, ...], data: dict[str, Any]
    ) -> list[tuple[tuple[str, ...], SubgraphStatus, str | None]]:
        """Return and remove tracked children closed by this task result."""
        result_id = data.get("id")
        if not result_id:
            return []
        transitions: list[tuple[tuple[str, ...], SubgraphStatus, str | None]] = []
        for child_ns, parent_task_id in list(self._open.items()):
            if child_ns[:-1] != ns or parent_task_id != result_id:
                continue
            status, error = _terminal_from_result(data)
            transitions.append((child_ns, status, error))
            del self._open[child_ns]
        return transitions

    def _handle_task_result(self, ns: tuple[str, ...], data: dict[str, Any]) -> None:
        for child_ns, status, error in self._pop_terminal_transitions(ns, data):
            self._on_terminal(child_ns, status, error)

    def finalize(self) -> None:
        """Close any tracked namespace still open at run end."""
        for ns in list(self._open):
            self._on_terminal(ns, "completed", None)
        self._open.clear()

    def fail(self, err: BaseException) -> None:
        """Emit a terminal status for any tracked namespace still open."""
        status, error = _status_from_exception(err)
        for ns in list(self._open):
            self._on_terminal(ns, status, error)
        self._open.clear()


class LifecycleTransformer(_TasksLifecycleBase):
    """Surface subgraph lifecycle as `lifecycle` protocol events.

    Tracks subgraphs at every depth below `scope`, so a graph -> subgraph -> subgraph chain reports
    both levels in one flat stream. Native: `run.lifecycle` is a direct attribute.
    """

    _native = True

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._channel: StreamChannel[LifecyclePayload] = StreamChannel("lifecycle")

    def init(self) -> dict[str, Any]:
        return {"lifecycle": self._channel}

    def _should_track(self, ns: tuple[str, ...]) -> bool:
        depth = len(self.scope)
        return len(ns) > depth and ns[:depth] == self.scope

    def _on_started(
        self,
        ns: tuple[str, ...],
        graph_name: str | None,
        trigger_call_id: str | None,
    ) -> None:
        if trigger_call_id is None:
            # Without a task id the parent result can't be correlated; finalize/fail closes it.
            return
        payload: LifecyclePayload = {"event": "started", "namespace": list(ns)}
        if graph_name:
            payload["graph_name"] = graph_name
        payload["trigger_call_id"] = trigger_call_id
        self._channel.push(payload)

    def _on_terminal(
        self,
        ns: tuple[str, ...],
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        payload: LifecyclePayload = {"event": status, "namespace": list(ns)}
        if error is not None:
            payload["error"] = error
        self._channel.push(payload)


class SubgraphTransformer(_TasksLifecycleBase):
    """Discover direct-child subgraphs as in-process navigation handles.

    Each discovered subgraph gets a `SubgraphRunStream` (or `AsyncSubgraphRunStream`) wrapping a
    child mini-mux scoped to its namespace; nested subgraphs land on the child handle.
    """

    _native = True
    supports_sync = True

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[Any] = StreamChannel()
        self._handles: dict[tuple[str, ...], Any] = {}
        self._mux: Any = None

    def init(self) -> dict[str, Any]:
        return {"subgraphs": self._log}

    def _on_register(self, mux: Any) -> None:
        self._mux = mux

    def _should_track(self, ns: tuple[str, ...]) -> bool:
        # Direct children only; grandchildren belong to the child mini-mux.
        depth = len(self.scope)
        return len(ns) == depth + 1 and ns[:depth] == self.scope

    def _on_started(
        self,
        ns: tuple[str, ...],
        graph_name: str | None,
        trigger_call_id: str | None,
    ) -> None:
        if self._mux is None:
            return
        try:
            child_mux = self._mux._make_child(ns)
        except RuntimeError:
            return
        from .run_stream import AsyncSubgraphRunStream, SubgraphRunStream

        handle_cls = AsyncSubgraphRunStream if child_mux.is_async else SubgraphRunStream
        handle = handle_cls(
            mux=child_mux,
            path=ns,
            graph_name=graph_name,
            trigger_call_id=trigger_call_id,
        )
        self._handles[ns] = handle
        self._log.push(handle)

    def _on_terminal(
        self,
        ns: tuple[str, ...],
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        handle = self._handles.get(ns)
        if handle is None or not self._mark_terminal(handle, status, error):
            return
        self._close_or_fail_handle(handle, status, error)

    def _mark_terminal(self, handle: Any, status: SubgraphStatus, error: str | None) -> bool:
        """Mark a handle terminal once; returns True on the first transition."""
        if handle._seen_terminal:
            return False
        handle.status = status
        if error is not None and handle.error is None:
            handle.error = error
        handle._seen_terminal = True
        return True

    def _close_or_fail_handle(
        self, handle: Any, status: SubgraphStatus, error: str | None
    ) -> None:
        if handle._mux is None or handle._mux._events._closed:
            return
        if status == "failed":
            handle._mux.fail(RuntimeError(error or "Subgraph failed"))
        else:
            handle._mux.close()

    def _handle_for_event(self, event: ProtocolEvent) -> Any:
        ns = tuple(event["params"]["namespace"])
        depth = len(self.scope)
        if len(ns) < depth + 1:
            return None
        handle = self._handles.get(ns[: depth + 1])
        if handle is None or handle._mux is None or handle._mux._events._closed:
            return None
        return handle

    def process(self, event: ProtocolEvent) -> bool:
        keep = super().process(event)
        handle = self._handle_for_event(event)
        if handle is not None:
            handle._observe_event(event)
            handle._mux.push(event)
        return keep

class MessagesTransformer(StreamTransformer):
    """Capture messages events as `ChatModelStream` handles.

    `run.messages` yields one handle per chat-model call; each handle exposes `.text`, `.reasoning`,
    `.tool_calls`, `.usage` and `.output`.

    Three input shapes are handled:

    1. v3 protocol events (dict with `"event"`), routed to a handle by `metadata["run_id"]`.
    2. A whole `BaseMessage` payload, replayed through `message_to_events` into a completed handle.
    3. The extension's own `messages` stream parts, which carry plain message dicts: whole messages
       take the same replay path, and `AIMessageChunk` dicts are accumulated per node and flushed as
       a completed handle when the run ends (the engine does not emit token-level v3 events yet).
    """

    _native = True
    required_stream_modes = ("messages",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[Any] = StreamChannel()
        self._by_run: dict[str, Any] = {}
        self._ignored_runs: set[str] = set()
        self._pump_fn: Callable[[], bool] | None = None
        self._apump_fn: Callable[[], Awaitable[bool]] | None = None
        self._scope_list: list[str] = list(scope)
        # Extension bridge: accumulated chunk dicts per node, flushed on finalize/fail.
        # Open chat-model calls: message id (or node) -> {stream, node, text}.
        self._pending: dict[str, dict[str, Any]] = {}

    def init(self) -> dict[str, Any]:
        return {"messages": self._log}

    def _bind_pump(self, fn: Callable[[], bool]) -> None:
        """Wire the sync pull callback."""
        self._pump_fn = fn

    def _bind_apump(self, fn: Callable[[], Awaitable[bool]]) -> None:
        """Wire the async pull callback."""
        self._apump_fn = fn

    def _make_stream(self, *, node: str | None, message_id: str | None) -> Any:
        """Create a sync or async `ChatModelStream`, wiring whichever pump is bound."""
        try:
            from langchain_core.language_models.chat_model_stream import (
                AsyncChatModelStream,
                ChatModelStream,
            )
        except Exception:  # pragma: no cover - langchain-core absent
            raise RuntimeError(
                "MessagesTransformer requires langchain-core's ChatModelStream; "
                "use stream_mode='messages' instead."
            ) from None
        if self._apump_fn is not None:
            stream = AsyncChatModelStream(namespace=[], node=node, message_id=message_id)
            stream.set_arequest_more(self._apump_fn)
            return stream
        if self._pump_fn is not None:
            stream = ChatModelStream(namespace=[], node=node, message_id=message_id)
            stream.set_request_more(self._pump_fn)
            return stream
        return AsyncChatModelStream(namespace=[], node=node, message_id=message_id)

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "messages":
            return True
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True

        payload, metadata = params["data"]
        metadata = metadata or {}
        node = metadata.get("langgraph_node")
        run_id = str(metadata.get("run_id", ""))

        if isinstance(payload, dict) and "event" in payload:
            self._route_protocol_event(payload, run_id=run_id, node=node)
        elif self._is_message_object(payload):
            if self._is_chunk_object(payload):
                return True  # v1 chunk tuples are not projected (LangGraph behaviour)
            self._route_whole_message(payload, node=node)
        elif isinstance(payload, dict) and self._looks_like_message(payload):
            # Extension bridge: our engine serializes messages to plain dicts.
            if self._is_chunk_dict(payload):
                self._accumulate(payload, node=node)
            else:
                # A whole message closes the call's live handle when one is open; any content the
                # deltas have not covered yet is dispatched first so the text stays complete.
                key = str(payload.get("id") or node or "")
                entry = self._pending.get(key)
                if entry is not None:
                    whole = payload.get("content")
                    if isinstance(whole, str) and whole.startswith(entry["text"]):
                        remainder = whole[len(entry["text"]) :]
                        if remainder:
                            entry["text"] += remainder
                            entry["stream"].dispatch(
                                {
                                    "event": "content-block-delta",
                                    "index": 0,
                                    "delta": {"type": "text-delta", "text": remainder},
                                }
                            )
                    self._finish_entry(key)
                else:
                    self._route_whole_message(self._to_message(payload), node=node)
        return True

    @staticmethod
    def _is_message_object(payload: Any) -> bool:
        try:
            from langchain_core.messages import BaseMessage
        except Exception:  # pragma: no cover - langchain-core absent
            return False
        return isinstance(payload, BaseMessage)

    @staticmethod
    def _is_chunk_object(payload: Any) -> bool:
        try:
            from langchain_core.messages import AIMessageChunk
        except Exception:  # pragma: no cover
            return False
        return isinstance(payload, AIMessageChunk)

    @staticmethod
    def _looks_like_message(payload: dict[str, Any]) -> bool:
        return "content" in payload

    @staticmethod
    def _is_chunk_dict(payload: dict[str, Any]) -> bool:
        kind = str(payload.get("type") or payload.get("role") or "")
        if "Chunk" in kind:
            return True
        # The engine's LangChain callback handler emits token dicts without a `type`
        # (`{"content": token, "tags": [...], "metadata": {...}}`).
        return "type" not in payload and "role" not in payload

    @staticmethod
    def _to_message(payload: dict[str, Any]) -> Any:
        """Build a LangChain message from our serialized dict."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        kind = str(payload.get("type") or payload.get("role") or "ai")
        content = payload.get("content", "")
        if "Chunk" in kind:
            return AIMessage(content=content)
        if kind in {"human", "user"}:
            return HumanMessage(content=content)
        if kind in {"system", "SystemMessage"}:
            return SystemMessage(content=content)
        if kind in {"tool", "ToolMessage"}:
            return ToolMessage(content=content, tool_call_id=str(payload.get("tool_call_id", "")))
        return AIMessage(
            content=content,
            tool_calls=payload.get("tool_calls") or [],
        )

    def _open_stream(self, key: str, *, node: str | None) -> dict[str, Any]:
        """Create a handle for a chat-model call and emit its start events."""
        stream = self._make_stream(node=node, message_id=key or None)
        self._log.push(stream)
        stream.dispatch({"event": "message-start", "role": "ai", "id": key or None})
        stream.dispatch(
            {
                "event": "content-block-start",
                "index": 0,
                "content": {"type": "text", "text": ""},
            }
        )
        entry: dict[str, Any] = {"stream": stream, "node": node, "text": ""}
        self._pending[key] = entry
        return entry

    def _accumulate(self, payload: dict[str, Any], *, node: str | None) -> None:
        """Feed one chunk dict into its call's handle as a `content-block-delta`.

        The handle is created (and pushed to `run.messages`) on the first chunk, so consumers see it
        as soon as the first delta arrives instead of at the end of the run.
        """
        key = str(payload.get("id") or node or "")
        entry = self._pending.get(key) or self._open_stream(key, node=node)
        text = payload.get("content")
        if isinstance(text, str) and text:
            entry["text"] += text
            entry["stream"].dispatch(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "delta": {"type": "text-delta", "text": text},
                }
            )

    def _finish_entry(self, key: str, *, error: BaseException | None = None) -> None:
        """Close a call's handle: content-block-finish + message-finish, then fail on error."""
        entry = self._pending.pop(key, None)
        if entry is None:
            return
        stream = entry["stream"]
        if error is None:
            stream.dispatch(
                {
                    "event": "content-block-finish",
                    "index": 0,
                    "content": {"type": "text", "text": entry["text"]},
                }
            )
            stream.dispatch({"event": "message-finish"})
        else:
            stream.fail(error)

    def _flush_pending(self, *, error: BaseException | None = None) -> None:
        """Close every open call (run end or failure)."""
        for key in list(self._pending):
            self._finish_entry(key, error=error)

    def _route_protocol_event(
        self, event: dict[str, Any], *, run_id: str, node: str | None
    ) -> None:
        event_type = event.get("event")
        if event_type == "message-start":
            if event.get("role") == "tool":
                self._ignored_runs.add(run_id)
                return
            message_id = event.get("message_id")
            stream = self._make_stream(
                node=node, message_id=str(message_id) if message_id is not None else None
            )
            self._by_run[run_id] = stream
            self._log.push(stream)
            stream.dispatch(event)
        elif run_id in self._ignored_runs:
            if event_type == "message-finish":
                self._ignored_runs.discard(run_id)
        elif run_id in self._by_run:
            stream = self._by_run[run_id]
            stream.dispatch(event)
            if event_type == "message-finish":
                del self._by_run[run_id]

    def _route_whole_message(
        self, message: Any, *, node: str | None, stream: Any = None, push: bool = True
    ) -> None:
        from langchain_core.language_models._compat_bridge import message_to_events

        message_id = getattr(message, "id", None)
        if stream is None:
            stream = self._make_stream(node=node, message_id=message_id)
        for event in message_to_events(message, message_id=message_id):
            stream.dispatch(event)
        if push:
            self._log.push(stream)

    def finalize(self) -> None:
        """Flush accumulated chunks, then reset routing state."""
        self._flush_pending()
        self._by_run.clear()
        self._ignored_runs.clear()

    def fail(self, err: BaseException) -> None:
        """Propagate the run error to open and accumulated streams."""
        for stream in list(self._by_run.values()):
            stream.fail(err)
        self._by_run.clear()
        self._ignored_runs.clear()
        self._flush_pending(error=err)
__all__ = [
    "CheckpointsTransformer",
    "CustomTransformer",
    "DebugTransformer",
    "LifecyclePayload",
    "LifecycleTransformer",
    "MessagesTransformer",
    "SubgraphStatus",
    "SubgraphTransformer",
    "TasksTransformer",
    "UpdatesTransformer",
    "ValuesTransformer",
]

_logger = logging.getLogger(__name__)

# Names LangGraph's module also exposes: its own imports of the message/stream types.
try:  # pragma: no cover - langchain-core is optional
    from langchain_core.language_models.chat_model_stream import (  # noqa: F401
        AsyncChatModelStream,
        ChatModelStream,
    )
    from langchain_core.messages import AIMessageChunk, BaseMessage  # noqa: F401
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - langchain_protocol is optional
    from langchain_protocol.protocol import LifecycleCause, MessagesData  # noqa: F401
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover
    from ..errors import GraphDrained, GraphInterrupt  # noqa: F401
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - langchain-core is optional
    from langchain_core.language_models._compat_bridge import message_to_events  # noqa: F401
    from langchain_core.messages import ToolMessage  # noqa: F401
except Exception:  # pragma: no cover
    pass

from .run_stream import AsyncSubgraphRunStream, SubgraphRunStream  # noqa: E402,F401


SubgraphStatus = Literal["started", "completed", "failed", "interrupted", "drained"]
"""Status of a subgraph run."""


class LifecyclePayload(TypedDict, total=False):
    """Payload of the `lifecycle` transformer."""

    event: str
    ns: list[str]
    data: NotRequired[Any]


class ValuesTransformer(StreamTransformer):
    """Capture values events as a drainable stream of state snapshots.

    Provides the `run.values` projection; `run.output`, `run.interrupted` and `run.interrupts` are
    tracked here as well. Only events at the run's own scope are captured.
    """

    _native = True
    required_stream_modes = ("values",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[dict[str, Any]] = StreamChannel()
        self._latest: dict[str, Any] | None = None
        self._interrupted = False
        self._interrupts: list[Any] = []
        self._scope_list: list[str] = list(scope)

    def init(self) -> dict[str, Any]:
        return {"values": self._log}

    @property
    def error(self) -> BaseException | None:
        """The error that ended the run, or `None` if it succeeded."""
        return self._log._error

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "values":
            return True
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True
        self._latest = params["data"]
        interrupts = params.get("interrupts", ())
        if interrupts:
            self._interrupted = True
            self._interrupts.extend(interrupts)
        self._log.push(params["data"])
        return True


class CustomTransformer(StreamTransformer):
    """Capture custom events as a drainable stream of arbitrary payloads.

    Nodes emit custom data via `get_stream_writer()`; this surfaces them on `run.custom`.
    """

    _native = True
    required_stream_modes = ("custom",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[Any] = StreamChannel()
        self._scope_list: list[str] = list(scope)

    def init(self) -> dict[str, Any]:
        return {"custom": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "custom":
            return True
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True
        self._log.push(params["data"])
        return True


class UpdatesTransformer(StreamTransformer):
    """Capture updates events as a drainable stream of node outputs (`run.updates`)."""

    _native = True
    required_stream_modes = ("updates",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[dict[str, Any]] = StreamChannel()
        self._scope_list: list[str] = list(scope)

    def init(self) -> dict[str, Any]:
        return {"updates": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "updates":
            return True
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True
        self._log.push(params["data"])
        return True


class CheckpointsTransformer(StreamTransformer):
    """Capture checkpoint events as a drainable stream (`run.checkpoints`)."""

    _native = True
    required_stream_modes = ("checkpoints",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[dict[str, Any]] = StreamChannel()
        self._scope_list: list[str] = list(scope)

    def init(self) -> dict[str, Any]:
        return {"checkpoints": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "checkpoints":
            return True
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True
        self._log.push(params["data"])
        return True


class DebugTransformer(StreamTransformer):
    """Capture debug events as a drainable stream (`run.debug`)."""

    _native = True
    required_stream_modes = ("debug",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[dict[str, Any]] = StreamChannel()
        self._scope_list: list[str] = list(scope)

    def init(self) -> dict[str, Any]:
        return {"debug": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "debug":
            return True
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True
        self._log.push(params["data"])
        return True


class TasksTransformer(StreamTransformer):
    """Capture raw task events as a drainable stream (`run.tasks`)."""

    _native = True
    required_stream_modes = ("tasks",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[dict[str, Any]] = StreamChannel()
        self._scope_list: list[str] = list(scope)

    def init(self) -> dict[str, Any]:
        return {"tasks": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "tasks":
            return True
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True
        self._log.push(params["data"])
        return True


