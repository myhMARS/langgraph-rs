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


_MISSING = fill_unsupported(
    globals(),
    ["LifecycleTransformer", "MessagesTransformer", "SubgraphTransformer"],
    hint=(
        "These transformers need the stream_events(version='v3') run-stream handles; "
        "use graph.stream(stream_mode=...) meanwhile."
    ),
)