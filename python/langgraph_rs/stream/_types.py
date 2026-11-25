"""Stream protocol primitives, mirroring ``langgraph.stream._types``.

`ProtocolEvent` is the uniform envelope every transformer observes, and `StreamTransformer` is the
extension point: subclasses implement `init()` plus `process()`/`aprocess()` and optionally the
lifecycle hooks.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from typing import Any, ClassVar, Literal

try:  # Python 3.13 ships the aliases in typing.
    from typing import NotRequired, TypedDict
except ImportError:  # pragma: no cover - 3.11/3.12
    from typing_extensions import NotRequired, TypedDict

_logger = logging.getLogger(__name__)


class _ProtocolEventParams(TypedDict):
    """Parameters of a protocol event.

    `timestamp` is wall-clock milliseconds and can go backwards; use `ProtocolEvent["seq"]` for a
    total order.
    """

    namespace: list[str]
    timestamp: int
    data: Any
    interrupts: NotRequired[tuple[Any, ...]]


class ProtocolEvent(TypedDict):
    """A protocol event emitted by the streaming infrastructure."""

    type: Literal["event"]
    event_id: NotRequired[str]
    seq: NotRequired[int]
    method: str
    params: _ProtocolEventParams


class StreamTransformer(ABC):
    """Extension point for custom stream projections.

    A transformer observes protocol events and builds derived projections (usually
    `StreamChannel`s). Set `_native = True` to expose the projection keys as attributes on the run
    stream.

    Subclasses must implement `init` and override at least one of `process` / `aprocess`.
    """

    requires_async: ClassVar[bool] = False
    supports_sync: ClassVar[bool] = False
    required_stream_modes: ClassVar[tuple[str, ...]] = ()
    before_builtins: ClassVar[bool] = False
    _native: ClassVar[bool] = False

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        self.scope: tuple[str, ...] = scope

    @abstractmethod
    def init(self) -> dict[str, Any]:
        """Return the projection dict (keys become entries in `run.extensions`)."""
        ...

    def _on_register(self, mux: Any) -> None:
        """Called by the mux after this transformer is wired in (default no-op)."""

    def process(self, event: ProtocolEvent) -> bool:
        """Handle an event on the sync lane; return `False` to suppress it from the log."""
        raise NotImplementedError(f"{type(self).__name__} must override process() or aprocess()")

    async def aprocess(self, event: ProtocolEvent) -> bool:
        """Handle an event on the async lane (defaults to `process`)."""
        return self.process(event)

    def finalize(self) -> None:
        """Called when the run ends normally on the sync lane."""

    async def afinalize(self) -> None:
        """Called when the run ends normally on the async lane (defaults to `finalize`)."""
        self.finalize()

    def fail(self, err: BaseException) -> None:
        """Called when the run ends with an error on the sync lane."""

    async def afail(self, err: BaseException) -> None:
        """Called when the run ends with an error on the async lane (defaults to `fail`)."""
        self.fail(err)

    def schedule(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        on_error: Literal["log", "raise"] = "log",
    ) -> asyncio.Task[Any]:
        """Schedule a coroutine whose lifecycle the run owns."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                f"{type(self).__name__}.schedule() requires a running event loop; "
                "this transformer must run under astream(), not stream(). "
                "Set requires_async=True on the class so this fails at registration."
            ) from None

        wrapped = self._wrap_scheduled(coro) if on_error == "log" else coro
        task = asyncio.create_task(wrapped)
        tasks = self._scheduled_task_set()
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return task

    @staticmethod
    async def _wrap_scheduled(coro: Coroutine[Any, Any, Any]) -> Any:
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except BaseException:
            _logger.exception("Scheduled StreamTransformer task failed")

    def _scheduled_task_set(self) -> set[asyncio.Task[Any]]:
        tasks: set[asyncio.Task[Any]] | None = getattr(self, "_stream_scheduled_tasks", None)
        if tasks is None:
            tasks = set()
            self._stream_scheduled_tasks = tasks
        return tasks


def transformer_requires_async(transformer: StreamTransformer) -> bool:
    """Whether a transformer needs a running event loop."""
    if transformer.requires_async:
        return True
    if transformer.supports_sync:
        return False
    cls = type(transformer)
    for name in ("aprocess", "afinalize", "afail"):
        if getattr(cls, name) is not getattr(StreamTransformer, name):
            return True
    return False