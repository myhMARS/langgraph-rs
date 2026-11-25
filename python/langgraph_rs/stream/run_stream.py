"""Run-stream wrappers, mirroring ``langgraph.stream.run_stream``.

`GraphRunStream` is a caller-driven pump: iterating any projection pulls one graph event, converts
it to a `ProtocolEvent` and pushes it through the `StreamMux`. `output`, `interrupted` and
`interrupts` drive the run to completion on demand.

The subgraph handles (`SubgraphRunStream`, `AsyncSubgraphRunStream`) need the subgraph discovery
pipeline and stay placeholders.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from types import MappingProxyType, TracebackType
from typing import Any, NoReturn

try:  # LangGraph marks the v3 run streams as beta; keep the same module surface.
    from langchain_core._api import beta
except Exception:  # pragma: no cover - langchain-core absent
    def beta(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

from ._convert import convert_to_protocol_event
from ._mux import StreamMux
from ._types import ProtocolEvent
from .stream_channel import StreamChannel

__all__ = [
    "AsyncGraphRunStream",
    "AsyncSubgraphRunStream",
    "GraphRunStream",
    "SubgraphRunStream",
]


def _as_interrupt(item: Any) -> Any:
    """Normalize a values-event interrupt into `langgraph_rs.Interrupt` when possible.

    The v2 stream part carries `(value, id)` tuples; LangGraph's v3 protocol exposes `Interrupt`
    objects, so `run.interrupts` matches that shape.
    """
    try:
        from langgraph_rs.types import Interrupt
    except Exception:  # pragma: no cover - native module missing
        return item
    if isinstance(item, (tuple, list)) and len(item) == 1:
        return _as_interrupt(item[0])
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        value, interrupt_id = item[0], item[1]
        try:
            interrupt = Interrupt(value)
        except Exception:  # pragma: no cover - unexpected payload shape
            return item
        try:
            interrupt.id = str(interrupt_id)
        except Exception:  # pragma: no cover - id may be read-only
            pass
        return interrupt
    return item

def _drive_until_done(pump: Callable[[], bool]) -> None:
    """Call the sync pump until it returns False."""
    while pump():
        pass


async def _adrive_until_done(pump: Callable[[], Awaitable[bool]]) -> None:
    """Call the async pump until it returns False."""
    while await pump():
        pass


def _raise_missing_projection(run: object, name: str) -> NoReturn:
    """Raise after normal attribute lookup fails for a projection."""
    mux = run.__dict__.get("_mux")
    registered = sorted(mux.native_keys) if mux is not None else []
    raise AttributeError(
        f"{type(run).__name__!r} object has no attribute {name!r} "
        f"(registered projections: {', '.join(registered) or 'none'})"
    )


class GraphRunStream:
    """Sync run stream with caller-driven pumping.

    Projections are single-consumer — iterating `run.values` twice raises. Use `tee(n)` for fan-out.
    """

    values: StreamChannel[dict[str, Any]]
    updates: StreamChannel[dict[str, Any]]
    custom: StreamChannel[Any]
    checkpoints: StreamChannel[dict[str, Any]]
    debug: StreamChannel[dict[str, Any]]
    tasks: StreamChannel[dict[str, Any]]

    def __getattr__(self, name: str) -> StreamChannel[Any]:
        _raise_missing_projection(self, name)

    def __init__(
        self,
        graph_iter: Iterator[Any] | None,
        mux: StreamMux,
        *,
        wire_pump: bool = True,
    ) -> None:
        self._graph_iter = graph_iter
        self._mux = mux
        self.extensions: Mapping[str, Any] = MappingProxyType(mux.extensions)
        self._exhausted = False
        self._latest: dict[str, Any] | None = None
        self._interrupted = False
        self._interrupts: list[Any] = []
        self._scope_list: list[str] = list(mux.scope)
        for key in mux.native_keys:
            setattr(self, key, mux.extensions[key])
        if wire_pump:
            self._wire_request_more(mux)

    def _wire_request_more(self, mux: StreamMux) -> None:
        """Wire the sync pull callback through the mux."""
        mux.bind_pump(self._pump_next)

    def _observe_event(self, event: ProtocolEvent) -> None:
        """Track values-event state for output/interrupted/interrupts."""
        if event["method"] != "values":
            return
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return
        self._latest = params["data"]
        interrupts = params.get("interrupts", ())
        if interrupts:
            self._interrupted = True
            self._interrupts.extend(_as_interrupt(item) for item in interrupts)

    def _pump_next(self) -> bool:
        """Pull one event from the graph and push it through the mux."""
        if self._exhausted or self._graph_iter is None:
            return False
        try:
            part = next(self._graph_iter)
            event = convert_to_protocol_event(part)
            self._observe_event(event)
            self._mux.push(event)
            return True
        except StopIteration:
            self._mux.close()
            self._exhausted = True
            return False
        except Exception as error:  # noqa: BLE001 - surfaced through the channel
            self._mux.fail(error)
            self._exhausted = True
            return False

    def abort(self) -> None:
        """Stop the run early (idempotent)."""
        if self._exhausted:
            return
        self._exhausted = True
        graph_iter = self._graph_iter
        self._graph_iter = None
        if graph_iter is not None and (close := getattr(graph_iter, "close", None)) is not None:
            try:
                close()
            except Exception:  # noqa: BLE001,S110 - abort must not raise
                pass
        try:
            self._mux.close()
        except Exception:  # noqa: BLE001,S110 - abort must not raise
            pass

    def __enter__(self) -> GraphRunStream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.abort()

    @property
    def output(self) -> dict[str, Any] | None:
        """Drive the run to completion and return the final state."""
        _drive_until_done(self._pump_next)
        if (error := self._mux._events._error) is not None:
            raise error
        return self._latest

    @property
    def interrupted(self) -> bool:
        """Drive the run to completion, then report whether it was interrupted."""
        _drive_until_done(self._pump_next)
        if (error := self._mux._events._error) is not None:
            raise error
        return self._interrupted

    @property
    def interrupts(self) -> list[Any]:
        """Drive the run to completion, then return the interrupt payloads."""
        _drive_until_done(self._pump_next)
        if (error := self._mux._events._error) is not None:
            raise error
        return self._interrupts

    def __iter__(self) -> Iterator[ProtocolEvent]:
        """Subscribe to the main event log and iterate protocol events."""
        return iter(self._mux._events)

    def interleave(self, *names: str) -> Iterator[tuple[str, Any]]:
        """Iterate several projections in arrival order, yielding `(name, item)`."""
        channels = {name: getattr(self, name, None) for name in names}
        for name, channel in channels.items():
            if channel is None:
                raise AttributeError(f"no projection named {name!r}")
        cursors = {name: iter(channel) for name, channel in channels.items()}
        while cursors:
            progressed = False
            for name in list(cursors):
                try:
                    item = next(cursors[name])
                except StopIteration:
                    del cursors[name]
                    continue
                progressed = True
                yield name, item
            if not progressed:
                return


class AsyncGraphRunStream:
    """Async run stream with caller-driven pumping."""

    values: StreamChannel[dict[str, Any]]
    updates: StreamChannel[dict[str, Any]]
    custom: StreamChannel[Any]
    checkpoints: StreamChannel[dict[str, Any]]
    debug: StreamChannel[dict[str, Any]]
    tasks: StreamChannel[dict[str, Any]]

    def __getattr__(self, name: str) -> StreamChannel[Any]:
        _raise_missing_projection(self, name)

    def __init__(
        self,
        graph_aiter: Any,
        mux: StreamMux,
        *,
        wire_pump: bool = True,
    ) -> None:
        self._graph_aiter = graph_aiter
        self._mux = mux
        self.extensions: Mapping[str, Any] = MappingProxyType(mux.extensions)
        self._exhausted = False
        self._latest: dict[str, Any] | None = None
        self._interrupted = False
        self._interrupts: list[Any] = []
        self._scope_list: list[str] = list(mux.scope)
        self._pumping = False
        self._pump_cond = asyncio.Condition()
        for key in mux.native_keys:
            setattr(self, key, mux.extensions[key])
        if wire_pump:
            mux.bind_apump(self._apump_next)

    def _observe_event(self, event: ProtocolEvent) -> None:
        """Track values-event state for output/interrupted/interrupts."""
        if event["method"] != "values":
            return
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return
        self._latest = params["data"]
        interrupts = params.get("interrupts", ())
        if interrupts:
            self._interrupted = True
            self._interrupts.extend(_as_interrupt(item) for item in interrupts)

    async def _apump_next(self) -> bool:
        """Drive one pump step, or wait for the active pumper to drive one."""
        async with self._pump_cond:
            if self._exhausted or self._graph_aiter is None:
                return False
            if self._pumping:
                await self._pump_cond.wait()
                return not self._exhausted
            self._pumping = True
        try:
            try:
                part = await self._graph_aiter.__anext__()
                event = convert_to_protocol_event(part)
                self._observe_event(event)
                await self._mux.apush(event)
                return True
            except StopAsyncIteration:
                self._exhausted = True
                await self._mux.aclose()
                return False
            except Exception as error:  # noqa: BLE001 - surfaced through the channel
                self._exhausted = True
                await self._mux.afail(error)
                return False
        finally:
            async with self._pump_cond:
                self._pumping = False
                self._pump_cond.notify_all()

    async def abort(self) -> None:
        """Stop the run early (idempotent)."""
        async with self._pump_cond:
            if self._exhausted:
                return
            self._exhausted = True
            graph_aiter = self._graph_aiter
            self._graph_aiter = None
            self._pump_cond.notify_all()
        if graph_aiter is not None and (aclose := getattr(graph_aiter, "aclose", None)) is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001,S110 - abort must not raise
                pass
        try:
            await self._mux.aclose()
        except Exception:  # noqa: BLE001,S110 - abort must not raise
            pass

    async def __aenter__(self) -> AsyncGraphRunStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.abort()

    async def output(self) -> dict[str, Any] | None:
        """Drive the run to completion and return the final state."""
        await _adrive_until_done(self._apump_next)
        if (error := self._mux._events._error) is not None:
            raise error
        return self._latest

    async def interrupted(self) -> bool:
        """Drive the run to completion, then report whether it was interrupted."""
        await _adrive_until_done(self._apump_next)
        if (error := self._mux._events._error) is not None:
            raise error
        return self._interrupted

    async def interrupts(self) -> list[Any]:
        """Drive the run to completion, then return the interrupt payloads."""
        await _adrive_until_done(self._apump_next)
        if (error := self._mux._events._error) is not None:
            raise error
        return self._interrupts

    def __aiter__(self) -> AsyncIterator[ProtocolEvent]:
        """Subscribe to the main event log and iterate protocol events."""
        return self._mux._events.__aiter__()


class _SubgraphRunStreamMixin:
    """Subgraph metadata plus parent-pump delegation for both lanes.

    A handle is built on a child mini-mux (`StreamMux._make_child`) whose pump is the parent's, so
    iterating a handle's projections drives the whole run.
    """

    path: tuple[str, ...]
    graph_name: str | None
    trigger_call_id: str | None
    status: str
    error: str | None
    _seen_terminal: bool


class SubgraphRunStream(GraphRunStream, _SubgraphRunStreamMixin):
    """Sync handle for a discovered subgraph."""

    def __init__(
        self,
        mux: StreamMux,
        *,
        path: tuple[str, ...],
        graph_name: str | None = None,
        trigger_call_id: str | None = None,
    ) -> None:
        self._parent_pump_fn: Callable[[], bool] | None = mux._pump_fn
        super().__init__(graph_iter=None, mux=mux, wire_pump=False)
        self.path = path
        self.graph_name = graph_name
        self.trigger_call_id = trigger_call_id
        self.status = "started"
        self.error = None
        self._seen_terminal = False

    def _pump_next(self) -> bool:
        """Delegate to the parent's pump."""
        if (
            self._exhausted
            or self._seen_terminal
            or self._mux._events._closed
            or self._parent_pump_fn is None
        ):
            return False
        return self._parent_pump_fn()


class AsyncSubgraphRunStream(AsyncGraphRunStream, _SubgraphRunStreamMixin):
    """Async handle for a discovered subgraph."""

    def __init__(
        self,
        mux: StreamMux,
        *,
        path: tuple[str, ...],
        graph_name: str | None = None,
        trigger_call_id: str | None = None,
    ) -> None:
        self._parent_apump_fn: Callable[[], Awaitable[bool]] | None = mux._apump_fn
        super().__init__(graph_aiter=None, mux=mux, wire_pump=False)
        self.path = path
        self.graph_name = graph_name
        self.trigger_call_id = trigger_call_id
        self.status = "started"
        self.error = None
        self._seen_terminal = False

    async def _apump_next(self) -> bool:
        """Delegate to the parent's async pump."""
        if (
            self._exhausted
            or self._seen_terminal
            or self._mux._events._closed
            or self._parent_apump_fn is None
        ):
            return False
        return await self._parent_apump_fn()