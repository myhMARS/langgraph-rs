"""Central event dispatcher for the stream transformer pipeline.

Mirrors ``langgraph.stream._mux``: a `StreamMux` owns the main event log, routes every
`ProtocolEvent` through the registered transformers, merges their projections into `extensions`,
and auto-wires named `StreamChannel`s so a `push()` also lands in the event log.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ._types import ProtocolEvent, StreamTransformer, transformer_requires_async
from .stream_channel import StreamChannel

TransformerFactory = Callable[["tuple[str, ...]"], StreamTransformer]
"""Builds a scoped transformer; called once per mux (and per child scope)."""


class StreamMux:
    """Owns the main event log and the transformer pipeline."""

    def __init__(
        self,
        transformers: list[StreamTransformer] | None = None,
        *,
        is_async: bool = False,
        factories: list[TransformerFactory] | None = None,
        scope: tuple[str, ...] = (),
        _assign_seq: bool = True,
    ) -> None:
        self.is_async = is_async
        self.scope: tuple[str, ...] = scope
        self._assign_seq = _assign_seq
        self._events: StreamChannel[ProtocolEvent] = StreamChannel()
        self._events._bind(is_async=is_async)
        self._events._bind_mux(self)
        self._transformers: list[StreamTransformer] = []
        self._channels: list[StreamChannel[Any]] = []
        self._seq = 0
        self._push_seq = 0

        self.extensions: dict[str, Any] = {}
        self.native_keys: set[str] = set()
        self._projection_owners: dict[str, str] = {}
        self._transformer_by_key: dict[str, StreamTransformer] = {}

        self._factories: list[TransformerFactory] | None = (
            list(factories) if factories is not None else None
        )
        self._pump_fn: Callable[[], bool] | None = None
        self._apump_fn: Callable[[], Awaitable[bool]] | None = None

        pre: list[StreamTransformer] = []
        rest: list[StreamTransformer] = []
        if factories is not None:
            for factory in factories:
                transformer = factory(scope)
                (pre if getattr(transformer, "before_builtins", False) else rest).append(
                    transformer
                )
            for transformer in (*pre, *rest):
                self._register(transformer)
            pre.clear()
            rest.clear()
        for transformer in transformers or ():
            (pre if getattr(transformer, "before_builtins", False) else rest).append(
                transformer
            )
        for transformer in (*pre, *rest):
            self._register(transformer)

    def transformer_by_key(self, key: str) -> StreamTransformer | None:
        """Return the transformer that contributed `key` to the projection."""
        return self._transformer_by_key.get(key)

    def _next_push_seq(self) -> int:
        self._push_seq += 1
        return self._push_seq

    # -- pump wiring + mini-mux nesting ---------------------------------

    def bind_pump(self, fn: Callable[[], bool]) -> None:
        """Wire the sync pull callback onto every projection in this mux."""
        self._pump_fn = fn
        self._events._request_more = fn
        for channel in self._channels:
            channel._request_more = fn
        for transformer in self._transformers:
            bind = getattr(transformer, "_bind_pump", None)
            if bind is not None:
                bind(fn)

    def bind_apump(self, fn: Callable[[], Awaitable[bool]]) -> None:
        """Async counterpart to `bind_pump`."""
        self._apump_fn = fn
        self._events._arequest_more = fn
        for channel in self._channels:
            channel._arequest_more = fn
        for transformer in self._transformers:
            abind = getattr(transformer, "_bind_apump", None)
            if abind is not None:
                abind(fn)

    def _make_child(self, scope: tuple[str, ...]) -> StreamMux:
        """Build a mini-mux with the same factories scoped to `scope`."""
        if self._factories is None:
            raise RuntimeError(
                "StreamMux._make_child requires the mux to be constructed with `factories=`; "
                "pre-built transformers can't be cloned to a new scope."
            )
        child = StreamMux(
            factories=self._factories,
            is_async=self.is_async,
            scope=scope,
            _assign_seq=False,
        )
        if self._pump_fn is not None:
            child.bind_pump(self._pump_fn)
        if self._apump_fn is not None:
            child.bind_apump(self._apump_fn)
        return child

    def _register(self, transformer: StreamTransformer) -> None:
        """Call `init()`, bind/wire its channels and merge the projection."""
        if transformer_requires_async(transformer) and not self.is_async:
            raise RuntimeError(
                f"{type(transformer).__name__} requires an async run — it overrides "
                "aprocess/afinalize/afail or sets requires_async=True. Use astream(), not stream()."
            )
        projection = transformer.init()
        if not isinstance(projection, dict):
            raise TypeError(
                "StreamTransformer.init() must return a dict, "
                f"got {type(projection).__name__}"
            )
        conflicts = set(projection) & set(self.extensions)
        if conflicts:
            attributions = ", ".join(
                f"{key!r} (owned by {self._projection_owners[key]})"
                for key in sorted(conflicts)
            )
            raise ValueError(
                f"Transformer {type(transformer).__name__} returned projection keys that "
                f"conflict with already-registered keys: {attributions}"
            )
        is_native = bool(getattr(transformer, "_native", False))
        self._transformers.append(transformer)
        self._bind_and_wire(projection, native=is_native)
        self.extensions.update(projection)
        owner_name = type(transformer).__name__
        for key in projection:
            self._projection_owners[key] = owner_name
            self._transformer_by_key[key] = transformer
        if is_native:
            self.native_keys.update(projection.keys())
        transformer._on_register(self)

    # -- dispatch -------------------------------------------------------

    def push(self, event: ProtocolEvent) -> None:
        """Route an event through the transformers, then append it to the main log."""
        keep = True
        for transformer in self._transformers:
            if not transformer.process(event):
                keep = False
        if keep:
            if self._assign_seq:
                self._seq += 1
                event["seq"] = self._seq
            self._events.push(event)

    def close(self) -> None:
        """Finalize transformers, then close the projections and the main log."""
        first_error: BaseException | None = None
        for transformer in self._transformers:
            try:
                transformer.finalize()
            except BaseException as error:  # noqa: BLE001 - re-raised after cleanup
                if first_error is None:
                    first_error = error
        for channel in self._channels:
            if not channel._closed:
                channel.close()
        self._events.close()
        if first_error is not None:
            raise first_error

    def fail(self, err: BaseException) -> None:
        """Fail transformers, projections and the main log."""
        for transformer in self._transformers:
            try:
                transformer.fail(err)
            except BaseException:  # noqa: BLE001,S110 - cleanup must continue
                pass
        for channel in self._channels:
            if not channel._closed:
                channel.fail(err)
        self._events.fail(err)

    # -- async dispatch -------------------------------------------------

    async def apush(self, event: ProtocolEvent) -> None:
        """Dispatch an event on the async lane."""
        keep = True
        for transformer in self._transformers:
            if not await transformer.aprocess(event):
                keep = False
        if keep:
            if self._assign_seq:
                self._seq += 1
                event["seq"] = self._seq
            self._events.push(event)

    async def aclose(self) -> None:
        """Await scheduled tasks, finalize transformers, close the projections."""
        pending = self._collect_scheduled_tasks()
        if pending:
            results = await asyncio.gather(*pending, return_exceptions=True)
            first_error = next(
                (
                    result
                    for result in results
                    if isinstance(result, BaseException)
                    and not isinstance(result, asyncio.CancelledError)
                ),
                None,
            )
            if first_error is not None:
                raise first_error

        first_error = None
        for transformer in self._transformers:
            try:
                await transformer.afinalize()
            except BaseException as error:  # noqa: BLE001 - re-raised after cleanup
                if first_error is None:
                    first_error = error
        for channel in self._channels:
            if not channel._closed:
                channel.close()
        self._events.close()
        if first_error is not None:
            raise first_error

    async def afail(self, err: BaseException) -> None:
        """Cancel scheduled tasks, run the `afail` hooks and fail the channels."""
        pending = self._collect_scheduled_tasks()
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        for transformer in self._transformers:
            try:
                await transformer.afail(err)
            except BaseException:  # noqa: BLE001,S110 - cleanup must continue
                pass
        for channel in self._channels:
            if not channel._closed:
                channel.fail(err)
        if not self._events._closed:
            self._events.fail(err)

    def _collect_scheduled_tasks(self) -> list[asyncio.Task[Any]]:
        """Snapshot of in-flight tasks scheduled via transformers."""
        return [
            task
            for transformer in self._transformers
            for task in getattr(transformer, "_stream_scheduled_tasks", ())
            if not task.done()
        ]

    # -- channel binding + auto-wiring ----------------------------------

    def _bind_and_wire(self, projection: dict[str, Any], *, native: bool = False) -> None:
        """Bind every `StreamChannel` in a projection; wire named ones for forwarding."""
        for value in projection.values():
            if isinstance(value, StreamChannel):
                value._bind(is_async=self.is_async)
                value._bind_mux(self)
                self._channels.append(value)
                if value.name is not None:
                    method = value.name if native else f"custom:{value.name}"

                    def _make_forward(method_name: str) -> Callable[[Any], None]:
                        def _forward(item: Any) -> None:
                            self._forward(method_name, item)

                        return _forward

                    value._wire(_make_forward(method))

    def _forward(self, method: str, item: Any) -> None:
        """Inject a protocol event for a channel push (bypassing the pipeline)."""
        event: ProtocolEvent = {
            "type": "event",
            "method": method,
            "params": {
                "namespace": [],
                "timestamp": int(time.time() * 1000),
                "data": item,
            },
        }
        if self._assign_seq:
            self._seq += 1
            event["seq"] = self._seq
        self._events.push(event)