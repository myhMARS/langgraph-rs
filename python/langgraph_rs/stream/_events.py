"""`stream_events(version="v3")` for the compiled graph, built on top of `stream()`.

LangGraph drives its v3 protocol with a `StreamMux` fed by the graph's own stream. The extension
implements the same shape in Python: the union of the registered transformers' required stream modes
is requested from `stream()`, every part is converted to a `ProtocolEvent`, and the caller's
iteration on a projection is what pulls the graph forward.

`messages`, `lifecycle` and `subgraphs` are not registered yet (they need the model-stream and
subgraph-handle machinery), so a run only exposes the capture projections.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from ._convert import convert_to_protocol_event
from ._mux import StreamMux
from .run_stream import GraphRunStream
from .transformers import (
    CheckpointsTransformer,
    CustomTransformer,
    DebugTransformer,
    LifecycleTransformer,
    SubgraphTransformer,
    TasksTransformer,
    UpdatesTransformer,
    ValuesTransformer,
)

# LangGraph registers `values` (plus its messages/lifecycle/subgraphs transformers, which the
# extension does not implement yet) by default; the other capture projections are opt-in through
# `transformers=` or the graph's compile-time transformers.
DEFAULT_TRANSFORMERS: tuple[Callable[[tuple[str, ...]], Any], ...] = (
    ValuesTransformer,
    LifecycleTransformer,
    SubgraphTransformer,
)


def _factories(
    extra: Sequence[Callable[[tuple[str, ...]], Any]] | None,
    compile_time: Sequence[Callable[[tuple[str, ...]], Any]] | None = None,
) -> list[Callable[[tuple[str, ...]], Any]]:
    """Built-ins, then the graph's compile-time transformers, then call-site ones."""
    factories = list(DEFAULT_TRANSFORMERS)
    for transformer in compile_time or ():
        factories.append(transformer)
    for transformer in extra or ():
        factories.append(transformer)
    return factories


def _required_modes(factories: Sequence[Callable[[tuple[str, ...]], Any]]) -> list[str]:
    modes: list[str] = []
    for factory in factories:
        transformer = factory(())
        for mode in getattr(transformer, "required_stream_modes", ()):
            if mode not in modes:
                modes.append(mode)
    return modes


def _needs_subgraphs(factories: Sequence[Callable[[tuple[str, ...]], Any]]) -> bool:
    """Whether the registered transformers need child-namespace events."""
    for factory in factories:
        transformer = factory(())
        if type(transformer).__name__ in {"SubgraphTransformer", "LifecycleTransformer"}:
            return True
    return False


def _live_parts(
    graph: Any,
    input: Any,
    bridge: Any,
    *,
    thread_id: str | None,
    modes: list[str],
    subgraphs: bool,
    kwargs: dict[str, Any],
) -> Iterator[Any]:
    """Yield live `messages` parts while the graph runs, then the engine's own parts.

    The graph runs on a worker thread so this generator can hand out tokens as the callback handler
    pushes them. Message parts from the engine are dropped once the bridge captured any, since the
    bridge is the more timely source for the same tokens.
    """
    engine: "queue.Queue[Any]" = queue.Queue()
    result: dict[str, Any] = {}

    def run() -> None:
        try:
            for part in graph.stream(
                input,
                thread_id=thread_id,
                stream_mode=modes,
                version="v2",
                subgraphs=subgraphs,
                **kwargs,
            ):
                engine.put(part)
        except BaseException as error:  # noqa: BLE001 - re-raised in the consumer
            result["error"] = error
        finally:
            engine.put(None)

    from .._live import activate, deactivate

    activate(bridge)
    worker = threading.Thread(target=run, name="langgraph_rs-stream-events", daemon=True)
    worker.start()
    try:
        finished = False
        deferred: list[Any] = []
        while True:
            for payload, metadata in bridge.drain():
                yield {
                    "type": "messages",
                    "ns": (),
                    "data": (payload, metadata),
                    "interrupts": (),
                }
            try:
                part = engine.get_nowait()
            except queue.Empty:
                if bridge._closed.is_set() and not worker.is_alive():
                    break
                continue
            if part is None:
                finished = True
                break
            if isinstance(part, dict) and part.get("type") == "messages":
                if bridge.count:
                    continue
            deferred.append(part)
        # Everything the engine produced (non-message parts, plus messages when the bridge was idle).
        if finished:
            while True:
                try:
                    part = engine.get_nowait()
                except queue.Empty:
                    break
                if part is None:
                    continue
                deferred.append(part)
        for part in deferred:
            yield part
        error = result.get("error")
        if error is not None:
            raise error
    finally:
        deactivate(bridge)
        worker.join(timeout=5)


def stream_events(
    graph: Any,
    input: Any = None,
    *,
    version: str = "v3",
    transformers: Sequence[Callable[[tuple[str, ...]], Any]] | None = None,
    compile_transformers: Sequence[Callable[[tuple[str, ...]], Any]] | None = None,
    thread_id: str | None = None,
    live: bool = True,
    **kwargs: Any,
) -> GraphRunStream:
    """Return a `GraphRunStream` the caller drives by iterating its projections."""
    if version != "v3":
        raise NotImplementedError(
            "sync stream_events only supports version='v3'; langchain-core has no sync v1/v2 "
            "event streaming either — use astream_events for those versions."
        )
    for rejected in ("stream_mode", "subgraphs"):
        if rejected in kwargs:
            raise TypeError(
                f"stream_events(version='v3') does not accept {rejected!r}; "
                "v3 owns the stream modes."
            )
    factories = _factories(transformers, compile_transformers)
    mux = StreamMux(factories=factories, is_async=False)
    # `version="v2"` makes every part a `StreamPart` dict (type/ns/data), which is what the
    # protocol conversion expects; `subgraphs=True` is what surfaces child namespaces.
    modes = _required_modes(factories)
    subgraphs = _needs_subgraphs(factories)

    if live and "messages" in modes:
        from .._live import LiveBridge

        return GraphRunStream(
            _live_parts(
                graph,
                input,
                LiveBridge(),
                thread_id=thread_id,
                modes=modes,
                subgraphs=subgraphs,
                kwargs=kwargs,
            ),
            mux,
        )

    def lazy_stream() -> Iterator[Any]:
        # Deferred so a graph that fails on the first step reports through the run stream
        # (`lifecycle` / handle failure) instead of raising before the stream exists.
        yield from graph.stream(
            input,
            thread_id=thread_id,
            stream_mode=modes,
            version="v2",
            subgraphs=subgraphs,
            **kwargs,
        )

    return GraphRunStream(lazy_stream(), mux)


async def astream_events(
    graph: Any,
    input: Any = None,
    *,
    version: str = "v3",
    transformers: Sequence[Callable[[tuple[str, ...]], Any]] | None = None,
    compile_transformers: Sequence[Callable[[tuple[str, ...]], Any]] | None = None,
    thread_id: str | None = None,
    **kwargs: Any,
) -> Any:
    """Async counterpart; returns an `AsyncGraphRunStream` wrapping the async stream."""
    if version != "v3":
        raise NotImplementedError(
            "astream_events only supports version='v3' in the extension."
        )
    for rejected in ("stream_mode", "subgraphs"):
        if rejected in kwargs:
            raise TypeError(
                f"astream_events(version='v3') does not accept {rejected!r}; v3 owns the stream modes."
            )
    factories = _factories(transformers, compile_transformers)
    mux = StreamMux(factories=factories, is_async=True)
    modes = _required_modes(factories)
    subgraphs = _needs_subgraphs(factories)

    async def lazy_astream():
        async for part in graph.astream(
            input,
            thread_id=thread_id,
            stream_mode=modes,
            version="v2",
            subgraphs=subgraphs,
            **kwargs,
        ):
            yield part

    return AsyncGraphRunStream(lazy_astream(), mux)


from .run_stream import AsyncGraphRunStream  # noqa: E402  (circular-safe tail import)