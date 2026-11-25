"""`stream_events(version="v3")` for the compiled graph, built on top of `stream()`.

LangGraph drives its v3 protocol with a `StreamMux` fed by the graph's own stream. The extension
implements the same shape in Python: the union of the registered transformers' required stream modes
is requested from `stream()`, every part is converted to a `ProtocolEvent`, and the caller's
iteration on a projection is what pulls the graph forward.

`messages`, `lifecycle` and `subgraphs` are not registered yet (they need the model-stream and
subgraph-handle machinery), so a run only exposes the capture projections.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
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
            # `stream_live` pushes every chunk as the engine produces it (node writes included);
            # graphs without it fall back to the collected `stream()`.
            source = getattr(graph, "stream_live", None)
            if source is not None:
                iterator = source(
                    input,
                    thread_id=thread_id,
                    stream_mode=modes,
                    version="v2",
                    subgraphs=subgraphs,
                    **kwargs,
                )
            else:
                iterator = graph.stream(
                    input,
                    thread_id=thread_id,
                    stream_mode=modes,
                    version="v2",
                    subgraphs=subgraphs,
                    **kwargs,
                )
            if hasattr(iterator, "poll"):
                # Pull-based live iterator: waiting happens here so the GIL is released for the
                # worker thread that runs the graph.
                while True:
                    try:
                        part = iterator.poll()
                    except StopIteration:
                        break
                    if part is None:
                        time.sleep(0.001)
                        continue
                    engine.put(part)
            else:
                for part in iterator:
                    engine.put(part)
        except BaseException as error:  # noqa: BLE001 - re-raised in the consumer
            result["error"] = error
        finally:
            engine.put(None)

    from .._live import activate, deactivate

    activate(bridge, thread_id)
    worker = threading.Thread(target=run, name="langgraph_rs-stream-events", daemon=True)
    worker.start()
    try:
        finished = False
        deferred: list[Any] = []
        while True:
            # Poll instead of blocking: a blocking drain here would delay engine chunks by that
            # timeout even when nothing is streaming.
            for payload, metadata in bridge.drain(timeout=0.0):
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
                time.sleep(0.001)
                continue
            if part is None:
                finished = True
                break

            if isinstance(part, dict) and part.get("type") == "messages":
                if bridge.count:
                    continue
            # Engine parts are produced in order, so hand them out as they arrive.
            yield part
        # Tokens pushed after the last drain must not be lost.
        for payload, metadata in bridge.drain():
            yield {
                "type": "messages",
                "ns": (),
                "data": (payload, metadata),
                "interrupts": (),
            }
        # Drain anything the worker queued after the sentinel check.
        while True:
            try:
                part = engine.get_nowait()
            except queue.Empty:
                break
            if part is None:
                continue
            if isinstance(part, dict) and part.get("type") == "messages" and bridge.count:
                continue
            yield part
        error = result.get("error")
        if error is not None:
            raise error
    finally:
        deactivate(bridge, thread_id)
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


async def _live_aparts(
    graph: Any,
    input: Any,
    bridge: Any,
    *,
    thread_id: str | None,
    modes: list[str],
    subgraphs: bool,
    kwargs: dict[str, Any],
) -> AsyncIterator[Any]:
    """Async twin of `_live_parts`: tokens are yielded while the model is still generating."""
    loop = asyncio.get_running_loop()
    engine: asyncio.Queue[Any] = asyncio.Queue()
    result: dict[str, Any] = {}

    async def run() -> None:
        try:
            async for part in graph.astream(
                input,
                thread_id=thread_id,
                stream_mode=modes,
                version="v2",
                subgraphs=subgraphs,
                **kwargs,
            ):
                engine.put_nowait(part)
        except BaseException as error:  # noqa: BLE001 - re-raised in the consumer
            result["error"] = error
        finally:
            engine.put_nowait(None)

    from .._live import activate, deactivate

    activate(bridge, thread_id)
    task = asyncio.ensure_future(run())
    deferred: list[Any] = []
    try:
        while True:
            for payload, metadata in bridge.drain(timeout=0.0):
                yield {
                    "type": "messages",
                    "ns": (),
                    "data": (payload, metadata),
                    "interrupts": (),
                }
            try:
                part = engine.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0)
                if not task.done() and engine.empty():
                    await asyncio.sleep(0.005)
                continue
            if part is None:
                break
            if isinstance(part, dict) and part.get("type") == "messages" and bridge.count:
                continue
            deferred.append(part)
        while True:
            try:
                part = engine.get_nowait()
            except asyncio.QueueEmpty:
                break
            if part is not None:
                deferred.append(part)
        for part in deferred:
            yield part
        error = result.get("error")
        if error is not None:
            raise error
    finally:
        deactivate(bridge, thread_id)
        if not task.done():
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001,S110 - cleanup
                pass

async def astream_events(
    graph: Any,
    input: Any = None,
    *,
    version: str = "v3",
    transformers: Sequence[Callable[[tuple[str, ...]], Any]] | None = None,
    compile_transformers: Sequence[Callable[[tuple[str, ...]], Any]] | None = None,
    thread_id: str | None = None,
    live: bool = True,
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

    if live and "messages" in modes:
        from .._live import LiveBridge

        return AsyncGraphRunStream(
            _live_aparts(
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