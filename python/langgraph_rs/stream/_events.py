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
    MessagesTransformer,
    SubgraphTransformer,
    TasksTransformer,
    UpdatesTransformer,
    ValuesTransformer,
)

# LangGraph registers `values`, `messages`, `lifecycle` and `subgraphs` by default; the remaining
# capture projections (`updates`, `custom`, `checkpoints`, `debug`, `tasks`) are opt-in through
# `transformers=` or the graph's compile-time transformers.
DEFAULT_TRANSFORMERS: tuple[Callable[[tuple[str, ...]], Any], ...] = (
    ValuesTransformer,
    MessagesTransformer,
    LifecycleTransformer,
    SubgraphTransformer,
)


def _factories(
    extra: Sequence[Callable[[tuple[str, ...]], Any]] | None,
    compile_time: Sequence[Callable[[tuple[str, ...]], Any]] | None = None,
) -> list[Callable[[tuple[str, ...]], Any]]:
    """Built-ins, then the graph's compile-time transformers, then call-site ones.

    A transformer type is registered once: passing `MessagesTransformer` explicitly must not
    collide with the built-in registration of the same projection.
    """
    factories: list[Callable[[tuple[str, ...]], Any]] = list(DEFAULT_TRANSFORMERS)
    seen = {_transformer_type(factory) for factory in factories}
    for transformer in (*((compile_time) or ()), *((extra) or ())):
        kind = _transformer_type(transformer)
        if kind in seen:
            continue
        seen.add(kind)
        factories.append(transformer)
    return factories


def _transformer_type(factory: Callable[[tuple[str, ...]], Any]) -> Any:
    """Class of the transformer a factory builds (factories close over their class)."""
    try:
        return type(factory(()))
    except Exception:  # pragma: no cover - exotic factory
        return factory


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

    # Only the token/custom lanes need live delivery; values/tasks-only runs keep the
    # collected path (their chunk shapes are assembled per step and ordered with it).
    if live and ("messages" in modes or "custom" in modes):
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


async def ainvoke_via_sync(graph: Any, input: Any, config: Any = None, kwargs: Any = None) -> Any:
    """Await the synchronous ``invoke`` on a worker thread (LangGraph's ``ainvoke`` shape)."""
    loop = asyncio.get_running_loop()
    kwargs = dict(kwargs or {})
    return await loop.run_in_executor(None, lambda: graph.invoke(input, config, **kwargs))


async def call_via_sync(graph: Any, method: str, kwargs: Any = None) -> Any:
    """Await one of the graph's synchronous methods on a worker thread."""
    loop = asyncio.get_running_loop()
    kwargs = dict(kwargs or {})
    return await loop.run_in_executor(None, lambda: getattr(graph, method)(**kwargs))


async def abatch_via_sync(graph: Any, inputs: Any, config: Any = None, kwargs: Any = None) -> Any:
    """Await the synchronous ``batch`` on a worker thread (LangGraph's ``abatch`` shape)."""
    loop = asyncio.get_running_loop()
    kwargs = dict(kwargs or {})
    return await loop.run_in_executor(None, lambda: graph.batch(inputs, config, **kwargs))


async def astream_via_sync(graph: Any, input: Any, config: Any = None, kwargs: Any = None):
    """Async generator over the synchronous ``stream`` result (LangGraph's ``astream`` shape)."""
    loop = asyncio.get_running_loop()
    kwargs = dict(kwargs or {})
    chunks = await loop.run_in_executor(None, lambda: graph.stream(input, config, **kwargs))
    for chunk in chunks:
        yield chunk


async def _aiter_collected(chunks: Any) -> AsyncIterator[Any]:
    """Expose a collected sync chunk list as an async iterator (``CompiledGraph.astream``)."""
    for chunk in chunks:
        yield chunk


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
    # A thread-safe queue: the live sink is called from an engine thread, and the async lane must
    # not depend on which loop owns the consumer.
    engine: "queue.Queue[Any]" = queue.Queue()
    result: dict[str, Any] = {}

    def sink(part: Any) -> None:
        engine.put(part)

    async def drive() -> None:
        call = dict(
            thread_id=thread_id,
            stream_mode=modes,
            version="v2",
            subgraphs=subgraphs,
            **kwargs,
        )
        try:
            stream = graph.astream(input, live_sink=sink, **call)
        except TypeError:
            # A graph without a live sink (the synchronous engine's facade) still streams its
            # parts, just without token-level liveness.
            stream = graph.astream(input, **call)
        async for part in stream:
            engine.put(part)

    def run() -> None:
        # The graph gets its own loop on its own thread: driving it on the consumer's loop would
        # keep the loop busy and the live sink could only be drained once the node finished.
        try:
            asyncio.run(drive())
        except BaseException as error:  # noqa: BLE001 - re-raised in the consumer
            result["error"] = error
        finally:
            engine.put(None)

    from .._live import activate, deactivate

    activate(bridge, thread_id)
    task = threading.Thread(target=run, name="langgraph_rs-astream", daemon=True)
    task.start()
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
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue
            if part is None:
                break
            if isinstance(part, dict) and part.get("type") == "messages" and bridge.count:
                continue
            # Engine parts are produced in order, so hand them out as they arrive.
            yield part
        while True:
            try:
                part = engine.get_nowait()
            except queue.Empty:
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
        task.join(timeout=5)

def astream_events(
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
    """LangGraph's `astream_events`: awaitable for v3, async generator for v2."""
    if version == "v2":
        return astream_events_v2(graph, input, thread_id=thread_id, **kwargs)
    return _astream_events_v3(
        graph,
        input,
        version=version,
        transformers=transformers,
        compile_transformers=compile_transformers,
        thread_id=thread_id,
        live=live,
        **kwargs,
    )


async def _astream_events_v3(
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
            "astream_events only supports version='v3' (and 'v2') in the extension."
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

    # Only the token/custom lanes need live delivery; values/tasks-only runs keep the
    # collected path (their chunk shapes are assembled per step and ordered with it).
    if live and ("messages" in modes or "custom" in modes):
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


async def astream_events_v2(
    graph: Any,
    input: Any = None,
    *,
    thread_id: str | None = None,
    **kwargs: Any,
) -> Any:
    """LangChain-style callback events (`on_chain_*`), as LangGraph's v2 streaming emits.

    LangGraph drives these from its `Runnable` callbacks; the extension derives them from the
    streamed supersteps, which carry the same information (task start/end, updates, metadata).
    """
    from uuid import uuid4

    graph_run = str(uuid4())
    graph_name = getattr(graph, "name", None) or "LangGraph"

    def event(kind: str, name: str, run_id: str, data: dict, metadata: dict, parents: list) -> dict:
        return {
            "event": kind,
            "name": name,
            "run_id": run_id,
            "tags": [],
            "metadata": metadata,
            "data": data,
            "parent_ids": parents,
        }

    yield event(
        "on_chain_start", graph_name, graph_run, {"input": input}, {"ls_integration": "langgraph"}, []
    )

    def parts() -> Iterator[Any]:
        # The engine streams `(namespace, mode, payload)` tuples through the same v2 protocol the
        # rest of the module uses.
        collected: dict[str, Any] = {}
        try:
            for part in graph.stream(
                input,
                thread_id=thread_id,
                stream_mode=["updates", "values"],
                version="v2",
                subgraphs=True,
                **kwargs,
            ):
                collected["last"] = part
                yield part
        finally:
            collected.clear()

    node_runs: dict[tuple, str] = {}
    collected: Any = None
    # Subgraph runs, keyed by their namespace path, so nested events nest like LangGraph's.
    child_runs: dict[tuple, dict[str, Any]] = {}

    def node_metadata(namespace: tuple, node: str, step: Any = None) -> dict:
        # LangGraph tags every node event with the integration, superstep and node name.
        metadata: dict[str, Any] = {"ls_integration": "langgraph"}
        if step is not None:
            metadata["langgraph_step"] = step
        metadata["langgraph_node"] = node
        if namespace:
            metadata["langgraph_checkpoint_ns"] = "|".join(namespace)
        return metadata

    def graph_chunk(payload: Any, namespace: tuple = ()) -> Any:
        metadata: dict[str, Any] = {"ls_integration": "langgraph"}
        if namespace:
            metadata["langgraph_checkpoint_ns"] = "|".join(namespace)
        return event("on_chain_stream", graph_name, graph_run, {"chunk": payload}, metadata, [])

    def close_children(namespace: tuple) -> Iterator[Any]:
        """End every child graph run opened under `namespace` or deeper."""
        for path in sorted(child_runs, key=len, reverse=True):
            if len(path) <= len(namespace) and path != namespace:
                continue
            if path[: len(namespace)] != namespace:
                continue
            state = child_runs.pop(path)
            yield event(
                "on_chain_end",
                graph_name,
                state["run_id"],
                {"output": state.get("values")},
                {"ls_integration": "langgraph", "langgraph_checkpoint_ns": "|".join(path)},
                list(state["parents"]),
            )

    async for part in _aiter_collected(parts()):
        step = None
        if isinstance(part, dict) and "type" in part:
            # v2 parts are `{"type", "ns", "data"}` (updates also carry their superstep).
            namespace, mode, payload = part.get("ns") or (), part["type"], part.get("data")
            step = part.get("step")
        elif isinstance(part, tuple) and len(part) == 3:
            namespace, mode, payload = part
        else:
            namespace, mode, payload = (), "values", part
        namespace = tuple(namespace)
        if mode == "values":
            if namespace:
                state = child_runs.get(namespace)
                if state is not None:
                    state["values"] = payload
            else:
                collected = payload
            continue
        if namespace and namespace not in child_runs:
            # First activity inside a subgraph: its parent node starts, then the child graph.
            node = namespace[-1].split(":")[0]
            parent_node_run = node_runs.get((namespace[:-1], node))
            if parent_node_run is None:
                parent_node_run = str(uuid4())
                node_runs[(namespace[:-1], node)] = parent_node_run
                yield event(
                    "on_chain_start",
                    node,
                    parent_node_run,
                    {"input": input},
                    node_metadata(namespace[:-1], node, step),
                    [graph_run],
                )
            child_run = str(uuid4())
            child_runs[namespace] = {
                "run_id": child_run,
                "node": node,
                "parents": [graph_run, parent_node_run],
                "values": None,
            }
            yield event(
                "on_chain_start",
                graph_name,
                child_run,
                {"input": input},
                {"ls_integration": "langgraph", "langgraph_checkpoint_ns": "|".join(namespace)},
                [graph_run, parent_node_run],
            )
        child = child_runs.get(namespace)
        parents = [graph_run] if child is None else list(child["parents"]) + [child["run_id"]]
        if not namespace:
            # A subgraph node's own stream/end follow the child graph's end event.
            for event_chunk in close_children(()):
                yield event_chunk
        for node, update in (payload or {}).items():
            key = (namespace, node)
            run_id = node_runs.get(key)
            metadata = node_metadata(namespace, node, step)
            if run_id is None:
                run_id = str(uuid4())
                node_runs[key] = run_id
                yield event("on_chain_start", node, run_id, {"input": input}, metadata, parents)
            yield event("on_chain_stream", node, run_id, {"chunk": update}, metadata, parents)
            yield event(
                "on_chain_end",
                node,
                run_id,
                {"input": input, "output": update},
                metadata,
                parents,
            )
        if not namespace:
            # LangGraph's graph-level chunk is this superstep's updates, not the whole state.
            yield graph_chunk(payload)
    for event_chunk in close_children(()):
        yield event_chunk
    yield event(
        "on_chain_end",
        graph_name,
        graph_run,
        {"output": collected},
        {"ls_integration": "langgraph"},
        [],
    )


from .run_stream import AsyncGraphRunStream  # noqa: E402  (circular-safe tail import)