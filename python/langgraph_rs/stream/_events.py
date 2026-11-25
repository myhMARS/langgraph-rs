"""`stream_events(version="v3")` for the compiled graph, built on top of `stream()`.

LangGraph drives its v3 protocol with a `StreamMux` fed by the graph's own stream. The extension
implements the same shape in Python: the union of the registered transformers' required stream modes
is requested from `stream()`, every part is converted to a `ProtocolEvent`, and the caller's
iteration on a projection is what pulls the graph forward.

`messages`, `lifecycle` and `subgraphs` are not registered yet (they need the model-stream and
subgraph-handle machinery), so a run only exposes the capture projections.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

from ._convert import convert_to_protocol_event
from ._mux import StreamMux
from .run_stream import GraphRunStream
from .transformers import (
    CheckpointsTransformer,
    CustomTransformer,
    DebugTransformer,
    TasksTransformer,
    UpdatesTransformer,
    ValuesTransformer,
)

# LangGraph registers `values` (plus its messages/lifecycle/subgraphs transformers, which the
# extension does not implement yet) by default; the other capture projections are opt-in through
# `transformers=` or the graph's compile-time transformers.
DEFAULT_TRANSFORMERS: tuple[Callable[[tuple[str, ...]], Any], ...] = (ValuesTransformer,)


def _factories(
    extra: Sequence[Callable[[tuple[str, ...]], Any]] | None,
) -> list[Callable[[tuple[str, ...]], Any]]:
    factories = list(DEFAULT_TRANSFORMERS)
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


def stream_events(
    graph: Any,
    input: Any = None,
    *,
    version: str = "v3",
    transformers: Sequence[Callable[[tuple[str, ...]], Any]] | None = None,
    thread_id: str | None = None,
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
    factories = _factories(transformers)
    mux = StreamMux(factories=factories, is_async=False)
    # `version="v2"` makes every part a `StreamPart` dict (type/ns/data), which is what the
    # protocol conversion expects.
    graph_iter: Iterator[Any] = graph.stream(
        input,
        thread_id=thread_id,
        stream_mode=_required_modes(factories),
        version="v2",
        **kwargs,
    )
    return GraphRunStream(graph_iter, mux)


async def astream_events(
    graph: Any,
    input: Any = None,
    *,
    version: str = "v3",
    transformers: Sequence[Callable[[tuple[str, ...]], Any]] | None = None,
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
    factories = _factories(transformers)
    mux = StreamMux(factories=factories, is_async=True)
    graph_iter = graph.astream(
        input,
        thread_id=thread_id,
        stream_mode=_required_modes(factories),
        version="v2",
        **kwargs,
    )
    return AsyncGraphRunStream(graph_iter, mux)


from .run_stream import AsyncGraphRunStream  # noqa: E402  (circular-safe tail import)