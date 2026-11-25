"""Run-stream wrappers, mirroring ``langgraph.stream.run_stream``.

LangGraph's run streams are driven by `graph.stream_events(version="v3")` and a `StreamMux`; the
extension does not implement that pipeline yet, so these names exist for import compatibility and
raise with a pointer to `stream(stream_mode=...)`.
"""

from __future__ import annotations

from .._compat import fill_unsupported

__all__ = [
    "AsyncGraphRunStream",
    "AsyncSubgraphRunStream",
    "GraphRunStream",
    "SubgraphRunStream",
]

_MISSING = fill_unsupported(
    globals(),
    __all__,
    hint=(
        "stream_events(version='v3') is not implemented; use graph.stream(stream_mode=...) which "
        "supports values/updates/checkpoints/tasks/debug/messages/custom."
    ),
)