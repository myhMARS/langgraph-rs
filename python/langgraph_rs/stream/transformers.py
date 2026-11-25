"""Built-in stream transformers, mirroring ``langgraph.stream.transformers``.

The transformer classes project protocol events into `StreamChannel`s. They are part of the
`stream_events(version="v3")` pipeline, which the extension does not implement yet, so they raise
when used. `LifecyclePayload` and `SubgraphStatus` are payload types and are available.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from .._compat import fill_unsupported

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


class LifecyclePayload(TypedDict, total=False):
    """Payload of the `lifecycle` transformer."""

    event: str
    ns: list[str]
    data: NotRequired[Any]


SubgraphStatus = str
"""Status of a subgraph run (`"start"` / `"end"` / `"error"`)."""

_MISSING = fill_unsupported(
    globals(),
    [
        name
        for name in __all__
        if name not in {"LifecyclePayload", "SubgraphStatus"}
    ],
    hint=(
        "Built-in transformers need the stream_events(version='v3') pipeline; use "
        "graph.stream(stream_mode=...) instead."
    ),
)