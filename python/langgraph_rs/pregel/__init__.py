"""LangGraph ``langgraph.pregel`` equivalents.

LangGraph exposes ``Pregel`` (the compiled-graph runtime) and ``NodeBuilder`` here. The extension
runs its own engine, so these names exist for import compatibility and raise when instantiated;
``CompiledGraph`` is the object to use instead.
"""

from __future__ import annotations

import abc as _abc

from .._compat import fill_unsupported, module_getattr

__all__ = [
    "ChannelWrite",
    "ChannelWriteEntry",
    "NodeBuilder",
    "Pregel",
    "PregelNode",
    "SyncAsyncFuture",
    "debug",
    "main",
    "protocol",
    "remote",
]

try:  # LangGraph's compiled graph is a `Runnable`; `Pregel` is its base class.
    from langchain_core.runnables import Runnable as _Runnable
except Exception:  # pragma: no cover - langchain_core is optional
    class _Runnable:  # type: ignore[no-redef]
        pass


class NodeBuilder:
    """Fluent builder for a node's read/write channels (LangGraph internal)."""

    def __init__(self) -> None:
        self._writes: list[Any] = []
        self._reads: list[str] = []
        self._subscribers: list[str] = []
        self._metadata: dict[str, Any] = {}
        self._cache_policy: Any = None
        self._retry_policies: Any = (None,)
        self._timeout: Any = None

    def subscribe_to(self, *channels: str) -> "NodeBuilder":
        self._subscribers.extend(channels)
        return self

    def read_from(self, *channels: str) -> "NodeBuilder":
        self._reads.extend(channels)
        return self

    def write_to(self, *writes: Any) -> "NodeBuilder":
        self._writes.extend(writes)
        return self

    def meta(self, **metadata: Any) -> "NodeBuilder":
        self._metadata.update(metadata)
        return self

    def subscribe_only(self, channel: str) -> "NodeBuilder":
        return self.subscribe_to(channel)

    def add_cache_policy(self, cache_policy: Any) -> "NodeBuilder":
        self._cache_policy = cache_policy
        return self

    def add_retry_policies(self, *retry_policies: Any) -> "NodeBuilder":
        self._retry_policies = retry_policies or (None,)
        return self

    def set_timeout(self, timeout: Any) -> "NodeBuilder":
        self._timeout = timeout
        return self

    def do(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "NodeBuilder is internal to LangGraph; build nodes with StateGraph.add_node()."
        )

    def build(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "NodeBuilder is internal to LangGraph; build nodes with StateGraph.add_node()."
        )


class Pregel(_Runnable, metaclass=_abc.ABCMeta):
    """Base class of the compiled graph, mirroring LangGraph's `Pregel` surface.

    The extension's engine exposes the same methods on `CompiledStateGraph`; this class only
    provides the names (and the `isinstance` relationship) so code written against LangGraph's
    type keeps working. Instantiating it directly is not supported.
    """

    name: str = "LangGraph"
    config: dict[str, Any] = {}
    checkpointer: Any = None
    store: Any = None
    cache: Any = None
    cache_policy: Any = None
    retry_policy: Any = None
    step_timeout: Any = None
    config_schema: Any = None
    context_schema: Any = None
    input_schema: Any = None
    output_schema: Any = None
    stream_mode: Any = None
    stream_channels: Any = None
    stream_channels_asis: Any = None
    stream_channels_list: Any = None
    stream_eager: bool = False
    config_specs: list[Any] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Use the compiled graph from StateGraph.compile() instead of instantiating Pregel."
        )

    # The methods a compiled graph implements; declared here so `dir(Pregel)` matches LangGraph.
    def _unimplemented(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    invoke = stream = astream = ainvoke = _unimplemented
    stream_events = astream_events = astream_log = _unimplemented
    batch = abatch = batch_as_completed = abatch_as_completed = _unimplemented
    transform = atransform = _unimplemented
    get_state = aget_state = get_state_history = aget_state_history = _unimplemented
    update_state = aupdate_state = bulk_update_state = abulk_update_state = _unimplemented
    get_subgraphs = aget_subgraphs = get_graph = aget_graph = _unimplemented
    clear_cache = aclear_cache = _unimplemented
    validate = _unimplemented
    get_input_jsonschema = get_output_jsonschema = _unimplemented
    get_config_jsonschema = get_context_jsonschema = _unimplemented
    get_input_schema = get_output_schema = _unimplemented
    get_prompts = get_name = copy = _unimplemented


# LangGraph keeps these in private modules (`langgraph.pregel._write`, ...); they are importable
# there, so the same names exist here for import compatibility even though the extension runs its
# own engine.
_MISSING = fill_unsupported(
    globals(),
    ["ChannelWrite", "ChannelWriteEntry", "PregelNode", "SyncAsyncFuture"],
    hint="Use the compiled graph returned by StateGraph.compile() instead.",
)
__getattr__ = module_getattr(__name__, globals(), _MISSING)

# LangGraph's package re-exports its internal submodules.
from . import _write, debug, main, protocol, remote  # noqa: E402,F401