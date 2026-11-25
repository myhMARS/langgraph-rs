"""LangGraph ``langgraph.pregel`` equivalents.

LangGraph exposes ``Pregel`` (the compiled-graph runtime) and ``NodeBuilder`` here. The extension
runs its own engine, so these names exist for import compatibility and raise when instantiated;
``CompiledGraph`` is the object to use instead.
"""

from __future__ import annotations

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

# LangGraph keeps these in private modules (`langgraph.pregel._write`, ...); they are importable
# there, so the same names exist here for import compatibility even though the extension runs its
# own engine.
_MISSING = fill_unsupported(
    globals(),
    ["ChannelWrite", "ChannelWriteEntry", "NodeBuilder", "Pregel", "PregelNode", "SyncAsyncFuture"],
    hint="Use the compiled graph returned by StateGraph.compile() instead.",
)
__getattr__ = module_getattr(__name__, globals(), _MISSING)

# LangGraph's package re-exports its internal submodules.
from . import _write, debug, main, protocol, remote  # noqa: E402,F401