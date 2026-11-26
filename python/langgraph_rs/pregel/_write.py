"""LangGraph ``langgraph.pregel._write`` equivalents (import compatibility).

The extension runs its own engine, so these names raise when instantiated; graph writes are declared
through ``StateGraph.add_node`` and the compiled graph.
"""

from __future__ import annotations

from .._compat import fill_unsupported, module_getattr
from . import ChannelWrite, ChannelWriteEntry

__all__ = ["ChannelWrite", "ChannelWriteEntry", "ChannelWriteTupleEntry"]

_MISSING = fill_unsupported(
    globals(),
    ["ChannelWriteTupleEntry"],
    hint="Use the compiled graph returned by StateGraph.compile() instead.",
)
__getattr__ = module_getattr(__name__, globals(), _MISSING)
