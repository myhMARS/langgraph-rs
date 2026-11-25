"""LangGraph ``langgraph.pregel._write`` equivalents (import compatibility).

The extension runs its own engine, so these names raise when instantiated; graph writes are declared
through ``StateGraph.add_node`` and the compiled graph.
"""

from __future__ import annotations

from . import ChannelWrite, ChannelWriteEntry

__all__ = ["ChannelWrite", "ChannelWriteEntry"]
