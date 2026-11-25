"""Internal module kept for import compatibility.

LangGraph ships ``langgraph.checkpoint.serde.types``; the extension has no counterpart, so the
module exists only so ``import`` statements keep working."""

from __future__ import annotations


def __getattr__(name: str):
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}: this LangGraph-internal module"
        " has no counterpart in the langgraph_rs extension"
    )
