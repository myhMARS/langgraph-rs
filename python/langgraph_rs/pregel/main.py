"""``langgraph.pregel.main`` import-compatibility module.

LangGraph defines ``Pregel`` (its compiled-graph runtime) and ``NodeBuilder`` in
``langgraph.pregel.main``. The extension runs its own engine, so both names are importable
placeholders that raise with guidance when used; ``CompiledGraph`` is the object to use instead.
"""

from __future__ import annotations

from .._compat import fill_unsupported, module_getattr

__all__ = ["NodeBuilder", "Pregel"]

_MISSING = fill_unsupported(
    globals(),
    ["NodeBuilder", "Pregel"],
    hint="Use the compiled graph returned by StateGraph.compile() instead.",
)
__getattr__ = module_getattr(__name__, globals(), _MISSING)
