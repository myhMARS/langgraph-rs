"""Exports package version (LangGraph ``langgraph.version``)."""

from __future__ import annotations

from importlib import metadata

__all__ = ("__version__",)

try:
    __version__ = metadata.version("langgraph_rs")
except Exception:  # pragma: no cover - metadata is absent for source checkouts
    __version__ = ""
del metadata
