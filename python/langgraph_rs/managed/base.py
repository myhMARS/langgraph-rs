"""LangGraph ``langgraph.managed.base`` (internal) equivalents."""

from __future__ import annotations

from typing import Any

__all__ = ["ManagedValue"]


class ManagedValue:
    """Duck-typed base of LangGraph's managed values (annotation marker)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs