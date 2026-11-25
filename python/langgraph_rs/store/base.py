"""LangGraph ``langgraph.store.base`` equivalents.

`BaseStore` mirrors LangGraph's abstract store so the same type annotations and subclassing work.
Subclassing is **not** required to use a custom store: any object implementing ``get`` / ``put`` /
``delete`` / ``search`` / ``list_namespaces`` can be passed to ``compile(store=...)`` and the
extension bridges to it (LangGraph's own ``InMemoryStore`` works that way too).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from .._compat import module_getattr

__all__ = [
    "BaseStore",
    "Item",
    "SearchItem",
]


class BaseStore(ABC):
    """Abstract long-term store."""

    @abstractmethod
    def get(self, namespace: tuple[str, ...], key: str) -> Any:
        """Return the item stored under `namespace`/`key`, or ``None``."""

    @abstractmethod
    def put(
        self, namespace: tuple[str, ...], key: str, value: dict[str, Any], index: Any = None
    ) -> None:
        """Insert or replace a value."""

    @abstractmethod
    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete a value."""

    @abstractmethod
    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> Sequence[Any]:
        """Search items under a namespace prefix."""

    def list_namespaces(self, **kwargs: Any) -> Sequence[tuple[str, ...]]:
        """List namespaces that contain at least one item."""
        return ()


class Item:
    """Value stored in a store (LangGraph's ``Item`` shape)."""

    __slots__ = ("namespace", "key", "value", "created_at", "updated_at", "score")

    def __init__(
        self,
        namespace: Sequence[str] | None = None,
        key: str = "",
        value: Any = None,
        created_at: Any = None,
        updated_at: Any = None,
        score: float | None = None,
    ) -> None:
        self.namespace = list(namespace or [])
        self.key = key
        self.value = value
        self.created_at = created_at
        self.updated_at = updated_at
        self.score = score

    def __repr__(self) -> str:
        return (
            f"Item(namespace={self.namespace!r}, key={self.key!r}, value={self.value!r}, "
            f"created_at={self.created_at!r}, updated_at={self.updated_at!r}, score={self.score!r})"
        )


SearchItem = Item

__getattr__ = module_getattr(
    __name__,
    globals(),
    ["GetOp", "IndexConfig", "ListNamespacesOp", "Op", "PutOp", "SearchOp", "TTLConfig"],
)