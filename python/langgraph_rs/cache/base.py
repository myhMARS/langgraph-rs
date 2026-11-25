"""LangGraph ``langgraph.cache.base`` equivalents.

`BaseCache` is the abstract interface LangGraph's node caches implement: a cache maps
``(namespace..., key)`` to a value, and ``set``/``get``/``clear`` work in batches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, NamedTuple, Sequence, TypeVar

from ..checkpoint.serde import JsonPlusSerializer, SerializerProtocol

__all__ = [
    "BaseCache",
    "FullKey",
    "JsonPlusSerializer",
    "Namespace",
    "SerializerProtocol",
    "ValueT",
]

ValueT = TypeVar("ValueT")

Namespace = tuple[str, ...]


class FullKey(NamedTuple):
    """Cache key: a namespace prefix plus the node-local key."""

    prefix: tuple[str, ...]
    key: str


class BaseCache(Generic[ValueT], ABC):
    """Abstract node cache."""

    @abstractmethod
    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Return the cached values for `keys` (missing keys are omitted)."""

    @abstractmethod
    def set(self, pairs: Sequence[tuple[FullKey, ValueT]]) -> None:
        """Store `pairs`."""

    @abstractmethod
    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Drop every entry, or only the given namespaces."""


CacheKey = FullKey
ValueSerializer = Callable[[Any], Any]
