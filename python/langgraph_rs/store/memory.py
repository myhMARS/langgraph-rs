"""``langgraph.store.memory`` equivalents."""

from __future__ import annotations

from typing import Any, Sequence

from .._compat import module_getattr, re_export
from .base import (
    BaseStore,
    Embeddings,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

__all__ = [
    "BaseStore",
    "Embeddings",
    "GetOp",
    "InMemoryStore",
    "IndexConfig",
    "Item",
    "ListNamespacesOp",
    "MatchCondition",
    "Op",
    "PutOp",
    "Result",
    "SearchItem",
    "SearchOp",
    "Store",
    "ensure_embeddings",
    "get_text_at_path",
    "tokenize_path",
]



from ..langgraph_rs import InMemoryStore as _NativeStore  # noqa: E402  (native implementation)


class InMemoryStore(_NativeStore):
    """LangGraph's `InMemoryStore`, with its batch and async surface."""

    supports_ttl = False
    ttl_config = None

    def __new__(cls, *, index: Any = None, ttl: Any = None) -> "InMemoryStore":
        # The native type builds itself in `tp_new`, so the index goes there rather than `__init__`.
        if index is None:
            return super().__new__(cls)
        return super().__new__(cls, index)

    def __init__(self, *, index: Any = None, ttl: Any = None) -> None:
        self.ttl = ttl
        # LangGraph exposes the constructor's index settings on the store.
        self.index_config = dict(index) if isinstance(index, dict) else index
        self.embeddings = (
            self.index_config.get("embed") if isinstance(self.index_config, dict) else None
        )

    def put(self, namespace: Any, key: str, value: Any, index: Any = None, *, ttl: Any = None) -> None:
        """Store a value; LangGraph's in-memory store rejects a TTL."""
        if ttl is not None and not self.supports_ttl:
            raise NotImplementedError(
                "TTL is not supported by InMemoryStore. Use a store implementation that supports "
                "TTL or set ttl=None."
            )
        super().put(tuple(namespace), key, value, index=index)

    def batch(self, ops: Sequence[Any]) -> list[Any]:
        """Run a list of ops, mirroring LangGraph's dispatch.

        LangGraph applies reads against the state the batch started from, so a `PutOp` in the same
        batch is not visible to a later `GetOp`/`SearchOp`/`ListNamespacesOp`; writes run after the
        reads here to reproduce that.
        """
        results: list[Any] = [None] * len(ops)
        writes: list[Any] = []
        for index, op in enumerate(ops):
            name = type(op).__name__
            if name == "GetOp":
                results[index] = self.get(tuple(op.namespace), op.key)
            elif name == "PutOp":
                writes.append(op)
            elif name == "SearchOp":
                results[index] = self.search(
                    tuple(op.namespace_prefix),
                    filter=op.filter,
                    limit=op.limit,
                    offset=op.offset,
                    query=op.query,
                )
            elif name == "ListNamespacesOp":
                results[index] = self.list_namespaces(
                    max_depth=op.max_depth, limit=op.limit, offset=op.offset
                )
            else:
                # LangGraph rejects anything that is not an op object.
                raise ValueError(f"Unknown operation type: {type(op)}")
        for op in writes:
            self.put(tuple(op.namespace), op.key, op.value, ttl=op.ttl)
        return results

    async def abatch(self, ops: Sequence[Any]) -> list[Any]:
        return self.batch(ops)

    async def aget(self, namespace: Any, key: str, **kwargs: Any) -> Any:
        return self.get(tuple(namespace), key, **kwargs)

    async def aput(
        self, namespace: Any, key: str, value: Any, index: Any = None, *, ttl: Any = None
    ) -> None:
        self.put(tuple(namespace), key, value, index=index, ttl=ttl)

    async def adelete(self, namespace: Any, key: str) -> None:
        self.delete(tuple(namespace), key)

    async def asearch(
        self, namespace_prefix: Any, *, query: Any = None, filter: Any = None,
        limit: int = 10, offset: int = 0,
    ) -> list[Any]:
        return self.search(
            tuple(namespace_prefix), filter=filter, limit=limit, offset=offset, query=query
        )

    async def alist_namespaces(
        self, *, prefix: Any = None, suffix: Any = None, max_depth: Any = None,
        limit: int = 100, offset: int = 0,
    ) -> list[Any]:
        return self.list_namespaces(
            prefix=prefix, suffix=suffix, max_depth=max_depth, limit=limit, offset=offset
        )


#: LangGraph's older name for the in-memory store.
Store = InMemoryStore


_MISSING = re_export(globals(), [])
__getattr__ = module_getattr(__name__, globals(), _MISSING)
