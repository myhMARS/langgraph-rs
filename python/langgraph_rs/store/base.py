"""``langgraph.store.base`` has no Rust counterpart.

The extension ships concrete stores; a custom Python store is not supported. Importing these
names works, using them raises ``NotImplementedError`` with that explanation.
"""

from __future__ import annotations

from .._compat import fill_unsupported, module_getattr

__all__ = [
    "BaseStore",
    "Item",
    "Op",
    "SearchOp",
    "TTLConfig",
    "ensure_embeddings",
    "get_filter_params",
]

_MISSING = fill_unsupported(
    globals(),
    list(__all__),
    hint="Use one of the shipped stores instead (InMemoryStore, SqliteStore, RedisStore, PostgresStore).",
)
__getattr__ = module_getattr(__name__, globals(), _MISSING)