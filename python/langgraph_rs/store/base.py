"""LangGraph ``langgraph.store.base`` equivalents.

`BaseStore` mirrors LangGraph's abstract store, and the pure helpers used by embedding-indexed
stores (`tokenize_path`, `get_text_at_path`, `override`, `validate_op_namespace`) are faithful
ports: they are differential-tested against LangGraph's own functions.

Subclassing `BaseStore` is optional. Any object implementing ``get`` / ``put`` / ``delete`` /
``search`` / ``list_namespaces`` can be passed to ``compile(store=...)`` and the extension bridges
to it (LangGraph's own ``InMemoryStore`` works that way too).
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from .._compat import module_getattr

__all__ = [
    "AEmbeddingsFunc",
    "BaseStore",
    "Embeddings",
    "EmbeddingsFunc",
    "GetOp",
    "IndexConfig",
    "InvalidNamespaceError",
    "Item",
    "ListNamespacesOp",
    "MatchCondition",
    "NOT_PROVIDED",
    "NamespaceMatchType",
    "NamespacePath",
    "NotProvided",
    "Op",
    "PutOp",
    "Result",
    "SearchItem",
    "SearchOp",
    "TTLConfig",
    "ensure_embeddings",
    "get_text_at_path",
    "override",
    "tokenize_path",
    "validate_op_namespace",
]

try:  # pragma: no cover - langchain_core is optional
    from langchain_core.embeddings import Embeddings as _LangChainEmbeddings
except Exception:  # pragma: no cover - fallback protocol
    _LangChainEmbeddings = None


class InvalidNamespaceError(ValueError):
    """Raised when a namespace label would be ambiguous when serialized."""


def _validate_namespace_labels(namespace: Sequence[Any]) -> None:
    for label in namespace:
        if not isinstance(label, str) or not label or "." in label:
            raise InvalidNamespaceError(
                "namespace labels must be non-empty strings without periods, "
                f"got {label!r} in {tuple(namespace)!r}"
            )


def _validate_namespace(namespace: Any) -> None:
    """Validate a namespace path, like LangGraph's `_validate_namespace`."""
    if not isinstance(namespace, tuple):
        raise TypeError(f"namespace must be a tuple of strings, got {type(namespace)}")
    _validate_namespace_labels(namespace)


class BaseStore(ABC):
    """LangGraph's store protocol.

    A store implements `batch` (and optionally `abatch`); `get` / `put` / `delete` / `search` /
    `list_namespaces` are derived from it, exactly like LangGraph's base class.
    """

    supports_ttl: bool = False
    ttl_config: Any = None

    def batch(self, ops: Sequence[Any]) -> list[Any]:
        """Run a list of ops in one round trip."""
        raise NotImplementedError

    async def abatch(self, ops: Sequence[Any]) -> list[Any]:
        """Async twin of `batch`; the default runs the sync version."""
        return self.batch(ops)

    def get(self, namespace: tuple[str, ...], key: str, *, refresh_ttl: Any = None) -> Any:
        """Return the item stored under `namespace`/`key`, or ``None``."""
        _validate_namespace_labels(namespace)
        return self.batch([GetOp(tuple(namespace), str(key))])[0]

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Any = None,
        *,
        ttl: Any = None,
    ) -> None:
        """Insert or replace a value."""
        _validate_namespace(tuple(namespace))
        if ttl is not None and not self.supports_ttl:
            raise NotImplementedError(
                f"TTL is not supported by {type(self).__name__}. "
                f"Use a store implementation that supports TTL or set ttl=None."
            )
        self.batch([PutOp(tuple(namespace), str(key), value, index=index, ttl=ttl)])

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete a value."""
        _validate_namespace_labels(namespace)
        # LangGraph encodes a delete as a `PutOp` with `value=None`.
        self.batch([PutOp(tuple(namespace), str(key), None, ttl=None)])

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Any = None,
    ) -> Sequence[Any]:
        """Search items under a namespace prefix."""
        _validate_namespace_labels(namespace_prefix)
        return self.batch(
            [SearchOp(tuple(namespace_prefix), filter, limit, offset, query=query)]
        )[0]

    def list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[tuple[str, ...]]:
        """List namespaces that contain at least one item."""
        _validate_namespace_labels(prefix or ())
        _validate_namespace_labels(suffix or ())
        match_conditions = []
        if prefix:
            match_conditions.append(MatchCondition("prefix", tuple(prefix)))
        if suffix:
            match_conditions.append(MatchCondition("suffix", tuple(suffix)))
        return self.batch(
            [
                ListNamespacesOp(
                    match_conditions=tuple(match_conditions),
                    max_depth=max_depth,
                    limit=limit,
                    offset=offset,
                )
            ]
        )[0]

    async def aget(self, namespace: Any, key: str, *, refresh_ttl: Any = None) -> Any:
        _validate_namespace_labels(namespace)
        return (await self.abatch([GetOp(tuple(namespace), str(key))]))[0]

    async def aput(
        self,
        namespace: Any,
        key: str,
        value: dict[str, Any],
        index: Any = None,
        *,
        ttl: Any = None,
    ) -> None:
        _validate_namespace(tuple(namespace))
        if ttl is not None and not self.supports_ttl:
            raise NotImplementedError(
                f"TTL is not supported by {type(self).__name__}. "
                f"Use a store implementation that supports TTL or set ttl=None."
            )
        await self.abatch([PutOp(tuple(namespace), str(key), value, index=index, ttl=ttl)])

    async def adelete(self, namespace: Any, key: str) -> None:
        _validate_namespace_labels(namespace)
        await self.abatch([PutOp(tuple(namespace), str(key), None, ttl=None)])

    async def asearch(
        self,
        namespace_prefix: Any,
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Any = None,
    ) -> Sequence[Any]:
        _validate_namespace_labels(namespace_prefix)
        return (
            await self.abatch(
                [SearchOp(tuple(namespace_prefix), filter, limit, offset, query=query)]
            )
        )[0]

    async def alist_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[tuple[str, ...]]:
        _validate_namespace_labels(prefix or ())
        _validate_namespace_labels(suffix or ())
        match_conditions = []
        if prefix:
            match_conditions.append(MatchCondition("prefix", tuple(prefix)))
        if suffix:
            match_conditions.append(MatchCondition("suffix", tuple(suffix)))
        return (
            await self.abatch(
                [
                    ListNamespacesOp(
                        match_conditions=tuple(match_conditions),
                        max_depth=max_depth,
                        limit=limit,
                        offset=offset,
                    )
                ]
            )
        )[0]


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

NamespaceMatchType = Literal["prefix", "suffix"]
NamespacePath = tuple[str | Literal["*"], ...]
Result = Any

_Func = TypeVar("_Func", bound=Callable[..., Any])


class NotProvided:
    """Sentinel type for "argument not supplied"."""

    def __repr__(self) -> str:
        return "<not provided>"


NOT_PROVIDED: NotProvided = NotProvided()


@dataclass
class MatchCondition:
    """One namespace match condition (`prefix` or `suffix`)."""

    match_type: NamespaceMatchType
    path: NamespacePath


@dataclass
class TTLConfig:
    """TTL policy of a store."""

    refresh_on_read: bool = True
    omit_expired: bool = True
    default_ttl: float | None = None
    sweep_interval_minutes: int | None = None


@dataclass
class IndexConfig:
    """Embedding index configuration of a store."""

    dims: int
    embed: Any  # Embeddings | EmbeddingsFunc | AEmbeddingsFunc | str
    fields: list[str] | None = None


@dataclass
class GetOp:
    """Batch op: read one key."""

    namespace: tuple[str, ...]
    key: str
    refresh_ttl: bool = False


@dataclass
class PutOp:
    """Batch op: write one key."""

    namespace: tuple[str, ...]
    key: str
    value: Mapping[str, Any] | None = None
    index: Literal[False] | list[str] | None = None
    ttl: float | None = None


@dataclass
class SearchOp:
    """Batch op: search a namespace prefix."""

    namespace_prefix: tuple[str, ...]
    filter: dict[str, Any] | None = None
    limit: int = 10
    offset: int = 0
    query: str | None = None
    refresh_ttl: bool = False


@dataclass
class ListNamespacesOp:
    """Batch op: list namespaces."""

    match_conditions: tuple[MatchCondition, ...] | None = None
    max_depth: int | None = None
    limit: int = 100
    offset: int = 0


Op = GetOp | PutOp | SearchOp | ListNamespacesOp


def override(method: _Func, /) -> _Func:
    """Mark a method as an intended override (no runtime checking, like LangGraph)."""
    try:
        method.__override__ = True  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass
    return method


def tokenize_path(path: str) -> list[str]:
    """Split a path expression into field / index / multi-field tokens."""
    tokens: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(path):
        char = path[i]
        if char == "[":  # Handle array indexing
            if current:
                tokens.append("".join(current))
                current = []
            bracket_count = 1
            index_chars = ["["]
            i += 1
            while i < len(path) and bracket_count > 0:
                if path[i] == "[":
                    bracket_count += 1
                elif path[i] == "]":
                    bracket_count -= 1
                index_chars.append(path[i])
                i += 1
            tokens.append("".join(index_chars))
            continue
        if char == "{":  # Handle multi-field selection
            if current:
                tokens.append("".join(current))
                current = []
            brace_count = 1
            field_chars = ["{"]
            i += 1
            while i < len(path) and brace_count > 0:
                if path[i] == "{":
                    brace_count += 1
                elif path[i] == "}":
                    brace_count -= 1
                field_chars.append(path[i])
                i += 1
            tokens.append("".join(field_chars))
            continue
        if char == ".":  # Handle regular field
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
        i += 1
    if current:
        tokens.append("".join(current))
    return tokens


def get_text_at_path(obj: Any, path: str | list[str]) -> list[str]:
    """Extract text from an object using LangGraph's path expression syntax."""
    if isinstance(obj, Mapping) and not isinstance(obj, dict):
        obj = dict(obj)

    if not path or path == "$":
        return [json.dumps(obj, sort_keys=True, ensure_ascii=False)]

    tokens = tokenize_path(path) if isinstance(path, str) else path

    def _extract_from_obj(current_obj: Any, tokens: list[str], pos: int) -> list[str]:
        if pos >= len(tokens):
            if isinstance(current_obj, (str, int, float, bool)):
                return [str(current_obj)]
            if current_obj is None:
                return []
            if isinstance(current_obj, (list, dict)):
                return [json.dumps(current_obj, sort_keys=True, ensure_ascii=False)]
            return []

        token = tokens[pos]
        results: list[str] = []

        if token.startswith("[") and token.endswith("]"):
            if not isinstance(current_obj, list):
                return []
            index = token[1:-1]
            if index == "*":
                for item in current_obj:
                    results.extend(_extract_from_obj(item, tokens, pos + 1))
            else:
                try:
                    idx = int(index)
                    if idx < 0:
                        idx = len(current_obj) + idx
                    if 0 <= idx < len(current_obj):
                        results.extend(_extract_from_obj(current_obj[idx], tokens, pos + 1))
                except (ValueError, IndexError):
                    return []
        elif token.startswith("{") and token.endswith("}"):
            if not isinstance(current_obj, dict):
                return []
            fields = [f.strip() for f in token[1:-1].split(",")]
            for field_name in fields:
                nested_tokens = tokenize_path(field_name)
                if nested_tokens:
                    nested: Any = current_obj
                    for nested_token in nested_tokens:
                        if isinstance(nested, dict) and nested_token in nested:
                            nested = nested[nested_token]
                        else:
                            nested = None
                            break
                    if nested is not None:
                        if isinstance(nested, (str, int, float, bool)):
                            results.append(str(nested))
                        elif isinstance(nested, (list, dict)):
                            results.append(
                                json.dumps(nested, sort_keys=True, ensure_ascii=False)
                            )
        elif token == "*":
            if isinstance(current_obj, dict):
                for value in current_obj.values():
                    results.extend(_extract_from_obj(value, tokens, pos + 1))
            elif isinstance(current_obj, list):
                for item in current_obj:
                    results.extend(_extract_from_obj(item, tokens, pos + 1))
        elif isinstance(current_obj, dict) and token in current_obj:
            results.extend(_extract_from_obj(current_obj[token], tokens, pos + 1))

        return results

    return _extract_from_obj(obj, tokens, 0)


def validate_op_namespace(op: Op) -> None:
    """Reject batch ops whose namespace labels would serialize ambiguously."""
    if isinstance(op, (GetOp, PutOp)):
        _validate_namespace_labels(op.namespace)
    elif isinstance(op, SearchOp):
        _validate_namespace_labels(op.namespace_prefix)
    elif isinstance(op, ListNamespacesOp):
        for condition in op.match_conditions or ():
            _validate_namespace_labels(condition.path)


Embeddings = _LangChainEmbeddings if _LangChainEmbeddings is not None else object
EmbeddingsFunc = Callable[[Sequence[str]], list[list[float]]]
AEmbeddingsFunc = Callable[[Sequence[str]], Any]


class _EmbeddingsLambda:
    """Adapts a plain embedding function to LangChain's `Embeddings` interface."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        result = self._func(list(texts))
        if asyncio.iscoroutine(result):
            return asyncio.get_event_loop().run_until_complete(result)
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        result = self._func(list(texts))
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]


def ensure_embeddings(
    embed: Any,
) -> Any:
    """Wrap an embedding function so it satisfies LangChain's `Embeddings` interface."""
    if embed is None:
        raise ValueError("embed must be provided")
    if isinstance(embed, str):
        raise ValueError(
            f"Could not load embeddings from string {embed!r}: loading embeddings by "
            "provider:identifier strings requires langchain>=0.3.9 and the provider-specific "
            "package. Alternatively pass an Embeddings object or a Python function."
        )
    if _LangChainEmbeddings is not None and isinstance(embed, _LangChainEmbeddings):
        return embed
    return _EmbeddingsLambda(embed)


# LangGraph exposes its embedding helpers as the submodule `store.base.embed`.
import sys as _sys
import types as _types

embed = _types.ModuleType(__name__ + ".embed")
embed.ensure_embeddings = ensure_embeddings
embed.EmbeddingsLambda = _EmbeddingsLambda
embed.Embeddings = Embeddings
embed.EmbeddingsFunc = EmbeddingsFunc
embed.AEmbeddingsFunc = AEmbeddingsFunc
_sys.modules[embed.__name__] = embed

__all__.append("embed")

__getattr__ = module_getattr(__name__, globals(), [])