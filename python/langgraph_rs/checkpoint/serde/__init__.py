"""LangGraph ``langgraph.checkpoint.serde`` equivalents.

`JsonPlusSerializer` is a JSON-only serializer: it round-trips JSON values plus the common Python
extras (``bytes`` as base64, ``datetime``/``date``/``time`` and ``set`` as tagged objects).
Tuples come back as lists because JSON has no tuple type, and LangGraph's msgpack payloads are not
supported -- unsupported objects raise instead of being silently mangled.
"""

from __future__ import annotations

import base64
import datetime as _datetime
import json
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "EncryptedSerializer",
    "JsonPlusSerializer",
    "Serializable",
    "SerializerProtocol",
]

_TAG = "__langgraph_serialized__"


@runtime_checkable
class Serializable(Protocol):
    """Object that knows how to serialize itself (`to_json`/`from_json`)."""

    def to_json(self) -> dict[str, Any]: ...


class SerializerProtocol(Protocol):
    """The interface savers use to encode checkpoints."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]: ...

    def loads_typed(self, data: tuple[str, bytes]) -> Any: ...

    def dumps(self, obj: Any) -> bytes: ...

    def loads(self, data: bytes) -> Any: ...


def _default(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {_TAG: "bytes", "data": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (_datetime.datetime, _datetime.date, _datetime.time)):
        return {_TAG: type(value).__name__, "data": value.isoformat()}
    if isinstance(value, (set, frozenset)):
        return {_TAG: "set", "data": sorted(value, key=repr)}

    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        return {_TAG: "serializable", "type": type(value).__name__, "data": to_json()}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _object_hook(value: dict[str, Any]) -> Any:
    tag = value.get(_TAG)
    if tag is None:
        return value
    data = value.get("data")
    if tag == "bytes":
        return base64.b64decode(data)
    if tag == "datetime":
        return _datetime.datetime.fromisoformat(data)
    if tag == "date":
        return _datetime.date.fromisoformat(data)
    if tag == "time":
        return _datetime.time.fromisoformat(data)
    if tag == "set":
        return set(data)
    if tag == "tuple":
        return tuple(data)
    return data


class JsonPlusSerializer:
    """JSON serializer used by custom savers and caches."""

    def __init__(self, *, pickle_fallback: bool = False, allowed_msgpack_modules: Any = None) -> None:
        self.pickle_fallback = pickle_fallback
        self._allowed_msgpack_modules = allowed_msgpack_modules

    def with_msgpack_allowlist(self, extra_allowlist: Any) -> "JsonPlusSerializer":
        """Return a serializer that also accepts these msgpack types, like LangGraph."""
        base = self._allowed_msgpack_modules
        if base is True or base is False:
            return self
        merged = set(base or ()) | set(extra_allowlist or ())
        if merged == set(base or ()):
            return self
        clone = JsonPlusSerializer(
            pickle_fallback=self.pickle_fallback, allowed_msgpack_modules=merged
        )
        return clone

    def with_allowlist(self, extra_allowlist: Any) -> "JsonPlusSerializer":
        """Alias of `with_msgpack_allowlist` (LangGraph exposes both)."""
        return self.with_msgpack_allowlist(extra_allowlist)

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return ("json", self.dumps(obj))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        kind, payload = data
        if kind not in ("json", "null"):
            raise ValueError(f"JsonPlusSerializer cannot decode payload type {kind!r}")
        return self.loads(payload)

    def dumps(self, obj: Any) -> bytes:
        return json.dumps(obj, default=_default, ensure_ascii=False).encode("utf-8")

    def loads(self, data: bytes) -> Any:
        if not data:
            return None
        return json.loads(data.decode("utf-8"), object_hook=_object_hook)
# LangGraph ships these as submodules too.
from . import base, encrypted, event_hooks, jsonplus, types  # noqa: E402,F401
from .encrypted import EncryptedSerializer  # noqa: E402