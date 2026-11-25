"""Serializer protocol definitions (``langgraph.checkpoint.serde.base``)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["CipherProtocol", "Serializable", "SerializerProtocol"]


@runtime_checkable
class Serializable(Protocol):
    """Object that knows how to serialize itself."""

    def to_json(self) -> dict[str, Any]: ...


class SerializerProtocol(Protocol):
    """Interface savers use to encode checkpoints."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]: ...

    def loads_typed(self, data: tuple[str, bytes]) -> Any: ...

    def dumps(self, obj: Any) -> bytes: ...

    def loads(self, data: bytes) -> Any: ...


class CipherProtocol(Protocol):
    """Encrypts and decrypts serialized payloads."""

    def encrypt(self, plaintext: bytes) -> bytes: ...

    def decrypt(self, ciphertext: bytes) -> bytes: ...
