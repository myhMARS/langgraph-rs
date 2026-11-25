"""``langgraph.checkpoint.serde.encrypted`` equivalents.

`EncryptedSerializer` wraps another serializer and encrypts the payload, like LangGraph's
implementation. LangGraph builds its cipher from ``pycryptodome``; here the cipher comes from
``cryptography`` (Fernet) instead, and a missing dependency is reported explicitly.
"""

from __future__ import annotations

import base64
from typing import Any

from . import JsonPlusSerializer

__all__ = ["EncryptedSerializer"]


class EncryptedSerializer:
    """Serializer that encrypts the payload produced by an inner serializer."""

    def __init__(self, cipher: Any = None, serde: Any = None) -> None:
        self.serde = serde or JsonPlusSerializer()
        self._cipher = cipher
        self._fernet_instance: Any = None

    @classmethod
    def from_pycryptodome_aes(
        cls, key: bytes | None = None, serde: Any = None
    ) -> "EncryptedSerializer":
        """LangGraph builds a pycryptodome cipher here; pass a Fernet key instead."""
        return cls(cipher=key, serde=serde)

    def _fernet(self) -> Any:
        if self._fernet_instance is not None:
            return self._fernet_instance
        try:
            from cryptography.fernet import Fernet
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "EncryptedSerializer needs the `cryptography` package: pip install cryptography"
            ) from error
        cipher = self._cipher
        if cipher is None:
            self._fernet_instance = Fernet(Fernet.generate_key())
        elif hasattr(cipher, "encrypt") and hasattr(cipher, "decrypt"):
            self._fernet_instance = cipher
        else:
            self._fernet_instance = Fernet(cipher)
        return self._fernet_instance

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        kind, payload = self.serde.dumps_typed(obj)
        token = self._fernet().encrypt(payload)
        return (f"encrypted+{kind}", token)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        kind, payload = data
        if kind.startswith("encrypted+"):
            kind = kind.split("+", 1)[1]
            payload = self._fernet().decrypt(payload)
        return self.serde.loads_typed((kind, payload))

    def dumps(self, obj: Any) -> bytes:
        return base64.b64encode(self.dumps_typed(obj)[1])

    def loads(self, data: bytes) -> Any:
        return self.serde.loads(base64.b64decode(data))