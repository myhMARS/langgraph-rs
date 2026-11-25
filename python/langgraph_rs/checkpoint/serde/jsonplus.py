"""``langgraph.checkpoint.serde.jsonplus`` equivalents."""

from __future__ import annotations

# MessagePack extension type ids, matching LangGraph's `jsonplus` module.
EXT_CONSTRUCTOR_SINGLE_ARG = 0
EXT_CONSTRUCTOR_POS_ARGS = 1
EXT_CONSTRUCTOR_KW_ARGS = 2
EXT_METHOD_SINGLE_ARG = 3
EXT_PYDANTIC_V1 = 4
EXT_PYDANTIC_V2 = 5
EXT_NUMPY_ARRAY = 6
EXT_DELTA_SNAPSHOT = 7

from . import JsonPlusSerializer
from .base import Serializable, SerializerProtocol

__all__ = ["JsonPlusSerializer", "Serializable", "SerializerProtocol"]
