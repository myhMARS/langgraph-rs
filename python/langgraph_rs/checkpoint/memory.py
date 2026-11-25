"""``langgraph.checkpoint.memory`` equivalents."""

from __future__ import annotations

from .._compat import module_getattr, re_export
from .base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)
from .serde import SerializerProtocol

ChannelVersions = dict
DeltaChannelHistory = list


class PersistentDict(dict):
    """Dict that survives process restarts in LangGraph; here a plain ``dict`` subclass."""

__all__ = [
    "AsyncInMemorySaver",
    "BaseCheckpointSaver",
    "Checkpoint",
    "CheckpointMetadata",
    "ChannelVersions",
    "CheckpointTuple",
    "DeltaChannelHistory",
    "PersistentDict",
    "InMemorySaver",
    "MemorySaver",
    "PendingWrite",
    "SerializerProtocol",
    "WRITES_IDX_MAP",
    "get_checkpoint_id",
    "get_checkpoint_metadata",
    "writes_sort_key",
]

_MISSING = re_export(globals(), ["AsyncInMemorySaver", "InMemorySaver"])
# LangGraph's legacy name for the in-memory saver.
MemorySaver = globals().get("InMemorySaver")
__getattr__ = module_getattr(__name__, globals(), _MISSING)
