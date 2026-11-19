"""``langgraph.checkpoint.base`` has no Rust counterpart.

The extension ships concrete savers (in-memory, SQLite, Redis, Postgres); writing a custom
saver in Python is not supported. Importing these names works, using them raises
``NotImplementedError`` with that explanation.
"""

from __future__ import annotations

from .._compat import fill_unsupported, module_getattr

__all__ = [
    "BaseCheckpointSaver",
    "Checkpoint",
    "CheckpointMetadata",
    "CheckpointTuple",
    "SerializerProtocol",
    "copy_checkpoint",
    "create_checkpoint",
    "get_checkpoint_id",
    "get_checkpoint_metadata",
]

_MISSING = fill_unsupported(
    globals(),
    list(__all__),
    hint="Use one of the shipped savers instead (InMemorySaver, SqliteSaver, RedisSaver, PostgresSaver).",
)
__getattr__ = module_getattr(__name__, globals(), _MISSING)