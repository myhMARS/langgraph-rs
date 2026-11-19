"""``langgraph.checkpoint.base`` placeholders.

Subclassing ``BaseCheckpointSaver`` is not how custom savers work with this engine: pass a plain
object implementing LangGraph's saver protocol (``put`` / ``put_writes`` / ``get_tuple`` /
``list``) to ``compile(checkpointer=...)`` and the extension bridges to it. LangGraph's own saver
instances work the same way. The names below only exist so imports keep resolving; using them
raises ``NotImplementedError`` with that explanation.
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
    hint=(
        "Pass a plain object implementing the LangGraph saver protocol to compile(checkpointer=...) "
        "instead of subclassing this class."
    ),
)
__getattr__ = module_getattr(__name__, globals(), _MISSING)
