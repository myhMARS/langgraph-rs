"""``AsyncStream``: an awaitable that is also an async iterator.

The native async graph returns an awaitable resolving to a list of chunks (the run has to finish
before the chunks are converted); LangGraph callers expect an async iterator, so the facade wraps it
in this class. Defining it here keeps the extension free of embedded Python source.
"""

from __future__ import annotations

from typing import Any


class AsyncStream:
    """Await once for the collected chunks, or iterate them as they resolve."""

    def __init__(self, awaitable: Any) -> None:
        self._awaitable = awaitable
        self._iter: Any = None
        self._aiter: Any = None
        # A native async iterator streams; an awaitable resolves to the collected chunks.
        if hasattr(awaitable, "__anext__") and not hasattr(awaitable, "__await__"):
            self._aiter = awaitable.__aiter__()

    def __await__(self) -> Any:
        return self._awaitable.__await__()

    def __aiter__(self) -> "AsyncStream":
        return self

    async def __anext__(self) -> Any:
        if self._aiter is not None:
            return await self._aiter.__anext__()
        if self._iter is None:
            resolved = await self._awaitable
            if hasattr(resolved, "__anext__"):
                self._aiter = resolved.__aiter__()
                return await self._aiter.__anext__()
            self._iter = iter(resolved)
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None
