"""Single-consumer drainable queue, mirroring ``langgraph.stream.stream_channel``."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class StreamChannel(Generic[T]):
    """Single-consumer drainable queue for streaming events.

    Items are popped off the front as the consumer advances. Exactly one subscriber is allowed;
    `tee(n)` / `atee(n)` fan out. The channel starts unbound: the run stream calls `_bind(is_async)`
    before iteration, and the other iteration protocol then raises `TypeError`.
    """

    def __init__(self, name: str | None = None, *, maxlen: int | None = None) -> None:
        if maxlen is not None and maxlen <= 0:
            raise ValueError("StreamChannel maxlen must be a positive int or None")
        self.name = name
        self._items: deque[tuple[int, T]] = deque()
        self._maxlen: int | None = maxlen
        self._closed = False
        self._error: BaseException | None = None

        self._is_async: bool | None = None
        self._subscribed = False

        self._request_more: Callable[[], bool] | None = None
        self._arequest_more: Callable[[], Awaitable[bool]] | None = None

        self._wire_fn: Callable[[T], None] | None = None
        self._mux = None

    # -- binding ---------------------------------------------------------

    def _bind_mux(self, mux) -> None:
        self._mux = mux

    def _bind(self, *, is_async: bool) -> None:
        if self._is_async is not None:
            raise RuntimeError("StreamChannel is already bound")
        self._is_async = is_async

    def _wire(self, fn: Callable[[T], None]) -> None:
        self._wire_fn = fn

    # -- producer --------------------------------------------------------

    def push(self, item: T) -> None:
        if self._subscribed:
            if self._closed:
                raise RuntimeError("Cannot push to a closed StreamChannel")
            stamp = self._mux._next_push_seq() if self._mux is not None else 0
            self._items.append((stamp, item))
        if self._wire_fn is not None:
            self._wire_fn(item)

    def close(self) -> None:
        self._closed = True

    def fail(self, err: BaseException) -> None:
        self._error = err
        self._closed = True

    # -- sync iteration --------------------------------------------------

    def __iter__(self) -> Iterator[T]:
        if self._is_async is None:
            raise TypeError(
                "StreamChannel has not been bound yet. "
                "Register the transformer with a StreamMux first."
            )
        if self._is_async:
            raise TypeError("This StreamChannel is bound to async mode — use 'async for' instead.")
        if self._subscribed:
            raise RuntimeError("StreamChannel already has a subscriber; use .tee(n) for fan-out.")
        self._subscribed = True
        return self._sync_cursor()

    def _sync_cursor(self) -> Iterator[T]:
        while True:
            if self._items:
                _stamp, item = self._items.popleft()
                yield item
            elif self._closed:
                if self._error is not None:
                    raise self._error
                return
            elif self._request_more is not None:
                if not self._request_more():
                    if not self._items and not self._closed:
                        return
            else:
                return

    # -- async iteration -------------------------------------------------

    def __aiter__(self) -> AsyncIterator[T]:
        if self._is_async is None:
            raise TypeError(
                "StreamChannel has not been bound yet. "
                "Register the transformer with a StreamMux first."
            )
        if not self._is_async:
            raise TypeError("This StreamChannel is bound to sync mode — use 'for' instead.")
        if self._subscribed:
            raise RuntimeError(
                "StreamChannel already has a subscriber; use .atee(n) for fan-out."
            )
        self._subscribed = True
        return self._async_cursor()

    async def _async_cursor(self) -> AsyncIterator[T]:
        while True:
            if self._items:
                _stamp, item = self._items.popleft()
                yield item
            elif self._closed:
                if self._error is not None:
                    raise self._error
                return
            elif self._arequest_more is not None:
                if not await self._arequest_more():
                    if not self._items and not self._closed:
                        return
            else:
                return

    # -- fan-out ---------------------------------------------------------

    def tee(self, n: int = 2) -> tuple[Iterator[T], ...]:
        if n < 1:
            raise ValueError("tee() requires n >= 1")
        source = self.__iter__()
        buffers: list[deque[T]] = [deque() for _ in range(n)]
        exhausted = [False]

        def branch(i: int) -> Iterator[T]:
            buf = buffers[i]
            while True:
                if buf:
                    yield buf.popleft()
                elif exhausted[0]:
                    return
                else:
                    try:
                        item = next(source)
                    except StopIteration:
                        exhausted[0] = True
                        return
                    for other in buffers:
                        other.append(item)

        return tuple(branch(i) for i in range(n))

    def atee(self, n: int = 2) -> tuple[AsyncIterator[T], ...]:
        if n < 1:
            raise ValueError("atee() requires n >= 1")
        source = self.__aiter__()
        buffers: list[deque[T]] = [deque() for _ in range(n)]
        exhausted = [False]
        error: list[BaseException | None] = [None]
        lock = asyncio.Lock()

        async def branch(i: int) -> AsyncIterator[T]:
            buf = buffers[i]
            while True:
                if buf:
                    yield buf.popleft()
                    continue
                if exhausted[0]:
                    if error[0] is not None:
                        raise error[0]
                    return
                async with lock:
                    if buf or exhausted[0]:
                        continue
                    try:
                        item = await source.__anext__()
                    except StopAsyncIteration:
                        exhausted[0] = True
                        continue
                    except Exception as exc:  # noqa: BLE001 - surfaced to every branch
                        error[0] = exc
                        exhausted[0] = True
                        continue
                    for other in buffers:
                        other.append(item)

        return tuple(branch(i) for i in range(n))