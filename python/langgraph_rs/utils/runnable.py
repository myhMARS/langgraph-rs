"""Backwards-compatible runnable helpers (LangGraph ``langgraph.utils.runnable``)."""

from __future__ import annotations

from typing import Any

__all__ = ["RunnableCallable", "RunnableLike"]

RunnableLike = Any
"""Anything callable that can act as a runnable (function, coroutine function or Runnable)."""


class RunnableCallable:
    """Wrap a function as a small runnable with `invoke` / `ainvoke`.

    LangGraph's own version wires tracing and config plumbing; the extension only needs the
    `invoke`/`ainvoke` surface for compatibility code.
    """

    def __init__(
        self,
        func: Any = None,
        afunc: Any = None,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.func = func
        self.afunc = afunc
        self.name = name or getattr(func, "__name__", None) or getattr(afunc, "__name__", None)
        self.kwargs = kwargs

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        if self.func is None:
            raise TypeError(f"{type(self).__name__} has no synchronous function")
        try:
            return self.func(input, config) if config is not None else self.func(input)
        except TypeError:
            return self.func(input)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        if self.afunc is not None:
            result = self.afunc(input, config) if config is not None else self.afunc(input)
        elif self.func is not None:
            result = self.invoke(input, config)
        else:
            raise TypeError(f"{type(self).__name__} has no callable")
        if hasattr(result, "__await__"):
            return await result
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs) if self.func is not None else self.afunc(*args, **kwargs)

    def __repr__(self) -> str:
        return f"RunnableCallable(name={self.name!r})"
