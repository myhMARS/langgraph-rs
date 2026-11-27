"""LangChain callback handlers the engine attaches to model calls.

The extension used to embed this module as a Rust string; it lives here so the reverse import of
``langgraph_rs._live`` stays ordinary Python. The extension asks for a handler through the
``langchain_callback_handler`` factory, which the shim registers at import time.
"""

from __future__ import annotations

from typing import Any

from . import langgraph_rs as _native


try:  # LangChain's base classes: `_StreamingCallbackHandler` makes `invoke` stream tokens.
    from langchain_core.callbacks import BaseCallbackHandler as _LcBase
    from langchain_core.tracers._streaming import _StreamingCallbackHandler as _LcStreaming
except Exception:  # pragma: no cover - langchain_core is optional
    class _LcBase:  # type: ignore[no-redef]
        pass

    class _LcStreaming:  # type: ignore[no-redef]
        pass


class _CallbackHandlerBody(_LcBase):
    """Bridge LangChain callbacks into the engine's stream."""

    raise_error = False
    ignore_llm = False
    ignore_chat_model = False
    ignore_agent = False
    ignore_retriever = False
    ignore_chain = False
    ignore_retry = False
    ignore_custom_event = False

    def __init__(self, model=None, tags=None):
        self.model = model
        self.tags = list(tags or [])
        # LLM runs that already streamed tokens; a run without tokens still has to reach
        # the `messages` stream once, the way LangGraph reports non-streaming models.
        self._streamed = set()
        self._run_metadata = {}

    def _emit(self, token, kwargs=None, complete=False):
        metadata = {}
        if self.model:
            metadata["model"] = self.model
        event_metadata = (kwargs or {}).get("metadata") or {}
        for key, value in event_metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata.setdefault(key, value)
        for key, value in self._run_metadata.items():
            metadata.setdefault(key, value)
        # `checkpoint_ns` is the task's own namespace (`<node>:<task id>`); `langgraph_path` only
        # appends *subgraph* segments, so it keeps using the run's namespace.
        checkpoint_ns = _native.current_checkpoint_ns()
        if checkpoint_ns:
            metadata["langgraph_checkpoint_ns"] = checkpoint_ns
            metadata["checkpoint_ns"] = checkpoint_ns
        namespace = _native.current_namespace()
        node = _native.current_node()
        if node:
            metadata["langgraph_node"] = node
        step = _native.current_step()
        if step is not None:
            # LangGraph counts supersteps from 1 in this metadata (its first superstep is step 1),
            # while the engine's step counter starts at 0.
            metadata["langgraph_step"] = step + 1
        triggers = _native.current_triggers()
        if triggers:
            metadata["langgraph_triggers"] = tuple(triggers)
        if node:
            # LangGraph reports the task's pull path; subgraph namespaces extend it segment by segment.
            path = ["__pregel_pull", node]
            if namespace:
                path.extend(part for part in namespace.split("|") if part)
            metadata["langgraph_path"] = tuple(path)
        # LangGraph yields a message *chunk*; keep the fields needed to rebuild one on the
        # Python side instead of flattening the token to a string.
        if hasattr(token, "content"):
            payload = {
                "content": token.content,
                # LangGraph emits the message itself when it was not streamed token by token.
                "chunk": type(token).__name__.endswith("Chunk"),
                "id": getattr(token, "id", None),
                "tool_call_chunks": list(getattr(token, "tool_call_chunks", []) or []),
                "additional_kwargs": dict(getattr(token, "additional_kwargs", {}) or {}),
                "response_metadata": dict(getattr(token, "response_metadata", {}) or {}),
                "tags": self.tags,
                "metadata": metadata,
            }
        else:
            payload = {
                "content": token,
                "chunk": not complete,
                "tags": self.tags,
                "metadata": metadata,
            }
        # A live `stream_events` run reads tokens straight from this bridge, so `run.messages`
        # shows deltas while the model is still generating rather than when the node returns.
        try:
            from . import _live as live

            active = live.active_bridge(_native.current_thread_id())
            if active is not None:
                active.push(payload, metadata)
        except Exception:  # pragma: no cover - the shim always ships `_live`
            pass
        _native.message(payload)

    def _pick_model(self, kwargs):
        metadata = kwargs.get("metadata") or {}
        name = metadata.get("ls_model_name") or metadata.get("model")
        if name:
            self.model = name
        # LangChain reports the provider details on the start event; token events do not repeat
        # them, so they are remembered here and merged into every chunk's metadata.
        for key in ("ls_provider", "ls_model_name", "ls_model_type", "ls_integration"):
            value = kwargs.get(key) or metadata.get(key)
            if value is not None:
                self._run_metadata[key] = value
        for key in ("lc_versions",):
            value = metadata.get(key)
            if value is not None:
                self._run_metadata[key] = value

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._pick_model(kwargs)
        if not self.model and isinstance(serialized, dict):
            self.model = serialized.get("name") or serialized.get("model")

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._pick_model(kwargs)

    def on_llm_new_token(self, token, **kwargs):
        if isinstance(token, str):
            run_id = kwargs.get("run_id")
            if run_id is not None:
                self._streamed.add(run_id)
            self._emit(token, kwargs)

    def _final_message_text(self, response):
        try:
            generation = response.generations[0][0]
        except Exception:
            return None
        message = getattr(generation, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                return "".join(parts) or None
            return None
        text = getattr(generation, "text", None)
        return text if isinstance(text, str) else None

    def on_llm_end(self, response, **kwargs):
        usage = None
        try:
            usage = (getattr(response, "llm_output", None) or {}).get("token_usage")
        except Exception:
            usage = None
        if not usage:
            try:
                message = response.generations[0][0].message
                usage = getattr(message, "usage_metadata", None)
            except Exception:
                usage = None
        if usage:
            input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
            output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
            if input_tokens or output_tokens:
                                _native.record_tokens(int(input_tokens), int(output_tokens))
        # LangGraph emits the completed message (not a chunk) when the model did not stream tokens.
        run_id = kwargs.get("run_id")
        if run_id not in self._streamed:
            text = self._final_message_text(response)
            if text:
                self._emit(text, kwargs, complete=True)

    def on_llm_error(self, error, **kwargs):
        pass

    def on_chain_start(self, serialized, inputs, **kwargs):
        pass

    def on_chain_end(self, outputs, **kwargs):
        pass

    def on_chain_error(self, error, **kwargs):
        pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        pass

    def on_tool_end(self, output, **kwargs):
        pass

    def on_tool_error(self, error, **kwargs):
        pass

    def on_retriever_start(self, serialized, query, **kwargs):
        pass

    def on_retriever_end(self, documents, **kwargs):
        pass

    def on_retriever_error(self, error, **kwargs):
        pass

    def on_agent_action(self, action, **kwargs):
        pass

    def on_agent_finish(self, finish, **kwargs):
        pass

    def on_text(self, text, **kwargs):
        pass


class LangChainCallbackHandler(_CallbackHandlerBody):
    """Plain variant: reports completion only, like LangGraph without a streaming mode."""


class StreamingLangChainCallbackHandler(_CallbackHandlerBody, _LcStreaming):
    """LangGraph's streaming variant: makes `model.invoke` take the streaming path."""
