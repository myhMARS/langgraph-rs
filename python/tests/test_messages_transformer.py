"""`MessagesTransformer`: `run.messages` yields one `ChatModelStream` per chat-model call.

Two input shapes matter: v3 protocol events (`message-start` / `message-delta` / `message-finish`)
and whole messages. The extension's engine serializes messages to plain dicts, so chunk dicts are
accumulated per call and flushed as a completed handle when the run ends.
"""

from __future__ import annotations

import asyncio
import unittest

import langgraph_rs as rs
from langgraph_rs.stream._mux import StreamMux
from langgraph_rs.stream.transformers import MessagesTransformer

try:
    from langchain_core.messages import AIMessage, AIMessageChunk

    _LANGCHAIN = True
except Exception:  # pragma: no cover - langchain-core optional
    _LANGCHAIN = False

try:
    from langgraph.stream._mux import StreamMux as LgStreamMux
    from langgraph.stream.transformers import MessagesTransformer as LgMessagesTransformer

    _LANGGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    _LANGGRAPH = False


def _messages_event(data, namespace=()):
    return {
        "type": "event",
        "method": "messages",
        "params": {"namespace": list(namespace), "timestamp": 0, "data": data},
    }


META = {"langgraph_node": "model", "run_id": "r1"}


def _protocol_events():
    """The content-block lifecycle LangChain's own bridge produces."""
    return [
        _messages_event(({"event": "message-start", "role": "ai", "id": "m1"}, META)),
        _messages_event(
            ({"event": "content-block-start", "index": 0, "content": {"type": "text", "text": ""}}, META)
        ),
        _messages_event(
            (
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "delta": {"type": "text-delta", "text": "hi"},
                },
                META,
            )
        ),
        _messages_event(
            ({"event": "message-finish"}, META),
        ),
    ]


def _collect_chunks(handle):
    """The individual text deltas a handle yields (empty when the lane is async-only)."""
    text = handle.text

    async def drain():
        return [chunk async for chunk in text]

    if hasattr(text, "__await__"):
        return asyncio.run(drain())
    return list(text)


def _collect_text(handle):
    """Resolve a handle's text projection on whichever lane it was created on."""
    text = handle.text

    async def drain():
        return await text

    if hasattr(text, "__await__"):
        return asyncio.run(drain())
    try:
        chunks = list(text)
    except TypeError:
        return text
    return "".join(str(chunk) for chunk in chunks)


def _resolve_output(handle):
    output = handle.output

    async def drain():
        return await output

    if hasattr(output, "__await__"):
        return asyncio.run(drain())
    return output


def _run(transformer_cls, events, *, mux_cls=StreamMux):
    transformer = transformer_cls()
    mux = mux_cls([transformer])
    cursor = iter(mux.extensions["messages"])
    for event in events:
        mux.push(event)
    mux.close()
    return list(cursor)


@unittest.skipUnless(_LANGCHAIN, "langchain-core is not installed")
class MessagesTransformerTest(unittest.TestCase):
    def test_protocol_events_close_a_handle_with_text(self):
        handles = _run(MessagesTransformer, _protocol_events())
        self.assertEqual(len(handles), 1)
        handle = handles[0]
        self.assertEqual(handle.node, "model")
        self.assertEqual(handle.message_id, "m1")
        self.assertEqual(_collect_text(handle), "hi")

    def test_tool_role_messages_are_ignored(self):
        events = [
            _messages_event(
                ({"event": "message-start", "message_id": "t1", "role": "tool"}, META)
            ),
            _messages_event(({"event": "message-finish"}, META)),
        ]
        self.assertEqual(_run(MessagesTransformer, events), [])

    def test_whole_message_dict_is_replayed(self):
        events = [
            _messages_event(
                ({"type": "ai", "content": "whole", "id": "m2"}, {"langgraph_node": "model"})
            )
        ]
        handles = _run(MessagesTransformer, events)
        self.assertEqual(len(handles), 1)
        self.assertEqual(_collect_text(handles[0]), "whole")

    def test_chunk_dicts_stream_as_separate_deltas(self):
        events = [
            _messages_event(
                (
                    {"type": "AIMessageChunk", "id": "m3", "content": piece},
                    {"langgraph_node": "model"},
                )
            )
            for piece in ("a", "b", "c")
        ]
        handles = _run(MessagesTransformer, events)
        self.assertEqual(len(handles), 1)
        self.assertEqual(handles[0].message_id, "m3")
        # Each chunk is dispatched as its own delta, not replayed as one joined message.
        self.assertEqual(_collect_chunks(handles[0]), ["a", "b", "c"])
        self.assertEqual(_collect_text(handles[0]), "abc")

    def test_handle_exists_after_the_first_chunk(self):
        transformer = MessagesTransformer()
        mux = StreamMux([transformer])
        cursor = iter(mux.extensions["messages"])
        meta = {"langgraph_node": "model"}
        mux.push(_messages_event(({"type": "AIMessageChunk", "id": "m7", "content": "a"}, meta)))
        # The handle is pushed on the first chunk, before the run ends.
        handle = next(cursor)
        self.assertEqual(handle.message_id, "m7")
        mux.push(_messages_event(({"type": "AIMessageChunk", "id": "m7", "content": "b"}, meta)))
        mux.close()
        self.assertEqual(_collect_text(handle), "ab")

    def test_whole_message_closes_the_live_handle(self):
        events = [
            _messages_event(
                ({"type": "AIMessageChunk", "id": "m8", "content": "Hel"}, {"langgraph_node": "model"})
            ),
            _messages_event(({"type": "ai", "content": "Hello", "id": "m8"}, {"langgraph_node": "model"})),
        ]
        handles = _run(MessagesTransformer, events)
        # The final message closes the open handle instead of adding a second one.
        self.assertEqual(len(handles), 1)
        # The closing message fills in the content the deltas had not covered yet.\n        self.assertEqual(_collect_chunks(handles[0]), ["Hel", "lo"])
        self.assertEqual(_collect_text(handles[0]), "Hello")

    def test_separate_calls_get_separate_handles(self):
        events = []
        for message_id, piece in (("x", "1"), ("y", "2")):
            events.append(
                _messages_event(
                    (
                        {"type": "AIMessageChunk", "id": message_id, "content": piece},
                        {"langgraph_node": "model"},
                    )
                )
            )
        handles = _run(MessagesTransformer, events)
        self.assertEqual([handle.message_id for handle in handles], ["x", "y"])
        self.assertEqual([_collect_text(handle) for handle in handles], ["1", "2"])

    def test_deeper_namespace_events_are_ignored(self):
        events = [
            _messages_event(
                ({"type": "ai", "content": "nested", "id": "m4"}, {"langgraph_node": "inner"}),
                namespace=["sub"],
            )
        ]
        self.assertEqual(_run(MessagesTransformer, events), [])

    def test_graph_emitted_chunks_reach_run_messages(self):
        graph = rs.StateGraph({"messages": "messages"})

        def model(state):
            rs.message({"type": "AIMessageChunk", "id": "m5", "content": "hello "})
            rs.message({"type": "AIMessageChunk", "id": "m5", "content": "world"})
            return {"messages": []}

        graph.add_node("model", model)
        graph.set_entry_point("model")
        graph.set_finish_point("model")
        app = graph.compile()

        with app.stream_events(
            {"messages": []}, version="v3", transformers=[MessagesTransformer]
        ) as run:
            handles = list(run.messages)
        self.assertEqual(len(handles), 1)
        self.assertEqual(_collect_chunks(handles[0]), ["hello ", "world"])
        self.assertEqual(_collect_text(handles[0]), "hello world")


@unittest.skipUnless(_LANGGRAPH and _LANGCHAIN, "langgraph/langchain-core not installed")
class MessagesTransformerParityTest(unittest.TestCase):
    def test_protocol_events_match_langgraph(self):
        ours = _run(MessagesTransformer, _protocol_events())
        theirs = _run(LgMessagesTransformer, _protocol_events(), mux_cls=LgStreamMux)
        self.assertEqual(len(ours), len(theirs))
        self.assertEqual(_collect_text(ours[0]), _collect_text(theirs[0]))
        self.assertEqual(_resolve_output(ours[0]).text, _resolve_output(theirs[0]).text)
        self.assertEqual(_resolve_output(ours[0]).id, _resolve_output(theirs[0]).id)

    def test_whole_message_matches_langgraph(self):
        # LangGraph only replays real `BaseMessage` payloads (dicts are ignored there); the
        # extension also accepts its own serialized dicts.
        events = [
            _messages_event((AIMessage(content="whole", id="m9"), {"langgraph_node": "model"}))
        ]
        ours = _run(MessagesTransformer, events)
        theirs = _run(LgMessagesTransformer, events, mux_cls=LgStreamMux)
        self.assertEqual(_collect_text(ours[0]), _collect_text(theirs[0]))


if __name__ == "__main__":
    unittest.main()
@unittest.skipUnless(_LANGCHAIN, "langchain-core is not installed")
class ReactAgentStreamingTest(unittest.TestCase):
    """`create_react_agent` streams its model, so `run.messages` sees token deltas."""

    @staticmethod
    def _model_class():
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        class BindableFake(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        return BindableFake

    def _app(self):
        from langchain_core.messages import AIMessage

        model = self._model_class()(messages=iter([AIMessage(content="Hello world")]))
        # `create_react_agent` returns the native compiled graph; wrap it the way `compile()` does.
        return rs.CompiledStateGraph(rs.create_react_agent(model, []))

    def test_model_chunks_arrive_as_deltas(self):
        with self._app().stream_events(
            {"messages": [{"role": "user", "content": "hi"}]},
            version="v3",
            transformers=[MessagesTransformer],
        ) as run:
            handles = list(run.messages)
        self.assertEqual(len(handles), 1)
        self.assertEqual(_collect_chunks(handles[0]), ["Hello", " ", "world"])
        self.assertEqual(_collect_text(handles[0]), "Hello world")

    def test_stream_mode_messages_matches(self):
        parts = list(
            self._app().stream(
                {"messages": [{"role": "user", "content": "hi"}]},
                stream_mode="messages",
                version="v2",
            )
        )
        self.assertEqual(
            [part["data"][0].get("content") for part in parts], ["Hello", " ", "world"]
        )
@unittest.skipUnless(_LANGCHAIN, "langchain-core is not installed")
class LiveTokenDeliveryTest(unittest.TestCase):
    """Tokens reach `run.messages` while the model is still generating.

    LangChain's callback handler feeds a live bridge, so the first delta arrives before the model's
    stream ends — the engine's own per-node flush would only deliver it afterwards.
    """

    def test_first_delta_arrives_before_generation_finishes(self):
        import time

        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage, HumanMessage

        finished = {"generation": False}

        class Slow(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

            def _stream(self, messages, stop=None, run_manager=None, **kwargs):
                for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                    time.sleep(0.05)
                    yield chunk
                finished["generation"] = True

        model = Slow(messages=iter([AIMessage(content="Hello world")]))
        graph = rs.StateGraph({"messages": "messages"})

        def model_node(state):
            for _chunk in model.stream([HumanMessage(content="hi")]):
                pass
            return {"messages": []}

        graph.add_node("model", model_node)
        graph.set_entry_point("model")
        graph.set_finish_point("model")
        app = graph.compile()

        observed_at_first_delta = {}
        with app.stream_events(
            {"messages": []}, version="v3", transformers=[MessagesTransformer]
        ) as run:
            handle = next(iter(run.messages))
            for index, _chunk in enumerate(handle.text):
                if index == 0:
                    observed_at_first_delta["generation"] = finished["generation"]

        self.assertFalse(
            observed_at_first_delta["generation"],
            "the first delta should arrive while the model is still generating",
        )
@unittest.skipUnless(_LANGCHAIN, "langchain-core is not installed")
class AsyncLiveTokenDeliveryTest(unittest.TestCase):
    """The async run stream also delivers tokens while the model is generating."""

    def test_first_async_delta_arrives_before_generation_finishes(self):
        import time

        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        finished = {"generation": False}

        class Slow(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

            def _stream(self, messages, stop=None, run_manager=None, **kwargs):
                for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                    time.sleep(0.05)
                    yield chunk
                finished["generation"] = True

        async def scenario():
            model = Slow(messages=iter([AIMessage(content="Hello world")]))
            graph = rs.AsyncStateGraph({"messages": "messages"})

            async def node(state):
                for _chunk in model.stream([{"role": "user", "content": "hi"}]):
                    pass
                return {"messages": []}

            graph.add_node("node", node)
            graph.set_entry_point("node")
            graph.set_finish_point("node")
            app = graph.compile()

            run = await app.astream_events(
                {"messages": []}, version="v3", transformers=[MessagesTransformer]
            )
            observed = {}
            async for handle in run.messages:
                index = 0
                async for _chunk in handle.text:
                    if index == 0:
                        observed["generation"] = finished["generation"]
                    index += 1
                break
            return observed

        observed = asyncio.run(scenario())
        self.assertFalse(
            observed.get("generation", True),
            "the first async delta should arrive while the model is still generating",
        )
@unittest.skipUnless(_LANGCHAIN, "langchain-core is not installed")
class ConcurrentLiveRunsTest(unittest.TestCase):
    """Concurrent runs do not exchange tokens (bridges are keyed by `thread_id`)."""

    def test_two_runs_keep_their_own_tokens(self):
        import threading
        import time

        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        class Slow(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

            def _stream(self, messages, stop=None, run_manager=None, **kwargs):
                for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                    time.sleep(0.02)
                    yield chunk

        def run_one(thread_id, text, out):
            model = Slow(messages=iter([AIMessage(content=text)]))
            graph = rs.StateGraph({"messages": "messages"})

            def node(state):
                for _chunk in model.stream([{"role": "user", "content": "hi"}]):
                    pass
                return {"messages": []}

            graph.add_node("node", node)
            graph.set_entry_point("node")
            graph.set_finish_point("node")
            app = graph.compile(checkpointer=rs.InMemorySaver())
            with app.stream_events(
                {"messages": []},
                version="v3",
                thread_id=thread_id,
                transformers=[MessagesTransformer],
            ) as run:
                chunks: list[str] = []
                for handle in run.messages:
                    chunks.extend(handle.text)
                out[thread_id] = "".join(chunks)

        out: dict[str, str] = {}
        workers = [
            threading.Thread(target=run_one, args=("t1", "alpha beta", out)),
            threading.Thread(target=run_one, args=("t2", "gamma delta", out)),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)
        self.assertEqual(out, {"t1": "alpha beta", "t2": "gamma delta"})
class NodeWriteLivenessTest(unittest.TestCase):
    """Chunks a node writes itself also reach `run.messages` while the node is still running."""

    def test_first_node_delta_arrives_before_the_node_finishes(self):
        import time

        graph = rs.StateGraph({"messages": "messages"})
        finished = {"node": False}

        def node(state):
            for piece in ("a", "b"):
                rs.message({"type": "AIMessageChunk", "id": "m1", "content": piece})
                time.sleep(0.05)
            finished["node"] = True
            return {"messages": []}

        graph.add_node("node", node)
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        app = graph.compile()

        observed = {}
        with app.stream_events(
            {"messages": []}, version="v3", transformers=[MessagesTransformer]
        ) as run:
            handle = next(iter(run.messages))
            for index, chunk in enumerate(handle.text):
                if index == 0:
                    observed["finished"] = finished["node"]
                observed.setdefault("text", "")
                observed["text"] += chunk

        self.assertFalse(
            observed["finished"],
            "the first delta should arrive while the node is still running",
        )
        self.assertEqual(observed["text"], "ab")

class AsyncNodeWriteLivenessTest(unittest.TestCase):
    """The async lane also hands out node-written chunks before the node finishes."""

    def test_first_async_delta_precedes_node_completion(self):
        import asyncio
        import time

        async def scenario():
            graph = rs.AsyncStateGraph({"messages": "messages"})
            finished = {"node": False}

            async def node(state):
                for piece in ("a", "b"):
                    rs.message({"type": "AIMessageChunk", "id": "m1", "content": piece})
                    await asyncio.sleep(0.05)
                finished["node"] = True
                return {"messages": []}

            graph.add_node("node", node)
            graph.set_entry_point("node")
            graph.set_finish_point("node")
            app = graph.compile()

            run = await app.astream_events(
                {"messages": []}, version="v3", transformers=[MessagesTransformer]
            )
            observed = {"text": ""}
            async for handle in run.messages:
                index = 0
                async for chunk in handle.text:
                    if index == 0:
                        observed["finished"] = finished["node"]
                    observed["text"] += chunk
                    index += 1
                break
            return observed

        observed = asyncio.run(scenario())
        self.assertFalse(
            observed["finished"],
            "the first async delta should arrive while the node is still running",
        )
        self.assertEqual(observed["text"], "ab")


class CustomLivenessTest(unittest.TestCase):
    """`get_stream_writer()` output is delivered to `run.custom` while the node runs."""

    def test_custom_values_arrive_before_the_node_finishes(self):
        import time

        from langgraph_rs.stream.transformers import CustomTransformer

        graph = rs.StateGraph({"messages": "messages"})
        finished = {"node": False}

        def node(state):
            writer = rs.get_stream_writer()
            for piece in ("x", "y"):
                writer({"progress": piece})
                time.sleep(0.05)
            finished["node"] = True
            return {"messages": []}

        graph.add_node("node", node)
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        app = graph.compile()

        seen = []
        with app.stream_events(
            {"messages": []}, version="v3", transformers=[CustomTransformer]
        ) as run:
            for payload in run.custom:
                seen.append((payload, finished["node"]))

        self.assertEqual([payload for payload, _ in seen], [{"progress": "x"}, {"progress": "y"}])
        self.assertFalse(
            seen[0][1], "the first custom payload should arrive while the node is still running"
        )


class LivePathInterruptTest(unittest.TestCase):
    """A pause is reported through the live path as well as the collected one."""

    def _app(self):
        graph = rs.StateGraph({"messages": "messages", "answer": "last"})

        def node(state):
            rs.message({"type": "AIMessageChunk", "id": "m1", "content": "x"})
            return {"answer": rs.interrupt({"question": "ok?"})}

        graph.add_node("node", node)
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        return graph.compile(checkpointer=rs.InMemorySaver())

    def test_live_run_reports_the_interrupt(self):
        with self._app().stream_events(
            {"messages": []}, version="v3", thread_id="live", transformers=[MessagesTransformer]
        ) as run:
            list(run.values)
            self.assertTrue(run.interrupted)
            self.assertEqual(len(run.interrupts), 1)
            self.assertEqual(run.interrupts[0].value, {"question": "ok?"})
