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

    def test_chunk_dicts_are_accumulated_per_call(self):
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
        self.assertEqual(_collect_text(handles[0]), "abc")

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