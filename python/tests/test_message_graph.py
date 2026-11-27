"""`MessageGraph` reproduces LangGraph's deprecated single-channel message builder.

LangGraph semantics: the graph has exactly one `messages` channel, nodes receive the message
list, may return one message or a list of messages, and `invoke` returns the resulting list.
"""

from __future__ import annotations

import unittest

import langgraph_rs as lg
from langgraph_rs.graph import END, START, MessageGraph
from langgraph_rs.graph.state import StateGraph


class MessageGraphTest(unittest.TestCase):
    def _graph(self):
        graph = MessageGraph()
        graph.add_node("reply", lambda messages: [{"role": "assistant", "content": "pong"}])
        graph.add_node("one", lambda messages: {"role": "assistant", "content": "single"})
        graph.add_edge(START, "reply")
        graph.add_edge("reply", "one")
        graph.add_edge("one", END)
        return graph

    def test_invoke_returns_the_message_list(self):
        compiled = self._graph().compile()
        out = compiled.invoke([{"role": "user", "content": "ping"}])
        self.assertIsInstance(out, list)
        self.assertEqual([m.content for m in out], ["ping", "pong", "single"])

    def test_nodes_receive_the_message_list(self):
        seen: list[object] = []

        def node(messages):
            seen.append(messages)
            return {"role": "assistant", "content": "done"}

        graph = MessageGraph()
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        graph.add_edge("node", END)
        graph.compile().invoke([{"role": "user", "content": "hi"}])
        self.assertTrue(seen and isinstance(seen[0], list))
        self.assertEqual(seen[0][0].content, "hi")

    def test_state_mapping_input_is_accepted(self):
        compiled = self._graph().compile()
        out = compiled.invoke({"messages": [{"role": "user", "content": "ping"}]})
        self.assertEqual([m.content for m in out], ["ping", "pong", "single"])

    def test_updates_stream_is_node_keyed(self):
        graph = MessageGraph()
        graph.add_node("reply", lambda messages: {"role": "assistant", "content": "pong"})
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        chunks = list(graph.compile().stream([{"role": "user", "content": "hi"}], stream_mode="updates"))
        self.assertEqual(list(chunks[0]), ["reply"])

    def test_builder_is_a_state_graph_and_chains(self):
        graph = MessageGraph()
        self.assertIsInstance(graph, StateGraph)
        returned = graph.add_node("n", lambda messages: messages)
        self.assertIsInstance(returned, MessageGraph)

    def test_interrupt_pause_keeps_messages_and_interrupt(self):
        graph = MessageGraph()
        graph.add_node(
            "ask",
            lambda messages: {"messages": [{"role": "assistant", "content": "answer"}]}
            if lg.interrupt({"question": "q"}) is None
            else {"messages": []},
        )
        graph.add_edge(START, "ask")
        graph.add_edge("ask", END)
        compiled = graph.compile(checkpointer=lg.InMemorySaver())
        paused = compiled.invoke([{"role": "user", "content": "hi"}], thread_id="mg")
        self.assertIn("__interrupt__", paused)
        self.assertEqual(paused["messages"][0].content, "hi")


if __name__ == "__main__":
    unittest.main()