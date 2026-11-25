"""`stream_mode="tools"`: the engine emits tool-call lifecycle events.

`ToolNode` reports `tool-started` / `tool-finished` / `tool-error` on the `tools` mode, which
`ToolCallTransformer` turns into handles on `run.tool_calls`.
"""

from __future__ import annotations

import asyncio
import unittest

import langgraph_rs as rs
from langgraph_rs.prebuilt import ToolCallTransformer

try:
    from langchain_core.tools import tool as lc_tool

    _LANGCHAIN = True
except Exception:  # pragma: no cover - langchain-core optional
    _LANGCHAIN = False


def _agent_graph(tool_node, *, state_schema=None):
    """`agent` proposes one `add` call; `tools` executes it."""

    def agent(state):
        return {
            "messages": [
                {
                    "role": "ai",
                    "content": "",
                    "tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1"}],
                }
            ]
        }

    graph = rs.StateGraph(state_schema or {"messages": "messages"})
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(rs.START, "agent")
    graph.add_edge("agent", "tools")
    graph.add_edge("tools", rs.END)
    return graph.compile()


@unittest.skipUnless(_LANGCHAIN, "langchain-core is not installed")
class ToolsStreamTest(unittest.TestCase):
    def _tool_node(self):
        @lc_tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        return rs.ToolNode([add])

    def _failing_tool_node(self):
        @lc_tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            raise RuntimeError("tool exploded")

        return rs.ToolNode([add])

    def test_stream_mode_tools_parts(self):
        app = _agent_graph(self._tool_node())
        parts = [
            part
            for part in app.stream({"messages": []}, stream_mode="tools", version="v2")
            if part.get("type") == "tools"
        ]
        events = [part["data"]["event"] for part in parts]
        self.assertEqual(events, ["tool-started", "tool-finished"])
        self.assertEqual(parts[0]["data"]["tool_call_id"], "call_1")
        self.assertEqual(parts[0]["data"]["tool_name"], "add")
        self.assertEqual(parts[1]["data"]["output"], "3")

    def test_tool_call_transformer_projects_a_completed_handle(self):
        app = _agent_graph(self._tool_node())
        with app.stream_events(
            {"messages": []}, version="v3", transformers=[ToolCallTransformer]
        ) as run:
            handles = list(run.tool_calls)
        self.assertEqual(len(handles), 1)
        handle = handles[0]
        self.assertEqual(handle.tool_call_id, "call_1")
        self.assertEqual(handle.tool_name, "add")
        self.assertEqual(handle.input, {"a": 1, "b": 2})
        self.assertEqual(handle.output, "3")
        self.assertIsNone(handle.error)
        self.assertTrue(handle.completed)

    def test_failing_tool_reports_an_error_handle(self):
        app = _agent_graph(self._failing_tool_node())
        with app.stream_events(
            {"messages": []}, version="v3", transformers=[ToolCallTransformer]
        ) as run:
            handles = list(run.tool_calls)
        self.assertEqual(len(handles), 1)
        self.assertTrue(handles[0].completed)
        self.assertIn("tool exploded", handles[0].error)

    def test_messages_mode_is_unaffected(self):
        app = _agent_graph(self._tool_node())
        result = app.invoke({"messages": []})
        self.assertEqual(result["messages"][-1]["content"], "3")

    def test_async_graph_emits_tool_events(self):
        tools = self._tool_node()

        async def agent(state):
            return {
                "messages": [
                    {
                        "role": "ai",
                        "content": "",
                        "tool_calls": [
                            {"name": "add", "args": {"a": 2, "b": 3}, "id": "call_a"}
                        ],
                    }
                ]
            }

        graph = rs.AsyncStateGraph({"messages": "messages"})
        graph.add_node("agent", agent)
        graph.add_node("tools", tools)
        graph.add_edge(rs.START, "agent")
        graph.add_edge("agent", "tools")
        graph.add_edge("tools", rs.END)
        app = graph.compile()

        async def collect():
            run = await app.astream_events(
                {"messages": []}, version="v3", transformers=[ToolCallTransformer]
            )
            return [handle async for handle in run.tool_calls]

        handles = asyncio.run(collect())
        self.assertEqual(len(handles), 1)
        self.assertEqual(handles[0].output, "5")


if __name__ == "__main__":
    unittest.main()