"""The ``langgraph_rs`` package exposes the LangGraph module layout.

These tests pin the compatibility surface: the flat ``langgraph_rs.StateGraph`` API keeps
working, every supported ``langgraph_rs.<submodule>`` import resolves, a graph built through
the submodule imports runs, and LangGraph-only names fail loudly instead of silently.
"""

from __future__ import annotations

import operator
import typing
import unittest

import langgraph_rs


class PackageLayoutTest(unittest.TestCase):
    def test_flat_api_is_reexported(self):
        self.assertTrue(hasattr(langgraph_rs, "StateGraph"))
        self.assertTrue(hasattr(langgraph_rs, "InMemorySaver"))
        self.assertEqual(langgraph_rs.START, "__start__")
        self.assertEqual(langgraph_rs.END, "__end__")
        self.assertIn("StateGraph", langgraph_rs.__all__)

    def test_langgraph_style_imports_resolve(self):
        from langgraph_rs.cache.memory import InMemoryCache  # noqa: F401
        from langgraph_rs.checkpoint.memory import AsyncInMemorySaver, InMemorySaver  # noqa: F401
        from langgraph_rs.checkpoint.postgres import PostgresSaver  # noqa: F401
        from langgraph_rs.checkpoint.redis import RedisSaver  # noqa: F401
        from langgraph_rs.checkpoint.sqlite import AsyncSqliteSaver, SqliteSaver  # noqa: F401
        from langgraph_rs.config import get_config, get_store, get_stream_writer  # noqa: F401
        from langgraph_rs.constants import END, START  # noqa: F401
        from langgraph_rs.func import entrypoint, task  # noqa: F401
        from langgraph_rs.graph import MessagesState, StateGraph, add_messages  # noqa: F401
        from langgraph_rs.graph.message import RemoveMessage  # noqa: F401
        from langgraph_rs.graph.state import CompiledStateGraph  # noqa: F401
        from langgraph_rs.managed import IsLastStep, RemainingSteps  # noqa: F401
        from langgraph_rs.prebuilt import ToolNode, create_react_agent, tools_condition  # noqa: F401
        from langgraph_rs.prebuilt.tool_node import ToolNode as ToolNodeFromModule  # noqa: F401
        from langgraph_rs.store.memory import InMemoryStore  # noqa: F401
        from langgraph_rs.types import (  # noqa: F401
            CachePolicy,
            Command,
            RetryPolicy,
            Send,
            StateSnapshot,
            TimeoutPolicy,
            interrupt,
        )

    def test_graph_runs_through_submodule_imports(self):
        from langgraph_rs.checkpoint.memory import InMemorySaver
        from langgraph_rs.graph import END, START, StateGraph

        class State(typing.TypedDict):
            log: typing.Annotated[list, operator.add]
            count: int

        def step(state):
            return {"count": state.get("count", 0) + 1, "log": ["step"]}

        def route(state):
            return "step" if state["count"] < 3 else END

        builder = StateGraph(State)
        builder.add_node("step", step)
        builder.add_edge(START, "step")
        builder.add_conditional_edges("step", route, {"step": "step", END: END})
        app = builder.compile(checkpointer=InMemorySaver())

        cfg = {"configurable": {"thread_id": "layout"}}
        result = app.invoke({"log": [], "count": 0}, cfg)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["log"], ["step", "step", "step"])
        self.assertEqual(app.get_state(cfg).values["count"], 3)
        self.assertEqual(
            [chunk for chunk in app.stream({"log": [], "count": 0}, cfg, stream_mode="values")][-1]["count"],
            3,
        )
        self.assertGreaterEqual(len(list(app.get_state_history(cfg))), 1)

    def test_unsupported_names_import_but_raise(self):
        from langgraph_rs.checkpoint.base import BaseCheckpointSaver

        with self.assertRaises(NotImplementedError):
            BaseCheckpointSaver()
        with self.assertRaises(NotImplementedError):
            type("MySaver", (BaseCheckpointSaver,), {})

    def test_unknown_attribute_message(self):
        with self.assertRaises(AttributeError) as raised:
            langgraph_rs.definitely_not_a_symbol
        self.assertIn("definitely_not_a_symbol", str(raised.exception))


if __name__ == "__main__":
    unittest.main()