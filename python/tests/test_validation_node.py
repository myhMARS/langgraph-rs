"""`ValidationNode`, `ToolRuntime` injection and the encrypted serializer."""

from __future__ import annotations

import typing
import unittest

from langchain_core.tools import tool

import langgraph_rs as lg
from langgraph_rs.checkpoint.serde import EncryptedSerializer
from langgraph_rs.prebuilt import ToolNode, ToolRuntime, ValidationNode

try:  # LangGraph is optional; the comparison test skips when it is absent.
    from langgraph.graph import END as _LG_END
    from langgraph.graph import START as _LG_START
    from langgraph.graph import StateGraph as _LgGraph
    from langgraph.graph.message import add_messages as _lg_add_messages
    from langgraph.prebuilt import ValidationNode as _LgValidationNode

    class _LgState(typing.TypedDict):
        messages: typing.Annotated[list, _lg_add_messages]

except Exception:  # pragma: no cover - langgraph is optional
    _LgState = None


@tool
def search(query: str) -> str:
    """Search for something."""
    return "result"


@tool
def inspect_state(runtime: ToolRuntime) -> str:
    """Report what the runtime injected."""
    return f"{runtime.state.get('user')}|{runtime.store is not None}"


def _call(name: str, args: dict, extra: dict | None = None) -> dict:
    payload = {
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [{"name": name, "args": args, "id": "c1"}]}
        ]
    }
    payload.update(extra or {})
    return payload


def _compile(node, schema=None):
    graph = lg.StateGraph(schema or {"messages": "add_messages"})
    graph.add_node("node", node)
    graph.set_entry_point("node")
    graph.set_finish_point("node")
    return graph


class ValidationNodeTest(unittest.TestCase):
    def test_valid_arguments_are_echoed_as_compact_json(self):
        result = _compile(ValidationNode([search])).compile().invoke(_call("search", {"query": "x"}))
        tool_messages = [m for m in result["messages"] if getattr(m, "type", None) == "tool"]
        self.assertEqual([m.content for m in tool_messages], ['{"query":"x"}'])
        self.assertEqual(tool_messages[0].tool_call_id, "c1")

    def test_invalid_arguments_report_the_validation_error(self):
        result = _compile(ValidationNode([search])).compile().invoke(_call("search", {}))
        content = [m for m in result["messages"] if getattr(m, "type", None) == "tool"][0].content
        self.assertIn("validation error", content)
        self.assertIn("query", content)

    def test_unknown_tool_fails_the_node(self):
        # LangGraph looks the schema up by name, so an unknown tool raises `KeyError`.
        with self.assertRaises(Exception) as raised:
            _compile(ValidationNode([search])).compile().invoke(_call("other", {}))
        self.assertIn("KeyError", str(raised.exception) + type(raised.exception).__name__)

    def test_matches_langgraph(self):
        if _LgState is None:  # pragma: no cover - langgraph is optional here
            self.skipTest("langgraph is not installed")
        payload = _call("search", {"query": "x"})
        payload["messages"][0]["tool_calls"][0]["type"] = "tool_call"
        lg_graph = _LgGraph(_LgState)
        lg_graph.add_node("node", _LgValidationNode([search]))
        lg_graph.add_edge(_LG_START, "node")
        lg_graph.add_edge("node", _LG_END)
        lg_result = lg_graph.compile().invoke(payload)
        rs_result = _compile(ValidationNode([search])).compile().invoke(payload)
        lg_content = [
            m.content if not isinstance(m, dict) else m.get("content")
            for m in lg_result["messages"]
            if (getattr(m, "type", None) or m.get("role")) == "tool"
        ]
        rs_content = [m.content for m in rs_result["messages"] if getattr(m, "type", None) == "tool"]
        self.assertEqual(rs_content, lg_content)


class ToolRuntimeTest(unittest.TestCase):
    def test_runtime_is_injected_with_state_and_store(self):
        compiled = _compile(ToolNode([inspect_state]), {"messages": "add_messages", "user": "last_value"}).compile(
            store=lg.InMemoryStore()
        )
        result = compiled.invoke(_call("inspect_state", {}, {"user": "ada"}))
        tool_messages = [m for m in result["messages"] if getattr(m, "type", None) == "tool"]
        self.assertEqual([m.content for m in tool_messages], ["ada|True"])


class EncryptedSerializerTest(unittest.TestCase):
    def test_round_trip(self):
        try:
            from cryptography.fernet import Fernet
        except ImportError:  # pragma: no cover - optional dependency
            self.skipTest("cryptography is not installed")
        serializer = EncryptedSerializer.from_pycryptodome_aes(Fernet.generate_key())
        kind, blob = serializer.dumps_typed({"a": 1, "b": [2]})
        self.assertTrue(kind.startswith("encrypted+"))
        self.assertEqual(serializer.loads_typed((kind, blob)), {"a": 1, "b": [2]})
        self.assertNotIn(b'"a"', blob)


if __name__ == "__main__":
    unittest.main()