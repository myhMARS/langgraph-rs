"""`InjectedState` / `InjectedStore` tool arguments are filled by the runtime.

LangGraph removes these parameters from the schema the model sees and substitutes the graph state
or the long-term store when the tool runs; the same behaviour is verified here, including a
side-by-side comparison with LangGraph's own `ToolNode`.
"""

from __future__ import annotations

import typing
import unittest

from langchain_core.tools import tool

import langgraph_rs as lg
from langgraph_rs.prebuilt import InjectedState, InjectedStore, ToolNode

try:  # LangGraph is optional; the comparison test skips when it is absent.
    from langgraph.graph import END as _LG_END
    from langgraph.graph import START as _LG_START
    from langgraph.graph import StateGraph as _LgStateGraph
    from langgraph.graph.message import add_messages as _lg_add_messages
    from langgraph.prebuilt import InjectedState as _LgInjectedState
    from langgraph.prebuilt import ToolNode as _LgToolNode

    @tool
    def _lg_lookup(query: str, state: typing.Annotated[dict, _LgInjectedState]) -> str:
        """Look something up in the graph state (LangGraph marker)."""
        return f"{query}|{state.get('user')}"

    class _LgState(typing.TypedDict):
        messages: typing.Annotated[list, _lg_add_messages]
        user: str

except Exception:  # pragma: no cover - langgraph is optional
    _LgState = None


class _Recorder:
    def __init__(self) -> None:
        self.data: dict[tuple[tuple[str, ...], str], object] = {}

    def get(self, namespace, key):
        value = self.data.get((tuple(namespace), key))
        return None if value is None else {"namespace": tuple(namespace), "key": key, "value": value}

    def put(self, namespace, key, value, index=None):
        self.data[(tuple(namespace), key)] = value

    def delete(self, namespace, key):
        self.data.pop((tuple(namespace), key), None)

    def search(self, namespace_prefix, query=None, filter=None, limit=10, offset=0):
        prefix = tuple(namespace_prefix)
        return [
            {"namespace": namespace, "key": key, "value": value}
            for (namespace, key), value in self.data.items()
            if namespace[: len(prefix)] == prefix
        ][:limit]

    def list_namespaces(self, **kwargs):
        return sorted({namespace for (namespace, _key) in self.data})


@tool
def lookup(query: str, state: typing.Annotated[dict, InjectedState]) -> str:
    """Look something up in the graph state."""
    return f"{query}|{state.get('user')}"


@tool
def remember(value: str, store: typing.Annotated[typing.Any, InjectedStore]) -> str:
    """Remember a value in the long-term store."""
    store.put(("mem",), "k", value)
    return "stored"


def _state(message: str, name: str, args: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "", "tool_calls": [{"name": name, "args": args, "id": "c1"}]}
        ],
        "user": message,
    }


class InjectionTest(unittest.TestCase):
    def test_injected_parameters_are_hidden_from_the_model(self):
        schema = lookup.tool_call_schema.model_json_schema()["properties"]
        self.assertEqual(list(schema), ["query"])
        schema = remember.tool_call_schema.model_json_schema()["properties"]
        self.assertEqual(list(schema), ["value"])

    def test_injected_state_reaches_the_tool(self):
        output = ToolNode([lookup])(_state("ada", "lookup", {"query": "q"}))
        self.assertEqual(output["messages"][0]["content"], "q|ada")

    def test_injected_store_reaches_the_tool(self):
        store = _Recorder()
        graph = lg.StateGraph({"messages": "add_messages"})
        graph.add_node("tools", ToolNode([remember]))
        graph.set_entry_point("tools")
        graph.set_finish_point("tools")
        result = graph.compile(store=store).invoke(_state("", "remember", {"value": "hi"}))
        self.assertEqual(
            [message.content for message in result["messages"] if getattr(message, "type", None) == "tool"],
            ["stored"],
        )
        self.assertEqual(store.data[(("mem",), "k")], "hi")

    def test_plain_callable_injection_uses_keyword_arguments(self):
        seen: dict[str, object] = {}

        def direct(query, state):
            seen["query"] = query
            seen["user"] = state["user"]
            return "ok"

        direct.__annotations__ = {
            "query": str,
            "state": typing.Annotated[dict, InjectedState],
        }
        output = ToolNode([direct])(_state("grace", "direct", {"query": "q"}))
        self.assertEqual(output["messages"][0]["content"], "ok")
        self.assertEqual(seen, {"query": "q", "user": "grace"})

    def test_matches_langgraph_tool_node(self):
        try:
            from langgraph.prebuilt import ToolNode as LgToolNode
        except ImportError:  # pragma: no cover - langgraph is optional here
            self.skipTest("langgraph is not installed")
        if _LgState is None:
            self.skipTest("langgraph is not installed")
        from langchain_core.messages import AIMessage

        # LangGraph expects an AIMessage; the extension takes the JSON form of the same call.
        lg_payload = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "_lg_lookup", "args": {"query": "q"}, "id": "c1"}],
                )
            ],
            "user": "ada",
        }
        payload = _state("ada", "lookup", {"query": "q"})
        # LangGraph's ToolNode only runs inside a compiled graph (it needs the run config).
        lg_graph = _LgStateGraph(_LgState)
        lg_graph.add_node("tools", _LgToolNode([_lg_lookup]))
        lg_graph.add_edge(_LG_START, "tools")
        lg_graph.add_edge("tools", _LG_END)
        expected = lg_graph.compile().invoke(lg_payload)

        # The state declares `user` too: LangGraph drops keys the state does not declare.
        rs_graph = lg.StateGraph({"messages": "add_messages", "user": "last_value"})
        rs_graph.add_node("tools", ToolNode([lookup]))
        rs_graph.set_entry_point("tools")
        rs_graph.set_finish_point("tools")
        actual = rs_graph.compile().invoke(payload)

        def contents(messages):
            # LangGraph returns ToolMessage objects, the extension returns their JSON form.
            return [
                message.get("content") if isinstance(message, dict) else message.content
                for message in messages
            ]

        self.assertEqual(contents(actual["messages"]), contents(expected["messages"]))
        self.assertIn("q|ada", contents(actual["messages"]))


if __name__ == "__main__":
    unittest.main()