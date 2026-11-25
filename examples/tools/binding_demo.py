"""Tools — langgraph_rs Python binding.

Same graph as `langgraph_demo.py` (one `ToolNode`, two tools, `InjectedState` hidden from the model
schema). Dict-based messages are used instead of `AIMessage` objects — the engine's `ToolNode` reads
the `tool_calls` field either way, and the demo compares the tool *results*.

Run:  python examples/tools/binding_demo.py
      python examples/tools/binding_demo.py --bench 500
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Annotated, TypedDict

from langchain_core.tools import tool

from langgraph_rs import END, START, StateGraph
from langgraph_rs.prebuilt import InjectedState, ToolNode


def append(left, right):
    return (left or []) + (right or [])


class State(TypedDict):
    messages: Annotated[list, append]
    user: str


# `@tool` keeps the LangChain calling convention: arguments arrive as keyword arguments.
@tool
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


@tool
def lookup(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Look something up for the current user."""
    return f"{query}|{state.get('user')}"


graph = StateGraph(State)
graph.add_node("tools", ToolNode([add, lookup]))
graph.add_edge(START, "tools")
graph.add_edge("tools", END)
COMPILED = graph.compile()


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": name, "args": args, "id": call_id}],
    }


def _content(message) -> str:
    return message["content"] if isinstance(message, dict) else message.content


def _role(message) -> str:
    return message.get("role") if isinstance(message, dict) else getattr(message, "type", None)


def run_once():
    messages = [_tool_call("add", {"a": 2, "b": 3}, "c1")]
    state = COMPILED.invoke({"messages": messages, "user": "ada"})
    messages = list(state["messages"]) + [_tool_call("lookup", {"query": "q"}, "c2")]
    state = COMPILED.invoke({"messages": messages, "user": "ada"})
    results = [_content(message) for message in state["messages"] if _role(message) == "tool"]
    return {"results": results}


def bench(iterations: int) -> float:
    for _ in range(50):
        run_once()
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            run_once()
        samples.append((time.perf_counter_ns() - start) / iterations)
    return statistics.median(samples)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        iterations = int(sys.argv[sys.argv.index("--bench") + 1])
        print(json.dumps({"engine": "langgraph_rs binding", "ns_per_op": bench(iterations)}))
    else:
        print(json.dumps(run_once(), sort_keys=True))