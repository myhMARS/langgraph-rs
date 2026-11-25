"""Streaming — LangGraph reference implementation.

`compiled.stream(..., stream_mode="updates")` yields one `{node: writes}` map per node, in
execution order. This demo collects the update stream and prints it as JSON, so the binding and the
native implementation can be compared chunk by chunk.

Run:  python examples/streaming/langgraph_demo.py
      python examples/streaming/langgraph_demo.py --mode values
      python examples/streaming/langgraph_demo.py --bench 500
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


def append(left, right):
    return (left or []) + (right or [])


class State(TypedDict):
    log: Annotated[list, append]
    plan: str
    result: str
    summary: str


def plan(state: State):
    return {"plan": "3 steps", "log": ["plan"]}


def work(state: State):
    return {"result": f"did {state['plan']}", "log": ["work"]}


def report(state: State):
    return {"summary": f"{state['result']} / {len(state['log'])} events", "log": ["report"]}


graph = StateGraph(State)
graph.add_node("plan", plan)
graph.add_node("work", work)
graph.add_node("report", report)
graph.add_edge(START, "plan")
graph.add_edge("plan", "work")
graph.add_edge("work", "report")
graph.add_edge("report", END)
compiled = graph.compile()


def _mode() -> str:
    if "--mode" in sys.argv:
        return sys.argv[sys.argv.index("--mode") + 1]
    return "updates"


def collect(mode: str = "updates"):
    """Return the stream as a list of JSON-able chunks."""
    chunks = []
    for chunk in compiled.stream({"log": []}, stream_mode=mode):
        chunks.append(_jsonable(chunk))
    return chunks


def _jsonable(chunk):
    if isinstance(chunk, dict):
        return {key: _jsonable(value) for key, value in chunk.items()}
    if isinstance(chunk, (list, tuple)):
        return [_jsonable(value) for value in chunk]
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump()
    if hasattr(chunk, "dict"):
        return chunk.dict()
    if hasattr(chunk, "__dict__") and type(chunk).__module__.startswith(("langgraph", "langchain")):
        return {key: _jsonable(value) for key, value in vars(chunk).items()}
    return chunk


def run_once():
    return {"chunks": collect(_mode()), "mode": _mode()}


def bench(iterations: int) -> float:
    for _ in range(50):
        collect("updates")
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            collect("updates")
        samples.append((time.perf_counter_ns() - start) / iterations)
    return statistics.median(samples)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        iterations = int(sys.argv[sys.argv.index("--bench") + 1])
        print(json.dumps({"engine": "langgraph", "ns_per_op": bench(iterations)}))
    else:
        print(json.dumps(run_once(), sort_keys=True))