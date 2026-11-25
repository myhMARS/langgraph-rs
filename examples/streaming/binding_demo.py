"""Streaming — langgraph_rs Python binding.

Same graph and the same `stream(..., stream_mode=...)` call as `langgraph_demo.py`; the chunks come
from the Rust engine.

Run:  python examples/streaming/binding_demo.py
      python examples/streaming/binding_demo.py --mode values
      python examples/streaming/binding_demo.py --bench 500
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Annotated, TypedDict

from langgraph_rs import END, START, StateGraph


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
    return [_jsonable(chunk) for chunk in compiled.stream({"log": []}, stream_mode=mode)]


def _jsonable(chunk):
    if isinstance(chunk, dict):
        return {key: _jsonable(value) for key, value in chunk.items()}
    if isinstance(chunk, (list, tuple)):
        return [_jsonable(value) for value in chunk]
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump()
    if hasattr(chunk, "dict"):
        return chunk.dict()
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
        print(json.dumps({"engine": "langgraph_rs binding", "ns_per_op": bench(iterations)}))
    else:
        print(json.dumps(run_once(), sort_keys=True))