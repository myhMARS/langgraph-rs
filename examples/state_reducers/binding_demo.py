"""State and reducers — langgraph_rs Python binding.

Identical to `langgraph_demo.py` except for the import line: the reducer contract
(`Annotated[list, append]`, `Annotated[int, merge_max]`) is interpreted by the Rust engine.

Run:  python examples/state_reducers/binding_demo.py
      python examples/state_reducers/binding_demo.py --bench 500
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


def merge_max(left, right):
    return max(left or 0, right or 0)


class State(TypedDict):
    items: Annotated[list, append]
    total: Annotated[int, merge_max]
    note: str


def left(state: State):
    return {"items": ["left-1", "left-2"], "total": 3}


def right(state: State):
    return {"items": ["right-1", "right-2"], "total": 7}


def summarize(state: State):
    return {"note": f"{len(state['items'])} items, total={state['total']}"}


graph = StateGraph(State)
graph.add_node("left", left)
graph.add_node("right", right)
graph.add_node("summarize", summarize)
graph.add_edge(START, "left")
graph.add_edge(START, "right")
graph.add_edge("left", "summarize")
graph.add_edge("right", "summarize")
graph.add_edge("summarize", END)
compiled = graph.compile()


def run_once():
    state = compiled.invoke({"items": [], "total": 0, "note": ""})
    return {
        "items": sorted(state["items"]),
        "note": state["note"],
        "total": state["total"],
    }


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