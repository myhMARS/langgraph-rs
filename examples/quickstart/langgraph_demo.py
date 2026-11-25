"""Quickstart demo — LangGraph (reference implementation).

A three-node state graph with a conditional loop: `seed` sets the target, `increment` bumps the
counter, and a conditional edge loops back until `count == target`.

Run:

    python examples/quickstart/langgraph_demo.py          # one run
    python examples/quickstart/langgraph_demo.py --bench 2000   # ns/op

The output is the canonical result the other three implementations must reproduce.
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
    count: int
    target: int
    steps: Annotated[list, append]


def seed(state: State):
    target = state["target"] or 3
    return {"target": target, "steps": ["seed"]}


def increment(state: State):
    return {"count": state["count"] + 1, "steps": [f"increment -> {state['count'] + 1}"]}


def keep_going(state: State):
    return "increment" if state["count"] < state["target"] else END


graph = StateGraph(State)
graph.add_node("seed", seed)
graph.add_node("increment", increment)
graph.add_edge(START, "seed")
graph.add_edge("seed", "increment")
graph.add_conditional_edges("increment", keep_going)
compiled = graph.compile()


def run_once():
    return compiled.invoke({"count": 0, "target": 3, "steps": []})


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
        print(json.dumps({"engine": "langgraph", "ns_per_op": bench(iterations)}))
    else:
        print(json.dumps(run_once(), sort_keys=True))