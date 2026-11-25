"""Subgraphs and `Send` fan-out — langgraph_rs Python binding.

Same graph as `langgraph_demo.py` (only the imports change): four `Send` worker tasks feed an append
reducer, then a compiled child graph runs as a subgraph and maps its write back into the parent.

Run:  python examples/subgraph/binding_demo.py
      python examples/subgraph/binding_demo.py --bench 200
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Annotated, TypedDict

from langgraph_rs import END, START, StateGraph
from langgraph_rs.types import Send

NUMBERS = [3, 8, 1, 6]


def append(left, right):
    return (left or []) + (right or [])


class State(TypedDict):
    results: Annotated[list, append]
    sorted_results: list
    total: int
    factor: int
    scaled: list


def dispatch(state: State):
    return {}


def worker(state):
    number = state["number"]
    return {"results": [number * number]}


def reduce(state: State):
    results = sorted(state["results"])
    return {"sorted_results": results, "total": sum(results)}


class SubState(TypedDict):
    total: int
    factor: int
    scaled: list


def scale(state: SubState):
    return {"scaled": [state["total"] * state["factor"]]}


def build_subgraph():
    sub = StateGraph(SubState)
    sub.add_node("scale", scale)
    sub.add_edge(START, "scale")
    sub.add_edge("scale", END)
    return sub.compile()


def build():
    graph = StateGraph(State)
    graph.add_node("dispatch", dispatch)
    graph.add_node("worker", worker)
    graph.add_node("reduce", reduce)
    graph.add_node("scaled", build_subgraph())
    graph.add_edge(START, "dispatch")
    graph.add_conditional_edges("dispatch", lambda state: [Send("worker", {"number": n}) for n in NUMBERS])
    graph.add_edge("worker", "reduce")
    graph.add_edge("reduce", "scaled")
    graph.add_edge("scaled", END)
    return graph.compile()


COMPILED = build()


def run_once():
    state = COMPILED.invoke({"results": [], "sorted_results": [], "total": 0, "factor": 2, "scaled": []})
    return {
        "scaled": state["scaled"],
        "sorted_results": state["sorted_results"],
        "total": state["total"],
    }


def bench(iterations: int) -> float:
    for _ in range(20):
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