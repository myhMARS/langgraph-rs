"""Routing — langgraph_rs Python binding.

Same graph as `langgraph_demo.py`: one conditional edge plus a node that returns
`Command(goto=...)`. Only the imports change.

Run:  python examples/routing/binding_demo.py
      python examples/routing/binding_demo.py --bench 500
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Annotated, TypedDict

from langgraph_rs import END, START, StateGraph
from langgraph_rs.types import Command


def append(left, right):
    return (left or []) + (right or [])


class State(TypedDict):
    question: str
    answer: str
    route: str
    trace: Annotated[list, append]


def classify(state: State):
    size = len(state["question"])
    return {"route": "fast" if size < 20 else "slow", "trace": ["classify"]}


def route_of(state: State):
    return state["route"]


def fast(state: State):
    return {"answer": "quick answer", "trace": ["fast"]}


def slow(state: State):
    return {"answer": "researched answer", "trace": ["slow"]}


def polish(state: State):
    return Command(
        update={"answer": state["answer"].upper(), "trace": ["polish"]},
        goto=END,
    )


graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("fast", fast)
graph.add_node("slow", slow)
graph.add_node("polish", polish)
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route_of, {"fast": "fast", "slow": "slow"})
graph.add_edge("fast", "polish")
graph.add_edge("slow", "polish")
COMPILED = graph.compile()


def run_once(question: str = "What is a superstep?"):
    state = COMPILED.invoke(
        {"question": question, "answer": "", "route": "", "trace": []}
    )
    return {"answer": state["answer"], "route": state["route"], "trace": state["trace"]}


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