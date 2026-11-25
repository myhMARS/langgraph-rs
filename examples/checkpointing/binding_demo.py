"""Checkpointing — langgraph_rs Python binding.

Same thread as `langgraph_demo.py`: three runs, an `update_state` fork, a continuation, and the
checkpoint history (stderr). Only the imports change.

Run:  python examples/checkpointing/binding_demo.py
      python examples/checkpointing/binding_demo.py --bench 200
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Annotated, TypedDict

from langgraph_rs import END, START, StateGraph
from langgraph_rs.checkpoint import InMemorySaver


def append(left, right):
    return (left or []) + (right or [])


class State(TypedDict):
    count: int
    log: Annotated[list, append]


def increment(state: State):
    return {"count": state["count"] + 1, "log": [f"count={state['count'] + 1}"]}


def build():
    graph = StateGraph(State)
    graph.add_node("increment", increment)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    return graph.compile(checkpointer=InMemorySaver())


def run_once(thread: str = "demo", report: bool = False):
    app = build()
    count = 0
    for _ in range(3):
        state = app.invoke({"count": count}, thread_id=thread)
        count = state["count"]
    app.update_state(thread, {"count": 42})
    forked = app.invoke({"count": 42}, thread_id=thread)
    if report:
        history = list(app.get_state_history(thread))
        print(f"history checkpoints: {len(history)}", file=sys.stderr)
        for snapshot in history:
            values = snapshot["values"] if isinstance(snapshot, dict) else snapshot.values
            print(f"  count={values.get('count')}", file=sys.stderr)
    return {"final": forked["count"], "log": forked["log"]}


def bench(iterations: int) -> float:
    for index in range(10):
        run_once(f"warm-{index}")
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for index in range(iterations):
            run_once(f"bench-{index}")
        samples.append((time.perf_counter_ns() - start) / iterations)
    return statistics.median(samples)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        iterations = int(sys.argv[sys.argv.index("--bench") + 1])
        print(json.dumps({"engine": "langgraph_rs binding", "ns_per_op": bench(iterations)}))
    else:
        print(json.dumps(run_once(report=True), sort_keys=True))