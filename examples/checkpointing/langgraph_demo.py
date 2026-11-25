"""Checkpointing — LangGraph reference implementation.

One node, one thread, and the four checkpoint operations you actually use:

1. three runs accumulate `count` (each run starts from the previous checkpoint),
2. `update_state` forks the thread with a new value (no node runs),
3. the next run continues from that forked state,
4. `get_state_history` lists every checkpoint of the thread (printed to stderr so it never
   interferes with the cross-implementation comparison).

Run:  python examples/checkpointing/langgraph_demo.py
      python examples/checkpointing/langgraph_demo.py --bench 200
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


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
    config = {"configurable": {"thread_id": thread}}
    count = 0
    for _ in range(3):
        state = app.invoke({"count": count}, config)
        count = state["count"]
    # Fork the thread without running anything.
    app.update_state(config, {"count": 42})
    forked = app.invoke({"count": 42}, config)
    if report:
        history = list(app.get_state_history(config))
        print(f"history checkpoints: {len(history)}", file=sys.stderr)
        for snapshot in history:
            values = snapshot.values
            print(f"  step={snapshot.metadata.get('step')} count={values.get('count')}", file=sys.stderr)
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
        print(json.dumps({"engine": "langgraph", "ns_per_op": bench(iterations)}))
    else:
        print(json.dumps(run_once(report=True), sort_keys=True))