"""Human-in-the-loop — langgraph_rs Python binding.

Same review graph as `langgraph_demo.py`; only the imports change. `invoke` returns the pending
`__interrupt__` instead of raising, and `Command(resume=...)` continues the run.

Run:  python examples/interrupt/binding_demo.py
      python examples/interrupt/binding_demo.py --bench 100
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import TypedDict

from langgraph_rs import END, START, StateGraph, interrupt
from langgraph_rs.checkpoint import InMemorySaver
from langgraph_rs.types import Command


class State(TypedDict):
    draft: str
    approved: bool
    feedback: str


def draft(state: State):
    return {"draft": "Quarterly report v1"}


def review(state: State):
    answer = interrupt(
        {"question": f"Approve `{state['draft']}`?", "options": ["approve", "reject"]}
    )
    return {
        "approved": bool(answer.get("approved")),
        "feedback": answer.get("comment", ""),
    }


def publish(state: State):
    label = "published" if state["approved"] else "sent back for revision"
    return {"draft": f"{state['draft']} [{label}]"}


def build(saver):
    graph = StateGraph(State)
    graph.add_node("draft", draft)
    graph.add_node("review", review)
    graph.add_node("publish", publish)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "review")
    graph.add_edge("review", "publish")
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=saver)


COMPILED = build(InMemorySaver())
ANSWER = {"approved": True, "comment": "Looks good, ship it!"}


def _next_node(snapshot):
    pending = getattr(snapshot, "next", None)
    if pending is None:
        try:
            pending = snapshot["next"]
        except Exception:  # pragma: no cover - defensive
            pending = ()
    return next(iter(pending), "")


def run_once(thread="demo"):
    paused = COMPILED.invoke(
        {"draft": "", "approved": False, "feedback": ""}, thread_id=thread
    )
    snapshot = COMPILED.get_state(thread)
    state = COMPILED.invoke(Command(resume=ANSWER), thread_id=thread)
    return {
        "approved": state["approved"],
        "draft": state["draft"],
        "feedback": state["feedback"],
        "paused_at": _next_node(snapshot),
        "payload": paused["__interrupt__"][0].value,
    }


def bench(iterations: int) -> float:
    for index in range(20):
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
        print(json.dumps(run_once(), sort_keys=True))