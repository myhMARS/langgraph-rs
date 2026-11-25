"""Routing — LangGraph reference implementation.

Two ways to decide where a graph goes next:

1. **Conditional edges** — `add_conditional_edges("classify", router)` returns a node name.
2. **`Command(goto=...)`** — the node itself returns a `Command`, which also carries a state update.

```text
START -> classify --(conditional)--> fast | slow --> polish --(Command goto)--> END
```

Run:  python examples/routing/langgraph_demo.py
      python examples/routing/langgraph_demo.py --bench 500
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


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
    # `Command` replaces the conditional-edge style: the node picks its successor itself.
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
        print(json.dumps({"engine": "langgraph", "ns_per_op": bench(iterations)}))
    else:
        print(json.dumps(run_once(), sort_keys=True))