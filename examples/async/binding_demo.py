"""Async execution — langgraph_rs Python binding.

Same async graph as `langgraph_demo.py`; the coroutines are driven by the Rust/Tokio runtime
(`AsyncStateGraph` + `ainvoke`), and concurrency across nodes of one superstep comes from Tokio.

Run:  python examples/async/binding_demo.py
      python examples/async/binding_demo.py --bench 500
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from typing import Annotated, TypedDict

from langgraph_rs import END, START, AsyncStateGraph


def append(left, right):
    return (left or []) + (right or [])


class State(TypedDict):
    question: str
    answer: str
    trace: Annotated[list, append]


async def retrieve(state: State):
    await asyncio.sleep(0)
    return {"answer": f"context for `{state['question']}`", "trace": ["retrieve"]}


async def generate(state: State):
    await asyncio.sleep(0)
    return {"answer": f"answer based on: {state['answer']}", "trace": ["generate"]}


async def judge(state: State):
    await asyncio.sleep(0)
    return {"trace": ["judge"]}


graph = AsyncStateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("judge", judge)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "judge")
graph.add_edge("judge", END)
COMPILED = graph.compile()


async def run_once():
    state = await COMPILED.ainvoke({"question": "What is a superstep?", "answer": "", "trace": []})
    return {"answer": state["answer"], "trace": state["trace"]}


def bench(iterations: int) -> float:
    async def measure() -> float:
        for _ in range(20):
            await run_once()
        samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            for _ in range(iterations):
                await run_once()
            samples.append((time.perf_counter_ns() - start) / iterations)
        return statistics.median(samples)

    return asyncio.run(measure())


if __name__ == "__main__":
    if "--bench" in sys.argv:
        iterations = int(sys.argv[sys.argv.index("--bench") + 1])
        print(json.dumps({"engine": "langgraph_rs binding", "ns_per_op": bench(iterations)}))
    else:
        print(json.dumps(asyncio.run(run_once()), sort_keys=True))