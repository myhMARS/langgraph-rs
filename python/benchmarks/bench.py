"""Benchmarks for Python LangGraph and for the langgraph-rs Python bindings.

The scenarios mirror ``benches/bench_rust.rs`` (same node counts, same iterations),
so the three engines can be compared on the same machine:

    python python/benchmarks/bench.py --engine langgraph --json out_langgraph.json
    python python/benchmarks/bench.py --engine binding --json out_binding.json

One JSON object per scenario is printed to stdout; a human summary goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Annotated, TypedDict

LINEAR_NODES = 10
LINEAR_ITERATIONS = 1000
LOOP_TARGET = 50
LOOP_RECURSION_LIMIT = 100
LOOP_ITERATIONS = 200
FANOUT_ITERATIONS = 1000
CHECKPOINT_ITERATIONS = 200
CHECKPOINT_NODES = 5
BUILD_ITERATIONS = 500

RESULT_FIELDS = ("iterations", "total_ms", "ns_per_op", "ops_per_sec")


def measure(warmup: int, iterations: int, operation) -> float:
    for _ in range(warmup):
        operation()
    started = time.perf_counter()
    for _ in range(iterations):
        operation()
    return time.perf_counter() - started


def report(engine: str, scenario: str, iterations: int, seconds: float, results: list) -> None:
    ns_per_op = seconds * 1e9 / iterations
    ops_per_sec = iterations / seconds
    record = {
        "engine": engine,
        "scenario": scenario,
        "iterations": iterations,
        "total_ms": round(seconds * 1000.0, 3),
        "ns_per_op": round(ns_per_op, 1),
        "ops_per_sec": round(ops_per_sec, 1),
    }
    results.append(record)
    print(json.dumps(record), flush=True)
    print(
        f"{engine:<11} {scenario:<18} {ns_per_op:>12.0f} ns/op {ops_per_sec:>12.0f} ops/s",
        file=sys.stderr,
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Python LangGraph engine
# --------------------------------------------------------------------------- #
class CounterState(TypedDict):
    value: int


def increment(state):
    return {"value": state["value"] + 1}


def langgraph_linear_10():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(CounterState)
    names = []
    for index in range(LINEAR_NODES):
        name = f"n{index:02d}"
        names.append(name)
        graph.add_node(name, increment)
    graph.add_edge(START, names[0])
    for previous, current in zip(names, names[1:]):
        graph.add_edge(previous, current)
    graph.add_edge(names[-1], END)
    return graph.compile()


def langgraph_loop_50():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(CounterState)
    graph.add_node("increment", increment)
    graph.add_edge(START, "increment")
    graph.add_conditional_edges(
        "increment",
        lambda state: "increment" if state["value"] < LOOP_TARGET else END,
    )
    return graph.compile()


def langgraph_fanout_4():
    from langgraph.graph import END, START, StateGraph

    class FanState(TypedDict):
        value: int
        a: int
        b: int
        c: int
        d: int

    graph = StateGraph(FanState)

    def dispatch(state):
        return {}

    def worker(key):
        def write(state):
            return {key: 1}

        return write

    def join(state):
        return {"value": sum(state.get(key, 0) for key in "abcd")}

    graph.add_node("dispatch", dispatch)
    graph.add_edge(START, "dispatch")
    for key in "abcd":
        graph.add_node(key, worker(key))
        graph.add_edge("dispatch", key)
        graph.add_edge(key, "join")
    graph.add_node("join", join)
    graph.add_edge("join", END)
    return graph.compile()


def langgraph_checkpoint_5():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(CounterState)
    names = []
    for index in range(CHECKPOINT_NODES):
        name = f"n{index:02d}"
        names.append(name)
        graph.add_node(name, increment)
    graph.add_edge(START, names[0])
    for previous, current in zip(names, names[1:]):
        graph.add_edge(previous, current)
    graph.add_edge(names[-1], END)
    return graph.compile(checkpointer=MemorySaver())


def run_langgraph() -> list:
    results: list = []

    linear = langgraph_linear_10()
    input_state = {"value": 0}
    seconds = measure(LINEAR_ITERATIONS // 10, LINEAR_ITERATIONS, lambda: linear.invoke(input_state))
    report("langgraph", "linear_10", LINEAR_ITERATIONS, seconds, results)

    looping = langgraph_loop_50()
    seconds = measure(
        LOOP_ITERATIONS // 10,
        LOOP_ITERATIONS,
        lambda: looping.invoke(input_state, {"recursion_limit": LOOP_RECURSION_LIMIT}),
    )
    report("langgraph", "loop_50", LOOP_ITERATIONS, seconds, results)

    fanout = langgraph_fanout_4()
    fan_input = {"value": 0, "a": 0, "b": 0, "c": 0, "d": 0}
    seconds = measure(FANOUT_ITERATIONS // 10, FANOUT_ITERATIONS, lambda: fanout.invoke(fan_input))
    report("langgraph", "fanout_4", FANOUT_ITERATIONS, seconds, results)

    checkpointed = langgraph_checkpoint_5()
    counter = {"value": 0}

    def checkpointed_run():
        counter["value"] += 1
        checkpointed.invoke(
            input_state,
            {"configurable": {"thread_id": f"bench-{counter['value']}"}},
        )

    seconds = measure(CHECKPOINT_ITERATIONS // 10, CHECKPOINT_ITERATIONS, checkpointed_run)
    report("langgraph", "checkpoint_5", CHECKPOINT_ITERATIONS, seconds, results)

    seconds = measure(BUILD_ITERATIONS // 10, BUILD_ITERATIONS, langgraph_linear_10)
    report("langgraph", "build_compile", BUILD_ITERATIONS, seconds, results)
    return results


# --------------------------------------------------------------------------- #
# langgraph-rs Python bindings
# --------------------------------------------------------------------------- #
def binding_linear_10():
    import langgraph_rs as lg

    graph = lg.StateGraph({"value": "last_value"})
    names = []
    for index in range(LINEAR_NODES):
        name = f"n{index:02d}"
        names.append(name)
        graph.add_node(name, lambda state: {"value": state["value"] + 1})
    graph.add_edge(lg.START, names[0])
    for previous, current in zip(names, names[1:]):
        graph.add_edge(previous, current)
    graph.add_edge(names[-1], lg.END)
    return graph.compile()


def binding_loop_50():
    import langgraph_rs as lg

    graph = lg.StateGraph({"value": "last_value"})
    graph.add_node("increment", lambda state: {"value": state["value"] + 1})
    graph.set_entry_point("increment")
    graph.add_conditional_edges(
        "increment",
        lambda state: "increment" if state["value"] < LOOP_TARGET else None,
    )
    return graph.compile(recursion_limit=LOOP_RECURSION_LIMIT)


def binding_fanout_4():
    import langgraph_rs as lg

    graph = lg.StateGraph(
        {key: "last_value" for key in ("value", "a", "b", "c", "d")}
    )
    graph.add_node("dispatch", lambda state: {})
    graph.set_entry_point("dispatch")

    def worker(key):
        return lambda state: {key: 1}

    for key in "abcd":
        graph.add_node(key, worker(key))
        graph.add_edge("dispatch", key)
        graph.add_edge(key, "join")
    graph.add_node(
        "join",
        lambda state: {
            "value": sum(state.get(key, 0) for key in "abcd")
        },
    )
    graph.set_finish_point("join")
    return graph.compile()


def binding_checkpoint_5():
    import langgraph_rs as lg

    graph = lg.StateGraph({"value": "last_value"})
    names = []
    for index in range(CHECKPOINT_NODES):
        name = f"n{index:02d}"
        names.append(name)
        graph.add_node(name, lambda state: {"value": state["value"] + 1})
    graph.set_entry_point(names[0])
    for previous, current in zip(names, names[1:]):
        graph.add_edge(previous, current)
    graph.set_finish_point(names[-1])
    return graph.compile(checkpointer=True)


def run_binding() -> list:
    results: list = []

    linear = binding_linear_10()
    input_state = {"value": 0}
    seconds = measure(LINEAR_ITERATIONS // 10, LINEAR_ITERATIONS, lambda: linear.invoke(input_state))
    report("binding", "linear_10", LINEAR_ITERATIONS, seconds, results)

    looping = binding_loop_50()
    seconds = measure(LOOP_ITERATIONS // 10, LOOP_ITERATIONS, lambda: looping.invoke(input_state))
    report("binding", "loop_50", LOOP_ITERATIONS, seconds, results)

    fanout = binding_fanout_4()
    fan_input = {"value": 0, "a": 0, "b": 0, "c": 0, "d": 0}
    seconds = measure(FANOUT_ITERATIONS // 10, FANOUT_ITERATIONS, lambda: fanout.invoke(fan_input))
    report("binding", "fanout_4", FANOUT_ITERATIONS, seconds, results)

    checkpointed = binding_checkpoint_5()
    counter = {"value": 0}

    def checkpointed_run():
        counter["value"] += 1
        checkpointed.invoke(input_state, thread_id=f"bench-{counter['value']}")

    seconds = measure(CHECKPOINT_ITERATIONS // 10, CHECKPOINT_ITERATIONS, checkpointed_run)
    report("binding", "checkpoint_5", CHECKPOINT_ITERATIONS, seconds, results)

    seconds = measure(BUILD_ITERATIONS // 10, BUILD_ITERATIONS, binding_linear_10)
    report("binding", "build_compile", BUILD_ITERATIONS, seconds, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["langgraph", "binding", "both"], default="both")
    parser.add_argument("--json", help="write all benchmark records to this JSON file")
    args = parser.parse_args()

    results: list = []
    if args.engine in ("langgraph", "both"):
        results.extend(run_langgraph())
    if args.engine in ("binding", "both"):
        results.extend(run_binding())

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()