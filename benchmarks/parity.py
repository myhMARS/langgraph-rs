"""Behavior + performance parity between Python LangGraph and langgraph-rs bindings.

Each scenario builds the same graph on both engines, compares observable behavior, and
records a small benchmark. Run from the repository root after building the extension:

    python benchmarks/parity.py

The script writes ``benchmarks/parity_results.json`` and ``benchmarks/PARITY.md``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TypedDict

import langgraph_rs as rs
from langgraph.checkpoint.memory import InMemorySaver as LgInMemorySaver
from langgraph.graph import END, START, StateGraph as LgStateGraph
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.types import Command as LgCommand, RetryPolicy as LgRetryPolicy
from langgraph.types import interrupt as lg_interrupt

HERE = Path(__file__).resolve().parent


class LgCommandState(TypedDict):
    value: int


class LgManagedState(TypedDict, total=False):
    value: str
    remaining_steps: RemainingSteps
    is_last_step: IsLastStep
WARMUP = 3
ITERATIONS = 100


def measure(operation) -> dict:
    for _ in range(WARMUP):
        operation()
    started = time.perf_counter()
    for _ in range(ITERATIONS):
        operation()
    seconds = time.perf_counter() - started
    return {
        "iterations": ITERATIONS,
        "ns_per_op": round(seconds * 1e9 / ITERATIONS, 1),
        "ops_per_sec": round(ITERATIONS / seconds, 1),
    }


def lg_linear():
    graph = LgStateGraph(dict)
    for index in range(10):
        name = f"n{index}"
        graph.add_node(name, lambda state: {"value": state.get("value", 0) + 1})
        graph.add_edge(START if index == 0 else f"n{index - 1}", name)
    graph.add_edge("n9", END)
    return graph.compile()


def rs_linear():
    graph = rs.StateGraph({"value": "last_value"})
    for index in range(10):
        graph.add_node(f"n{index}", lambda state: {"value": state["value"] + 1})
    graph.set_entry_point("n0")
    graph.add_edge("n9", rs.END)
    for index in range(9):
        graph.add_edge(f"n{index}", f"n{index + 1}")
    return graph.compile()


def lg_conditional():
    graph = LgStateGraph(dict)
    graph.add_node("start", lambda state: {"value": 1})
    graph.add_node("big", lambda state: {"value": state["value"] + 10})
    graph.add_node("small", lambda state: {"value": state["value"] + 1})
    graph.add_edge(START, "start")
    graph.add_conditional_edges(
        "start", lambda state: "big" if state["value"] >= 1 else "small"
    )
    graph.add_edge("big", END)
    graph.add_edge("small", END)
    return graph.compile()


def rs_conditional():
    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("start", lambda state: {"value": 1})
    graph.add_node("big", lambda state: {"value": state["value"] + 10})
    graph.add_node("small", lambda state: {"value": state["value"] + 1})
    graph.set_entry_point("start")
    graph.add_conditional_edges(
        "start", lambda state: "big" if state["value"] >= 1 else "small"
    )
    graph.add_edge("big", rs.END)
    graph.add_edge("small", rs.END)
    return graph.compile()


def lg_loop():
    graph = LgStateGraph(dict)
    graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    graph.add_edge(START, "inc")
    graph.add_conditional_edges(
        "inc", lambda state: "inc" if state["value"] < 5 else END
    )
    return graph.compile()


def rs_loop():
    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    graph.set_entry_point("inc")
    graph.add_conditional_edges(
        "inc", lambda state: "inc" if state["value"] < 5 else None
    )
    return graph.compile()


def lg_interrupt_graph():
    def ask(state):
        return {"value": lg_interrupt({"q": "value"})}

    graph = LgStateGraph(dict)
    graph.add_node("ask", ask)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    return graph.compile(checkpointer=LgInMemorySaver())


def rs_interrupt_graph():
    def ask(state):
        return {"value": rs.interrupt({"q": "value"})}

    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("ask", ask)
    graph.set_entry_point("ask")
    graph.set_finish_point("ask")
    return graph.compile(checkpointer=rs.InMemorySaver())


def lg_update_state():
    graph = LgStateGraph(dict)
    graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    graph.add_edge(START, "inc")
    graph.add_edge("inc", END)
    return graph.compile(checkpointer=LgInMemorySaver())


def rs_update_state():
    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    graph.set_entry_point("inc")
    graph.set_finish_point("inc")
    return graph.compile(checkpointer=rs.InMemorySaver())


def lg_subgraph():
    child = LgStateGraph(dict)
    child.add_node("inc", lambda state: {"value": state["value"] + 1})
    child.add_edge(START, "inc")
    child.add_edge("inc", END)
    parent = LgStateGraph(dict)
    parent.add_node("child", child.compile())
    parent.add_node("double", lambda state: {"value": state["value"] * 2})
    parent.add_edge(START, "child")
    parent.add_edge("child", "double")
    parent.add_edge("double", END)
    return parent.compile()


def rs_subgraph():
    child = rs.StateGraph({"value": "last_value"})
    child.add_node("inc", lambda state: {"value": state["value"] + 1})
    child.set_entry_point("inc")
    child.set_finish_point("inc")
    parent = rs.StateGraph({"value": "last_value"})
    parent.add_subgraph("child", child.compile())
    parent.add_node("double", lambda state: {"value": state["value"] * 2})
    parent.set_entry_point("child")
    parent.add_edge("child", "double")
    parent.set_finish_point("double")
    return parent.compile()


def run_linear_scenarios() -> list[dict]:
    results = []

    lg_graph, rs_graph = lg_linear(), rs_linear()
    lg_value = lg_graph.invoke({"value": 0})
    rs_value = rs_graph.invoke({"value": 0})
    results.append(
        {
            "scenario": "linear_10",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(lambda: rs_graph.invoke({"value": 0})),
                "lg": measure(lambda: lg_graph.invoke({"value": 0})),
            },
        }
    )

    lg_graph, rs_graph = lg_conditional(), rs_conditional()
    lg_value = lg_graph.invoke({})
    rs_value = rs_graph.invoke({})
    results.append(
        {
            "scenario": "conditional_branch",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(lambda: rs_graph.invoke({})),
                "lg": measure(lambda: lg_graph.invoke({})),
            },
        }
    )

    lg_graph, rs_graph = lg_loop(), rs_loop()
    lg_value = lg_graph.invoke({"value": 0})
    rs_value = rs_graph.invoke({"value": 0})
    results.append(
        {
            "scenario": "loop_until_5",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(lambda: rs_graph.invoke({"value": 0})),
                "lg": measure(lambda: lg_graph.invoke({"value": 0})),
            },
        }
    )

    lg_graph, rs_graph = lg_linear(), rs_linear()
    lg_values = lg_graph.batch([{"value": 0}, {"value": 10}])
    rs_values = rs_graph.batch([{"value": 0}, {"value": 10}])
    results.append(
        {
            "scenario": "batch_2_inputs",
            "rs": rs_values,
            "lg": lg_values,
            "equivalent": rs_values == lg_values,
            "bench": {
                "rs": measure(
                    lambda: rs_graph.batch([{"value": 0}, {"value": 10}])
                ),
                "lg": measure(
                    lambda: lg_graph.batch([{"value": 0}, {"value": 10}])
                ),
            },
        }
    )
    # Managed values
    lg_managed = lg_managed_graph()
    rs_managed = rs_managed_graph()
    lg_value = lg_managed.invoke({}, {"recursion_limit": 5})
    rs_value = rs_managed.invoke({})
    results.append(
        {
            "scenario": "managed_values",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
        }
    )

    # Streaming values (final streamed state)
    lg_graph, rs_graph = lg_linear(), rs_linear()
    lg_chunks = list(lg_graph.stream({"value": 0}, stream_mode="values"))
    rs_chunks = rs_graph.stream({"value": 0}, stream_mode="values")
    lg_value = lg_chunks[-1]
    rs_value = [
        chunk["values"]
        for chunk in rs_chunks
        if chunk.get("type") == "values"
    ][-1]
    results.append(
        {
            "scenario": "stream_values",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
        }
    )

    # Command(update=...) starts at entry points
    lg_graph, rs_graph = lg_command_graph(), rs_command_graph()
    lg_value = lg_graph.invoke(LgCommand(update={"value": 10}))
    rs_value = rs_graph.invoke(rs.Command(update={"value": 10}))
    results.append(
        {
            "scenario": "command_update",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
        }
    )

    # Retry policy
    lg_graph, lg_calls = lg_retry_graph()
    rs_graph, rs_calls = rs_retry_graph()
    lg_value = lg_graph.invoke({"value": 0})
    rs_value = rs_graph.invoke({"value": 0})
    results.append(
        {
            "scenario": "retry_policy",
            "rs": {"value": rs_value, "calls": rs_calls()},
            "lg": {"value": lg_value, "calls": lg_calls()},
            "equivalent": rs_value == lg_value and rs_calls() == lg_calls(),
        }
    )
    return results


def lg_command_graph():
    graph = LgStateGraph(LgCommandState)
    graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    graph.add_edge(START, "inc")
    graph.add_edge("inc", END)
    return graph.compile()


def rs_command_graph():
    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    graph.set_entry_point("inc")
    graph.set_finish_point("inc")
    return graph.compile()


def lg_managed_graph():
    graph = LgStateGraph(LgManagedState)
    graph.add_node(
        "inspect",
        lambda state: {
            "value": f"{state['remaining_steps']}/{state['is_last_step']}"
        },
    )
    graph.add_edge(START, "inspect")
    graph.add_edge("inspect", END)
    return graph.compile()


def rs_managed_graph():
    graph = rs.StateGraph({"value": "last_value"}, managed_values=True)
    graph.add_node(
        "inspect",
        lambda state: {
            "value": f"{state['remaining_steps']}/{state['is_last_step']}"
        },
    )
    graph.set_entry_point("inspect")
    graph.set_finish_point("inspect")
    return graph.compile(recursion_limit=5)


def lg_retry_graph():
    calls = {"count": 0}

    def flaky(state):
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("transient")
        return {"value": 7}

    graph = LgStateGraph(dict)
    graph.add_node(
        "flaky",
        flaky,
        retry_policy=LgRetryPolicy(
            max_attempts=3, initial_interval=0.001, backoff_factor=1.0,
            retry_on=Exception,
        ),
    )
    graph.add_edge(START, "flaky")
    graph.add_edge("flaky", END)
    return graph.compile(), lambda: calls["count"]


def rs_retry_graph():
    calls = {"count": 0}

    def flaky(state):
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("transient")
        return {"value": 7}

    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node(
        "flaky",
        flaky,
        retry_policy=rs.RetryPolicy(
            max_attempts=3, initial_interval=0.001, backoff_factor=1.0,
            retry_on=["transient"],
        ),
    )
    graph.set_entry_point("flaky")
    graph.set_finish_point("flaky")
    return graph.compile(), lambda: calls["count"]


def run_checkpoint_scenarios() -> list[dict]:
    results = []

    lg_graph, rs_graph = lg_interrupt_graph(), rs_interrupt_graph()
    lg_thread = {"configurable": {"thread_id": "parity-lg"}}
    lg_error = None
    try:
        lg_graph.invoke({"value": 0}, lg_thread)
    except Exception as exc:  # noqa: BLE001
        lg_error = type(exc).__name__
    rs_error = None
    try:
        rs_graph.invoke({"value": 0}, thread_id="parity-rs")
    except Exception as exc:  # noqa: BLE001
        rs_error = type(exc).__name__
    lg_value = lg_graph.invoke(LgCommand(resume=5), lg_thread)
    rs_value = rs_graph.resume("parity-rs", 5)
    results.append(
        {
            "scenario": "interrupt_resume",
            "rs": {"error": rs_error, "value": rs_value},
            "lg": {"error": lg_error, "value": lg_value},
            "equivalent": rs_value == lg_value,
        }
    )

    lg_graph, rs_graph = lg_update_state(), rs_update_state()
    lg_thread = {"configurable": {"thread_id": "parity-lg-update"}}
    lg_graph.invoke({"value": 0}, lg_thread)
    lg_state = lg_graph.update_state(lg_thread, {"value": 42})
    rs_graph.invoke({"value": 0}, thread_id="parity-rs-update")
    rs_state = rs_graph.update_state("parity-rs-update", {"value": 42})
    lg_values = dict(lg_graph.get_state(lg_thread).values)
    results.append(
        {
            "scenario": "update_state",
            "rs": rs_state["values"],
            "lg": lg_values,
            "equivalent": rs_state["values"] == lg_values,
            # LangGraph returns a config dict here; the binding returns the snapshot.
            "note": "language API shape differs (snapshot vs config); state matches",
        }
    )

    lg_graph, rs_graph = lg_update_state(), rs_update_state()
    lg_thread = {"configurable": {"thread_id": "parity-lg-history"}}
    lg_graph.invoke({"value": 0}, lg_thread)
    lg_graph.invoke({"value": 0}, lg_thread)
    rs_graph.invoke({"value": 0}, thread_id="parity-rs-history")
    rs_graph.invoke({"value": 0}, thread_id="parity-rs-history")
    lg_history = list(lg_graph.get_state_history(lg_thread))
    rs_history = rs_graph.get_state_history("parity-rs-history")
    results.append(
        {
            "scenario": "checkpoint_history",
            "rs": len(rs_history),
            "lg": len(lg_history),
            "equivalent": len(rs_history) == len(lg_history),
            # LangGraph writes an extra input checkpoint before the first superstep.
            "known_difference": "LangGraph records one extra input checkpoint per fresh run",
        }
    )
    return results


def run_subgraph_scenario() -> list[dict]:
    lg_graph, rs_graph = lg_subgraph(), rs_subgraph()
    lg_value = lg_graph.invoke({"value": 1})
    rs_value = rs_graph.invoke({"value": 1})
    return [
        {
            "scenario": "subgraph",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(lambda: rs_graph.invoke({"value": 1})),
                "lg": measure(lambda: lg_graph.invoke({"value": 1})),
            },
        }
    ]


def main() -> None:
    results = []
    for runner in (
        run_linear_scenarios,
        run_checkpoint_scenarios,
        run_subgraph_scenario,
    ):
        try:
            results.extend(runner())
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "scenario": runner.__name__,
                    "rs": None,
                    "lg": None,
                    "equivalent": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    failures = [
        item
        for item in results
        if not item["equivalent"] and "known_difference" not in item
    ]
    (HERE / "parity_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Behavior / performance parity",
        "",
        "Generated by `python benchmarks/parity.py`.",
        "",
        "| Scenario | Equivalent | rs ns/op | lg ns/op | rs speedup |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in results:
        bench = item.get("bench")
        status = "yes" if item["equivalent"] else ("known" if item.get("known_difference") else "no")
        if bench:
            rs_ns = bench["rs"]["ns_per_op"]
            lg_ns = bench["lg"]["ns_per_op"]
            speedup = f"{lg_ns / rs_ns:.2f}x" if rs_ns else "n/a"
            lines.append(
                f"| {item['scenario']} | {status} | "
                f"{rs_ns:.0f} | {lg_ns:.0f} | {speedup} |"
            )
        else:
            lines.append(
                f"| {item['scenario']} | {status} | - | - | - |"
            )
    (HERE / "PARITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    equivalent = sum(1 for item in results if item["equivalent"])
    known = sum(1 for item in results if not item["equivalent"] and "known_difference" in item)
    print(f"\n{equivalent}/{len(results)} scenarios equivalent ({known} known differences)")
    for failure in failures:
        print("MISMATCH:", json.dumps(failure, default=str))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()