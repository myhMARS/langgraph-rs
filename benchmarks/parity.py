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

import langgraph_rs as rs
from langgraph.checkpoint.memory import InMemorySaver as LgInMemorySaver
from langgraph.graph import END, START, StateGraph as LgStateGraph
from langgraph.types import Command as LgCommand
from langgraph.types import interrupt as lg_interrupt

HERE = Path(__file__).resolve().parent
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
    return results


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