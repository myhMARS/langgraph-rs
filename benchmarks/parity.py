"""Behavior + performance parity between Python LangGraph and langgraph-rs bindings.

Each scenario builds the same graph on both engines, compares observable behavior, and
records a small benchmark. Run from the repository root after building the extension:

    python benchmarks/parity.py

The script writes ``benchmarks/parity_results.json`` and ``benchmarks/PARITY.md``.
"""

from __future__ import annotations

import asyncio
import json
import operator
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Annotated, TypedDict

import langgraph_rs as rs
from langgraph.checkpoint.memory import InMemorySaver as LgInMemorySaver
from langgraph.graph import END, START, StateGraph as LgStateGraph
from langgraph.cache.memory import InMemoryCache as LgCache
from langgraph.config import get_store as lg_get_store
from langgraph.func import entrypoint as lg_entrypoint, task as lg_task
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.store.memory import InMemoryStore as LgInMemoryStore
from langgraph.types import CachePolicy as LgCachePolicy, TimeoutPolicy as LgTimeoutPolicy
from langgraph.types import Command as LgCommand, RetryPolicy as LgRetryPolicy, Send as LgSend
from langgraph.types import interrupt as lg_interrupt

HERE = Path(__file__).resolve().parent


class LgPendingState(TypedDict):
    log: Annotated[list, operator.add]


class LgDeferredState(TypedDict):
    log: Annotated[list, operator.add]


class LgSendState(TypedDict):
    log: Annotated[list, operator.add]
    value: int


class LgCommandState(TypedDict, total=False):
    value: int
    seeded: int


class LgManagedState(TypedDict, total=False):
    value: str
    remaining_steps: RemainingSteps
    is_last_step: IsLastStep
def run_big_state_scenario() -> list[dict]:
    """A large (256 KB) channel value must survive every superstep and every checkpoint.

    This covers the value-sharing paths (checkpoint `Arc`, persisted values, snapshot reads):
    the graph rewrites the blob on each step, and both the final state and a `get_state` read
    are compared against LangGraph.
    """
    results = []
    blob = "x" * (256 * 1024)

    class LgBigState(TypedDict):
        blob: str
        value: int

    def lg_graph():
        graph = LgStateGraph(LgBigState)
        for index in range(4):
            graph.add_node(f"n{index}", lambda state: {"blob": state["blob"], "value": state["value"] + 1})
            graph.add_edge(f"n{index}", f"n{index + 1}" if index < 3 else END)
        graph.add_edge(START, "n0")
        return graph.compile(LgInMemorySaver())

    def rs_graph():
        graph = rs.StateGraph({"blob": "last_value", "value": "last_value"})
        for index in range(4):
            graph.add_node(
                f"n{index}",
                lambda state: {"blob": state["blob"], "value": state["value"] + 1},
            )
            if index == 0:
                graph.add_edge(rs.START, f"n{index}")
            if index < 3:
                graph.add_edge(f"n{index}", f"n{index + 1}")
        graph.set_finish_point("n3")
        return graph.compile(checkpointer=rs.InMemorySaver())

    lg_compiled = lg_graph()
    rs_compiled = rs_graph()
    lg_thread = {"configurable": {"thread_id": "parity-lg-big"}}
    lg_value = lg_compiled.invoke({"blob": blob, "value": 0}, lg_thread)
    rs_value = rs_compiled.invoke({"blob": blob, "value": 0}, thread_id="parity-rs-big")
    lg_state = lg_compiled.get_state(lg_thread).values
    rs_state = rs_compiled.get_state("parity-rs-big").values
    results.append(
        {
            "scenario": "big_state",
            "rs": {"value": rs_value["value"], "blob_len": len(rs_value["blob"]), "state_len": len(rs_state["blob"])},
            "lg": {"value": lg_value["value"], "blob_len": len(lg_value["blob"]), "state_len": len(lg_state["blob"])},
            "equivalent": (
                rs_value["value"] == lg_value["value"]
                and rs_value["blob"] == lg_value["blob"]
                and rs_state["blob"] == lg_state["blob"]
            ),
            "bench": {
                "rs": measure(lambda: rs_compiled.invoke({"blob": blob, "value": 0}, thread_id="parity-rs-big-bench")),
                "lg": measure(lambda: lg_compiled.invoke({"blob": blob, "value": 0}, {"configurable": {"thread_id": "parity-lg-big-bench"}})),
            },
        }
    )
    return results


def run_reducer_scenario() -> list[dict]:
    """`Annotated[list, operator.add]` must behave the same on both engines.

    The RS schema uses the callable reducer directly, which is the path that now
    concatenates lists natively; this scenario keeps it honest against LangGraph.
    """
    results = []

    class LgReducedState(TypedDict):
        log: Annotated[list, operator.add]
        value: int

    def lg_graph():
        graph = LgStateGraph(LgReducedState)
        for index in range(6):
            graph.add_node(f"n{index}", lambda state: {"log": ["x"], "value": state["value"] + 1})
            graph.add_edge(f"n{index}", f"n{index + 1}" if index < 5 else END)
        graph.add_edge(START, "n0")
        return graph.compile()

    def rs_graph():
        graph = rs.StateGraph({"log": operator.add, "value": "last_value"})
        for index in range(6):
            graph.add_node(
                f"n{index}",
                lambda state: {"log": ["x"], "value": state.get("value", 0) + 1},
            )
            if index == 0:
                graph.add_edge(rs.START, f"n{index}")
            if index < 5:
                graph.add_edge(f"n{index}", f"n{index + 1}")
        graph.set_finish_point("n5")
        return graph.compile()

    lg_compiled = lg_graph()
    rs_compiled = rs_graph()
    lg_value = lg_compiled.invoke({"log": [], "value": 0})
    rs_value = rs_compiled.invoke({"log": [], "value": 0})
    results.append(
        {
            "scenario": "reducer_add",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(lambda: rs_compiled.invoke({"log": [], "value": 0})),
                "lg": measure(lambda: lg_compiled.invoke({"log": [], "value": 0})),
            },
        }
    )
    return results


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


def measure_async(operation) -> dict:
    async def run() -> float:
        for _ in range(WARMUP):
            await operation()
        started = time.perf_counter()
        for _ in range(ITERATIONS):
            await operation()
        return time.perf_counter() - started

    seconds = asyncio.run(run())
    return {
        "iterations": ITERATIONS,
        "ns_per_op": round(seconds * 1e9 / ITERATIONS, 1),
        "ops_per_sec": round(ITERATIONS / seconds, 1),
    }


def lg_async_linear():
    graph = LgStateGraph(dict)
    for index in range(5):
        name = f"n{index}"
        graph.add_node(name, lambda state: {"value": state.get("value", 0) + 1})
        graph.add_edge(START if index == 0 else f"n{index - 1}", name)
    graph.add_edge("n4", END)
    return graph.compile()


def rs_async_linear():
    graph = rs.AsyncStateGraph({"value": "last_value"})
    for index in range(5):
        graph.add_node(
            f"n{index}", lambda state: {"value": state["value"] + 1}
        )
    graph.set_entry_point("n0")
    graph.add_edge("n4", rs.END)
    for index in range(4):
        graph.add_edge(f"n{index}", f"n{index + 1}")
    return graph.compile()


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


def parallel_interrupt_pair() -> tuple[object, object]:
    """Two parallel nodes that both interrupt: LangGraph surfaces both and resumes by id."""

    class LgState(TypedDict, total=False):
        a: int
        b: int

    def lg_node_a(state):
        return {"a": lg_interrupt({"ask": "a"})}

    def lg_node_b(state):
        return {"b": lg_interrupt({"ask": "b"})}

    lg_graph = LgStateGraph(LgState)
    lg_graph.add_node("na", lg_node_a)
    lg_graph.add_node("nb", lg_node_b)
    lg_graph.add_edge(START, "na")
    lg_graph.add_edge(START, "nb")
    lg_graph.add_edge("na", END)
    lg_graph.add_edge("nb", END)
    lg = lg_graph.compile(checkpointer=LgInMemorySaver())

    def rs_node_a(state):
        return {"a": rs.interrupt({"ask": "a"})}

    def rs_node_b(state):
        return {"b": rs.interrupt({"ask": "b"})}

    rs_graph = rs.StateGraph({"a": "last", "b": "last"})
    rs_graph.add_node("na", rs_node_a)
    rs_graph.add_node("nb", rs_node_b)
    rs_graph.set_entry_point("na")
    rs_graph.set_entry_point("nb")
    rs_graph.set_finish_point("na")
    rs_graph.set_finish_point("nb")
    return lg, rs_graph.compile(checkpointer=rs.InMemorySaver())


def subgraph_updates_pair():
    """A parent node that runs a subgraph reports its own `updates` chunk, as LangGraph does."""

    class LgState(TypedDict, total=False):
        value: int

    lg_sub = LgStateGraph(LgState)
    lg_sub.add_node("inner", lambda state: {"value": (state.get("value") or 0) + 1})
    lg_sub.add_edge(START, "inner")
    lg_sub.add_edge("inner", END)
    lg_par = LgStateGraph(LgState)
    lg_par.add_node("first", lambda state: {"value": 5})
    lg_par.add_node("sub", lg_sub.compile())
    lg_par.add_node("after", lambda state: {"value": (state.get("value") or 0) + 10})
    lg_par.add_edge(START, "first")
    lg_par.add_edge("first", "sub")
    lg_par.add_edge("sub", "after")
    lg_par.add_edge("after", END)

    rs_sub = rs.StateGraph({"value": "last"})
    rs_sub.add_node("inner", lambda state: {"value": (state.get("value") or 0) + 1})
    rs_sub.set_entry_point("inner")
    rs_sub.set_finish_point("inner")
    rs_par = rs.StateGraph({"value": "last"})
    rs_par.add_node("first", lambda state: {"value": 5})
    rs_par.add_node("sub", rs_sub.compile())
    rs_par.add_node("after", lambda state: {"value": (state.get("value") or 0) + 10})
    rs_par.set_entry_point("first")
    rs_par.add_edge("first", "sub")
    rs_par.add_edge("sub", "after")
    rs_par.set_finish_point("after")
    return lg_par.compile(), rs_par.compile()


def task_triggers_pair():
    """Task triggers name the scheduling channel: `branch:to:<node>` and `__pregel_push`."""

    class LgState(TypedDict, total=False):
        value: int
        log: Annotated[list, operator.add]

    lg_linear = LgStateGraph(LgState)
    lg_linear.add_node("a", lambda state: {"value": 1})
    lg_linear.add_node("b", lambda state: {"value": 2})
    lg_linear.add_edge(START, "a")
    lg_linear.add_edge("a", "b")
    lg_linear.add_edge("b", END)

    rs_linear = rs.StateGraph({"value": "last", "log": "append"})
    rs_linear.add_node("a", lambda state: {"value": 1})
    rs_linear.add_node("b", lambda state: {"value": 2})
    rs_linear.set_entry_point("a")
    rs_linear.add_edge("a", "b")
    rs_linear.set_finish_point("b")

    lg_fan = LgStateGraph(LgState)
    lg_fan.add_node("work", lambda state: {"log": [state.get("value")]})
    lg_fan.add_conditional_edges(
        START, lambda state: [LgSend("work", {"value": 1}), LgSend("work", {"value": 2})]
    )
    lg_fan.add_edge("work", END)

    rs_fan = rs.StateGraph({"value": "last", "log": "append"})
    rs_fan.add_node("work", lambda state: {"log": [state.get("value")]})
    rs_fan.add_conditional_edges(
        START, lambda state: [rs.Send("work", {"value": 1}), rs.Send("work", {"value": 2})]
    )
    rs_fan.set_finish_point("work")
    return lg_linear.compile(), rs_linear.compile(), lg_fan.compile(), rs_fan.compile()


def run_graph_export_scenario() -> list[dict]:
    """`get_graph()` exposes LangGraph's node/edge structure (including conditional edges)."""

    class LgState(TypedDict, total=False):
        value: int

    def lg_build():
        graph = LgStateGraph(LgState)
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": 2})
        graph.add_edge(START, "a")
        graph.add_conditional_edges("a", lambda state: "b", {"b": "b"})
        graph.add_edge("b", END)
        return graph.compile()

    def rs_build():
        graph = rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": 2})
        graph.set_entry_point("a")
        graph.add_conditional_edges("a", lambda state: "b", {"b": "b"})
        graph.set_finish_point("b")
        return graph.compile()

    def describe(app):
        graph = app.get_graph()
        return {
            "type": f"{type(graph).__module__}.{type(graph).__name__}",
            "nodes": sorted(graph.nodes),
            "edges": sorted(
                (edge.source, edge.target, bool(edge.conditional)) for edge in graph.edges
            ),
        }

    lg_view = describe(lg_build())
    rs_view = describe(rs_build())
    return [
        {
            "scenario": "graph_export",
            "rs": rs_view,
            "lg": lg_view,
            "equivalent": rs_view == lg_view,
            "bench": {
                "rs": measure(lambda: rs_build().get_graph()),
                "lg": measure(lambda: lg_build().get_graph()),
            },
        }
    ]


def run_error_handler_scenario() -> list[dict]:
    """A callable `error_handler` runs `__error_handler__<node>` and ends that branch."""

    class LgState(TypedDict, total=False):
        value: int
        note: str

    def failing(state):
        raise ValueError("nope")

    def lg_build():
        graph = LgStateGraph(LgState)
        graph.add_node("boom", failing, error_handler=lambda state: {"note": f"handled:{state}"})
        graph.add_node("after", lambda state: {"value": 99})
        graph.add_edge(START, "boom")
        graph.add_edge("boom", "after")
        graph.add_edge("after", END)
        return graph.compile()

    def rs_build():
        graph = rs.StateGraph({"value": "last_value", "note": "last_value"})
        graph.add_node(
            "boom", failing, error_handler=lambda state: {"note": f"handled:{state}"}
        )
        graph.add_node("after", lambda state: {"value": 99})
        graph.set_entry_point("boom")
        graph.add_edge("boom", "after")
        graph.set_finish_point("after")
        return graph.compile()

    lg_result = lg_build().invoke({"value": 0})
    rs_result = rs_build().invoke({"value": 0})
    lg_names = sorted(lg_build().get_graph().nodes)
    rs_names = sorted(node["id"] for node in rs_build().get_graph().to_dict()["nodes"])
    return [
        {
            "scenario": "error_handler",
            "rs": {"result": rs_result, "nodes": rs_names},
            "lg": {"result": lg_result, "nodes": lg_names},
            "equivalent": rs_result == lg_result and rs_names == lg_names,
            "bench": {
                "rs": measure(lambda: rs_build().invoke({"value": 0})),
                "lg": measure(lambda: lg_build().invoke({"value": 0})),
            },
        }
    ]


def run_debug_stream_scenario() -> list[dict]:
    """`stream_mode="debug"` mirrors LangGraph: same entry kinds, steps and payload keys."""

    class LgState(TypedDict, total=False):
        value: int

    def lg_build(checkpointer):
        graph = LgStateGraph(LgState)
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": 2})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        saver = LgInMemorySaver() if checkpointer else None
        return graph.compile(checkpointer=saver)

    def rs_build(checkpointer):
        graph = rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": 2})
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")
        saver = rs.InMemorySaver() if checkpointer else None
        return graph.compile(checkpointer=saver)

    def describe(app, label, checkpointer):
        call = {"configurable": {"thread_id": f"debug-{label}"}} if checkpointer else {}
        out = []
        for part in app.stream({"value": 0}, call, stream_mode="debug"):
            payload = part.get("payload") or {}
            out.append((part["type"], part["step"], tuple(sorted(payload.keys()))))
        return out

    comparisons = {}
    for checkpointer in (False, True):
        key = "with_checkpointer" if checkpointer else "plain"
        comparisons[key] = {
            "rs": describe(rs_build(checkpointer), "rs", checkpointer),
            "lg": describe(lg_build(checkpointer), "lg", checkpointer),
        }
    return [
        {
            "scenario": "debug_stream",
            "rs": {name: item["rs"] for name, item in comparisons.items()},
            "lg": {name: item["lg"] for name, item in comparisons.items()},
            "equivalent": all(item["rs"] == item["lg"] for item in comparisons.values()),
            "bench": {
                "rs": measure(lambda: list(rs_build(False).stream({"value": 0}, stream_mode="debug"))),
                "lg": measure(lambda: list(lg_build(False).stream({"value": 0}, stream_mode="debug"))),
            },
        }
    ]


def run_durability_pause_scenario() -> list[dict]:
    """`durability="exit"` keeps one checkpoint at a pause and one after the resume."""

    class LgState(TypedDict, total=False):
        value: int
        log: list

    def lg_build():
        graph = LgStateGraph(LgState)
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("gate", lambda state: {"log": [lg_interrupt("ok?")]})
        graph.add_edge(START, "a")
        graph.add_edge("a", "gate")
        graph.add_edge("gate", END)
        return graph.compile(checkpointer=LgInMemorySaver())

    def rs_build():
        graph = rs.StateGraph({"value": "last_value", "log": "append"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("gate", lambda state: {"log": [rs.interrupt("ok?")]})
        graph.set_entry_point("a")
        graph.add_edge("a", "gate")
        graph.set_finish_point("gate")
        return graph.compile(checkpointer=rs.InMemorySaver())

    def history(app, config):
        return [(snap.metadata.get("source"), snap.metadata.get("step"), list(snap.next))
                for snap in app.get_state_history(config)]

    results = {}
    for label, build in (("rs", rs_build), ("lg", lg_build)):
        app = build()
        config = {"configurable": {"thread_id": f"durability-{label}"}}
        payload = {"value": 0, "log": []}
        app.invoke(payload, config, durability="exit")
        paused = history(app, config)
        app.invoke(LgCommand(resume="yes") if label == "lg" else rs.Command(resume="yes"),
                   config, durability="exit")
        results[label] = {
            "paused": paused,
            "resumed": history(app, config),
            "final": dict(app.get_state(config).values),
        }
    return [
        {
            "scenario": "durability_pause",
            "rs": results["rs"],
            "lg": results["lg"],
            "equivalent": results["rs"] == results["lg"],
            "bench": {
                "rs": measure(
                    lambda: rs_build().invoke(
                        {"value": 0, "log": []},
                        {"configurable": {"thread_id": "durability-bench-rs"}},
                    )
                ),
                "lg": measure(
                    lambda: lg_build().invoke(
                        {"value": 0, "log": []},
                        {"configurable": {"thread_id": "durability-bench-lg"}},
                    )
                ),
            },
        }
    ]


def run_message_stream_scenario() -> list[dict]:
    """`stream_mode="messages"` yields a message chunk plus node/step/trigger metadata."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage as LcAIMessage

    class LgState(TypedDict, total=False):
        messages: list

    def lg_build():
        model = GenericFakeChatModel(messages=iter([LcAIMessage("second")]))
        graph = LgStateGraph(LgState)
        graph.add_node("a", lambda state: {"messages": []})
        graph.add_node("b", lambda state: {"messages": [model.invoke(state["messages"])]})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        return graph.compile()

    def rs_build():
        model = GenericFakeChatModel(messages=iter([LcAIMessage("second")]))
        graph = rs.StateGraph({"messages": "append"})
        graph.add_node("a", lambda state: {"messages": []})
        graph.add_node("b", lambda state: {"messages": [model.invoke(state["messages"])]})
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")
        return graph.compile()

    keys = ("langgraph_node", "langgraph_step", "langgraph_triggers", "langgraph_path")

    def describe(chunk, metadata):
        content = getattr(chunk, "content", chunk)
        return {
            "type": type(chunk).__name__,
            "content": content,
            "meta": {key: tuple(metadata[key]) if isinstance(metadata.get(key), (list, tuple))
                     else metadata.get(key) for key in keys},
        }

    lg_chunk, lg_meta = next(iter(lg_build().stream({"messages": []}, stream_mode="messages")))
    rs_chunk, rs_meta = next(iter(rs_build().stream({"messages": []}, stream_mode="messages")))
    lg_view = describe(lg_chunk, lg_meta)
    rs_view = describe(rs_chunk, rs_meta)
    return [
        {
            "scenario": "message_stream",
            "rs": rs_view,
            "lg": lg_view,
            "equivalent": rs_view == lg_view,
            "bench": {
                "rs": measure(
                    lambda: list(rs_build().stream({"messages": []}, stream_mode="messages"))
                ),
                "lg": measure(
                    lambda: list(lg_build().stream({"messages": []}, stream_mode="messages"))
                ),
            },
        }
    ]


def run_bulk_update_scenario() -> list[dict]:
    """`bulk_update_state` writes one checkpoint per superstep and unions each superstep's next."""

    class LgState(TypedDict, total=False):
        value: int
        note: str

    def lg_build():
        graph = LgStateGraph(LgState)
        graph.add_node("a", lambda state: {"value": (state.get("value") or 0) + 1})
        graph.add_node("b", lambda state: {"value": (state.get("value") or 0) + 10})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        return graph.compile(checkpointer=LgInMemorySaver())

    def rs_build():
        graph = rs.StateGraph({"value": "last", "note": "last"})
        graph.add_node("a", lambda state: {"value": (state.get("value") or 0) + 1})
        graph.add_node("b", lambda state: {"value": (state.get("value") or 0) + 10})
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")
        return graph.compile(checkpointer=rs.InMemorySaver())

    supersteps = [
        [({"value": 100}, "a", None)],
        [({"note": "n1"}, "a", None), ({"value": 200}, "b", None)],
    ]
    lg, rs_graph = lg_build(), rs_build()
    lg_cfg = {"configurable": {"thread_id": "bulk-lg"}}
    rs_cfg = {"configurable": {"thread_id": "bulk-rs"}}
    lg.invoke({"value": 0}, lg_cfg)
    rs_graph.invoke({"value": 0}, rs_cfg)
    before_lg = len(list(lg.get_state_history(lg_cfg)))
    before_rs = len(list(rs_graph.get_state_history(rs_cfg)))
    lg.bulk_update_state(lg_cfg, supersteps)
    rs_graph.bulk_update_state(rs_cfg, supersteps)
    lg_after = len(list(lg.get_state_history(lg_cfg))) - before_lg
    rs_after = len(list(rs_graph.get_state_history(rs_cfg))) - before_rs
    lg_state = dict(lg.get_state(lg_cfg).values)
    rs_state = dict(rs_graph.get_state(rs_cfg).values)
    lg_next = tuple(lg.get_state(lg_cfg).next)
    rs_next = tuple(rs_graph.get_state(rs_cfg).next)
    return [
        {
            "scenario": "bulk_update_state",
            "rs": {"added": rs_after, "values": rs_state, "next": rs_next},
            "lg": {"added": lg_after, "values": lg_state, "next": lg_next},
            "equivalent": (rs_after, rs_state, rs_next) == (lg_after, lg_state, lg_next),
            "bench": {
                "rs": measure(
                    lambda: rs_build().invoke(
                        {"value": 0}, {"configurable": {"thread_id": "bulk-bench-rs"}}
                    )
                ),
                "lg": measure(
                    lambda: lg_build().invoke(
                        {"value": 0}, {"configurable": {"thread_id": "bulk-bench-lg"}}
                    )
                ),
            },
        }
    ]


def run_task_triggers_scenario() -> list[dict]:
    lg_linear, rs_linear, lg_fan, rs_fan = task_triggers_pair()

    def starts(app, payload):
        out = []
        for part in app.stream(payload, stream_mode="tasks", version="v2"):
            data = part["data"] if isinstance(part, dict) else part
            if data.get("result") is None:
                out.append((data["name"], tuple(data.get("triggers") or ())))
        return sorted(out)

    lg_linear_tasks = starts(lg_linear, {"value": 0})
    rs_linear_tasks = starts(rs_linear, {"value": 0, "log": []})
    lg_fan_tasks = starts(lg_fan, {"value": 0, "log": []})
    rs_fan_tasks = starts(rs_fan, {"value": 0, "log": []})
    return [
        {
            "scenario": "task_triggers",
            "rs": {"linear": rs_linear_tasks, "fan": rs_fan_tasks},
            "lg": {"linear": lg_linear_tasks, "fan": lg_fan_tasks},
            "equivalent": rs_linear_tasks == lg_linear_tasks and rs_fan_tasks == lg_fan_tasks,
            "bench": {
                "rs": measure(lambda: rs_linear.invoke({"value": 0, "log": []})),
                "lg": measure(lambda: lg_linear.invoke({"value": 0, "log": []})),
            },
        }
    ]


def run_subgraph_updates_scenario() -> list[dict]:
    lg, rs_graph = subgraph_updates_pair()
    lg_flat = [
        part["data"]
        for part in lg.stream({"value": 0}, stream_mode="updates", version="v2")
    ]
    rs_flat = [
        part["data"]
        for part in rs_graph.stream({"value": 0}, stream_mode="updates", version="v2")
    ]
    lg_ns = [
        (part["ns"], part["data"])
        for part in lg.stream({"value": 0}, stream_mode="updates", subgraphs=True, version="v2")
    ]
    rs_ns = [
        (part["ns"], part["data"])
        for part in rs_graph.stream(
            {"value": 0}, stream_mode="updates", subgraphs=True, version="v2"
        )
    ]
    # Namespace ids are opaque (LangGraph uses uuids), so compare the names only.
    def names(parts):
        return [
            (tuple(item.split(":")[0] for item in ns), data) for ns, data in parts
        ]

    return [
        {
            "scenario": "subgraph_updates",
            "rs": {"flat": rs_flat, "ns": names(rs_ns)},
            "lg": {"flat": lg_flat, "ns": names(lg_ns)},
            "equivalent": rs_flat == lg_flat and names(rs_ns) == names(lg_ns),
            "bench": {
                "rs": measure(lambda: rs_graph.invoke({"value": 0})),
                "lg": measure(lambda: lg.invoke({"value": 0})),
            },
        }
    ]


def run_parallel_interrupt_scenario() -> list[dict]:
    lg, rs_graph = parallel_interrupt_pair()
    lg_cfg = {"configurable": {"thread_id": "par-lg"}}
    rs_cfg = {"configurable": {"thread_id": "par-rs"}}
    lg_out = lg.invoke({}, lg_cfg)
    rs_out = rs_graph.invoke({}, rs_cfg)
    lg_payloads = sorted(interrupt.value["ask"] for interrupt in lg_out["__interrupt__"])
    rs_payloads = sorted(interrupt.value["ask"] for interrupt in rs_out["__interrupt__"])
    lg_resume = lg.invoke(
        LgCommand(resume={item.id: f"v{item.value['ask']}" for item in lg_out["__interrupt__"]}),
        lg_cfg,
    )
    rs_resume = rs_graph.invoke(
        rs.Command(resume={item.id: f"v{item.value['ask']}" for item in rs_out["__interrupt__"]}),
        rs_cfg,
    )
    lg_state = {"a": lg_resume.get("a"), "b": lg_resume.get("b")}
    rs_state = {"a": rs_resume.get("a"), "b": rs_resume.get("b")}
    return [
        {
            "scenario": "parallel_interrupts",
            "rs": {"payloads": rs_payloads, "resumed": rs_state},
            "lg": {"payloads": lg_payloads, "resumed": lg_state},
            "equivalent": rs_payloads == lg_payloads and rs_state == lg_state,
            "bench": {
                "rs": measure(
                    lambda: rs_graph.invoke({}, {"configurable": {"thread_id": "par-bench-rs"}})
                ),
                "lg": measure(
                    lambda: lg.invoke({}, {"configurable": {"thread_id": "par-bench-lg"}})
                ),
            },
        }
    ]


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
            "bench": {
                "rs": measure(lambda: rs_managed.invoke({})),
                "lg": measure(lambda: lg_managed.invoke({}, {"recursion_limit": 5})),
            },
        }
    )

    # Streaming values (final streamed state)
    lg_graph, rs_graph = lg_linear(), rs_linear()
    lg_chunks = list(lg_graph.stream({"value": 0}, stream_mode="values"))
    rs_chunks = rs_graph.stream({"value": 0}, stream_mode="values")
    lg_value = lg_chunks[-1]
    # A single requested mode yields raw payloads; multi-mode yields (mode, payload) parts.
    def _unpack(chunk):
        if isinstance(chunk, tuple):
            return chunk[-1]
        if isinstance(chunk, dict) and "type" in chunk:
            return chunk["data"]
        return chunk

    rs_value = [_unpack(chunk) for chunk in rs_chunks][-1]
    results.append(
        {
            "scenario": "stream_values",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(
                    lambda: rs_graph.stream({"value": 0}, stream_mode="values")
                ),
                "lg": measure(
                    lambda: list(lg_graph.stream({"value": 0}, stream_mode="values"))
                ),
            },
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
            "bench": {
                "rs": measure(
                    lambda: rs_graph.invoke(rs.Command(update={"value": 10}))
                ),
                "lg": measure(
                    lambda: lg_graph.invoke(LgCommand(update={"value": 10}))
                ),
            },
        }
    )

    # Command(goto=...) starts at the target and applies its update.
    lg_graph, rs_graph = lg_command_goto_graph(), rs_command_goto_graph()
    lg_value = lg_graph.invoke(
        LgCommand(goto="target")
    )
    rs_value = rs_graph.invoke(
        rs.Command(goto="target")
    )
    results.append(
        {
            "scenario": "command_goto",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(
                    lambda: rs_graph.invoke(
                        rs.Command(goto="target")
                    )
                ),
                "lg": measure(
                    lambda: lg_graph.invoke(
                        LgCommand(goto="target")
                    )
                ),
            },
        }
    )

    # Retry policy
    lg_graph, lg_calls, lg_counter = lg_retry_graph()
    rs_graph, rs_calls, rs_counter = rs_retry_graph()
    lg_value = lg_graph.invoke({"value": 0})
    rs_value = rs_graph.invoke({"value": 0})
    results.append(
        {
            "scenario": "retry_policy",
            "rs": {"value": rs_value, "calls": rs_calls()},
            "lg": {"value": lg_value, "calls": lg_calls()},
            "equivalent": rs_value == lg_value and rs_calls() == lg_calls(),
            "bench": {
                # The node counter is shared across calls, so reset it before each measured
                # run; otherwise only the first call retries and the bench measures a
                # straight-line invoke.
                "rs": measure(lambda: (rs_counter.__setitem__("count", 0), rs_graph.invoke({"value": 0}))[1]),
                "lg": measure(lambda: (lg_counter.__setitem__("count", 0), lg_graph.invoke({"value": 0}))[1]),
            },
        }
    )

    # Error handler routes a failed node to a fallback.
    lg_graph = lg_error_handler_graph()
    rs_graph = rs_error_handler_graph()
    lg_value = lg_graph.invoke({"value": 0})
    rs_value = rs_graph.invoke({"value": 0})
    results.append(
        {
            "scenario": "error_handler",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(lambda: rs_graph.invoke({"value": 0})),
                "lg": measure(lambda: lg_graph.invoke({"value": 0})),
            },
        }
    )

    # Unhandled node errors surface as failures on both engines.
    lg_failed, lg_message = raise_check(lg_failing_graph(), is_lg=True)
    rs_failed, rs_message = raise_check(rs_failing_graph(), is_lg=False)
    results.append(
        {
            "scenario": "error_propagation",
            "rs": {"failed": rs_failed, "contains_boom": "boom" in rs_message},
            "lg": {"failed": lg_failed, "contains_boom": "boom" in lg_message},
            "equivalent": (
                lg_failed
                == rs_failed
                and ("boom" in lg_message) == ("boom" in rs_message)
            ),
            "bench": {
                "rs": measure(lambda: raise_check(rs_failing_graph(), False)),
                "lg": measure(lambda: raise_check(lg_failing_graph(), True)),
            },
        }
    )

    # Node timeout: LangGraph async cancellation vs the binding's sync worker timeout.
    lg_failed, lg_message = run_lg_timeout()
    rs_failed, rs_message = run_rs_timeout()
    results.append(
        {
            "scenario": "node_timeout",
            "rs": {"failed": rs_failed, "contains_timeout": timeout_marker(rs_message)},
            "lg": {"failed": lg_failed, "contains_timeout": timeout_marker(lg_message)},
            "equivalent": (
                lg_failed == rs_failed
                and timeout_marker(lg_message) == timeout_marker(rs_message)
            ),
            "note": "both engines cancel the async node with a timeout error",
            "bench": {
                "rs": measure_async(run_rs_timeout_async),
                "lg": measure_async(run_lg_timeout_async),
            },
        }
    )

    # Node cache: second identical invocation reuses the cached output.
    lg_graph, lg_cache_calls = lg_cache_graph()
    rs_graph, rs_cache_calls = rs_cache_graph()
    lg_first = lg_graph.invoke({"value": 0})
    lg_second = lg_graph.invoke({"value": 0})
    rs_first = rs_graph.invoke({"value": 0})
    rs_second = rs_graph.invoke({"value": 0})
    results.append(
        {
            "scenario": "node_cache",
            "rs": {"value": rs_second, "calls": rs_cache_calls()},
            "lg": {"value": lg_second, "calls": lg_cache_calls()},
            "equivalent": (
                rs_first == lg_first
                and rs_second == lg_second
                and rs_cache_calls() == lg_cache_calls()
            ),
            "bench": {
                "rs": measure(lambda: rs_graph.invoke({"value": 0})),
                "lg": measure(lambda: lg_graph.invoke({"value": 0})),
            },
        }
    )

    # SQLite-backed node cache survives across compiled graphs sharing the file.
    lg_value, rs_value = sqlite_cache_values()
    results.append(
        {
            "scenario": "cache_sqlite",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(rs_sqlite_cache_invoke),
                "lg": measure(lg_sqlite_cache_invoke),
            },
        }
    )

    # v2 stream: compare the final values part (the binding also emits its usage
    # extension chunk, which LangGraph has no equivalent for).
    lg_graph, rs_graph = lg_linear(), rs_linear()
    lg_parts = list(
        lg_graph.stream({"value": 0}, stream_mode=["values"], version="v2")
    )
    rs_parts = rs_graph.stream(
        {"value": 0}, stream_mode=["values"], version="v2"
    )
    lg_value = [part for part in lg_parts if part["type"] == "values"][-1]["data"]
    rs_value = [part for part in rs_parts if part["type"] == "values"][-1]["data"]
    results.append(
        {
            "scenario": "stream_v2_values",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "note": "binding also emits a usage extension chunk",
            "bench": {
                "rs": measure(
                    lambda: rs_graph.stream(
                        {"value": 0}, stream_mode=["values"], version="v2"
                    )
                ),
                "lg": measure(
                    lambda: list(
                        lg_graph.stream(
                            {"value": 0}, stream_mode=["values"], version="v2"
                        )
                    )
                ),
            },
        }
    )

    # LangChain message/token stream. Both engines inject the callback context around node
    # calls, so a plain `model.invoke(...)` reaches the `messages` stream without extra setup.
    lg_tokens = lg_message_tokens()
    rs_tokens = rs_message_tokens()
    results.append(
        {
            "scenario": "messages_stream",
            "rs": rs_tokens,
            "lg": lg_tokens,
            "equivalent": rs_tokens == lg_tokens,
            "bench": {
                "rs": measure(rs_message_tokens),
                "lg": measure(lg_message_tokens),
            },
        }
    )

    # Graph export shape (nodes and static edges, excluding START/END sentinels).
    lg_value = lg_graph_export()
    rs_value = rs_graph_export()
    results.append(
        {
            "scenario": "graph_export",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(rs_graph_export),
                "lg": measure(lg_graph_export),
            },
        }
    )

    # Time travel: replay from an earlier checkpoint.
    lg_value = lg_time_travel()
    rs_value = rs_time_travel()
    results.append(
        {
            "scenario": "time_travel",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(rs_time_travel),
                "lg": measure(lg_time_travel),
            },
        }
    )

    # Send(timeout=...) on async workers.
    lg_failed, lg_message = run_lg_send_timeout()
    rs_failed, rs_message = run_rs_send_timeout()
    results.append(
        {
            "scenario": "send_timeout",
            "rs": {"failed": rs_failed, "contains_timeout": timeout_marker(rs_message)},
            "lg": {"failed": lg_failed, "contains_timeout": timeout_marker(lg_message)},
            "equivalent": (
                lg_failed == rs_failed
                and timeout_marker(lg_message) == timeout_marker(rs_message)
            ),
            "bench": {
                "rs": measure_async(run_rs_send_timeout_async),
                "lg": measure_async(run_lg_send_timeout_async),
            },
        }
    )

    # Two sequential interrupts in one node, resumed with two values.
    lg_value = lg_multi_interrupt()
    rs_value = rs_multi_interrupt()
    results.append(
        {
            "scenario": "multi_interrupt",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(rs_multi_interrupt),
                "lg": measure(lg_multi_interrupt),
            },
        }
    )

    # Snapshot metadata: source and next tasks (step numbering differs: LangGraph
    # counts input checkpoints across the thread, the binding resets per run).
    lg_meta = lg_snapshot_metadata()
    rs_meta = rs_snapshot_metadata()
    results.append(
        {
            "scenario": "snapshot_metadata",
            "rs": rs_meta,
            "lg": lg_meta,
            "equivalent": rs_meta == lg_meta,
            "note": "source/next/step aligned",
            "bench": {
                "rs": measure(rs_snapshot_metadata),
                "lg": measure(lg_snapshot_metadata),
            },
        }
    )

    # Pending writes: a successful parallel branch survives an interrupt in another.
    lg_value = lg_pending_writes()
    rs_value = rs_pending_writes()
    results.append(
        {
            "scenario": "pending_writes",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(rs_pending_writes),
                "lg": measure(lg_pending_writes),
            },
        }
    )

    # Send fan-out.
    lg_graph = lg_send_graph()
    rs_graph = rs_send_graph()
    lg_value = lg_graph.invoke({"log": [], "value": 0})
    rs_value = rs_graph.invoke({"log": [], "value": 0})
    results.append(
        {
            "scenario": "send_fanout",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(lambda: rs_graph.invoke({"log": [], "value": 0})),
                "lg": measure(lambda: lg_graph.invoke({"log": [], "value": 0})),
            },
        }
    )

    # Deferred node runs after regular nodes.
    lg_graph = lg_deferred_graph()
    rs_graph = rs_deferred_graph()
    lg_value = lg_graph.invoke({"log": []})
    rs_value = rs_graph.invoke({"log": []})
    results.append(
        {
            "scenario": "deferred_node",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(lambda: rs_graph.invoke({"log": []})),
                "lg": measure(lambda: lg_graph.invoke({"log": []})),
            },
        }
    )

    # Functional API (@entrypoint / @task).
    lg_graph = lg_functional_graph()
    rs_graph = rs_functional_graph()
    lg_value = lg_graph.invoke({"value": 1})
    rs_value = rs_graph.invoke({"value": 1})
    results.append(
        {
            "scenario": "functional_api",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(lambda: rs_graph.invoke({"value": 1})),
                "lg": measure(lambda: lg_graph.invoke({"value": 1})),
            },
        }
    )

    # In-memory store round trip
    lg_value = lg_store_graph().invoke({})
    rs_value = rs_store_graph().invoke({})
    results.append(
        {
            "scenario": "store_roundtrip",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(lambda: rs_store_graph().invoke({})),
                "lg": measure(lambda: lg_store_graph().invoke({})),
            },
        }
    )

    # update_state addressed to a specific pending task.
    lg_value = lg_update_state_task()
    rs_value = rs_update_state_task()
    results.append(
        {
            "scenario": "update_state_task",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(rs_update_state_task),
                "lg": measure(lg_update_state_task),
            },
        }
    )

    # SQLite-backed long-term store round trip.
    lg_store_value = lg_sqlite_store_invoke()
    rs_store_value = rs_sqlite_store_invoke()
    results.append(
        {
            "scenario": "store_sqlite",
            "rs": rs_store_value,
            "lg": lg_store_value,
            "equivalent": rs_store_value == lg_store_value,
            "bench": {
                "rs": measure(rs_sqlite_store_invoke),
                "lg": measure(lg_sqlite_store_invoke),
            },
        }
    )

    # SQLite persistence round trip
    lg_value, rs_value, persistence_bench = persistence_values()
    results.append(
        {
            "scenario": "sqlite_persistence",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": persistence_bench,
        }
    )

    # Async execution
    lg_async = lg_async_linear()
    rs_async = rs_async_linear()
    lg_value = asyncio.run(lg_async.ainvoke({"value": 0}))
    async def run_rs_async():
        return await rs_async.ainvoke({"value": 0})

    rs_value = asyncio.run(run_rs_async())
    results.append(
        {
            "scenario": "async_linear_5",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure_async(
                    lambda: rs_async.ainvoke({"value": 0})
                ),
                "lg": measure_async(
                    lambda: lg_async.ainvoke({"value": 0})
                ),
            },
        }
    )
    return results


def lg_message_tokens():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    model = FakeListChatModel(responses=["hello world"])

    def node(state):
        model.invoke("hi")
        return {"text": "done"}

    graph = LgStateGraph(dict)
    graph.add_node("model", node)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    chunks = graph.compile().stream({"text": ""}, stream_mode="messages")
    return "".join(chunk[0].content for chunk in chunks if chunk[0].content)


def rs_message_tokens():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    model = FakeListChatModel(responses=["hello world"])

    def node(state):
        model.invoke("hi")
        return {"text": "done"}

    graph = rs.StateGraph({"text": "last_value"})
    graph.add_node("model", node)
    graph.set_entry_point("model")
    graph.set_finish_point("model")
    chunks = graph.compile().stream({"text": ""}, stream_mode="messages")
    # v1 message chunks are ``(message, metadata)`` tuples just like LangGraph's; the message
    # itself is a dict here (the binding exposes plain JSON objects).
    parts = []
    for chunk in chunks:
        if not isinstance(chunk, tuple):
            continue
        message = chunk[0]
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if content:
            parts.append(content)
    return "".join(parts)


def _normalize_export(nodes, edges):
    ignored = {"__start__", "__end__"}
    return {
        "nodes": sorted(node for node in nodes if node not in ignored),
        "edges": sorted(
            (source, target)
            for source, target in edges
            if source not in ignored and target not in ignored
        ),
    }


def lg_graph_export():
    graph = LgStateGraph(dict)
    graph.add_node("a", lambda state: {})
    graph.add_node("b", lambda state: {})
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    view = graph.compile().get_graph()
    return _normalize_export(
        view.nodes.keys(), [(edge.source, edge.target) for edge in view.edges]
    )


def rs_graph_export():
    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("a", lambda state: {})
    graph.add_node("b", lambda state: {})
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.set_finish_point("b")
    view = graph.compile().get_graph().to_dict()
    return _normalize_export(
        [node["id"] for node in view["nodes"]],
        [(edge["source"], edge["target"]) for edge in view["edges"]],
    )


def run_lg_send_timeout():
    return asyncio.run(run_lg_send_timeout_async())


async def run_lg_send_timeout_async():
    async def worker(state):
        await asyncio.sleep(0.2)
        return {"log": [1]}

    def route(state):
        return [LgSend("worker", {"value": 1}, timeout=0.05)]

    graph = LgStateGraph(dict)
    graph.add_node("dispatch", lambda state: {})
    graph.add_node("worker", worker)
    graph.add_conditional_edges("dispatch", route)
    graph.add_edge(START, "dispatch")
    graph.set_finish_point("worker")
    compiled = graph.compile()
    try:
        await compiled.ainvoke({"log": []})
        return False, ""
    except Exception as exc:  # noqa: BLE001
        return True, str(exc)


def run_rs_send_timeout():
    return asyncio.run(run_rs_send_timeout_async())


async def run_rs_send_timeout_async():
    async def worker(state):
        await asyncio.sleep(0.2)
        return {"log": [1]}

    def route(state):
        return [rs.Send("worker", {"value": 1}, timeout=0.05)]

    graph = rs.AsyncStateGraph({"log": "append"})
    graph.add_node("dispatch", lambda state: {})
    graph.add_node("worker", worker)
    graph.add_conditional_edges("dispatch", route)
    graph.set_entry_point("dispatch")
    graph.set_finish_point("worker")
    compiled = graph.compile()
    try:
        await compiled.ainvoke({"log": []})
        return False, ""
    except Exception as exc:  # noqa: BLE001
        return True, str(exc)


_multi_counter = [0]


def lg_multi_interrupt():
    _multi_counter[0] += 1
    thread = f"parity-lg-multi-{_multi_counter[0]}"

    def ask(state):
        first = lg_interrupt("first")
        second = lg_interrupt("second")
        return {"value": first + second}

    graph = LgStateGraph(dict)
    graph.add_node("ask", ask)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    compiled = graph.compile(checkpointer=LgInMemorySaver())
    config = {"configurable": {"thread_id": thread}}
    try:
        compiled.invoke({"value": 0}, config)
    except Exception:  # noqa: BLE001
        pass
    try:
        compiled.invoke(LgCommand(resume=5), config)
    except Exception:  # noqa: BLE001
        pass
    return compiled.invoke(LgCommand(resume=7), config)


def rs_multi_interrupt():
    _multi_counter[0] += 1
    thread = f"parity-rs-multi-{_multi_counter[0]}"

    def ask(state):
        first = rs.interrupt("first")
        second = rs.interrupt("second")
        return {"value": first + second}

    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("ask", ask)
    graph.set_entry_point("ask")
    graph.set_finish_point("ask")
    compiled = graph.compile(checkpointer=rs.InMemorySaver())
    try:
        compiled.invoke({"value": 0}, thread_id=thread)
    except Exception:  # noqa: BLE001
        pass
    try:
        compiled.resume(thread, 5)
    except Exception:  # noqa: BLE001
        pass
    return compiled.resume(thread, 7)


def lg_snapshot_metadata():
    graph = LgStateGraph(dict)
    graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    graph.add_edge(START, "inc")
    graph.add_edge("inc", END)
    compiled = graph.compile(checkpointer=LgInMemorySaver())
    config = {"configurable": {"thread_id": "parity-lg-meta"}}
    compiled.invoke({"value": 0}, config)
    compiled.invoke({"value": 10}, config)
    snapshot = compiled.get_state(config)
    return {
        "source": snapshot.metadata["source"],
        "next": list(snapshot.next),
        "step": snapshot.metadata["step"],
    }


def rs_snapshot_metadata():
    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    graph.set_entry_point("inc")
    graph.set_finish_point("inc")
    compiled = graph.compile(checkpointer=rs.InMemorySaver())
    compiled.invoke({"value": 0}, thread_id="parity-rs-meta")
    compiled.invoke({"value": 10}, thread_id="parity-rs-meta")
    snapshot = compiled.get_state("parity-rs-meta")
    return {
        "source": snapshot["metadata"]["source"],
        "next": list(snapshot["next"]),
        "step": snapshot["metadata"]["step"],
    }


def lg_pending_writes():
    saver = LgInMemorySaver()

    def branch_a(state):
        return {"log": ["a"]}

    def branch_b(state):
        return {"log": ["b", lg_interrupt("value")]}

    graph = LgStateGraph(LgPendingState)
    graph.add_node("a", branch_a)
    graph.add_node("b", branch_b)
    graph.add_edge(START, "a")
    graph.add_edge(START, "b")
    graph.add_edge("a", END)
    graph.add_edge("b", END)
    compiled = graph.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "parity-lg-pending"}}
    try:
        compiled.invoke({"log": []}, config)
    except Exception:  # noqa: BLE001
        pass
    return compiled.invoke(LgCommand(resume=5), config)


def rs_pending_writes():
    def branch_a(state):
        return {"log": ["a"]}

    def branch_b(state):
        return {"log": ["b", rs.interrupt("value")]}

    graph = rs.StateGraph({"log": "append"})
    graph.add_node("a", branch_a)
    graph.add_node("b", branch_b)
    graph.add_edge(rs.START, "a")
    graph.add_edge(rs.START, "b")
    graph.add_edge("a", rs.END)
    graph.add_edge("b", rs.END)
    compiled = graph.compile(checkpointer=rs.InMemorySaver())
    try:
        compiled.invoke({"log": []}, thread_id="parity-rs-pending")
    except Exception:  # noqa: BLE001
        pass
    return compiled.resume("parity-rs-pending", 5)


def lg_time_travel():
    graph = LgStateGraph(dict)
    graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    graph.add_edge(START, "inc")
    graph.add_edge("inc", END)
    saver = LgInMemorySaver()
    compiled = graph.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "parity-lg-time-travel"}}
    compiled.invoke({"value": 0}, config)
    compiled.invoke({"value": 10}, config)
    target = next(
        snapshot
        for snapshot in compiled.get_state_history(config)
        if snapshot.values.get("value") == 10 and snapshot.next
    )
    return compiled.invoke(None, target.config)


def rs_time_travel():
    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    graph.set_entry_point("inc")
    graph.set_finish_point("inc")
    compiled = graph.compile(checkpointer=rs.InMemorySaver())
    compiled.invoke({"value": 0}, thread_id="parity-rs-time-travel")
    compiled.invoke({"value": 10}, thread_id="parity-rs-time-travel")
    target = next(
        snapshot
        for snapshot in compiled.get_state_history("parity-rs-time-travel")
        if snapshot["values"].get("value") == 10 and snapshot["next"]
    )
    return compiled.resume_from(
        "parity-rs-time-travel", target["checkpoint_id"], None
    )


def lg_send_graph():
    graph = LgStateGraph(LgSendState)
    graph.add_node("dispatch", lambda state: {})
    graph.add_node("worker", lambda state: {"log": [state["value"]]})
    graph.add_conditional_edges(
        "dispatch",
        lambda state: [
            LgSend("worker", {"value": value}) for value in (1, 2, 3)
        ],
    )
    graph.add_edge(START, "dispatch")
    graph.set_finish_point("worker")
    return graph.compile()


def rs_send_graph():
    graph = rs.StateGraph({"log": "append", "value": "last_value"})
    graph.add_node("dispatch", lambda state: {})
    graph.add_node("worker", lambda state: {"log": [state["value"]]})
    graph.add_conditional_edges(
        "dispatch",
        lambda state: [
            rs.Send("worker", {"value": value}) for value in (1, 2, 3)
        ],
    )
    graph.set_entry_point("dispatch")
    graph.set_finish_point("worker")
    return graph.compile()


def lg_deferred_graph():
    graph = LgStateGraph(LgDeferredState)
    graph.add_node("regular", lambda state: {"log": ["regular"]})
    graph.add_node(
        "deferred", lambda state: {"log": ["deferred"]}, defer=True
    )
    graph.add_edge(START, "regular")
    graph.add_edge(START, "deferred")
    graph.add_edge("regular", END)
    graph.add_edge("deferred", END)
    return graph.compile()


def rs_deferred_graph():
    graph = rs.StateGraph({"log": "append"})
    graph.add_node("regular", lambda state: {"log": ["regular"]})
    graph.add_node(
        "deferred", lambda state: {"log": ["deferred"]}, deferred=True
    )
    graph.set_entry_point("regular")
    graph.add_edge("regular", "deferred")
    return graph.compile()


def lg_functional_graph():
    @lg_task
    def add(x):
        return x + 1

    @lg_entrypoint()
    def workflow(state):
        return {"value": add(state["value"]).result()}

    return workflow


def rs_functional_graph():
    add = rs.task(lambda x: x + 1, name="add")
    return rs.entrypoint(
        lambda state: {"value": add(state["value"])}
    ).compile()


def lg_update_state_task():
    def ask(state):
        lg_interrupt({"question": "value?"})
        return {}

    def finish(state):
        return {}

    graph = LgStateGraph(dict)
    graph.add_node("ask", ask)
    graph.add_node("finish", finish)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", "finish")
    graph.add_edge("finish", END)
    compiled = graph.compile(checkpointer=LgInMemorySaver())
    config = {"configurable": {"thread_id": "parity-lg-task-update"}}
    try:
        compiled.invoke({}, config)
    except Exception:  # noqa: BLE001
        pass
    task = compiled.get_state(config).tasks[0]
    compiled.update_state(config, {"value": 5}, task_id=task.id)
    snapshot = compiled.get_state(config)
    return {"value": snapshot.values.get("value"), "next": list(snapshot.next)}


def rs_update_state_task():
    def ask(state):
        rs.interrupt({"question": "value?"})
        return {}

    def finish(state):
        return {}

    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("ask", ask)
    graph.add_node("finish", finish)
    graph.set_entry_point("ask")
    graph.add_edge("ask", "finish")
    graph.set_finish_point("finish")
    compiled = graph.compile(checkpointer=rs.InMemorySaver())
    try:
        compiled.invoke({}, thread_id="parity-rs-task-update")
    except Exception:  # noqa: BLE001
        pass
    task = compiled.get_state("parity-rs-task-update")["tasks"][0]
    compiled.update_state(
        "parity-rs-task-update", {"value": 5}, task_id=task["id"]
    )
    snapshot = compiled.get_state("parity-rs-task-update")
    return {"value": snapshot["values"].get("value"), "next": list(snapshot["next"])}


def lg_sqlite_store_invoke():
    import sqlite3

    from langgraph.store.sqlite import SqliteStore

    directory = Path(tempfile.mkdtemp())
    connection = sqlite3.connect(
        directory / "store.sqlite", check_same_thread=False, isolation_level=None
    )
    store = SqliteStore(connection)
    store.setup()

    def write(state):
        store = lg_get_store()
        store.put(("parity",), "key", {"value": 7})
        return {"value": store.get(("parity",), "key").value["value"]}

    graph = LgStateGraph(dict)
    graph.add_node("write", write)
    graph.add_edge(START, "write")
    graph.add_edge("write", END)
    return graph.compile(store=store).invoke({})


def rs_sqlite_store_invoke():
    directory = Path(tempfile.mkdtemp())
    store = rs.SqliteStore(str(directory / "store.sqlite"))

    def write(state):
        store = rs.get_store()
        store.put(("parity",), "key", {"value": 7})
        return {"value": store.get(("parity",), "key").value["value"]}

    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("write", write)
    graph.set_entry_point("write")
    graph.set_finish_point("write")
    return graph.compile(store=store).invoke({})


def lg_store_graph():
    graph = LgStateGraph(dict)
    store = LgInMemoryStore()

    def write(state):
        lg_get_store().put(("parity",), "key", {"value": 7})
        item = lg_get_store().get(("parity",), "key")
        return {"value": item.value["value"]}

    graph.add_node("write", write)
    graph.add_edge(START, "write")
    graph.add_edge("write", END)
    return graph.compile(store=store)


def rs_store_graph():
    graph = rs.StateGraph({"value": "last_value"})
    store = rs.InMemoryStore()

    def write(state):
        rs.get_store().put(("parity",), "key", {"value": 7})
        item = rs.get_store().get(("parity",), "key")
        return {"value": item.value["value"]}

    graph.add_node("write", write)
    graph.set_entry_point("write")
    graph.set_finish_point("write")
    return graph.compile(store=store)


def persistence_values():
    directory = Path(tempfile.mkdtemp())
    lg_path = directory / "lg.sqlite"
    rs_path = directory / "rs.sqlite"

    lg_graph = LgStateGraph(dict)
    lg_graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    lg_graph.add_edge(START, "inc")
    lg_graph.add_edge("inc", END)
    lg_conn = sqlite3.connect(lg_path, check_same_thread=False)
    lg_compiled = lg_graph.compile(
        checkpointer=_lg_sqlite_saver(lg_conn)
    )
    lg_config = {"configurable": {"thread_id": "parity-persist"}}
    lg_compiled.invoke({"value": 0}, lg_config)
    lg_value = lg_compiled.get_state(lg_config).values

    rs_graph = rs.StateGraph({"value": "last_value"})
    rs_graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    rs_graph.set_entry_point("inc")
    rs_graph.set_finish_point("inc")
    rs_compiled = rs_graph.compile(checkpointer=rs.SqliteSaver(str(rs_path)))
    rs_compiled.invoke({"value": 0}, thread_id="parity-persist")
    rs_value = rs_compiled.get_state("parity-persist")["values"]

    bench = {
        "rs": measure(
            lambda: rs_compiled.invoke(
                {"value": 0}, thread_id="parity-persist-bench"
            )
        ),
        "lg": measure(
            lambda: lg_compiled.invoke(
                {"value": 0},
                {"configurable": {"thread_id": "parity-persist-bench"}},
            )
        ),
    }
    lg_conn.close()
    return dict(lg_value), rs_value, bench


_interrupt_counter = [0]


def interrupt_bench(graph, prefix, use_lg):
    _interrupt_counter[0] += 1
    thread = f"{prefix}-interrupt-bench-{_interrupt_counter[0]}"
    if use_lg:
        config = {"configurable": {"thread_id": thread}}
        try:
            graph.invoke({"value": 0}, config)
        except Exception:  # noqa: BLE001
            pass
        return graph.invoke(LgCommand(resume=5), config)
    try:
        graph.invoke({"value": 0}, thread_id=thread)
    except Exception:  # noqa: BLE001
        pass
    return graph.resume(thread, 5)


def _lg_sqlite_saver(connection):
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver(connection)


def _cache_sqlite_pair():
    directory = Path(tempfile.mkdtemp())
    return directory / "lg-cache.sqlite", directory / "rs-cache.sqlite"


def lg_sqlite_cache_invoke():
    from langgraph.cache.sqlite import SqliteCache

    directory = Path(tempfile.mkdtemp())
    cache = SqliteCache(path=str(directory / "cache.sqlite"))
    calls = {"count": 0}

    def work(state):
        calls["count"] += 1
        return {"value": state["value"] + 1}

    graph = LgStateGraph(dict)
    graph.add_node("work", work, cache_policy=LgCachePolicy(ttl=60))
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    compiled = graph.compile(cache=cache)
    compiled.invoke({"value": 0})
    return compiled.invoke({"value": 0})


def rs_sqlite_cache_invoke():
    directory = Path(tempfile.mkdtemp())
    cache = rs.SqliteCache(str(directory / "cache.sqlite"))
    calls = {"count": 0}

    def work(state):
        calls["count"] += 1
        return {"value": state["value"] + 1}

    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("work", work, cache_policy=rs.CachePolicy(ttl=60))
    graph.set_entry_point("work")
    graph.set_finish_point("work")
    compiled = graph.compile(cache=cache)
    compiled.invoke({"value": 0})
    return compiled.invoke({"value": 0})


def sqlite_cache_values():
    lg_value = lg_sqlite_cache_invoke()
    rs_value = rs_sqlite_cache_invoke()
    return lg_value, rs_value


def lg_command_graph():
    graph = LgStateGraph(LgCommandState)
    graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    graph.add_edge(START, "inc")
    graph.add_edge("inc", END)
    return graph.compile()


def lg_command_goto_graph():
    graph = LgStateGraph(LgCommandState)
    graph.add_node("seed", lambda state: {"seeded": 1})
    graph.add_node("target", lambda state: {"value": 99})
    graph.add_edge(START, "seed")
    graph.add_edge("seed", "target")
    graph.add_edge("target", END)
    return graph.compile()


def rs_command_goto_graph():
    graph = rs.StateGraph({"value": "last_value", "seeded": "last_value"})
    graph.add_node("seed", lambda state: {"seeded": 1})
    graph.add_node("target", lambda state: {"value": 99})
    graph.set_entry_point("seed")
    graph.add_edge("seed", "target")
    graph.set_finish_point("target")
    return graph.compile()


def rs_command_graph():
    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    graph.set_entry_point("inc")
    graph.set_finish_point("inc")
    return graph.compile()


def lg_error_handler_graph():
    graph = LgStateGraph(dict)
    graph.add_node(
        "work",
        lambda state: (_ for _ in ()).throw(RuntimeError("boom")),
        error_handler=lambda state: {"value": -1},
    )
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    return graph.compile()


def rs_error_handler_graph():
    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node(
        "work",
        lambda state: (_ for _ in ()).throw(RuntimeError("boom")),
        error_handler="fallback",
    )
    graph.add_node("fallback", lambda state: {"value": -1})
    graph.set_entry_point("work")
    graph.set_finish_point("fallback")
    return graph.compile()


def lg_failing_graph():
    graph = LgStateGraph(dict)
    graph.add_node(
        "boom", lambda state: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    graph.add_edge(START, "boom")
    graph.add_edge("boom", END)
    return graph.compile()


def rs_failing_graph():
    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node(
        "boom", lambda state: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    graph.set_entry_point("boom")
    graph.set_finish_point("boom")
    return graph.compile()


def raise_check(graph, is_lg):
    try:
        if is_lg:
            graph.invoke({"value": 0})
        else:
            graph.invoke({"value": 0})
        return False, ""
    except Exception as exc:  # noqa: BLE001
        return True, str(exc)


def timeout_marker(message):
    lowered = message.lower()
    return "timeout" in lowered or "timed out" in lowered


def run_lg_timeout():
    return asyncio.run(run_lg_timeout_async())


async def run_lg_timeout_async():
    async def slow(state):
        await asyncio.sleep(0.2)
        return {"value": 1}

    graph = LgStateGraph(dict)
    graph.add_node(
        "slow", slow, timeout=LgTimeoutPolicy(run_timeout=0.02)
    )
    graph.add_edge(START, "slow")
    graph.add_edge("slow", END)
    compiled = graph.compile()
    try:
        await compiled.ainvoke({"value": 0})
        return False, ""
    except Exception as exc:  # noqa: BLE001
        return True, str(exc)


def run_rs_timeout():
    return asyncio.run(run_rs_timeout_async())


async def run_rs_timeout_async():
    async def slow(state):
        await asyncio.sleep(0.2)
        return {"value": 1}

    graph = rs.AsyncStateGraph({"value": "last_value"})
    graph.add_node(
        "slow", slow, timeout=rs.TimeoutPolicy(run_timeout=0.02)
    )
    graph.set_entry_point("slow")
    graph.set_finish_point("slow")
    try:
        await graph.compile().ainvoke({"value": 0})
        return False, ""
    except Exception as exc:  # noqa: BLE001
        return True, str(exc)


def lg_cache_graph():
    calls = {"count": 0}

    def work(state):
        calls["count"] += 1
        return {"value": state["value"] + 1}

    graph = LgStateGraph(dict)
    graph.add_node("work", work, cache_policy=LgCachePolicy(ttl=60))
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    return graph.compile(cache=LgCache()), lambda: calls["count"]


def rs_cache_graph():
    calls = {"count": 0}

    def work(state):
        calls["count"] += 1
        return {"value": state["value"] + 1}

    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("work", work, cache_policy=rs.CachePolicy(ttl=60))
    graph.set_entry_point("work")
    graph.set_finish_point("work")
    return graph.compile(cache=rs.InMemoryCache()), lambda: calls["count"]


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
    return graph.compile(), lambda: calls["count"], calls


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
    return graph.compile(), lambda: calls["count"], calls


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
            "bench": {
                "rs": measure(lambda: interrupt_bench(rs_graph, "rs", False)),
                "lg": measure(lambda: interrupt_bench(lg_graph, "lg", True)),
            },
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
            "bench": {
                "rs": measure(
                    lambda: (
                        rs_graph.invoke(
                            {"value": 0}, thread_id="parity-rs-update-bench"
                        ),
                        rs_graph.update_state(
                            "parity-rs-update-bench", {"value": 42}
                        ),
                    )
                ),
                "lg": measure(
                    lambda: (
                        lg_graph.invoke(
                            {"value": 0},
                            {"configurable": {"thread_id": "parity-lg-update-bench"}},
                        ),
                        lg_graph.update_state(
                            {"configurable": {"thread_id": "parity-lg-update-bench"}},
                            {"value": 42},
                        ),
                    )
                ),
            },
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
            "note": "counts match after aligning input-checkpoint creation",
            "bench": {
                "rs": measure(
                    lambda: (
                        rs_graph.invoke(
                            {"value": 0}, thread_id="parity-rs-history-bench"
                        ),
                        rs_graph.get_state_history("parity-rs-history-bench"),
                    )
                ),
                "lg": measure(
                    lambda: (
                        lg_graph.invoke(
                            {"value": 0},
                            {"configurable": {"thread_id": "parity-lg-history-bench"}},
                        ),
                        list(
                            lg_graph.get_state_history(
                                {"configurable": {"thread_id": "parity-lg-history-bench"}}
                            )
                        ),
                    )
                ),
            },
        }
    )
    return results


def lg_subgraph_interrupt_graph():
    saver = LgInMemorySaver()
    child = LgStateGraph(dict)
    child.add_node(
        "ask",
        lambda state: {"value": state.get("value", 0) + lg_interrupt("delta")},
    )
    child.add_edge(START, "ask")
    child.add_edge("ask", END)
    parent = LgStateGraph(dict)
    parent.add_node("child", child.compile(checkpointer=saver))
    parent.add_edge(START, "child")
    parent.add_edge("child", END)
    return parent.compile(checkpointer=saver)


def rs_subgraph_interrupt_graph():
    saver = rs.InMemorySaver()
    child = rs.StateGraph({"value": "last_value"})
    child.add_node(
        "ask",
        lambda state: {"value": state.get("value", 0) + rs.interrupt("delta")},
    )
    child.set_entry_point("ask")
    child.set_finish_point("ask")
    child_compiled = child.compile(checkpointer=saver)
    parent = rs.StateGraph({"value": "last_value"})
    parent.add_subgraph("child", child_compiled)
    parent.set_entry_point("child")
    parent.set_finish_point("child")
    return parent.compile(checkpointer=saver)


def run_command_parent_scenario() -> list[dict]:
    lg_child = LgStateGraph(LgCommandState)
    lg_child.add_node(
        "jump",
        lambda state: LgCommand(
            update={"value": 100},
            goto="parent_after",
            graph=LgCommand.PARENT,
        ),
    )
    lg_child.add_edge(START, "jump")
    lg_parent = LgStateGraph(LgCommandState)
    lg_parent.add_node("child", lg_child.compile())
    lg_parent.add_node(
        "parent_after", lambda state: {"value": state["value"] + 1}
    )
    lg_parent.add_edge(START, "child")
    lg_parent.add_edge("child", "parent_after")
    lg_parent.add_edge("parent_after", END)
    lg_compiled = lg_parent.compile()

    rs_child = rs.StateGraph({"value": "last_value"})
    rs_child.add_node(
        "jump",
        lambda state: rs.Command(
            update={"value": 100},
            goto="parent_after",
            graph=rs.Command.PARENT,
        ),
    )
    rs_child.set_entry_point("jump")
    rs_parent = rs.StateGraph({"value": "last_value"})
    rs_parent.add_subgraph("child", rs_child.compile())
    rs_parent.add_node(
        "parent_after", lambda state: {"value": state["value"] + 1}
    )
    rs_parent.set_entry_point("child")
    rs_parent.add_edge("child", "parent_after")
    rs_parent.set_finish_point("parent_after")
    rs_compiled = rs_parent.compile()

    lg_value = lg_compiled.invoke({"value": 0})
    rs_value = rs_compiled.invoke({"value": 0})
    return [
        {
            "scenario": "command_parent",
            "rs": rs_value,
            "lg": lg_value,
            "equivalent": rs_value == lg_value,
            "bench": {
                "rs": measure(lambda: rs_compiled.invoke({"value": 0})),
                "lg": measure(lambda: lg_compiled.invoke({"value": 0})),
            },
        }
    ]


def run_subgraph_interrupt_scenario() -> list[dict]:
    lg_graph = lg_subgraph_interrupt_graph()
    rs_graph = rs_subgraph_interrupt_graph()
    lg_thread = {"configurable": {"thread_id": "parity-lg-subgraph-int"}}
    try:
        lg_graph.invoke({"value": 0}, lg_thread)
        lg_error = None
    except Exception as exc:  # noqa: BLE001
        lg_error = type(exc).__name__
    try:
        rs_graph.invoke({"value": 0}, thread_id="parity-rs-subgraph-int")
        rs_error = None
    except Exception as exc:  # noqa: BLE001
        rs_error = type(exc).__name__
    lg_value = lg_graph.invoke(LgCommand(resume=5), lg_thread)
    rs_value = rs_graph.resume("parity-rs-subgraph-int", 5)
    return [
        {
            "scenario": "subgraph_interrupt",
            "rs": {"error": rs_error, "value": rs_value},
            "lg": {"error": lg_error, "value": lg_value},
            "equivalent": lg_value == rs_value,
            "bench": {
                "rs": measure(
                    lambda: interrupt_bench(rs_graph, "rs-subgraph", False)
                ),
                "lg": measure(
                    lambda: interrupt_bench(lg_graph, "lg-subgraph", True)
                ),
            },
        }
    ]


def run_remote_store_scenario() -> list[dict]:
    """Redis/Postgres store parity, enabled through the same PARITY_* env vars."""
    results = []
    postgres_url = os.environ.get("PARITY_POSTGRES_URL")
    if postgres_url:
        try:
            from langgraph.store.postgres import PostgresStore

            with PostgresStore.from_conn_string(postgres_url) as lg_store:
                lg_store.setup()
                lg_value = lg_store_invoke(lg_store, True)
                lg_bench = measure(lambda: lg_store_invoke(lg_store, True))
            rs_store = rs.PostgresStore(postgres_url)
            rs_value = lg_store_invoke(rs_store, False)
            results.append(
                {
                    "scenario": "store_postgres",
                    "rs": rs_value,
                    "lg": lg_value,
                    "equivalent": rs_value == lg_value,
                    "bench": {
                        "rs": measure(lambda: lg_store_invoke(rs_store, False)),
                        "lg": lg_bench,
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "scenario": "store_postgres",
                    "rs": None,
                    "lg": None,
                    "equivalent": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    redis_url = os.environ.get("PARITY_REDIS_URL")
    if redis_url:
        try:
            from langgraph.store.redis import RedisStore

            with RedisStore.from_conn_string(redis_url) as lg_store:
                lg_store.setup()
                lg_value = lg_store_invoke(lg_store, True)
            rs_store = rs.RedisStore(redis_url)
            rs_value = lg_store_invoke(rs_store, False)
            results.append(
                {
                    "scenario": "store_redis",
                    "rs": rs_value,
                    "lg": lg_value,
                    "equivalent": rs_value == lg_value,
                    "bench": {
                        "rs": measure(lambda: lg_store_invoke(rs_store, False)),
                        "lg": measure(lambda: lg_store_invoke(lg_store, True)),
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "scenario": "store_redis",
                    "rs": None,
                    "lg": None,
                    "equivalent": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


def lg_store_invoke(store, use_lg):
    if use_lg:
        def write(state):
            current = lg_get_store()
            current.put(("parity",), "key", {"value": 9})
            return {"value": current.get(("parity",), "key").value["value"]}

        graph = LgStateGraph(dict)
    else:
        def write(state):
            current = rs.get_store()
            current.put(("parity",), "key", {"value": 9})
            return {"value": current.get(("parity",), "key")["value"]}

        graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("write", write)
    if use_lg:
        graph.add_edge(START, "write")
        graph.add_edge("write", END)
    else:
        graph.set_entry_point("write")
        graph.set_finish_point("write")
    return graph.compile(store=store).invoke({})


def run_remote_persistence_scenario() -> list[dict]:
    """Redis/Postgres parity, enabled through PARITY_POSTGRES_URL / PARITY_REDIS_URL."""
    results = []
    postgres_url = os.environ.get("PARITY_POSTGRES_URL")
    if postgres_url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            with PostgresSaver.from_conn_string(postgres_url) as lg_saver:
                lg_saver.setup()
                lg_compiled = lg_linear_graph_small().compile(
                    checkpointer=lg_saver
                )
                lg_compiled.invoke(
                    {"value": 0},
                    {"configurable": {"thread_id": "parity-lg-pg"}},
                )
                lg_value = lg_compiled.get_state(
                    {"configurable": {"thread_id": "parity-lg-pg"}}
                ).values
                lg_bench = measure(
                    lambda: lg_compiled.invoke(
                        {"value": 0},
                        {"configurable": {"thread_id": "parity-lg-pg-bench"}},
                    )
                )
            rs_saver = rs.PostgresSaver(postgres_url)
            rs_compiled = rs_linear_graph_small().compile(checkpointer=rs_saver)
            rs_compiled.invoke({"value": 0}, thread_id="parity-rs-pg")
            rs_value = rs_compiled.get_state("parity-rs-pg")["values"]
            rs_bench = measure(
                lambda: rs_compiled.invoke(
                    {"value": 0}, thread_id="parity-rs-pg-bench"
                )
            )
            results.append(
                {
                    "scenario": "postgres_persistence",
                    "rs": rs_value,
                    "lg": dict(lg_value),
                    "equivalent": rs_value == dict(lg_value),
                    "bench": {"rs": rs_bench, "lg": lg_bench},
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "scenario": "postgres_persistence",
                    "rs": None,
                    "lg": None,
                    "equivalent": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    redis_url = os.environ.get("PARITY_REDIS_URL")
    if redis_url:
        try:
            from langgraph.checkpoint.redis import RedisSaver

            lg_saver = RedisSaver(redis_url=redis_url)
            lg_saver.setup()
            lg_compiled = lg_linear_graph_small().compile(checkpointer=lg_saver)
            lg_compiled.invoke(
                {"value": 0},
                {"configurable": {"thread_id": "parity-lg-redis"}},
            )
            lg_value = lg_compiled.get_state(
                {"configurable": {"thread_id": "parity-lg-redis"}}
            ).values
            lg_bench = measure(
                lambda: lg_compiled.invoke(
                    {"value": 0},
                    {"configurable": {"thread_id": "parity-lg-redis-bench"}},
                )
            )
            rs_saver = rs.RedisSaver(redis_url)
            rs_compiled = rs_linear_graph_small().compile(checkpointer=rs_saver)
            rs_compiled.invoke({"value": 0}, thread_id="parity-rs-redis")
            rs_value = rs_compiled.get_state("parity-rs-redis")["values"]
            rs_bench = measure(
                lambda: rs_compiled.invoke(
                    {"value": 0}, thread_id="parity-rs-redis-bench"
                )
            )
            results.append(
                {
                    "scenario": "redis_persistence",
                    "rs": rs_value,
                    "lg": dict(lg_value),
                    "equivalent": rs_value == dict(lg_value),
                    "bench": {"rs": rs_bench, "lg": lg_bench},
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "scenario": "redis_persistence",
                    "rs": None,
                    "lg": None,
                    "equivalent": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


def lg_linear_graph_small():
    graph = LgStateGraph(dict)
    graph.add_node("inc", lambda state: {"value": state.get("value", 0) + 1})
    graph.add_edge(START, "inc")
    graph.add_edge("inc", END)
    return graph


def rs_linear_graph_small():
    graph = rs.StateGraph({"value": "last_value"})
    graph.add_node("inc", lambda state: {"value": state["value"] + 1})
    graph.set_entry_point("inc")
    graph.set_finish_point("inc")
    return graph


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
        run_parallel_interrupt_scenario,
        run_subgraph_updates_scenario,
        run_task_triggers_scenario,
        run_bulk_update_scenario,
        run_message_stream_scenario,
        run_durability_pause_scenario,
        run_debug_stream_scenario,
        run_error_handler_scenario,
        run_graph_export_scenario,
        run_reducer_scenario,
        run_big_state_scenario,
        run_checkpoint_scenarios,
        run_subgraph_scenario,
        run_subgraph_interrupt_scenario,
        run_command_parent_scenario,
        run_remote_persistence_scenario,
        run_remote_store_scenario,
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