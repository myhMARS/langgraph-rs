"""Mega workflow built with the *Python LangGraph* reference implementation.

The topology, node bodies and phase sequence match ``build_rs.py`` (the Rust engine
through its PyO3 binding) and ``../mega_workflow_native.rs`` (pure Rust), so the three
timings measure the same amount of orchestration work.

Topology (13 nodes + one child subgraph node):

    START -> seed -> (count < 3 ? grow : fanout)
    grow -> (count < 3 ? grow : fanout)          # loop
    fanout -> Send(worker, n=1|2|3)              # fan-out
    worker -> join -> cached -> retry -> boom    # retry + error handler
    boom -(error)-> handled
    handled -> defer -> sub -> ask -> done       # deferred, subgraph, interrupt
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.store.memory import InMemoryStore
from langgraph.types import CachePolicy, Command, RetryPolicy, Send, interrupt

# Module-level counters keep node bodies identical to the Rust/PyO3 versions.
CALLS = {"retry": 0, "boom": 0, "cached": 0, "ask": 0, "child": 0}


class MegaState(TypedDict, total=False):
    count: int
    sink: Annotated[int, operator.add]
    log: Annotated[list[str], operator.add]
    flag: str
    child: int
    remaining_steps: RemainingSteps


async def _seed(state: MegaState) -> dict:
    store = get_store()
    previous = store.get(("mega",), "runs")
    runs = 0 if previous is None else previous.value["n"]
    store.put(("mega",), "runs", {"n": runs + 1})
    return {"log": ["seed"], "count": 0, "sink": 0, "child": 0, "flag": f"runs={runs + 1}"}


async def _grow(state: MegaState) -> dict:
    return {"count": state.get("count", 0) + 1, "log": ["grow"]}


def _fanout_sends():
    return [Send("worker", {"n": 1}), Send("worker", {"n": 2}), Send("worker", {"n": 3})]


def _route_after_grow(state: MegaState):
    """Loop while the counter is low, then fan out to three `Send` workers."""
    if state.get("count", 0) < 3:
        return "grow"
    return _fanout_sends()


async def _worker(state: MegaState) -> dict:
    n = state.get("n", 1)
    return {"log": ["worker"], "sink": n}


async def _join(state: MegaState) -> dict:
    return {"log": ["join"]}


async def _cached(state: MegaState) -> dict:
    CALLS["cached"] += 1
    return {"log": ["cached"]}


async def _retry(state: MegaState) -> dict:
    CALLS["retry"] += 1
    if CALLS["retry"] % 2 == 1:
        raise ValueError("transient failure")
    return {"log": ["retry"]}


async def _boom(state: MegaState) -> dict:
    CALLS["boom"] += 1
    raise ValueError("always fails")


async def _handled(state: MegaState) -> dict:
    return {"log": ["handled"]}


async def _defer(state: MegaState) -> dict:
    return {"log": ["deferred"]}


class ChildState(TypedDict, total=False):
    value: int
    log: Annotated[list[str], operator.add]


async def _child(state: ChildState) -> dict:
    CALLS["child"] += 1
    return {"value": state.get("value", 0) + 1, "log": ["child"]}


def _build_child():
    child = StateGraph(ChildState)
    child.add_node("child", _child)
    child.add_edge(START, "child")
    child.add_edge("child", END)
    return child.compile()


async def _ask(state: MegaState) -> dict:
    CALLS["ask"] += 1
    approval = interrupt("approve?")
    return {"log": ["ask"], "flag": f"approved={approval}"}


async def _done(state: MegaState) -> dict:
    return {"log": ["done"]}


def _subgraph_node() -> dict:
    """Run the child graph with an explicit input/output mapping (no shared channels)."""
    child = _build_child()

    async def sub(state: MegaState) -> dict:
        result = await child.ainvoke({"value": state.get("count", 0), "log": []})
        return {"child": result["value"], "log": ["sub"]}

    return {"node": sub}


def build() -> dict:
    """Return ``{"graph": compiled, "store": store, "saver": saver}``."""
    graph = StateGraph(MegaState)
    graph.add_node("seed", _seed)
    graph.add_node("grow", _grow)
    graph.add_node("worker", _worker)
    graph.add_node("join", _join)
    graph.add_node("cached", _cached, cache_policy=CachePolicy(ttl=60))
    graph.add_node("retry", _retry, retry_policy=RetryPolicy(max_attempts=3, retry_on=ValueError, initial_interval=0.0, backoff_factor=1.0, max_interval=0.0, jitter=False))
    graph.add_node("boom", _boom, error_handler=_handled)
    graph.add_node("handled", _handled)
    graph.add_node("defer", _defer, defer=True)
    graph.add_node("sub", _subgraph_node()["node"])
    graph.add_node("ask", _ask)
    graph.add_node("done", _done)

    graph.add_edge(START, "seed")
    graph.add_edge("seed", "grow")
    graph.add_conditional_edges("grow", _route_after_grow)
    graph.add_edge("worker", "join")
    graph.add_edge("join", "cached")
    graph.add_edge("cached", "retry")
    graph.add_edge("retry", "sub")
    graph.add_edge("sub", "ask")
    graph.add_edge("ask", "done")
    # Both tails are terminal: the deferred node cannot have outgoing edges, and `boom`
    # always fails so its error handler runs instead.
    graph.add_edge("done", "defer")
    graph.add_edge("done", "boom")

    saver = InMemorySaver()
    store = InMemoryStore()
    compiled = graph.compile(checkpointer=saver, store=store, cache=InMemoryCache())
    return {"graph": compiled, "store": store, "saver": saver, "Command": Command}