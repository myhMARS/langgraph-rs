"""Mega workflow built with the Rust engine through its PyO3 binding.

Mirrors ``build_lg.py`` node for node so the three timings cover the same work:

    seed -> grow (loop x3) -> Send(worker) x3 -> join -> cached -> retry
    -> defer (deferred node) -> sub (embedded child graph) -> ask (interrupt)
    -> done -> boom (always fails, error_handler)

Differences that are API-level only:

* the state schema is a dict of channel aliases instead of a ``TypedDict``;
* the child graph runs through an explicit input/output mapping (same as the
  LangGraph version, so no channel is shared and no reducer duplicates values);
* ``Send`` fan-out comes from the conditional router, as in LangGraph.
"""

from __future__ import annotations

import langgraph_rs as lg

CALLS = {"retry": 0, "boom": 0, "cached": 0, "ask": 0, "child": 0}

SCHEMA = {
    "count": "last_value",
    "sink": "add",
    "log": "append",
    "flag": "last_value",
    "child": "last_value",
}


async def _seed(state) -> dict:
    store = lg.get_store()
    previous = store.get(("mega",), "runs")
    if previous is None:
        runs = 0
    elif isinstance(previous, dict) and "value" in previous:
        runs = previous["value"].get("n", 0)
    else:
        runs = (previous or {}).get("n", 0)
    store.put(("mega",), "runs", {"n": runs + 1})
    return {"log": ["seed"], "count": 0, "sink": 0, "child": 0, "flag": f"runs={runs + 1}"}


async def _grow(state) -> dict:
    return {"count": state.get("count", 0) + 1, "log": ["grow"]}


def _route_after_grow(state):
    if state.get("count", 0) < 3:
        return "grow"
    return [lg.Send("worker", {"n": 1}), lg.Send("worker", {"n": 2}), lg.Send("worker", {"n": 3})]


async def _worker(state) -> dict:
    return {"log": ["worker"], "sink": state.get("n", 1)}


async def _join(state) -> dict:
    return {"log": ["join"]}


async def _cached(state) -> dict:
    CALLS["cached"] += 1
    return {"log": ["cached"]}


async def _retry(state) -> dict:
    CALLS["retry"] += 1
    if CALLS["retry"] % 2 == 1:
        raise ValueError("transient failure")
    return {"log": ["retry"]}


async def _boom(state) -> dict:
    CALLS["boom"] += 1
    raise ValueError("always fails")


async def _handled(state) -> dict:
    return {"log": ["handled"]}


async def _defer(state) -> dict:
    return {"log": ["deferred"]}


async def _child(state) -> dict:
    CALLS["child"] += 1
    return {"value": state.get("value", 0) + 1, "log": ["child"]}


def _child_sync(state) -> dict:
    CALLS["child"] += 1
    return {"value": state.get("value", 0) + 1, "log": ["child"]}


def _build_child():
    # A sync child graph: embedding an async child inside an async node would need a
    # second event loop, which asyncio rejects.
    child = lg.StateGraph({"value": "last_value", "log": "append"})
    child.add_node("child", _child_sync)
    child.add_edge(lg.START, "child")
    child.set_finish_point("child")
    return child.compile()


async def _ask(state) -> dict:
    CALLS["ask"] += 1
    approval = lg.interrupt("approve?")
    return {"log": ["ask"], "flag": f"approved={approval}"}


async def _done(state) -> dict:
    return {"log": ["done"]}


def build() -> dict:
    graph = lg.AsyncStateGraph(SCHEMA)
    graph.add_node("seed", _seed)
    graph.add_node("grow", _grow)
    graph.add_node("worker", _worker)
    graph.add_node("join", _join)
    graph.add_node("cached", _cached, cache_policy=lg.CachePolicy(ttl=60))
    graph.add_node("retry", _retry, retry_policy=lg.RetryPolicy(max_attempts=3, initial_interval=0.0, backoff_factor=1.0, max_interval=0.0, jitter=0.0, retry_on=["ValueError"]))
    graph.add_node("boom", _boom, error_handler="handled")
    graph.add_node("handled", _handled)
    graph.add_node("defer", _defer, deferred=True)

    child = _build_child()

    async def sub(state) -> dict:
        count = state.get("count", 0)
        result = child.invoke({"value": count, "log": []})
        return {"child": result.get("value", count + 1), "log": ["sub"]}

    graph.add_node("sub", sub)
    graph.add_node("ask", _ask)
    graph.add_node("done", _done)

    graph.add_edge(lg.START, "seed")
    graph.add_edge("seed", "grow")
    graph.add_conditional_edges("grow", _route_after_grow)
    graph.add_edge("worker", "join")
    graph.add_edge("join", "cached")
    graph.add_edge("cached", "retry")
    graph.add_edge("retry", "sub")
    graph.add_edge("sub", "ask")
    graph.add_edge("ask", "done")
    # Terminal tails: `defer` may not have outgoing edges and `boom` always fails.
    graph.add_edge("done", "defer")
    graph.add_edge("done", "boom")

    saver = lg.InMemorySaver()
    store = lg.InMemoryStore()
    compiled = graph.compile(checkpointer=saver, store=store, cache=True)
    return {"graph": compiled, "store": store, "saver": saver}