"""A LangGraph program that only changes its import lines.

The graph below is ordinary LangGraph code. Switching to the Rust engine means replacing::

    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command, interrupt

with::

    from langgraph_rs.graph import END, START, StateGraph
    from langgraph_rs.checkpoint.memory import InMemorySaver
    from langgraph_rs.types import Command, interrupt

Everything else -- the state class, the nodes, ``add_conditional_edges`` with a path map,
``compile(checkpointer=...)``, the ``{"configurable": {"thread_id": ...}}`` config, ``invoke``,
``stream``, ``get_state``, ``update_state`` and ``get_state_history`` -- stays untouched.

Run it with the package assembled::

    cargo build --release --features "python python-async sqlite redis postgres msgpack"
    python python/build_package.py --release
    PYTHONPATH=<repo>/target/python python python/examples/dropin_from_langgraph.py
"""

from __future__ import annotations

import operator
import typing

from langgraph_rs.checkpoint.memory import InMemorySaver
from langgraph_rs.graph import END, START, StateGraph
from langgraph_rs.types import Command, GraphInterrupt, interrupt


class State(typing.TypedDict):
    """A normal LangGraph state: `Annotated` reducers, plain channels."""

    messages: typing.Annotated[list, operator.add]
    attempts: int


def call_model(state: State) -> dict:
    attempt = state.get("attempts", 0) + 1
    return {"attempts": attempt, "messages": [f"model call {attempt}"]}


def review(state: State) -> dict:
    answer = interrupt({"question": "approve?", "attempts": state["attempts"]})
    return {"messages": [f"review: {answer}"]}


def route(state: State) -> str:
    return "review" if state["attempts"] >= 2 else "model"


builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_node("review", review)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", route, {"model": "model", "review": "review", END: END})
builder.add_edge("review", END)
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "dropin-demo"}}

# Difference from LangGraph: LangGraph returns the state with an ``__interrupt__`` entry,
# langgraph_rs raises ``GraphInterrupt`` with the same information. This is the one place
# where an import-only switch needs a small code change.
print("1) invoke stops at the interrupt")
try:
    print("   ", graph.invoke({"messages": [], "attempts": 0}, config))
except GraphInterrupt as interrupted:
    args = interrupted.args[0] if interrupted.args else None
    print("    GraphInterrupt:", args)
    print("    paused state  :", graph.get_state(config).values)

print("2) resume with the LangGraph `Command(resume=...)` shape")
print("   ", graph.invoke(Command(resume="ok"), config))

print("3) state / history / update_state")
snapshot = graph.get_state(config)
print("    values :", snapshot.values)
print("    history:", len(list(graph.get_state_history(config))))
graph.update_state(config, {"attempts": 99}, as_node="review")
print("    after update:", graph.get_state(config).values["attempts"])

print("4) streaming values (a list in langgraph_rs, a generator in LangGraph)")
chunks = graph.stream({"messages": [], "attempts": 0}, {"configurable": {"thread_id": "s"}}, stream_mode="values")
print("    last chunk:", chunks[-1])