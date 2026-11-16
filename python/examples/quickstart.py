"""Quickstart for the Python bindings.

Build the extension module first, for example with::

    maturin develop --features python
    # or: cargo build --features python, then copy the cdylib to langgraph_rs.pyd/.so

then run::

    python python/examples/quickstart.py
"""

import langgraph_rs as lg


def main() -> None:
    # 1. A stateful graph: every channel declares how writes merge.
    graph = lg.StateGraph({"count": "last_value", "log": "append"})

    # 2. Nodes are plain Python callables: state dict in, writes dict out.
    graph.add_node(
        "increment",
        lambda state: {"count": state["count"] + 1, "log": [f"increment -> {state['count'] + 1}"]},
    )
    graph.add_node("finish", lambda state: {"log": ["finish"]})

    # 3. Edges: START/END aliases are supported, routers may return None to stop.
    graph.add_edge("START", "increment")
    graph.add_conditional_edges(
        "increment",
        lambda state: "increment" if state["count"] < 3 else "finish",
    )
    graph.set_finish_point("finish")

    compiled = graph.compile()
    result = compiled.invoke({"count": 0, "log": []})
    print("final state:", result)

    # 4. Per-node consumption tracking: `run` returns state + report.
    usage_graph = lg.StateGraph({"answer": "last_value"})
    usage_graph.add_node(
        "model",
        lambda state: {
            "answer": "42",
            "__usage__": {
                "model": "gpt-4o-mini",
                "input_tokens": 128,
                "output_tokens": 16,
                "cost_usd": 0.0002,
            },
        },
    )
    usage_graph.set_entry_point("model")
    usage_graph.set_finish_point("model")
    outcome = usage_graph.compile().run({"answer": ""})
    print("report:", outcome["report"])

    # 5. Human in the loop: interrupt() parks the run, resume() continues it.
    hitl = lg.StateGraph({"draft": "last_value", "approved": "last_value"})
    hitl.add_node("draft", lambda state: {"draft": "quarterly report"})
    hitl.add_node(
        "review",
        lambda state: {
            "approved": lg.interrupt({"question": f"approve `{state['draft']}`?"})["approved"]
        },
    )
    hitl.set_entry_point("draft")
    hitl.add_edge("draft", "review")
    hitl.set_finish_point("review")
    reviewed = hitl.compile(checkpointer=True)

    try:
        reviewed.invoke({"draft": "", "approved": False}, thread_id="demo")
    except lg.GraphInterrupt as paused:
        print("paused with payload:", paused.args[0])
        snapshot = reviewed.get_state("demo")
        print("snapshot:", snapshot["interrupted"], snapshot["next"])
        final = reviewed.resume("demo", {"approved": True})
        print("resumed:", final)


if __name__ == "__main__":
    main()