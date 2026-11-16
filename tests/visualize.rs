//! Mermaid visualisation tests.

use langgraph_rs::{NodeOutput, PathMap, RetryPolicy, Route, State, StateGraph, StateSchema};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
struct VisualState {
    value: i64,
}

impl State for VisualState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

fn demo_graph() -> langgraph_rs::CompiledGraph<VisualState> {
    StateGraph::<VisualState>::new()
        .add_node("work", |_state: &VisualState| Ok(NodeOutput::new()))
        .add_node("finish", |_state: &VisualState| Ok(NodeOutput::new()))
        .add_node("fallback", |_state: &VisualState| Ok(NodeOutput::new()))
        .set_entry_point("work")
        .add_conditional_edges_mapped(
            "work",
            |_state: &VisualState| Ok(Route::node("ok")),
            PathMap::from([("ok", "finish")]),
        )
        .add_error_edge("work", "fallback")
        .set_retry(
            "work",
            RetryPolicy::new(3).with_max_backoff(std::time::Duration::from_millis(1)),
        )
        .compile_with(langgraph_rs::CompileOptions::new().interrupt_before(["work"]))
        .unwrap()
}

#[test]
fn mermaid_export_has_expected_structure() {
    let mermaid = demo_graph().to_mermaid();
    assert!(mermaid.starts_with("%% langgraph-rs graph\ngraph TD"));
    for expected in [
        "START([\"START\"])",
        "END([\"END\"])",
        "START --> n_work",
        "n_work[\"work<br/>retry x3<br/>pause before\"]",
        "n_work_router_0{\"router\"}",
        "n_work_router_0 -.->|\"ok\"| n_finish",
        "n_work -.->|\"on error\"| n_fallback",
    ] {
        assert!(
            mermaid.contains(expected),
            "mermaid output is missing `{expected}`:\n{mermaid}"
        );
    }
}

#[test]
fn mermaid_export_is_deterministic() {
    let graph = demo_graph();
    assert_eq!(graph.to_mermaid(), graph.to_mermaid());
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_graphs_export_mermaid_too() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let graph = AsyncStateGraph::<VisualState>::new()
        .add_node("only", |_state: VisualState, _ctx| async move {
            Ok(NodeOutput::new())
        })
        .set_entry_point("only")
        .set_finish_point("only")
        .compile()
        .unwrap();
    let mermaid = graph.to_mermaid();
    assert!(mermaid.contains("START --> n_only"));
    assert!(mermaid.contains("n_only --> END"));
}
