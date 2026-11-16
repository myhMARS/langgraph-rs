//! `Command(graph=PARENT)` from a subgraph node (sync graphs).

use std::sync::Arc;

use langgraph_rs::{JsonState, NodeOutput, RunConfig, StateGraph, StateSchema, json};

#[test]
fn subgraph_node_can_route_to_the_parent_graph() {
    let child = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("jump", |_state: &JsonState| {
            Ok(NodeOutput::new()
                .update("value", json!(100))
                .goto_parent("parent_after"))
        })
        .set_entry_point("jump")
        .compile()
        .unwrap();

    let parent = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_subgraph("child", Arc::new(child))
        .add_node("parent_after", |state: &JsonState| {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("child")
        .set_finish_point("parent_after")
        .compile()
        .unwrap();

    let output = parent.invoke(&JsonState::new(), &RunConfig::new()).unwrap();
    assert_eq!(output.get("value"), Some(&json!(101)));
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_graphs_reject_parent_commands() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let graph = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("jump", |_state: JsonState, _ctx| async move {
            Ok(NodeOutput::new().goto_parent("parent_after"))
        })
        .set_entry_point("jump")
        .compile()
        .unwrap();

    let error = graph
        .run(&JsonState::new(), &RunConfig::new())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("PARENT"));
}
