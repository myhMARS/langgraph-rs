//! Subgraph streaming (`RunConfig::with_stream_subgraphs`).

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, NodeOutput, RunConfig, State, StateGraph, StateSchema, StreamChunk, custom,
    json,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct StreamState {
    value: i64,
}

impl State for StreamState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

fn child_graph() -> langgraph_rs::CompiledGraph<StreamState> {
    StateGraph::<StreamState>::new()
        .add_node("work", |state: &StreamState| {
            custom(json!({ "token": "child" }))?;
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .set_entry_point("work")
        .set_finish_point("work")
        .compile()
        .unwrap()
}

fn parent_graph() -> langgraph_rs::CompiledGraph<StreamState> {
    let child = Arc::new(child_graph());
    StateGraph::<StreamState>::new()
        .add_node("seed", |_state: &StreamState| {
            Ok(NodeOutput::new().set("value", 10))
        })
        .add_subgraph("child", child)
        .set_entry_point("seed")
        .add_edge("seed", "child")
        .set_finish_point("child")
        .compile_with(CompileOptions::new())
        .unwrap()
}

#[test]
fn subgraph_streaming_forwards_namespaced_chunks() {
    let graph = parent_graph();
    let mut chunks = Vec::new();
    let output = graph
        .stream_with(
            &StreamState::default(),
            &RunConfig::new().with_stream_subgraphs(true),
            |chunk| chunks.push(chunk),
        )
        .unwrap();
    assert_eq!(output.value, 11);

    let child_chunks: Vec<&StreamChunk> = chunks
        .iter()
        .filter(|chunk| chunk.matches_modes(&[]))
        .filter(|chunk| chunk.namespace() == Some("child"))
        .collect();
    assert!(
        !child_chunks.is_empty(),
        "expected child chunks, got {chunks:?}"
    );
    assert!(
        child_chunks.iter().any(|chunk| matches!(
            chunk,
            StreamChunk::Subgraph { chunk, .. } if matches!(chunk.as_ref(), StreamChunk::Custom { .. })
        )),
        "expected a forwarded custom chunk"
    );
}

#[test]
fn subgraph_streaming_is_off_by_default() {
    let graph = parent_graph();
    let chunks = graph
        .stream(&StreamState::default(), &RunConfig::new())
        .unwrap();
    assert!(chunks.iter().all(|chunk| chunk.namespace().is_none()));
}

#[test]
fn nested_subgraph_namespaces_are_joined() {
    let grandchild = Arc::new(child_graph());
    let child = Arc::new(
        StateGraph::<StreamState>::new()
            .add_node("seed", |_state: &StreamState| {
                Ok(NodeOutput::new().set("value", 1))
            })
            .add_subgraph("grandchild", grandchild)
            .set_entry_point("seed")
            .add_edge("seed", "grandchild")
            .set_finish_point("grandchild")
            .compile()
            .unwrap(),
    );
    let parent = StateGraph::<StreamState>::new()
        .add_node("seed", |_state: &StreamState| {
            Ok(NodeOutput::new().set("value", 0))
        })
        .add_subgraph("child", child)
        .set_entry_point("seed")
        .add_edge("seed", "child")
        .set_finish_point("child")
        .compile()
        .unwrap();

    let chunks = parent
        .stream(
            &StreamState::default(),
            &RunConfig::new().with_stream_subgraphs(true),
        )
        .unwrap();
    let namespaces: Vec<&str> = chunks
        .iter()
        .filter_map(|chunk| chunk.namespace())
        .collect();
    assert!(
        namespaces.iter().any(|namespace| namespace.contains('|')),
        "expected a joined namespace, got {namespaces:?}"
    );
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_subgraph_streaming_forwards_chunks() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let child = Arc::new(
        AsyncStateGraph::<StreamState>::new()
            .add_node("work", |state: StreamState, ctx| async move {
                ctx.custom(json!({ "token": "async-child" }))?;
                Ok(NodeOutput::new().set("value", state.value + 1))
            })
            .set_entry_point("work")
            .set_finish_point("work")
            .compile()
            .unwrap(),
    );
    let parent = AsyncStateGraph::<StreamState>::new()
        .add_node("seed", |_state: StreamState, _ctx| async move {
            Ok(NodeOutput::new().set("value", 10))
        })
        .add_subgraph("child", child)
        .set_entry_point("seed")
        .add_edge("seed", "child")
        .set_finish_point("child")
        .compile()
        .unwrap();

    let mut chunks = Vec::new();
    let output = parent
        .stream_with(
            &StreamState::default(),
            &RunConfig::new().with_stream_subgraphs(true),
            |chunk| chunks.push(chunk),
        )
        .await
        .unwrap();
    assert_eq!(output.value, 11);
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk.namespace() == Some("child")),
        "expected child chunks, got {chunks:?}"
    );
}
