//! Stream mode tests: values, updates, custom, messages, tasks, debug, checkpoints.

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph, StateSchema,
    StreamChunk, StreamMode, custom, json, message, record_tokens,
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

fn writer_graph() -> langgraph_rs::CompiledGraph<StreamState> {
    StateGraph::<StreamState>::new()
        .add_node("emit", |state: &StreamState| {
            message(json!({ "token": "hello" }))?;
            custom(json!({ "progress": 0.5 }))?;
            record_tokens(3, 4)?;
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .set_entry_point("emit")
        .set_finish_point("emit")
        .compile()
        .unwrap()
}

fn modes_of(chunks: &[StreamChunk]) -> Vec<StreamMode> {
    chunks.iter().filter_map(StreamChunk::mode).collect()
}

#[test]
fn stream_all_contains_writer_and_lifecycle_chunks() {
    let chunks = writer_graph()
        .stream(&StreamState::default(), &RunConfig::new())
        .unwrap();
    assert!(
        chunks
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Custom { .. }))
    );
    assert!(chunks.iter().any(|chunk| matches!(chunk, StreamChunk::Message { node, message } if node == "emit" && message == &json!({ "token": "hello" }))));
    assert!(chunks.iter().any(
        |chunk| matches!(chunk, StreamChunk::Task { node, result: Some(_), .. } if node == "emit")
    ));
    assert!(
        chunks
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Debug { .. }))
    );
    assert!(
        chunks
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Values { .. }))
    );
    assert!(
        chunks
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Updates { .. }))
    );
    assert!(
        chunks
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Usage { .. }))
    );
}

#[test]
fn stream_modes_filter_chunks() {
    let graph = writer_graph();
    let config = RunConfig::new();

    let values = graph
        .stream_modes(&StreamState::default(), &config, &[StreamMode::Values])
        .unwrap();
    assert!(
        values
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Values { .. }))
    );
    assert!(
        !values
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Custom { .. }))
    );
    assert!(
        !values
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Message { .. }))
    );
    assert!(
        !values
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Task { .. }))
    );
    assert!(
        !values
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Debug { .. }))
    );

    let messages = graph
        .stream_modes(&StreamState::default(), &config, &[StreamMode::Messages])
        .unwrap();
    assert!(
        messages
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Message { .. }))
    );
    assert!(
        !messages
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Custom { .. }))
    );

    let tasks = graph
        .stream_modes(&StreamState::default(), &config, &[StreamMode::Tasks])
        .unwrap();
    assert!(
        tasks
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Task { .. }))
    );
    assert!(
        !tasks
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Values { .. }))
    );

    let debug = graph
        .stream_modes(&StreamState::default(), &config, &[StreamMode::Debug])
        .unwrap();
    assert!(
        debug
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Debug { .. }))
    );
    assert!(debug.iter().all(|chunk| matches!(
        chunk,
        StreamChunk::Debug { .. }
            | StreamChunk::Usage { .. }
            | StreamChunk::Interrupt { .. }
            | StreamChunk::Task { .. }
    )));
    assert!(
        !debug
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Values { .. }))
    );
    assert!(
        debug
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Task { .. }))
    );
}

#[test]
fn checkpoint_mode_requires_a_checkpointer() {
    let graph = StateGraph::<StreamState>::new()
        .add_node("work", |state: &StreamState| {
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .set_entry_point("work")
        .set_finish_point("work")
        .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
        .unwrap();
    let chunks = graph
        .stream_modes(
            &StreamState::default(),
            &RunConfig::new().with_thread("stream-cp"),
            &[StreamMode::Checkpoints],
        )
        .unwrap();
    assert!(chunks.iter().all(|chunk| matches!(
        chunk,
        StreamChunk::Checkpoint { .. } | StreamChunk::Usage { .. }
    )));
    assert!(!chunks.is_empty());
}

#[test]
fn stream_mode_names_parse() {
    for mode in [
        StreamMode::Values,
        StreamMode::Updates,
        StreamMode::Custom,
        StreamMode::Messages,
        StreamMode::Debug,
        StreamMode::Tasks,
        StreamMode::Checkpoints,
    ] {
        assert_eq!(StreamMode::parse(mode.as_str()), Some(mode));
    }
    assert_eq!(StreamMode::parse("nope"), None);
    assert!(
        modes_of(
            &writer_graph()
                .stream(&StreamState::default(), &RunConfig::new())
                .unwrap()
        )
        .contains(&StreamMode::Messages)
    );
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_graphs_support_stream_modes() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let graph = AsyncStateGraph::<StreamState>::new()
        .add_node("emit", |state: StreamState, ctx| async move {
            ctx.message(json!({ "token": "async-hello" }))?;
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .set_entry_point("emit")
        .set_finish_point("emit")
        .compile()
        .unwrap();

    let chunks = graph
        .stream_modes(
            &StreamState::default(),
            &RunConfig::new(),
            &[StreamMode::Messages],
        )
        .await
        .unwrap();
    assert!(
        chunks
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Message { .. }))
    );
    assert!(
        !chunks
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Values { .. }))
    );
}
