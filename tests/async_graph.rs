//! Async (Tokio) runtime adapter tests.
#![cfg(feature = "async")]

use std::time::{Duration, Instant};

use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{
    Error, JsonState, NodeOutput, RunConfig, State, StateGraph, StateSchema, StreamChunk, Usage,
    reducers,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct AsyncState {
    value: i64,
    log: Vec<String>,
}

impl State for AsyncState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("value")
            .with_reducer("log", reducers::append())
    }
}

#[tokio::test]
async fn async_linear_pipeline_and_report() {
    let graph = AsyncStateGraph::<AsyncState>::new()
        .add_node("first", |state: AsyncState, context| async move {
            context.record_tokens(10, 2);
            Ok(NodeOutput::new()
                .set("value", state.value + 1)
                .set("log", vec!["first".to_string()]))
        })
        .add_node("second", |mut state: AsyncState, context| async move {
            state.value += 1;
            context.record_usage(Usage::new().with_model("test-model"));
            Ok(NodeOutput::new()
                .set("value", state.value)
                .set("log", vec!["second".to_string()]))
        })
        .set_entry_point("first")
        .add_edge("first", "second")
        .set_finish_point("second")
        .compile()
        .expect("async graph compiles");

    let outcome = graph
        .run(&AsyncState::default(), &RunConfig::new())
        .await
        .unwrap();
    assert_eq!(outcome.state.value, 2);
    assert_eq!(outcome.state.log, vec!["first", "second"]);
    assert_eq!(outcome.report.executions().len(), 2);
    assert_eq!(outcome.report.total_tokens(), 12);
    assert_eq!(
        outcome.report.executions()[1]
            .usage
            .as_ref()
            .unwrap()
            .model
            .as_deref(),
        Some("test-model")
    );
}

#[tokio::test]
async fn async_conditional_loop_terminates() {
    let graph = AsyncStateGraph::<AsyncState>::new()
        .add_node("increment", |state: AsyncState, _context| async move {
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .set_entry_point("increment")
        .add_conditional_edges("increment", |state: &AsyncState| {
            Ok(if state.value < 5 {
                langgraph_rs::Route::node("increment")
            } else {
                langgraph_rs::Route::end()
            })
        })
        .compile()
        .unwrap();

    let output = graph.invoke(&AsyncState::default()).await.unwrap();
    assert_eq!(output.value, 5);
}

#[tokio::test]
async fn async_supersteps_run_concurrently() {
    let graph = AsyncStateGraph::<JsonState>::new()
        .add_node("start", |_state: JsonState, _context| async move {
            Ok(NodeOutput::new())
        })
        .add_node("slow_a", |_state: JsonState, _context| async move {
            tokio::time::sleep(Duration::from_millis(60)).await;
            Ok(NodeOutput::new().set("a", 1))
        })
        .add_node("slow_b", |_state: JsonState, _context| async move {
            tokio::time::sleep(Duration::from_millis(60)).await;
            Ok(NodeOutput::new().set("b", 2))
        })
        .add_node("join", |_state: JsonState, _context| async move {
            Ok(NodeOutput::new().set("done", true))
        })
        .set_entry_point("start")
        .add_edge("start", "slow_a")
        .add_edge("start", "slow_b")
        .add_edge("slow_a", "join")
        .add_edge("slow_b", "join")
        .set_finish_point("join")
        .compile()
        .unwrap();

    let started = Instant::now();
    let output = graph.invoke(&JsonState::default()).await.unwrap();
    let elapsed = started.elapsed();
    assert!(elapsed < Duration::from_millis(200), "took {elapsed:?}");
    assert_eq!(output.get("a"), Some(&serde_json::json!(1)));
    assert_eq!(output.get("b"), Some(&serde_json::json!(2)));
    assert_eq!(output.get("done"), Some(&serde_json::json!(true)));
}

#[tokio::test]
async fn async_streaming_and_custom_events() {
    let graph = AsyncStateGraph::<AsyncState>::new()
        .add_node("emit", |_state: AsyncState, context| async move {
            context.custom(serde_json::json!({ "progress": 0.5 }))?;
            context.record_usage(Usage::new().with_input_tokens(3));
            Ok(NodeOutput::new().set("value", 1))
        })
        .set_entry_point("emit")
        .set_finish_point("emit")
        .compile()
        .unwrap();

    let chunks = graph
        .stream(&AsyncState::default(), &RunConfig::new())
        .await
        .unwrap();
    assert!(chunks.iter().any(|chunk| matches!(
        chunk,
        StreamChunk::Custom { node, .. } if node == "emit"
    )));
    assert!(chunks.iter().any(|chunk| matches!(
        chunk,
        StreamChunk::Usage { execution } if execution.usage.as_ref().is_some_and(|usage| usage.total_tokens() == 3)
    )));
    assert!(
        chunks
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Values { .. }))
    );
}

#[tokio::test]
async fn async_failures_abort_the_run() {
    let graph = AsyncStateGraph::<AsyncState>::new()
        .add_node("broken", |_state: AsyncState, _context| async move {
            Err::<NodeOutput, _>(Error::other("async boom"))
        })
        .set_entry_point("broken")
        .set_finish_point("broken")
        .compile()
        .unwrap();

    let error = graph.invoke(&AsyncState::default()).await.unwrap_err();
    assert!(matches!(error, Error::Node { node, .. } if node == "broken"));
}

#[tokio::test]
async fn async_graph_interoperates_with_the_sync_graph() {
    // A sync graph can be used to prepare state for an async graph and vice versa.
    let sync_graph = StateGraph::<AsyncState>::new()
        .add_node("seed", |_state: &AsyncState| {
            Ok(NodeOutput::new().set("value", 10))
        })
        .set_entry_point("seed")
        .set_finish_point("seed")
        .compile()
        .unwrap();
    let seeded = sync_graph
        .invoke(&AsyncState::default(), &RunConfig::new())
        .unwrap();

    let async_graph = AsyncStateGraph::<AsyncState>::new()
        .add_node("double", |state: AsyncState, _context| async move {
            Ok(NodeOutput::new().set("value", state.value * 2))
        })
        .set_entry_point("double")
        .set_finish_point("double")
        .compile()
        .unwrap();
    let output = async_graph.invoke(&seeded).await.unwrap();
    assert_eq!(output.value, 20);
}
