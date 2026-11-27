//! Deferred nodes and per-`Send` timeouts.

use std::time::Duration;

use langgraph_rs::{
    CompileOptions, InMemorySaver, NodeOutput, RunConfig, Send, State, StateGraph, StateSchema,
    json, reducers,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct DeferredState {
    value: i64,
    log: Vec<String>,
}

impl State for DeferredState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("value")
            .with_reducer("log", reducers::append())
    }
}

fn deferred_graph(
    saver: Option<std::sync::Arc<InMemorySaver>>,
) -> langgraph_rs::CompiledGraph<DeferredState> {
    let builder = StateGraph::<DeferredState>::new()
        .add_node("inc", |state: &DeferredState| {
            Ok(NodeOutput::new()
                .set("value", state.value + 1)
                .set("log", vec!["inc".to_string()]))
        })
        .add_node_deferred("summary", |state: &DeferredState| {
            Ok(NodeOutput::new()
                .set("value", 100)
                .set("log", vec![format!("summary after {}", state.value)]))
        })
        .set_entry_point("inc")
        .add_edge("inc", "summary")
        .set_finish_point("inc");
    match saver {
        Some(saver) => builder
            .compile_with(CompileOptions::new().with_checkpointer(saver))
            .unwrap(),
        None => builder.compile().unwrap(),
    }
}

#[test]
fn deferred_node_runs_after_regular_nodes() {
    let graph = deferred_graph(None);
    let outcome = graph
        .run(&DeferredState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(outcome.state.value, 100);
    assert_eq!(outcome.state.log, vec!["inc", "summary after 1"]);
    let order: Vec<&str> = outcome
        .report
        .executions
        .iter()
        .map(|execution| execution.node.as_str())
        .collect();
    assert_eq!(order, vec!["inc", "summary"]);
}

#[test]
fn deferred_node_interrupt_can_be_resumed() {
    let saver = std::sync::Arc::new(InMemorySaver::new());
    let graph = StateGraph::<DeferredState>::new()
        .add_node("inc", |state: &DeferredState| {
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .add_node_deferred("summary", |state: &DeferredState| {
            let extra = langgraph_rs::interrupt(json!({ "question": "extra" }))?;
            Ok(NodeOutput::new().set("value", state.value + extra.as_i64().unwrap_or_default()))
        })
        .set_entry_point("inc")
        .add_edge("inc", "summary")
        .set_finish_point("inc")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &DeferredState::default(),
            &RunConfig::new().with_thread("deferred-hitl"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());
    assert_eq!(
        error.interrupt().map(|interrupt| interrupt.node.as_str()),
        Some("summary")
    );

    let output = graph.resume("deferred-hitl", json!(41)).unwrap();
    assert_eq!(output.value, 42);
}

#[test]
fn deferred_nodes_may_be_entries_and_have_edges() {
    // LangGraph accepts a deferred entry point and deferred nodes with outgoing edges; the deferred
    // task simply runs once no regular task is pending.
    let graph = StateGraph::<DeferredState>::new()
        .add_node_deferred("entry", |_state: &DeferredState| {
            Ok(NodeOutput::new().set("value", 7))
        })
        .set_entry_point("entry")
        .set_finish_point("entry")
        .compile()
        .expect("deferred entry point compiles");
    let output = graph.invoke(&DeferredState::default(), &RunConfig::new()).unwrap();
    assert_eq!(output.value, 7);

    let graph = StateGraph::<DeferredState>::new()
        .add_node("a", |_state: &DeferredState| {
            Ok(NodeOutput::new().set("log", vec!["a".to_string()]))
        })
        .add_node_deferred("summary", |_state: &DeferredState| {
            Ok(NodeOutput::new().set("value", 1))
        })
        .set_entry_point("a")
        .add_edge("a", "summary")
        .set_finish_point("summary")
        .compile()
        .expect("deferred node with outgoing edges compiles");
    let output = graph.invoke(&DeferredState::default(), &RunConfig::new()).unwrap();
    assert_eq!(output.value, 1);
    assert_eq!(output.log, vec!["a".to_string()]);
}

#[test]
fn send_timeout_overrides_the_node_policy() {
    let graph = StateGraph::<DeferredState>::new()
        .add_node("dispatch", |_state: &DeferredState| {
            Ok(NodeOutput::new().send(
                Send::with_arg("worker", json!({ "value": 1 }))
                    .with_timeout(Duration::from_millis(20)),
            ))
        })
        .add_node("worker", |_state: &DeferredState| {
            std::thread::sleep(Duration::from_millis(250));
            Ok(NodeOutput::new().set("value", 2))
        })
        .set_entry_point("dispatch")
        .set_finish_point("worker")
        .compile()
        .unwrap();

    let error = graph
        .run(&DeferredState::default(), &RunConfig::new())
        .unwrap_err()
        .to_string();
    assert!(error.contains("timed out"), "unexpected error: {error}");
}

#[test]
fn send_timeout_allows_fast_tasks() {
    let graph = StateGraph::<DeferredState>::new()
        .add_node("dispatch", |_state: &DeferredState| {
            Ok(NodeOutput::new().send(
                Send::with_arg("worker", json!({ "value": 5 }))
                    .with_timeout(Duration::from_secs(5)),
            ))
        })
        .add_node("worker", |state: &DeferredState| {
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .set_entry_point("dispatch")
        .set_finish_point("worker")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&DeferredState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.value, 6);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_deferred_node_runs_after_regular_nodes() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let graph = AsyncStateGraph::<DeferredState>::new()
        .add_node("inc", |state: DeferredState, _ctx| async move {
            Ok(NodeOutput::new()
                .set("value", state.value + 1)
                .set("log", vec!["inc".to_string()]))
        })
        .add_node("summary", |state: DeferredState, _ctx| async move {
            Ok(NodeOutput::new()
                .set("value", 50)
                .set("log", vec![format!("summary after {}", state.value)]))
        })
        .set_deferred("summary")
        .set_entry_point("inc")
        .add_edge("inc", "summary")
        .set_finish_point("inc")
        .compile()
        .unwrap();

    let outcome = graph
        .run(&DeferredState::default(), &RunConfig::new())
        .await
        .unwrap();
    assert_eq!(outcome.state.value, 50);
    assert_eq!(outcome.state.log, vec!["inc", "summary after 1"]);
}
