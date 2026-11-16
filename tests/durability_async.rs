//! `Durability::Async` background checkpoint writes.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use langgraph_rs::{
    CompileOptions, Durability, InMemorySaver, JsonState, NodeOutput, RunConfig, StateGraph,
    StateSchema, json,
};

fn graph(saver: Arc<InMemorySaver>) -> langgraph_rs::CompiledGraph<JsonState> {
    StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("a", |_state: &JsonState| {
            Ok(NodeOutput::new().update("value", json!(1)))
        })
        .add_node("b", |_state: &JsonState| {
            Ok(NodeOutput::new().update("value", json!(2)))
        })
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap()
}

#[test]
fn async_durability_flushes_the_full_history() {
    let sync_saver = Arc::new(InMemorySaver::new());
    let sync_graph = graph(sync_saver);
    sync_graph
        .run(
            &JsonState::new(),
            &RunConfig::new()
                .with_thread("async-sync")
                .with_durability(Durability::Sync),
        )
        .unwrap();
    let sync_history = sync_graph.get_state_history("async-sync").unwrap().len();

    let async_saver = Arc::new(InMemorySaver::new());
    let async_graph = graph(async_saver);
    let output = async_graph
        .run(
            &JsonState::new(),
            &RunConfig::new()
                .with_thread("async-durability")
                .with_durability(Durability::Async),
        )
        .unwrap();
    assert_eq!(output.state.get("value"), Some(&json!(2)));
    let async_history = async_graph.get_state_history("async-durability").unwrap();
    assert_eq!(async_history.len(), sync_history);
    assert_eq!(
        async_graph
            .get_state("async-durability")
            .unwrap()
            .unwrap()
            .values
            .get("value"),
        Some(&json!(2))
    );
}

#[test]
fn async_durability_flushes_on_failure() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("boom", move |_state: &JsonState| {
            counter.fetch_add(1, Ordering::SeqCst);
            Err(langgraph_rs::Error::other("boom"))
        })
        .set_entry_point("boom")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    assert!(
        graph
            .run(
                &JsonState::new(),
                &RunConfig::new()
                    .with_thread("async-failure")
                    .with_durability(Durability::Async),
            )
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(!graph.get_state_history("async-failure").unwrap().is_empty());
}
