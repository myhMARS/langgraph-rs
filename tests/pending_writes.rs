//! Pending writes: successful task results survive a single interrupt and replay on resume.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use langgraph_rs::{
    CompileOptions, InMemorySaver, JsonState, NodeOutput, RunConfig, StateGraph, StateSchema,
    interrupt, json,
};

#[test]
fn single_interrupt_defers_successful_writes_to_pending_writes() {
    let calls = Arc::new(AtomicUsize::new(0));
    let ok_calls = calls.clone();
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("ok", move |_state: &JsonState| {
            ok_calls.fetch_add(1, Ordering::SeqCst);
            Ok(NodeOutput::new().update("value", json!(1)))
        })
        .add_node("ask", |_state: &JsonState| {
            let answer = interrupt(json!({"question": "value"}))?;
            Ok(NodeOutput::new().update("value", answer.clone()))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask")
        .set_finish_point("ok")
        .set_finish_point("ask")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &JsonState::new(),
            &RunConfig::new().with_thread("pending-writes"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let snapshot = graph.get_state("pending-writes").unwrap().unwrap();
    assert_eq!(snapshot.tasks.len(), 1);
    assert_eq!(snapshot.tasks[0].node, "ask");

    let document = graph
        .export_checkpoint("pending-writes")
        .unwrap()
        .expect("checkpoint exists");
    let pending = document["pending_writes"].as_array().unwrap();
    assert!(
        pending
            .iter()
            .any(|write| write["node"] == "ok" && write["channel"] == "value")
    );

    let output = graph.resume("pending-writes", json!(5)).unwrap();
    assert_eq!(output.get("value"), Some(&json!(5)));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the successful task must not re-run on resume"
    );
}
