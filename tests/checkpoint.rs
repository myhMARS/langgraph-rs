//! Checkpointing and human-in-the-loop tests.

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, Error, InMemorySaver, NodeOutput, RunConfig, State, StateGraph, StateSchema,
    interrupt, reducers,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct ApprovalState {
    value: i64,
    approved: bool,
    log: Vec<String>,
}

impl State for ApprovalState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("value")
            .last_value("approved")
            .with_reducer("log", reducers::append())
    }
}

fn approval_graph(saver: Arc<InMemorySaver>) -> langgraph_rs::CompiledGraph<ApprovalState> {
    StateGraph::<ApprovalState>::new()
        .add_node("ask", |state: &ApprovalState| {
            let answer = interrupt(json!({ "question": "which value?" }))?;
            Ok(NodeOutput::new()
                .set("value", state.value + answer.as_i64().unwrap_or_default())
                .set("log", vec!["ask".to_string()]))
        })
        .add_node("finish", |_state: &ApprovalState| {
            Ok(NodeOutput::new()
                .set("approved", true)
                .set("log", vec!["finish".to_string()]))
        })
        .set_entry_point("ask")
        .add_edge("ask", "finish")
        .set_finish_point("finish")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap()
}

#[test]
fn interrupt_checkpoints_and_resume_continues_the_run() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = approval_graph(saver);

    let config = RunConfig::new().with_thread("approval");
    let error = graph
        .invoke(&ApprovalState::default(), &config)
        .unwrap_err();
    assert!(error.is_interrupt());
    let payload = error.interrupt().unwrap();
    assert_eq!(payload.node, "ask");
    assert_eq!(payload.value, json!({ "question": "which value?" }));

    let snapshot = graph.get_state("approval").unwrap().unwrap();
    assert!(snapshot.interrupted);
    assert_eq!(snapshot.next, vec!["ask".to_string()]);
    assert_eq!(snapshot.values.log, Vec::<String>::new());

    let output = graph.resume("approval", json!(41)).unwrap();
    assert_eq!(output.value, 41);
    assert!(output.approved);
    assert_eq!(output.log, vec!["ask", "finish"]);

    let final_state = graph.get_state("approval").unwrap().unwrap();
    assert!(!final_state.interrupted);
    assert!(final_state.next.is_empty());
}

#[test]
fn resume_can_apply_human_edits_before_continuing() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = approval_graph(saver);

    let error = graph
        .invoke(
            &ApprovalState::default(),
            &RunConfig::new().with_thread("edits"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let output = graph
        .resume_with_updates("edits", json!(7), [("value", json!(100))])
        .unwrap();
    // The node observes the human-provided state patch (100) plus the resume value (7).
    assert_eq!(output.value, 107);
    assert!(output.approved);
}

#[test]
fn interrupt_before_pauses_before_the_node_runs() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<ApprovalState>::new()
        .add_node("a", |_state: &ApprovalState| {
            Ok(NodeOutput::new().set("log", vec!["a".to_string()]))
        })
        .add_node("b", |_state: &ApprovalState| {
            Ok(NodeOutput::new().set("log", vec!["b".to_string()]))
        })
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile_with(
            CompileOptions::new()
                .with_checkpointer(saver)
                .interrupt_before(["b"]),
        )
        .unwrap();

    let error = graph
        .invoke(
            &ApprovalState::default(),
            &RunConfig::new().with_thread("before"),
        )
        .unwrap_err();
    assert_eq!(
        error.interrupt().map(|interrupt| interrupt.node.as_str()),
        Some("b")
    );

    let snapshot = graph.get_state("before").unwrap().unwrap();
    assert_eq!(snapshot.values.log, vec!["a"]);
    assert_eq!(snapshot.next, vec!["b".to_string()]);

    let output = graph.continue_thread("before").unwrap();
    assert_eq!(output.log, vec!["a", "b"]);
}

#[test]
fn interrupt_after_pauses_after_the_node_ran() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<ApprovalState>::new()
        .add_node("a", |_state: &ApprovalState| {
            Ok(NodeOutput::new().set("log", vec!["a".to_string()]))
        })
        .add_node("b", |_state: &ApprovalState| {
            Ok(NodeOutput::new().set("log", vec!["b".to_string()]))
        })
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile_with(
            CompileOptions::new()
                .with_checkpointer(saver)
                .interrupt_after(["a"]),
        )
        .unwrap();

    let error = graph
        .invoke(
            &ApprovalState::default(),
            &RunConfig::new().with_thread("after"),
        )
        .unwrap_err();
    assert_eq!(
        error.interrupt().map(|interrupt| interrupt.node.as_str()),
        Some("a")
    );

    let snapshot = graph.get_state("after").unwrap().unwrap();
    assert_eq!(snapshot.values.log, vec!["a"]);
    assert_eq!(snapshot.next, vec!["b".to_string()]);

    let output = graph.continue_thread("after").unwrap();
    assert_eq!(output.log, vec!["a", "b"]);
}

#[test]
fn resume_from_a_specific_checkpoint_replays_from_there() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = approval_graph(saver);

    let error = graph
        .invoke(
            &ApprovalState::default(),
            &RunConfig::new().with_thread("replay"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let history = graph.get_state_history("replay").unwrap();
    let interrupted = &history[0];
    assert!(interrupted.interrupted);

    let output = graph
        .resume_from("replay", &interrupted.checkpoint_id, json!(5))
        .unwrap();
    assert_eq!(output.value, 5);
    assert!(output.approved);
}

#[test]
fn continuing_a_finished_thread_returns_the_final_state() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = approval_graph(saver);
    let output = graph
        .invoke(
            &ApprovalState::default(),
            &RunConfig::new().with_thread("done"),
        )
        .unwrap_err();
    assert!(output.is_interrupt());

    let first = graph.resume("done", json!(1)).unwrap();
    let second = graph.continue_thread("done").unwrap();
    assert_eq!(first, second);
}

#[test]
fn resuming_an_unknown_thread_is_an_error() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = approval_graph(saver);
    let error = graph.resume("missing", json!(1)).unwrap_err();
    assert!(matches!(error, Error::Checkpoint(_)));
}

#[test]
fn checkpointer_is_required_for_state_queries() {
    let graph = StateGraph::<ApprovalState>::new()
        .add_node("a", |_state: &ApprovalState| Ok(NodeOutput::new()))
        .set_entry_point("a")
        .set_finish_point("a")
        .compile()
        .unwrap();

    assert!(matches!(
        graph.get_state("thread"),
        Err(Error::MissingCheckpointer(_))
    ));
    assert!(matches!(
        graph.get_state_history("thread"),
        Err(Error::MissingCheckpointer(_))
    ));
    assert!(matches!(
        graph.resume("thread", json!(1)),
        Err(Error::MissingCheckpointer(_))
    ));
}
