//! Checkpoint model parity tests: snapshot tasks/interrupts, update_state, history filters.

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph, StateSchema,
    interrupt, json,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct ModelState {
    value: i64,
}

impl State for ModelState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

fn hitl_graph() -> langgraph_rs::CompiledGraph<ModelState> {
    StateGraph::<ModelState>::new()
        .add_node("ask", |state: &ModelState| {
            let answer = interrupt(json!({ "question": "delta?" }))?;
            Ok(NodeOutput::new().set("value", state.value + answer.as_i64().unwrap_or_default()))
        })
        .add_node("finish", |_state: &ModelState| Ok(NodeOutput::new()))
        .set_entry_point("ask")
        .add_edge("ask", "finish")
        .set_finish_point("finish")
        .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
        .unwrap()
}

fn two_node_graph() -> langgraph_rs::CompiledGraph<ModelState> {
    StateGraph::<ModelState>::new()
        .add_node("a", |state: &ModelState| {
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .add_node("b", |state: &ModelState| {
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
        .unwrap()
}

#[test]
fn snapshots_expose_interrupts_tasks_and_parent_links() {
    let graph = hitl_graph();
    let error = graph
        .run(
            &ModelState::default(),
            &RunConfig::new().with_thread("model-hitl"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("model-hitl").unwrap().unwrap();
    assert!(snapshot.interrupted);
    assert_eq!(snapshot.next, vec!["ask".to_string()]);
    assert_eq!(snapshot.tasks.len(), 1);
    assert_eq!(snapshot.tasks[0].node, "ask");
    assert_eq!(snapshot.interrupts.len(), 1);
    assert_eq!(snapshot.interrupts[0].node, "ask");
    assert_eq!(
        snapshot.interrupts[0].value,
        json!({ "question": "delta?" })
    );
    assert_eq!(
        snapshot
            .interrupt()
            .map(|interrupt| interrupt.node.as_str()),
        Some("ask")
    );

    let resumed = graph.resume("model-hitl", json!(5)).unwrap();
    assert_eq!(resumed.value, 5);
    let last = graph.get_state("model-hitl").unwrap().unwrap();
    assert!(!last.interrupted);
    assert!(last.interrupts.is_empty());
    assert!(last.parent_checkpoint_id.is_some());
}

#[test]
fn update_state_recomputes_next_tasks_from_as_node() {
    let graph = two_node_graph();
    graph
        .run(
            &ModelState::default(),
            &RunConfig::new().with_thread("model-update"),
        )
        .unwrap();

    let snapshot = graph
        .update_state("model-update", [("value", json!(10))], Some("a"))
        .unwrap();
    assert_eq!(snapshot.values.value, 10);
    assert_eq!(snapshot.next, vec!["b".to_string()]);
    assert_eq!(snapshot.tasks.len(), 1);
    assert_eq!(snapshot.tasks[0].node, "b");
    assert_eq!(
        snapshot.metadata.get("source").and_then(|v| v.as_str()),
        Some("update")
    );

    let continued = graph.continue_thread("model-update").unwrap();
    assert_eq!(continued.value, 11);
}

#[test]
fn update_state_supports_end_and_input_as_node() {
    let graph = two_node_graph();
    graph
        .run(
            &ModelState::default(),
            &RunConfig::new().with_thread("model-update-end"),
        )
        .unwrap();

    // `as_node=END` records the update as coming from the end: nothing is scheduled.
    let snapshot = graph
        .update_state(
            "model-update-end",
            [("value", json!(3))],
            Some(langgraph_rs::END),
        )
        .unwrap();
    assert!(snapshot.next.is_empty());

    // `as_node=START` (or `__input__`) replays the entry points.
    let snapshot = graph
        .update_state(
            "model-update-end",
            [("value", json!(4))],
            Some(langgraph_rs::START),
        )
        .unwrap();
    assert_eq!(snapshot.next, vec!["a".to_string()]);
}

#[test]
fn update_state_infers_as_node_in_a_multi_node_graph() {
    let graph = two_node_graph();
    graph
        .run(
            &ModelState::default(),
            &RunConfig::new().with_thread("model-update-infer"),
        )
        .unwrap();

    // No `as_node`: the last writer must be inferred instead of raising ambiguity.
    let snapshot = graph
        .update_state("model-update-infer", [("value", json!(20))], None)
        .unwrap();
    assert_eq!(snapshot.values.value, 20);
    assert!(snapshot.next.is_empty());
}

#[test]
fn update_state_without_as_node_keeps_pending_tasks() {
    let graph = hitl_graph();
    let _ = graph
        .run(
            &ModelState::default(),
            &RunConfig::new().with_thread("model-update-pending"),
        )
        .unwrap_err();
    let snapshot = graph
        .update_state("model-update-pending", [("value", json!(99))], None)
        .unwrap();
    assert_eq!(snapshot.values.value, 99);
    assert_eq!(snapshot.next, vec!["ask".to_string()]);
    let resumed = graph.resume("model-update-pending", json!(1)).unwrap();
    assert_eq!(resumed.value, 100);
}

#[test]
fn history_supports_before_and_limit_filters() {
    let graph = two_node_graph();
    graph
        .run(
            &ModelState::default(),
            &RunConfig::new().with_thread("model-history"),
        )
        .unwrap();
    let full = graph.get_state_history("model-history").unwrap();
    assert!(full.len() >= 3);

    let limited = graph
        .get_state_history_filtered("model-history", None, Some(2))
        .unwrap();
    assert_eq!(limited.len(), 2);
    assert_eq!(limited[0].checkpoint_id, full[0].checkpoint_id);

    let before = full[1].checkpoint_id.clone();
    let filtered = graph
        .get_state_history_filtered("model-history", Some(&before), None)
        .unwrap();
    assert!(
        !filtered
            .iter()
            .any(|snapshot| snapshot.checkpoint_id == before)
    );
    assert_eq!(filtered.len(), full.len() - 2);
}
