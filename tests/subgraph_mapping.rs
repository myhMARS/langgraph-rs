//! Subgraph state mapping, checkpoint namespaces and nested interrupts.

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph, StateSchema,
    interrupt, json,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct ParentState {
    value: i64,
}

impl State for ParentState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct ChildState {
    x: i64,
}

impl State for ChildState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("x")
    }
}

fn child_graph(saver: Option<Arc<InMemorySaver>>) -> langgraph_rs::CompiledGraph<ChildState> {
    let builder = StateGraph::<ChildState>::new()
        .add_node("double", |state: &ChildState| {
            Ok(NodeOutput::new().set("x", state.x * 2))
        })
        .set_entry_point("double")
        .set_finish_point("double");
    match saver {
        Some(saver) => builder
            .compile_with(CompileOptions::new().with_checkpointer(saver))
            .unwrap(),
        None => builder.compile().unwrap(),
    }
}

fn mapped_parent(
    child: Arc<langgraph_rs::CompiledGraph<ChildState>>,
) -> langgraph_rs::CompiledGraph<ParentState> {
    StateGraph::<ParentState>::new()
        .add_node("seed", |_state: &ParentState| {
            Ok(NodeOutput::new().set("value", 5))
        })
        .add_subgraph_mapped(
            "child",
            child,
            |parent: &ParentState| Ok(ChildState { x: parent.value }),
            |child: &ChildState, _parent: &ParentState| Ok(NodeOutput::new().set("value", child.x)),
        )
        .set_entry_point("seed")
        .add_edge("seed", "child")
        .set_finish_point("child")
        .compile()
        .unwrap()
}

#[test]
fn mapped_subgraph_maps_between_state_types() {
    let parent = mapped_parent(Arc::new(child_graph(None)));
    let output = parent
        .invoke(&ParentState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.value, 10);
}

fn interrupt_child(saver: Arc<InMemorySaver>) -> langgraph_rs::CompiledGraph<ChildState> {
    StateGraph::<ChildState>::new()
        .add_node("ask", |state: &ChildState| {
            let delta = interrupt(json!({ "question": "delta?" }))?;
            Ok(NodeOutput::new().set("x", state.x + delta.as_i64().unwrap_or_default()))
        })
        .add_node("finish", |_state: &ChildState| Ok(NodeOutput::new()))
        .set_entry_point("ask")
        .add_edge("ask", "finish")
        .set_finish_point("finish")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap()
}

#[test]
fn nested_interrupt_pauses_parent_and_resumes_child() {
    let saver = Arc::new(InMemorySaver::new());
    let child = Arc::new(interrupt_child(saver.clone()));
    let parent = StateGraph::<ParentState>::new()
        .add_node("seed", |_state: &ParentState| {
            Ok(NodeOutput::new().set("value", 100))
        })
        .add_subgraph_mapped(
            "child",
            child.clone(),
            |parent: &ParentState| Ok(ChildState { x: parent.value }),
            |child: &ChildState, _parent: &ParentState| Ok(NodeOutput::new().set("value", child.x)),
        )
        .set_entry_point("seed")
        .add_edge("seed", "child")
        .set_finish_point("child")
        .compile_with(CompileOptions::new().with_checkpointer(saver.clone()))
        .unwrap();

    let error = parent
        .run(
            &ParentState::default(),
            &RunConfig::new().with_thread("nested"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());
    assert_eq!(
        error.interrupt().map(|interrupt| interrupt.node.as_str()),
        Some("child")
    );
    assert_eq!(
        error.interrupt().map(|interrupt| interrupt.value.clone()),
        Some(json!({ "question": "delta?" }))
    );

    // The child checkpoint lives in a derived namespace, the parent in the root one.
    let child_states = child.subgraph_states("nested").unwrap();
    let child_state = child_states
        .iter()
        .find(|state| state.checkpoint_ns.starts_with("child:"))
        .expect("child namespace checkpoint");
    assert!(child_state.interrupted);
    assert_eq!(child_state.values.x, 100);
    let parent_state = parent.get_state("nested").unwrap().unwrap();
    assert!(parent_state.interrupted);

    let resumed = parent.resume("nested", json!(7)).unwrap();
    assert_eq!(resumed.value, 107);
    assert!(!parent.get_state("nested").unwrap().unwrap().interrupted);
    assert!(
        child
            .subgraph_states("nested")
            .unwrap()
            .iter()
            .all(|state| !state.interrupted)
    );
}

#[test]
fn nested_subgraph_checkpoints_are_isolated_by_namespace() {
    let saver = Arc::new(InMemorySaver::new());
    let child = Arc::new(interrupt_child(saver.clone()));
    let parent = StateGraph::<ParentState>::new()
        .add_node("seed", |_state: &ParentState| {
            Ok(NodeOutput::new().set("value", 1))
        })
        .add_subgraph_mapped(
            "child",
            child.clone(),
            |parent: &ParentState| Ok(ChildState { x: parent.value }),
            |child: &ChildState, _parent: &ParentState| Ok(NodeOutput::new().set("value", child.x)),
        )
        .set_entry_point("seed")
        .add_edge("seed", "child")
        .set_finish_point("child")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let _ = parent
        .run(
            &ParentState::default(),
            &RunConfig::new().with_thread("nested-ns"),
        )
        .unwrap_err();
    // Root history only contains parent checkpoints; subgraph history lives under `child`.
    for snapshot in parent.get_state_history("nested-ns").unwrap() {
        assert!(snapshot.checkpoint_ns.is_empty());
    }
    let child_snapshots = child.subgraph_states("nested-ns").unwrap();
    assert!(!child_snapshots.is_empty());
    assert!(
        child_snapshots
            .iter()
            .all(|snapshot| snapshot.checkpoint_ns.starts_with("child:"))
    );

    parent.resume("nested-ns", json!(2)).unwrap();
    assert_eq!(
        parent.get_state("nested-ns").unwrap().unwrap().values.value,
        3
    );
}
