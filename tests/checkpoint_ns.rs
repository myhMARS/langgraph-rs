//! Checkpoint namespace and thread-management tests.

use std::sync::Arc;

use langgraph_rs::{
    Checkpointer, CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph,
    StateSchema, interrupt, json,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct NsState {
    value: i64,
}

impl State for NsState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

fn two_node_graph(saver: Arc<InMemorySaver>) -> langgraph_rs::CompiledGraph<NsState> {
    StateGraph::<NsState>::new()
        .add_node("a", |state: &NsState| {
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .add_node("b", |state: &NsState| {
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap()
}

fn hitl_graph(saver: Arc<InMemorySaver>) -> langgraph_rs::CompiledGraph<NsState> {
    StateGraph::<NsState>::new()
        .add_node("ask", |state: &NsState| {
            let answer = interrupt(json!({ "question": "delta?" }))?;
            Ok(NodeOutput::new().set("value", state.value + answer.as_i64().unwrap_or_default()))
        })
        .set_entry_point("ask")
        .set_finish_point("ask")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap()
}

#[test]
fn namespaces_are_isolated_within_a_thread() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = two_node_graph(saver.clone());
    let thread = "ns-thread";

    graph
        .run(
            &NsState::default(),
            &RunConfig::new()
                .with_thread(thread)
                .with_checkpoint_ns("sub/one"),
        )
        .unwrap();
    graph
        .run(
            &NsState { value: 10 },
            &RunConfig::new()
                .with_thread(thread)
                .with_checkpoint_ns("sub/two"),
        )
        .unwrap();

    assert!(graph.get_state_in(thread, "").unwrap().is_none());
    let one = graph.get_state_in(thread, "sub/one").unwrap().unwrap();
    let two = graph.get_state_in(thread, "sub/two").unwrap().unwrap();
    assert_eq!(one.values.value, 2);
    assert_eq!(two.values.value, 12);
    assert_eq!(one.checkpoint_ns, "sub/one");
    // LangGraph layout: raw input, post-input and one checkpoint per superstep.
    assert_eq!(
        graph.get_state_history_in(thread, "sub/one").unwrap().len(),
        4
    );
    assert_eq!(
        graph.get_state_history_in(thread, "sub/two").unwrap().len(),
        4
    );
    assert_eq!(saver.threads().unwrap(), vec![thread.to_string()]);
}

#[test]
fn resume_inside_a_namespace() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = hitl_graph(saver);
    let thread = "ns-resume";

    let error = graph
        .run(
            &NsState::default(),
            &RunConfig::new()
                .with_thread(thread)
                .with_checkpoint_ns("sub/approval"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let resumed = graph.resume_in(thread, "sub/approval", json!(5)).unwrap();
    assert_eq!(resumed.value, 5);
    assert!(graph.get_state_in(thread, "sub/other").unwrap().is_none());
}

#[test]
fn delete_and_copy_threads() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = two_node_graph(saver.clone());
    graph
        .run(&NsState::default(), &RunConfig::new().with_thread("source"))
        .unwrap();

    graph.copy_thread("source", "copy").unwrap();
    let original = graph.get_state("source").unwrap().unwrap();
    let copied = graph.get_state("copy").unwrap().unwrap();
    assert_eq!(original.values, copied.values);
    assert_eq!(copied.thread_id, "copy");
    assert!(copied.parent_checkpoint_id.is_some());

    graph.delete_thread("source").unwrap();
    assert!(graph.get_state("source").unwrap().is_none());
    assert!(graph.get_state("copy").unwrap().is_some());

    graph.delete_thread("copy").unwrap();
    assert!(saver.threads().unwrap().is_empty());
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_graphs_support_namespaces() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let graph = AsyncStateGraph::<NsState>::new()
        .add_node("work", |state: NsState, _ctx| async move {
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .set_entry_point("work")
        .set_finish_point("work")
        .with_checkpointer(Arc::new(InMemorySaver::new()))
        .compile()
        .unwrap();

    graph
        .run(
            &NsState::default(),
            &RunConfig::new()
                .with_thread("async-ns")
                .with_checkpoint_ns("sub/async"),
        )
        .await
        .unwrap();
    let snapshot = graph
        .get_state_in("async-ns", "sub/async")
        .unwrap()
        .unwrap();
    assert_eq!(snapshot.values.value, 1);
    assert!(graph.get_state_in("async-ns", "").unwrap().is_none());
}

#[test]
fn subgraph_states_returns_each_namespace() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = two_node_graph(saver);
    let thread = "ns-subgraphs";

    for namespace in ["", "sub/one", "sub/two"] {
        graph
            .run(
                &NsState::default(),
                &RunConfig::new()
                    .with_thread(thread)
                    .with_checkpoint_ns(namespace),
            )
            .unwrap();
    }

    let (root, subgraphs) = graph
        .get_state_subgraphs(thread)
        .unwrap()
        .expect("root state exists");
    assert_eq!(root.checkpoint_ns, "");
    assert_eq!(subgraphs.len(), 2);
    let namespaces: Vec<&str> = subgraphs
        .iter()
        .map(|snapshot| snapshot.checkpoint_ns.as_str())
        .collect();
    assert_eq!(namespaces, vec!["sub/one", "sub/two"]);
}
