//! Subgraph support tests.

use std::sync::Arc;

use langgraph_rs::{NodeOutput, RunConfig, State, StateGraph, StateSchema, reducers};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct NestedState {
    value: i64,
    log: Vec<String>,
}

impl State for NestedState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("value")
            .with_reducer("log", reducers::append())
    }
}

fn inner_graph() -> langgraph_rs::CompiledGraph<NestedState> {
    StateGraph::<NestedState>::new()
        .add_node("sub_one", |state: &NestedState| {
            Ok(NodeOutput::new()
                .set("value", state.value + 1)
                .set("log", vec!["sub_one".to_string()]))
        })
        .add_node("sub_two", |state: &NestedState| {
            Ok(NodeOutput::new()
                .set("value", state.value + 2)
                .set("log", vec!["sub_two".to_string()]))
        })
        .set_entry_point("sub_one")
        .add_edge("sub_one", "sub_two")
        .set_finish_point("sub_two")
        .compile()
        .unwrap()
}

#[test]
fn subgraph_runs_inside_a_parent_node_and_publishes_its_state() {
    let inner = Arc::new(inner_graph());
    let graph = StateGraph::<NestedState>::new()
        .add_node("pre", |_state: &NestedState| {
            Ok(NodeOutput::new()
                .set("value", 10)
                .set("log", vec!["pre".to_string()]))
        })
        .add_subgraph("inner", inner)
        .add_node("post", |state: &NestedState| {
            Ok(NodeOutput::new()
                .set("value", state.value * 2)
                .set("log", vec!["post".to_string()]))
        })
        .set_entry_point("pre")
        .add_edge("pre", "inner")
        .add_edge("inner", "post")
        .set_finish_point("post")
        .compile()
        .unwrap();

    let outcome = graph
        .run(&NestedState::default(), &RunConfig::new())
        .unwrap();
    // pre: 10, sub_one: 11, sub_two: 13, post: 26
    assert_eq!(outcome.state.value, 26);
    // the subgraph's log is published as a replacement, then `post` appends once
    assert_eq!(outcome.state.log, vec!["pre", "sub_one", "sub_two", "post"]);
    let nodes: Vec<&str> = outcome
        .report
        .executions()
        .iter()
        .map(|execution| execution.node.as_str())
        .collect();
    assert_eq!(nodes, vec!["pre", "inner", "post"]);
}

#[test]
fn a_subgraph_that_changes_nothing_produces_no_writes() {
    let inner = Arc::new(
        StateGraph::<NestedState>::new()
            .add_node("noop", |_state: &NestedState| Ok(NodeOutput::new()))
            .set_entry_point("noop")
            .set_finish_point("noop")
            .compile()
            .unwrap(),
    );
    let graph = StateGraph::<NestedState>::new()
        .add_node("seed", |_state: &NestedState| {
            Ok(NodeOutput::new().set("value", 5))
        })
        .add_subgraph("inner", inner)
        .set_entry_point("seed")
        .add_edge("seed", "inner")
        .set_finish_point("inner")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&NestedState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.value, 5);
    assert!(output.log.is_empty());
}
