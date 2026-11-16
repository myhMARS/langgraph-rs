//! Managed values (`remaining_steps` / `is_last_step`) and per-node timeout policies.

use std::time::Duration;

use langgraph_rs::{
    NodeOutput, RunConfig, State, StateGraph, StateSchema, TimeoutPolicy, is_last_step, reducers,
    remaining_steps,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct StepsState {
    value: i64,
    log: Vec<i64>,
}

impl State for StepsState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("value")
            .with_reducer("log", reducers::append())
    }
}

#[test]
fn remaining_steps_counts_down_across_supersteps() {
    let graph = StateGraph::<StepsState>::new()
        .add_node("a", |_state: &StepsState| {
            Ok(NodeOutput::new().set("log", vec![remaining_steps().unwrap_or(0) as i64]))
        })
        .add_node("b", |_state: &StepsState| {
            Ok(NodeOutput::new().set("log", vec![remaining_steps().unwrap_or(0) as i64]))
        })
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile_with(langgraph_rs::CompileOptions::new().with_recursion_limit(3))
        .unwrap();

    let output = graph
        .invoke(&StepsState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.log, vec![3, 2]);
}

#[test]
fn is_last_step_is_true_on_the_final_superstep() {
    let graph = StateGraph::<StepsState>::new()
        .add_node("only", |_state: &StepsState| {
            Ok(NodeOutput::new().set("value", i64::from(is_last_step())))
        })
        .set_entry_point("only")
        .set_finish_point("only")
        .compile_with(langgraph_rs::CompileOptions::new().with_recursion_limit(1))
        .unwrap();

    let output = graph
        .invoke(&StepsState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.value, 1);
}

#[test]
fn run_timeout_fails_a_slow_node() {
    let graph = StateGraph::<StepsState>::new()
        .add_node_with_timeout(
            "slow",
            TimeoutPolicy::new().with_run_timeout(Duration::from_millis(20)),
            |_state: &StepsState| {
                std::thread::sleep(Duration::from_millis(250));
                Ok(NodeOutput::new().set("value", 1))
            },
        )
        .set_entry_point("slow")
        .set_finish_point("slow")
        .compile()
        .unwrap();

    let error = graph
        .run(&StepsState::default(), &RunConfig::new())
        .unwrap_err()
        .to_string();
    assert!(error.contains("timed out"), "unexpected error: {error}");
}

#[test]
fn run_timeout_lets_fast_nodes_finish() {
    let graph = StateGraph::<StepsState>::new()
        .add_node_with_timeout(
            "fast",
            TimeoutPolicy::new().with_run_timeout(Duration::from_secs(5)),
            |_state: &StepsState| Ok(NodeOutput::new().set("value", 9)),
        )
        .set_entry_point("fast")
        .set_finish_point("fast")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&StepsState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.value, 9);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_run_timeout_fails_a_slow_node() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let graph = AsyncStateGraph::<StepsState>::new()
        .add_node("slow", |_state: StepsState, _ctx| async move {
            tokio::time::sleep(Duration::from_millis(250)).await;
            Ok(NodeOutput::new().set("value", 1))
        })
        .set_entry_point("slow")
        .set_finish_point("slow")
        .set_timeout(
            "slow",
            TimeoutPolicy::new().with_run_timeout(Duration::from_millis(20)),
        )
        .compile()
        .unwrap();

    let error = graph
        .run(&StepsState::default(), &RunConfig::new())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("timed out"), "unexpected error: {error}");
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_fast_nodes_finish_under_timeout() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let graph = AsyncStateGraph::<StepsState>::new()
        .add_node("fast", |_state: StepsState, _ctx| async move {
            Ok(NodeOutput::new().set("value", 4))
        })
        .set_entry_point("fast")
        .set_finish_point("fast")
        .set_timeout(
            "fast",
            TimeoutPolicy::new().with_run_timeout(Duration::from_secs(5)),
        )
        .compile()
        .unwrap();

    let output = graph
        .run(&StepsState::default(), &RunConfig::new())
        .await
        .unwrap()
        .state;
    assert_eq!(output.value, 4);
}
