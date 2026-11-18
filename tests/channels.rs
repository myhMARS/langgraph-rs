//! Channel semantics beyond last-value/topic/reducer: AnyValue, EphemeralValue,
//! UntrackedValue, NamedBarrierValue and explicit overwrites.

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph, StateSchema, json,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct BarrierState {
    barrier: bool,
    seen: i64,
}

impl State for BarrierState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .named_barrier("barrier", ["a", "b"])
            .last_value("seen")
    }
}

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct PulseState {
    pulse: i64,
    seen: i64,
}

impl State for PulseState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .ephemeral_value("pulse")
            .last_value("seen")
    }
}

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct SecretState {
    secret: i64,
    seen: i64,
}

impl State for SecretState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .untracked_value("secret")
            .last_value("seen")
    }
}

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct AnyState {
    value: i64,
}

impl State for AnyState {
    fn schema() -> StateSchema {
        StateSchema::new().any_value("value")
    }
}

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct LogState {
    log: Vec<String>,
}

impl State for LogState {
    fn schema() -> StateSchema {
        StateSchema::new().topic("log")
    }
}

#[test]
fn any_value_allows_concurrent_writes() {
    let graph = StateGraph::<AnyState>::new()
        .add_node("a", |_state: &AnyState| {
            Ok(NodeOutput::new().set("value", 1))
        })
        .add_node("b", |_state: &AnyState| {
            Ok(NodeOutput::new().set("value", 2))
        })
        .add_edge(langgraph_rs::START, "a")
        .add_edge(langgraph_rs::START, "b")
        .set_finish_point("a")
        .set_finish_point("b")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&AnyState::default(), &RunConfig::new())
        .unwrap();
    assert!(output.value == 1 || output.value == 2);
}

#[test]
fn ephemeral_value_is_cleared_before_the_next_step() {
    let graph = StateGraph::<PulseState>::new()
        .add_node("write", |_state: &PulseState| {
            Ok(NodeOutput::new().set("pulse", 42).set("seen", -1))
        })
        .add_node("read", |state: &PulseState| {
            Ok(NodeOutput::new().set("seen", state.pulse))
        })
        .set_entry_point("write")
        .add_edge("write", "read")
        .set_finish_point("read")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&PulseState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(
        output.seen, 0,
        "the read node must not see the ephemeral write"
    );
    assert_eq!(
        output.pulse, 0,
        "the ephemeral value is cleared after the step"
    );
}

#[test]
fn untracked_value_is_visible_in_run_but_not_checkpointed() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<SecretState>::new()
        .add_node("write", |_state: &SecretState| {
            Ok(NodeOutput::new().set("secret", 9))
        })
        .add_node("read", |state: &SecretState| {
            Ok(NodeOutput::new().set("seen", state.secret))
        })
        .set_entry_point("write")
        .add_edge("write", "read")
        .set_finish_point("read")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let output = graph
        .run(
            &SecretState::default(),
            &RunConfig::new().with_thread("untracked"),
        )
        .unwrap()
        .state;
    assert_eq!(
        output.seen, 9,
        "untracked values stay visible inside the run"
    );

    for snapshot in graph.get_state_history("untracked").unwrap() {
        assert_eq!(
            snapshot.values.secret, 0,
            "untracked channels must not be written to checkpoints"
        );
    }
}

#[test]
fn named_barrier_opens_after_every_writer() {
    let graph = StateGraph::<BarrierState>::new()
        .add_node("a", |_state: &BarrierState| {
            Ok(NodeOutput::new().set("barrier", "a"))
        })
        .add_node("b", |_state: &BarrierState| {
            Ok(NodeOutput::new().set("barrier", "b"))
        })
        .add_node("join", |state: &BarrierState| {
            Ok(NodeOutput::new().set("seen", i64::from(state.barrier)))
        })
        .add_edge(langgraph_rs::START, "a")
        .add_edge(langgraph_rs::START, "b")
        .add_edges_from(["a", "b"], "join")
        .set_finish_point("join")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&BarrierState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.seen, 1);
}

#[test]
fn named_barrier_is_consumed_after_use() {
    let graph = StateGraph::<BarrierState>::new()
        .add_node("a", |_state: &BarrierState| {
            Ok(NodeOutput::new().set("barrier", "a"))
        })
        .add_node("b", |_state: &BarrierState| {
            Ok(NodeOutput::new().set("barrier", "b"))
        })
        .add_node("join", |state: &BarrierState| {
            Ok(NodeOutput::new().set("seen", i64::from(state.barrier)))
        })
        .add_edge(langgraph_rs::START, "a")
        .add_edge(langgraph_rs::START, "b")
        .add_edges_from(["a", "b"], "join")
        .set_finish_point("join")
        .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
        .unwrap();

    let output = graph
        .invoke(
            &BarrierState::default(),
            &RunConfig::new().with_thread("barrier-consume"),
        )
        .unwrap();
    // The join node still observes the open barrier ...
    assert_eq!(output.seen, 1);
    // ... but the consumed barrier does not stay open in the checkpoint.
    let snapshot = graph.get_state("barrier-consume").unwrap().unwrap();
    assert!(!snapshot.values.barrier);
}

#[test]
fn overwrite_bypasses_the_reducer() {
    let graph = StateGraph::<LogState>::new()
        .add_node("rewrite", |_state: &LogState| {
            Ok(NodeOutput::new().overwrite("log", json!(["x"])))
        })
        .set_entry_point("rewrite")
        .set_finish_point("rewrite")
        .compile()
        .unwrap();

    let input = LogState {
        log: vec!["old".to_string()],
    };
    let output = graph.invoke(&input, &RunConfig::new()).unwrap();
    assert_eq!(output.log, vec!["x"]);
}
