//! Retry policies and error routing tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use langgraph_rs::{
    Error, NodeOutput, RetryPolicy, RunConfig, State, StateGraph, StateSchema, interrupt,
    last_error,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct WorkState {
    value: i64,
    error_message: String,
}

impl State for WorkState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("value")
            .last_value("error_message")
    }
}

#[test]
fn retry_succeeds_after_transient_failures() {
    let calls = Arc::new(AtomicUsize::new(0));
    let attempts = calls.clone();

    let graph = StateGraph::<WorkState>::new()
        .add_node_with_retry(
            "flaky",
            RetryPolicy::new(3).with_backoff(Duration::from_millis(1)),
            move |_state: &WorkState| {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                if attempt < 3 {
                    return Err(Error::other(format!("transient failure {attempt}")));
                }
                Ok(NodeOutput::new().set("value", 42))
            },
        )
        .set_entry_point("flaky")
        .set_finish_point("flaky")
        .compile()
        .unwrap();

    let outcome = graph.run(&WorkState::default(), &RunConfig::new()).unwrap();
    assert_eq!(outcome.state.value, 42);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(outcome.report.executions().len(), 1);
    assert_eq!(outcome.report.executions()[0].attempts, 3);
    assert!(outcome.report.executions()[0].error.is_none());
}

#[test]
fn exhausted_retries_fail_with_the_attempt_count() {
    let graph = StateGraph::<WorkState>::new()
        .add_node_with_retry("broken", RetryPolicy::new(2), |_state: &WorkState| {
            Err(Error::other("still broken"))
        })
        .set_entry_point("broken")
        .set_finish_point("broken")
        .compile()
        .unwrap();

    let error = graph
        .invoke(&WorkState::default(), &RunConfig::new())
        .unwrap_err();
    match error {
        Error::Node { node, message } => {
            assert_eq!(node, "broken");
            assert!(message.contains("still broken"));
            assert!(message.contains("failed after 2 attempt(s)"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn error_edges_route_failures_to_a_handler() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("work", |_state: &WorkState| {
            Err(Error::other("upstream exploded"))
        })
        .add_node("handle", |_state: &WorkState| {
            let failure = last_error().expect("handler sees the failure");
            Ok(NodeOutput::new()
                .set(
                    "error_message",
                    format!("{} failed: {}", failure.node, failure.message),
                )
                .set("value", -1))
        })
        .set_entry_point("work")
        .add_error_edge("work", "handle")
        .set_finish_point("handle")
        .compile()
        .unwrap();

    let outcome = graph.run(&WorkState::default(), &RunConfig::new()).unwrap();
    assert_eq!(outcome.state.value, -1);
    assert_eq!(
        outcome.state.error_message,
        "work failed: upstream exploded"
    );
    assert_eq!(outcome.report.executions().len(), 2);
    assert_eq!(outcome.report.executions()[0].node, "work");
    assert_eq!(
        outcome.report.executions()[0].error.as_deref(),
        Some("upstream exploded")
    );
    assert_eq!(outcome.report.executions()[1].node, "handle");
}

#[test]
fn interrupts_are_not_retried() {
    let calls = Arc::new(AtomicUsize::new(0));
    let attempts = calls.clone();

    let graph = StateGraph::<WorkState>::new()
        .add_node_with_retry("ask", RetryPolicy::new(5), move |_state: &WorkState| {
            attempts.fetch_add(1, Ordering::SeqCst);
            let answer = interrupt(json!({ "question": "value?" }))?;
            Ok(NodeOutput::new().set("value", answer.as_i64().unwrap_or_default()))
        })
        .set_entry_point("ask")
        .set_finish_point("ask")
        .compile_with(
            langgraph_rs::CompileOptions::new()
                .with_checkpointer(Arc::new(langgraph_rs::InMemorySaver::new())),
        )
        .unwrap();

    let error = graph
        .invoke(
            &WorkState::default(),
            &RunConfig::new().with_thread("retry-interrupt"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let output = graph.resume("retry-interrupt", json!(7)).unwrap();
    assert_eq!(output.value, 7);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn retry_policy_backoff_is_exponential_and_capped() {
    let policy = RetryPolicy::new(4)
        .with_backoff(Duration::from_millis(10))
        .with_multiplier(2.0)
        .with_max_backoff(Duration::from_millis(50));
    assert_eq!(policy.backoff_for(1), Duration::from_millis(10));
    assert_eq!(policy.backoff_for(2), Duration::from_millis(20));
    assert_eq!(policy.backoff_for(3), Duration::from_millis(40));
    assert_eq!(policy.backoff_for(4), Duration::from_millis(50));
    assert!(!RetryPolicy::new(1).is_retrying());
    assert!(RetryPolicy::new(3).is_retrying());
}
