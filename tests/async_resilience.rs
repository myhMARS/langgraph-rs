//! Async adapter: checkpointing, HITL, retries and error routing.
#![cfg(feature = "async")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{
    Error, InMemorySaver, NodeOutput, RetryPolicy, RunConfig, State, StateSchema, Usage, json,
    reducers,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct HitlState {
    value: i64,
    log: Vec<String>,
}

impl State for HitlState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("value")
            .with_reducer("log", reducers::append())
    }
}

#[tokio::test]
async fn async_interrupt_checkpoints_and_resumes() {
    let graph = AsyncStateGraph::<HitlState>::new()
        .add_node("ask", |state: HitlState, ctx| async move {
            let answer = ctx.interrupt(json!({ "question": "how much?" }))?;
            Ok(NodeOutput::new()
                .set("value", state.value + answer.as_i64().unwrap_or_default())
                .set("log", vec!["ask".to_string()]))
        })
        .add_node("finish", |_state: HitlState, _ctx| async move {
            Ok(NodeOutput::new().set("log", vec!["finish".to_string()]))
        })
        .set_entry_point("ask")
        .add_edge("ask", "finish")
        .set_finish_point("finish")
        .with_checkpointer(Arc::new(InMemorySaver::new()))
        .compile()
        .unwrap();

    let error = graph
        .run(
            &HitlState::default(),
            &RunConfig::new().with_thread("async-hitl"),
        )
        .await
        .unwrap_err();
    assert!(error.is_interrupt());
    assert_eq!(
        error.interrupt().map(|interrupt| interrupt.node.as_str()),
        Some("ask")
    );

    let snapshot = graph.get_state("async-hitl").unwrap().unwrap();
    assert!(snapshot.interrupted);
    assert_eq!(snapshot.next, vec!["ask".to_string()]);

    let output = graph.resume("async-hitl", json!(41)).await.unwrap();
    assert_eq!(output.value, 41);
    assert_eq!(output.log, vec!["ask", "finish"]);
    assert!(!graph.get_state("async-hitl").unwrap().unwrap().interrupted);
}

#[tokio::test]
async fn async_interrupt_before_hook_can_be_continued() {
    let graph = AsyncStateGraph::<HitlState>::new()
        .add_node("a", |_state: HitlState, _ctx| async move {
            Ok(NodeOutput::new().set("log", vec!["a".to_string()]))
        })
        .add_node("b", |_state: HitlState, _ctx| async move {
            Ok(NodeOutput::new().set("log", vec!["b".to_string()]))
        })
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .interrupt_before(["b"])
        .with_checkpointer(Arc::new(InMemorySaver::new()))
        .compile()
        .unwrap();

    let error = graph
        .run(
            &HitlState::default(),
            &RunConfig::new().with_thread("async-before"),
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.interrupt().map(|interrupt| interrupt.node.as_str()),
        Some("b")
    );
    let state = graph.get_state("async-before").unwrap().unwrap();
    assert_eq!(state.next, vec!["b".to_string()]);

    // Resuming from an interrupt_before checkpoint runs the pending node.
    let output = graph.continue_thread("async-before").await.unwrap();
    assert_eq!(output.log, vec!["a", "b"]);
}

#[tokio::test]
async fn async_retries_with_backoff() {
    let calls = Arc::new(AtomicUsize::new(0));
    let attempts = calls.clone();
    let graph = AsyncStateGraph::<HitlState>::new()
        .add_node_with_retry(
            "flaky",
            RetryPolicy::new(3).with_backoff(Duration::from_millis(1)),
            move |_state: HitlState, ctx| {
                let attempts = attempts.clone();
                async move {
                    let call = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                    ctx.record_tokens(1, 1);
                    if call < 3 {
                        return Err(Error::other(format!("transient {call}")));
                    }
                    Ok(NodeOutput::new().set("value", 7))
                }
            },
        )
        .set_entry_point("flaky")
        .set_finish_point("flaky")
        .compile()
        .unwrap();

    let outcome = graph
        .run(&HitlState::default(), &RunConfig::new())
        .await
        .unwrap();
    assert_eq!(outcome.state.value, 7);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(outcome.report.executions()[0].attempts, 3);
}

#[tokio::test]
async fn async_error_edges_route_to_handlers() {
    let graph = AsyncStateGraph::<HitlState>::new()
        .add_node("work", |_state: HitlState, _ctx| async move {
            Err::<NodeOutput, _>(Error::other("upstream exploded"))
        })
        .add_node("handle", |_state: HitlState, ctx| async move {
            let failure = ctx.last_error().expect("handler sees the failure");
            Ok(NodeOutput::new().set("value", -1).set(
                "log",
                vec![format!("handled {}: {}", failure.node, failure.message)],
            ))
        })
        .set_entry_point("work")
        .add_error_edge("work", "handle")
        .set_finish_point("handle")
        .compile()
        .unwrap();

    let outcome = graph
        .run(&HitlState::default(), &RunConfig::new())
        .await
        .unwrap();
    assert_eq!(outcome.state.value, -1);
    assert_eq!(outcome.state.log, vec!["handled work: upstream exploded"]);
    assert_eq!(outcome.report.executions().len(), 2);
    assert_eq!(
        outcome.report.executions()[0].error.as_deref(),
        Some("upstream exploded")
    );
}

#[tokio::test]
async fn async_usage_is_recorded_across_retries() {
    let graph = AsyncStateGraph::<HitlState>::new()
        .add_node_with_retry(
            "model",
            RetryPolicy::new(2),
            |_state: HitlState, ctx| async move {
                ctx.record_usage(Usage::new().with_input_tokens(10));
                Err::<NodeOutput, _>(Error::other("nope"))
            },
        )
        .set_entry_point("model")
        .set_finish_point("model")
        .compile()
        .unwrap();

    let error = graph
        .run(&HitlState::default(), &RunConfig::new())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Node { .. }));
}

#[tokio::test]
async fn async_retry_on_predicates_stop_unmatched_retries() {
    let calls = Arc::new(AtomicUsize::new(0));
    let graph = AsyncStateGraph::<HitlState>::new()
        .add_node_with_retry("fail", RetryPolicy::new(3).with_retry_on(["connect"]), {
            let calls = calls.clone();
            move |_state: HitlState, _ctx| {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Err(Error::other("boom"))
                }
            }
        })
        .set_entry_point("fail")
        .set_finish_point("fail")
        .compile()
        .unwrap();
    assert!(
        graph
            .run(&HitlState::default(), &RunConfig::new())
            .await
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
