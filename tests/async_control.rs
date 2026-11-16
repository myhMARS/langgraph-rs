//! Async RunControl cancellation.

#![cfg(feature = "async")]

use std::time::{Duration, Instant};

use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{JsonState, NodeOutput, RunConfig, RunControl, StateSchema, json};

#[tokio::test]
async fn abort_cancels_a_running_async_node() {
    let graph = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("slow", |_state: JsonState, _ctx| async move {
            tokio::time::sleep(Duration::from_millis(400)).await;
            Ok(NodeOutput::new().update("value", json!(1)))
        })
        .set_entry_point("slow")
        .set_finish_point("slow")
        .compile()
        .unwrap();

    let control = RunControl::new();
    let trigger = control.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        trigger.abort();
    });

    let started = Instant::now();
    let error = graph
        .run(&JsonState::new(), &RunConfig::new().with_control(control))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("RunControl"));
    assert!(
        started.elapsed() < Duration::from_millis(300),
        "node future was not cancelled promptly: {:?}",
        started.elapsed()
    );
}
