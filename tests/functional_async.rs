//! Async functional API: concurrent and memoized tasks.

#![cfg(feature = "async")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use langgraph_rs::JsonState;
use langgraph_rs::functional::{async_entrypoint, async_task};
use langgraph_rs::{RunConfig, json};

#[tokio::test]
async fn async_entrypoint_runs_independent_tasks_concurrently() {
    let calls = Arc::new(AtomicUsize::new(0));

    let first = async_task("first", {
        let calls = calls.clone();
        move |input: i64| {
            let calls = calls.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(120)).await;
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(input + 1)
            }
        }
    });
    let second = async_task("second", {
        let calls = calls.clone();
        move |input: i64| {
            let calls = calls.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(120)).await;
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(input + 2)
            }
        }
    });

    let graph = async_entrypoint(move |_state: JsonState| {
        let first = first.clone();
        let second = second.clone();
        async move {
            let (left, right) = tokio::join!(first.call(1), second.call(2));
            Ok(serde_json::from_value(json!({
                "value": left? + right?,
            }))?)
        }
    })
    .compile()
    .unwrap();

    let started = Instant::now();
    let outcome = graph
        .run(&JsonState::new(), &RunConfig::new())
        .await
        .unwrap();
    assert_eq!(outcome.state.get("value"), Some(&json!(6)));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(
        started.elapsed() < Duration::from_millis(220),
        "tasks did not overlap: {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn async_tasks_are_memoized_within_a_run() {
    let calls = Arc::new(AtomicUsize::new(0));
    let task = async_task("memo", {
        let calls = calls.clone();
        move |input: i64| {
            let calls = calls.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(input * 10)
            }
        }
    });

    let graph = async_entrypoint(move |_state: JsonState| {
        let task = task.clone();
        async move {
            let first = task.call(3).await?;
            let second = task.call(3).await?;
            Ok(serde_json::from_value(json!({"value": first + second}))?)
        }
    })
    .compile()
    .unwrap();

    let outcome = graph
        .run(&JsonState::new(), &RunConfig::new())
        .await
        .unwrap();
    assert_eq!(outcome.state.get("value"), Some(&json!(60)));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
