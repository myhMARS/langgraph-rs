//! Async subgraphs and async node caching.

#![cfg(feature = "async")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{
    CachePolicy, InMemoryCache, InMemorySaver, NodeOutput, RunConfig, State, StateSchema, json,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct AsyncState {
    value: i64,
}

impl State for AsyncState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

#[tokio::test]
async fn async_subgraph_nested_interrupt_and_resume() {
    let saver = Arc::new(InMemorySaver::new());
    let child =
        Arc::new(
            AsyncStateGraph::<AsyncState>::new()
                .add_node("ask", |state: AsyncState, ctx| async move {
                    let delta = ctx.interrupt(json!({ "question": "delta" }))?;
                    Ok(NodeOutput::new()
                        .set("value", state.value + delta.as_i64().unwrap_or_default()))
                })
                .set_entry_point("ask")
                .set_finish_point("ask")
                .with_checkpointer(saver.clone())
                .compile()
                .unwrap(),
        );
    let parent = AsyncStateGraph::<AsyncState>::new()
        .add_node("seed", |_state: AsyncState, _ctx| async move {
            Ok(NodeOutput::new().set("value", 10))
        })
        .add_subgraph("child", child.clone())
        .set_entry_point("seed")
        .add_edge("seed", "child")
        .set_finish_point("child")
        .with_checkpointer(saver)
        .compile()
        .unwrap();

    let error = parent
        .run(
            &AsyncState::default(),
            &RunConfig::new().with_thread("async-subgraph"),
        )
        .await
        .unwrap_err();
    assert!(error.is_interrupt());
    assert_eq!(
        error.interrupt().map(|interrupt| interrupt.node.as_str()),
        Some("child")
    );

    let child_state = parent
        .get_state_in("async-subgraph", "child")
        .unwrap()
        .unwrap();
    assert!(child_state.interrupted);
    assert_eq!(child_state.values.value, 10);

    let output = parent.resume("async-subgraph", json!(5)).await.unwrap();
    assert_eq!(output.value, 15);
    assert!(
        !parent
            .get_state_in("async-subgraph", "child")
            .unwrap()
            .unwrap()
            .interrupted
    );
}

#[tokio::test]
async fn async_node_cache_avoids_repeated_execution() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let graph = AsyncStateGraph::<AsyncState>::new()
        .add_node("work", move |state: AsyncState, _ctx| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(NodeOutput::new().set("value", state.value + 1))
            }
        })
        .set_entry_point("work")
        .set_finish_point("work")
        .with_cache(Arc::new(InMemoryCache::new()))
        .set_cache("work", CachePolicy::new().with_ttl(Duration::from_secs(60)))
        .compile()
        .unwrap();

    let first = graph
        .run(&AsyncState::default(), &RunConfig::new())
        .await
        .unwrap();
    let second = graph
        .run(&AsyncState::default(), &RunConfig::new())
        .await
        .unwrap();
    assert_eq!(first.state.value, 1);
    assert_eq!(second.state.value, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(second.report.executions.iter().any(|node| node.cached));
}

#[tokio::test]
async fn async_subgraph_without_checkpointer_still_runs() {
    let child = Arc::new(
        AsyncStateGraph::<AsyncState>::new()
            .add_node("double", |state: AsyncState, _ctx| async move {
                Ok(NodeOutput::new().set("value", state.value * 2))
            })
            .set_entry_point("double")
            .set_finish_point("double")
            .compile()
            .unwrap(),
    );
    let parent = AsyncStateGraph::<AsyncState>::new()
        .add_node("seed", |_state: AsyncState, _ctx| async move {
            Ok(NodeOutput::new().set("value", 21))
        })
        .add_subgraph("child", child)
        .set_entry_point("seed")
        .add_edge("seed", "child")
        .set_finish_point("child")
        .compile()
        .unwrap();

    let output = parent.invoke(&AsyncState::default()).await.unwrap();
    assert_eq!(output.value, 42);
}

struct FailingCache;

impl langgraph_rs::NodeCache for FailingCache {
    fn get(&self, _key: &str) -> langgraph_rs::Result<Option<langgraph_rs::CachedOutput>> {
        Err(langgraph_rs::Error::other("cache get failed"))
    }

    fn put(
        &self,
        _key: &str,
        _output: langgraph_rs::CachedOutput,
        _ttl: Option<Duration>,
    ) -> langgraph_rs::Result<()> {
        Err(langgraph_rs::Error::other("cache put failed"))
    }

    fn clear(&self) -> langgraph_rs::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn async_cache_lookup_errors_fail_the_node() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let graph = AsyncStateGraph::<AsyncState>::new()
        .add_node("work", move |state: AsyncState, _ctx| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(NodeOutput::new().set("value", state.value + 1))
            }
        })
        .set_entry_point("work")
        .set_finish_point("work")
        .with_cache(Arc::new(FailingCache))
        .set_cache("work", CachePolicy::new())
        .compile()
        .unwrap();

    let error = graph
        .run(&AsyncState::default(), &RunConfig::new())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("cache"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
