//! Async twin of `mega_workflow_native.rs`: same 15-node workflow, same phases, but built
//! with [`langgraph_rs::tokio_runtime::AsyncStateGraph`] and awaited on Tokio.
//!
//! ```text
//! cargo run --release --example mega_workflow_async_native --features async -- 10
//! ```
//!
//! Prints `FINAL ...` and one `PHASE=<name>:<ns>` line per phase, which
//! `examples/mega_workflow/bench_async.py` parses.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use langgraph_rs::tokio_runtime::{AsyncContext, AsyncStateGraph};
use langgraph_rs::{
    CachePolicy, InMemoryCache, InMemorySaver, InMemoryStore, JsonState, NodeOutput, RetryPolicy,
    Route, RunConfig, Send, StateGraph, StateSchema, json, reducers,
};

fn schema() -> StateSchema {
    StateSchema::dynamic()
        .with_reducer("log", reducers::append())
        .with_reducer("sink", reducers::add())
}

fn input() -> JsonState {
    JsonState::new()
        .with("count", json!(0))
        .with("log", json!([]))
        .with("sink", json!(0))
}

fn log(value: &str) -> NodeOutput {
    NodeOutput::new().update("log", json!([value]))
}

fn build_child() -> Arc<langgraph_rs::CompiledGraph<JsonState>> {
    let child = StateGraph::<JsonState>::with_schema(
        StateSchema::dynamic().with_reducer("log", reducers::append()),
    )
    .add_node("child", |state: &JsonState| {
        let value = state.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
        Ok(NodeOutput::new()
            .set("value", json!(value + 1))
            .update("log", json!(["child"])))
    })
    .set_entry_point("child")
    .set_finish_point("child")
    .compile()
    .expect("child graph compiles");
    Arc::new(child)
}

struct Counters {
    retry: AtomicUsize,
    boom: AtomicUsize,
    cached: AtomicUsize,
}

fn build(counters: Arc<Counters>) -> langgraph_rs::tokio_runtime::AsyncCompiledGraph<JsonState> {
    let child = build_child();
    let retry_counter = Arc::clone(&counters);
    let boom_counter = Arc::clone(&counters);
    let cached_counter = Arc::clone(&counters);

    AsyncStateGraph::<JsonState>::with_schema(schema())
        .add_node("seed", |_state: JsonState, context: AsyncContext| async move {
            let store = context.store().expect("store installed");
            let namespace = vec!["mega".to_string()];
            let previous = store.get(&namespace, "runs")?;
            let runs = previous
                .as_ref()
                .and_then(|value| value.get("n"))
                .and_then(|value| value.as_i64())
                .unwrap_or(0);
            store.put(&namespace, "runs", json!({ "n": runs + 1 }))?;
            Ok(NodeOutput::new()
                .update("log", json!(["seed"]))
                .set("count", json!(0))
                .set("sink", json!(0))
                .set("child", json!(0))
                .set("flag", json!(format!("runs={}", runs + 1))))
        })
        .add_node("grow", |state: JsonState, _context: AsyncContext| async move {
            let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(NodeOutput::new()
                .set("count", json!(count + 1))
                .update("log", json!(["grow"])))
        })
        .add_conditional_edges("grow", |state: &JsonState| {
            let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            if count < 3 {
                Ok(Route::node("grow"))
            } else {
                Ok(Route::sends([
                    Send::with_arg("worker", json!({ "n": 1 })),
                    Send::with_arg("worker", json!({ "n": 2 })),
                    Send::with_arg("worker", json!({ "n": 3 })),
                ]))
            }
        })
        .add_node("worker", |state: JsonState, _context: AsyncContext| async move {
            let n = state.get("n").and_then(|v| v.as_i64()).unwrap_or(1);
            Ok(NodeOutput::new()
                .update("log", json!(["worker"]))
                .set("sink", json!(n)))
        })
        .add_node("join", |_state: JsonState, _context: AsyncContext| async move {
            Ok(log("join"))
        })
        .add_node("cached", move |_state: JsonState, _context: AsyncContext| {
            let counter = Arc::clone(&cached_counter);
            async move {
                counter.cached.fetch_add(1, Ordering::SeqCst);
                Ok(log("cached"))
            }
        })
        .set_cache("cached", CachePolicy::new())
        .with_cache(Arc::new(InMemoryCache::new()))
        .add_node_with_retry(
            "retry",
            RetryPolicy::new(3).with_backoff(std::time::Duration::ZERO),
            move |_state: JsonState, _context: AsyncContext| {
                let counter = Arc::clone(&retry_counter);
                async move {
                    let attempt = counter.retry.fetch_add(1, Ordering::SeqCst) + 1;
                    if attempt % 2 == 1 {
                        return Err(langgraph_rs::errors::Error::other("transient failure"));
                    }
                    Ok(log("retry"))
                }
            },
        )
        .add_node("boom", move |_state: JsonState, _context: AsyncContext| {
            let counter = Arc::clone(&boom_counter);
            async move {
                counter.boom.fetch_add(1, Ordering::SeqCst);
                Err(langgraph_rs::errors::Error::other("always fails"))
            }
        })
        .add_error_edge("boom", "handled")
        .add_node("handled", |_state: JsonState, _context: AsyncContext| async move {
            Ok(log("handled"))
        })
        .add_node_deferred("defer", |_state: JsonState, _context: AsyncContext| async move {
            Ok(log("deferred"))
        })
        .add_node("sub", move |state: JsonState, _context: AsyncContext| {
            let child = Arc::clone(&child);
            async move {
                let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                let child_state = JsonState::new()
                    .with("value", json!(count))
                    .with("log", json!([]));
                let result = child.invoke(&child_state, &RunConfig::new())?;
                let value = result.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(NodeOutput::new()
                    .set("child", json!(value))
                    .update("log", json!(["sub"])))
            }
        })
        .add_node("ask", |_state: JsonState, context: AsyncContext| async move {
            let approval = context.interrupt(json!("approve?"))?;
            Ok(NodeOutput::new()
                .update("log", json!(["ask"]))
                .set("flag", json!(format!("approved={approval}"))))
        })
        .add_node("done", |_state: JsonState, _context: AsyncContext| async move {
            Ok(log("done"))
        })
        .add_edge(langgraph_rs::START, "seed")
        .add_edge("seed", "grow")
        .add_edge("worker", "join")
        .add_edge("join", "cached")
        .add_edge("cached", "retry")
        .add_edge("retry", "sub")
        .add_edge("sub", "ask")
        .add_edge("ask", "done")
        .add_edge("done", "defer")
        .add_edge("done", "boom")
        .with_checkpointer(Arc::new(InMemorySaver::new()))
        .with_store(Arc::new(InMemoryStore::new()))
        .compile()
        .expect("mega async graph compiles")
}

#[tokio::main]
async fn main() {
    let iterations: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);

    let counters = Arc::new(Counters {
        retry: AtomicUsize::new(0),
        boom: AtomicUsize::new(0),
        cached: AtomicUsize::new(0),
    });
    let graph = build(Arc::clone(&counters));
    let mut phases: HashMap<&str, Vec<u128>> = HashMap::new();

    for iteration in 0..iterations {
        let thread = format!("async-native-{iteration}");

        let started = Instant::now();
        let paused = graph
            .run(&input(), &RunConfig::new().with_thread(&thread))
            .await;
        assert!(paused.as_ref().is_err_and(|error| error.is_interrupt()));
        phases.entry("invoke").or_default().push(started.elapsed().as_nanos());

        let started = Instant::now();
        let final_state = graph
            .resume(&thread, json!("go"))
            .await
            .expect("resume completes the run");
        phases.entry("resume").or_default().push(started.elapsed().as_nanos());

        let started = Instant::now();
        let snapshot = graph.get_state(&thread).expect("state").expect("snapshot");
        let history = graph
            .get_state_history_filtered(&thread, None, Some(5))
            .expect("history");
        std::hint::black_box((snapshot.step, history.len()));
        phases.entry("state").or_default().push(started.elapsed().as_nanos());

        let started = Instant::now();
        graph
            .update_state(&thread, [("count", json!(99))], Some("done"))
            .await
            .expect("update_state");
        phases.entry("update_state").or_default().push(started.elapsed().as_nanos());

        let started = Instant::now();
        match graph
            .stream(
                &input(),
                &RunConfig::new().with_thread(format!("{thread}-stream")),
            )
            .await
        {
            Ok(chunks) => {
                std::hint::black_box(chunks.len());
            }
            Err(error) if error.is_interrupt() => {
                std::hint::black_box(0usize);
            }
            Err(error) => panic!("stream failed: {error}"),
        }
        phases.entry("stream").or_default().push(started.elapsed().as_nanos());

        let started = Instant::now();
        let batched = graph
            .batch(
                [input(), input()],
                &RunConfig::new().with_thread(format!("{thread}-batch")),
            )
            .await;
        std::hint::black_box(batched.len());
        phases.entry("batch").or_default().push(started.elapsed().as_nanos());

        if iteration == 0 {
            let count = final_state.get("count").and_then(|v| v.as_i64()).unwrap_or(-1);
            let sink = final_state.get("sink").and_then(|v| v.as_i64()).unwrap_or(-1);
            let child = final_state.get("child").and_then(|v| v.as_i64()).unwrap_or(-1);
            let entries = final_state
                .get("log")
                .and_then(|value| value.as_array())
                .map(|values| values.len())
                .unwrap_or(0);
            println!("FINAL count={count} sink={sink} child={child} log={entries}");
            println!(
                "COUNTERS cached={} retry={} boom={}",
                counters.cached.load(Ordering::SeqCst),
                counters.retry.load(Ordering::SeqCst),
                counters.boom.load(Ordering::SeqCst)
            );
        }
    }

    for (phase, mut samples) in phases {
        samples.sort_unstable();
        println!("PHASE={}:{}", phase, samples[samples.len() / 2]);
    }
}