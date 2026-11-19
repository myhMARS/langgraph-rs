//! Mega workflow (pure Rust) mirroring `examples/mega_workflow/build_lg.py` and
//! `build_rs.py`: same nodes, same order, same work per suite.
//!
//! ```text
//! seed -> grow (loop x3) -> Send(worker) x3 -> join -> cached -> retry
//!      -> sub (embedded child graph) -> ask (interrupt) -> done
//!      -> defer (deferred, terminal) and -> boom (always fails, error handler)
//! ```
//!
//! Run it with `cargo run --release --example mega_workflow`. It prints the number of
//! nanoseconds for each phase and a machine-readable `PHASE=<name>:<ns>` line pair for the
//! Python driver in `examples/mega_workflow/bench.py`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use langgraph_rs::{
    CachePolicy, CompileOptions, Durability, InMemoryCache, InMemorySaver, InMemoryStore,
    JsonState, NodeOutput, RetryPolicy, Route, RunConfig, Send, StateGraph, StateSchema,
    interrupt, json, reducers,
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

/// Child graph invoked by the `sub` node with an explicit input/output mapping.
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
    child: AtomicUsize,
}

fn build(counters: Arc<Counters>) -> langgraph_rs::CompiledGraph<JsonState> {
    let child = build_child();

    let retry_counter = Arc::clone(&counters);
    let boom_counter = Arc::clone(&counters);
    let cached_counter = Arc::clone(&counters);

    StateGraph::<JsonState>::with_schema(schema())
        .add_node("seed", |_state: &JsonState| {
            let store = langgraph_rs::get_store().expect("store installed");
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
        .add_node("grow", |state: &JsonState| {
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
        .add_node("worker", |state: &JsonState| {
            let n = state.get("n").and_then(|v| v.as_i64()).unwrap_or(1);
            Ok(NodeOutput::new()
                .update("log", json!(["worker"]))
                .set("sink", json!(n)))
        })
        .add_node("join", |_state: &JsonState| Ok(log("join")))
        .add_node("cached", move |_state: &JsonState| {
            cached_counter.cached.fetch_add(1, Ordering::SeqCst);
            Ok(log("cached"))
        })
        .set_cache("cached", CachePolicy::new())
        .with_cache(Arc::new(InMemoryCache::new()))
        .add_node_with_retry(
            "retry",
            RetryPolicy::new(3).with_backoff(std::time::Duration::ZERO),
            move |_state: &JsonState| {
                let attempt = retry_counter.retry.fetch_add(1, Ordering::SeqCst) + 1;
                if attempt % 2 == 1 {
                    return Err(langgraph_rs::errors::Error::other("transient failure"));
                }
                Ok(log("retry"))
            },
        )
        .add_node("boom", move |_state: &JsonState| {
            boom_counter.boom.fetch_add(1, Ordering::SeqCst);
            Err(langgraph_rs::errors::Error::other("always fails"))
        })
        .add_error_edge("boom", "handled")
        .add_node("handled", |_state: &JsonState| Ok(log("handled")))
        .add_node_deferred("defer", |_state: &JsonState| Ok(log("deferred")))
        .add_node("sub", move |state: &JsonState| {
            let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            let child_state = JsonState::new()
                .with("value", json!(count))
                .with("log", json!([]));
            let result = child.invoke(&child_state, &RunConfig::new())?;
            let value = result.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(NodeOutput::new()
                .set("child", json!(value))
                .update("log", json!(["sub"])))
        })
        .add_node("ask", |_state: &JsonState| {
            let approval = interrupt(json!("approve?"))?;
            Ok(NodeOutput::new()
                .update("log", json!(["ask"]))
                .set("flag", json!(format!("approved={approval}"))))
        })
        .add_node("done", |_state: &JsonState| Ok(log("done")))
        .add_edge(langgraph_rs::START, "seed")
        .add_edge("seed", "grow")
        .add_edge("worker", "join")
        .add_edge("join", "cached")
        .add_edge("cached", "retry")
        .add_edge("retry", "sub")
        .add_edge("sub", "ask")
        .add_edge("ask", "done")
        // Both tails are terminal: `defer` cannot have outgoing edges and `boom` fails.
        .add_edge("done", "defer")
        .add_edge("done", "boom")


        .compile_with(
            CompileOptions::new()
                .with_checkpointer(Arc::new(InMemorySaver::new()))
                .with_store(Arc::new(InMemoryStore::new())),
        )
        .expect("mega graph compiles")
}

fn count(value: &str) -> String {
    value.to_string()
}

fn main() {
    let iterations: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);

    let counters = Arc::new(Counters {
        retry: AtomicUsize::new(0),
        boom: AtomicUsize::new(0),
        cached: AtomicUsize::new(0),
        child: AtomicUsize::new(0),
    });
    let graph = build(Arc::clone(&counters));
    let cache = Arc::new(InMemoryCache::new());
    let _ = cache;

    // One warm-up suite so allocation/JIT-free caches are hot, then measure.
    let mut phases: HashMap<&str, Vec<u128>> = HashMap::new();
    for iteration in 0..iterations {
        let thread = format!("native-{iteration}");

        let started = Instant::now();
        let paused = graph.run(&input(), &RunConfig::new().with_thread(&thread));
        assert!(paused.as_ref().is_err_and(|error| error.is_interrupt()));
        phases.entry("invoke").or_default().push(started.elapsed().as_nanos());

        let started = Instant::now();
        let final_state = graph
            .resume(&thread, json!("go"))
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
            .expect("update_state");
        phases.entry("update_state").or_default().push(started.elapsed().as_nanos());

        let started = Instant::now();
        match graph.stream(&input(), &RunConfig::new().with_thread(format!("{thread}-stream"))) {
            Ok(chunks) => { std::hint::black_box(chunks.len()); }
            // The workflow pauses at `ask`; streaming reports the interrupt.
            Err(error) if error.is_interrupt() => { std::hint::black_box(0usize); }
            Err(error) => panic!("stream failed: {error}"),
        }
        phases.entry("stream").or_default().push(started.elapsed().as_nanos());

        let started = Instant::now();
        let batched = graph.batch(
            [input(), input()],
            &RunConfig::new().with_thread(format!("{thread}-batch")),
        );
        std::hint::black_box(batched.len());
        phases.entry("batch").or_default().push(started.elapsed().as_nanos());

        if iteration == 0 {
            let summary: Vec<i64> = ["count", "sink", "child"]
                .iter()
                .map(|key| {
                    final_state
                        .get(key)
                        .and_then(|value| value.as_i64())
                        .unwrap_or(-1)
                })
                .collect();
            let entry = format!(
                "FINAL count={} sink={} child={} log={}",
                summary[0],
                summary[1],
                summary[2],
                final_state
                    .get("log")
                    .and_then(|value| value.as_array())
                    .map(|entries| entries.len())
                    .unwrap_or(0)
            );
            println!("{entry}");
            println!(
                "COUNTERS cached={} retry={} boom={} child={}",
                counters.cached.load(Ordering::SeqCst),
                counters.retry.load(Ordering::SeqCst),
                counters.boom.load(Ordering::SeqCst),
                counters.child.load(Ordering::SeqCst)
            );
        }
    }

    for (phase, mut samples) in phases {
        samples.sort_unstable();
        let median = samples[samples.len() / 2];
        println!("PHASE={}:{}", count(phase), median);
    }

    // Keep the durability enum referenced so the example documents the default.
    let _ = Durability::default();
}