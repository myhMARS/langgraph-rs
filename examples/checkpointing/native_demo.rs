//! Checkpointing — Rust native API.
//!
//! Same thread as the Python demos: three runs, an `update_state` fork, one continuation, and the
//! checkpoint history (stderr, so the compared JSON stays clean).
//!
//! ```text
//! cargo run --release --example checkpointing_native
//! cargo run --release --example checkpointing_native -- --bench 200
//! ```

use std::sync::Arc;
use std::time::Instant;

use langgraph_rs::{
    InMemorySaver, NodeOutput, RunConfig, START, StateGraph,
};
use serde_json::{Value, json};

fn build() -> langgraph_rs::CompiledGraph<langgraph_rs::JsonState> {
    use langgraph_rs::{CompileOptions, JsonState, StateSchema};
    StateGraph::<JsonState>::with_schema(
        StateSchema::dynamic().with_reducer("log", langgraph_rs::reducers::append()),
    )
    .add_node("increment", |state: &JsonState| {
        let count: i64 = state.get_as("count").unwrap_or_default();
        Ok(NodeOutput::new()
            .set("count", count + 1)
            .set("log", json!([format!("count={}", count + 1)])))
    })
    .add_edge(START, "increment")
    .set_finish_point("increment")
    .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
    .expect("graph compiles")
}


fn run_once(
    graph: &langgraph_rs::CompiledGraph<langgraph_rs::JsonState>,
    thread: &str,
    report: bool,
) -> Value {
    let config = RunConfig::new().with_thread(thread);
    // The native API takes the *whole* state, so continuing a thread means reading the latest
    // checkpoint first (the Python binding and LangGraph merge the partial input for you).
    let mut count = 0;
    for _ in 0..3 {
        let mut input = graph
            .get_state(thread)
            .expect("state reads")
            .map(|snapshot| snapshot.values)
            .unwrap_or_default();
        input.insert("count".to_string(), json!(count));
        let state = graph.invoke(&input, &config).expect("graph runs");
        count = state.get_as("count").unwrap_or_default();
    }
    graph
        .update_state(thread, [("count", json!(42))], None)
        .expect("state updates");
    let mut input = graph
        .get_state(thread)
        .expect("state reads")
        .map(|snapshot| snapshot.values)
        .unwrap_or_default();
    input.insert("count".to_string(), json!(42));
    let forked = graph.invoke(&input, &config).expect("graph runs");
    if report {
        let history = graph.get_state_history(thread).expect("history reads");
        eprintln!("history checkpoints: {}", history.len());
        for snapshot in history {
            eprintln!("  count={:?}", snapshot.values.get_as::<i64>("count"));
        }
    }
    let final_count: i64 = forked.get_as("count").unwrap_or_default();
    let log: Vec<String> = forked.get_as("log").unwrap_or_default();
    json!({ "final": final_count, "log": log })
}
fn main() {
    let graph = build();
    let args: Vec<String> = std::env::args().collect();
    if let Some(index) = args.iter().position(|arg| arg == "--bench") {
        let iterations: u32 = args
            .get(index + 1)
            .and_then(|value| value.parse().ok())
            .unwrap_or(200);
        for index in 0..10 {
            let _ = run_once(&graph, &format!("warm-{index}"), false);
        }
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            for index in 0..iterations {
                let _ = run_once(&graph, &format!("bench-{index}"), false);
            }
            samples.push(start.elapsed().as_nanos() as f64 / iterations as f64);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        println!(
            "{{\"engine\":\"rust native\",\"ns_per_op\":{}}}",
            samples[samples.len() / 2]
        );
        return;
    }
    println!("{}", run_once(&graph, "demo", true));
}