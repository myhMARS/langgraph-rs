//! Streaming — Rust native API.
//!
//! `graph.stream(&input, &config)` returns every [`StreamChunk`] of the run. This demo keeps the
//! `Updates` chunks (the `stream_mode="updates"` equivalent) and prints them as JSON in the same
//! shape the Python demos print.
//!
//! ```text
//! cargo run --release --example streaming_native
//! cargo run --release --example streaming_native -- --bench 500
//! ```

use std::time::Instant;

use langgraph_rs::{
    NodeOutput, RunConfig, START, State, StateGraph, StateSchema, StreamChunk, reducers,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct DemoState {
    log: Vec<String>,
    plan: String,
    result: String,
    summary: String,
}

impl State for DemoState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .with_reducer("log", reducers::append())
            .last_value("plan")
            .last_value("result")
            .last_value("summary")
    }
}

fn build() -> langgraph_rs::CompiledGraph<DemoState> {
    StateGraph::<DemoState>::new()
        .add_node("plan", |_state: &DemoState| {
            Ok(NodeOutput::new()
                .set("plan", "3 steps")
                .set("log", json!(["plan"])))
        })
        .add_node("work", |state: &DemoState| {
            Ok(NodeOutput::new()
                .set("result", format!("did {}", state.plan))
                .set("log", json!(["work"])))
        })
        .add_node("report", |state: &DemoState| {
            Ok(NodeOutput::new()
                .set(
                    "summary",
                    format!("{} / {} events", state.result, state.log.len()),
                )
                .set("log", json!(["report"])))
        })
        .add_edge(START, "plan")
        .add_edge("plan", "work")
        .add_edge("work", "report")
        .set_finish_point("report")
        .compile()
        .expect("graph compiles")
}

fn collect(graph: &langgraph_rs::CompiledGraph<DemoState>) -> Vec<Value> {
    let chunks = graph
        .stream(&DemoState::default(), &RunConfig::new())
        .expect("graph streams");
    chunks
        .into_iter()
        .filter_map(|chunk| match chunk {
            StreamChunk::Updates { node, update, .. } => Some(json!({ node: update })),
            _ => None,
        })
        .collect()
}

fn main() {
    let graph = build();
    let args: Vec<String> = std::env::args().collect();
    if let Some(index) = args.iter().position(|arg| arg == "--bench") {
        let iterations: u32 = args
            .get(index + 1)
            .and_then(|value| value.parse().ok())
            .unwrap_or(500);
        for _ in 0..50 {
            let _ = collect(&graph);
        }
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = collect(&graph);
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
    println!("{}", json!({ "chunks": collect(&graph), "mode": "updates" }));
}