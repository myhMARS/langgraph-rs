//! Quickstart demo — Rust native API.
//!
//! Mirrors `langgraph_demo.py` / `binding_demo.py` with the typed Rust API:
//!
//! ```text
//! cargo run --release --example quickstart_native
//! cargo run --release --example quickstart_native -- --bench 2000
//! ```
//!
//! Without `--bench` it prints the same JSON state the Python demos print; with `--bench N` it
//! prints `ns_per_op` measured inside the process (no interpreter startup in the number).

use std::time::Instant;

use langgraph_rs::{NodeOutput, Route, RunConfig, StateGraph, StateSchema, reducers};
use serde::{Deserialize, Serialize};

/// Graph state: `count` / `target` are last-value channels, `steps` appends.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct DemoState {
    count: i64,
    target: i64,
    steps: Vec<String>,
}

impl langgraph_rs::State for DemoState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("count")
            .last_value("target")
            .with_reducer("steps", reducers::append())
    }
}

fn build() -> langgraph_rs::CompiledGraph<DemoState> {
    StateGraph::<DemoState>::new()
        .add_node("seed", |state: &DemoState| {
            let target = if state.target > 0 { state.target } else { 3 };
            Ok(NodeOutput::new()
                .set("target", target)
                .set("steps", vec!["seed".to_string()]))
        })
        .add_node("increment", |state: &DemoState| {
            Ok(NodeOutput::new()
                .set("count", state.count + 1)
                .set("steps", vec![format!("increment -> {}", state.count + 1)]))
        })
        .set_entry_point("seed")
        .add_edge("seed", "increment")
        .add_conditional_edges("increment", |state: &DemoState| {
            Ok(if state.count < state.target {
                Route::node("increment")
            } else {
                Route::end()
            })
        })
        .compile()
        .expect("graph compiles")
}

fn run_once(graph: &langgraph_rs::CompiledGraph<DemoState>) -> DemoState {
    let input = DemoState {
        count: 0,
        target: 3,
        steps: vec![],
    };
    graph
        .invoke(&input, &RunConfig::new())
        .expect("graph runs")
}

fn main() {
    let graph = build();
    let args: Vec<String> = std::env::args().collect();
    if let Some(index) = args.iter().position(|arg| arg == "--bench") {
        let iterations: u32 = args
            .get(index + 1)
            .and_then(|value| value.parse().ok())
            .unwrap_or(2000);
        for _ in 0..50 {
            let _ = run_once(&graph);
        }
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = run_once(&graph);
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
    let state = run_once(&graph);
    println!(
        "{{\"count\":{},\"steps\":{},\"target\":{}}}",
        state.count,
        serde_json::to_string(&state.steps).expect("steps serialize"),
        state.target
    );
}