//! State and reducers — Rust native API.
//!
//! Same graph as the Python demos: `left` and `right` run in one superstep and both write
//! `items` / `total`. The reducers decide the merge:
//!
//! * `items`  -> `reducers::append()`
//! * `total`  -> a custom `Reducer::new(..)` keeping the maximum
//! * `note`   -> `last_value` (single writer)
//!
//! ```text
//! cargo run --release --example state_reducers_native
//! cargo run --release --example state_reducers_native -- --bench 500
//! ```

use std::time::Instant;

use langgraph_rs::{NodeOutput, Reducer, RunConfig, START, State, StateGraph, StateSchema, reducers};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct DemoState {
    items: Vec<String>,
    total: i64,
    note: String,
}

impl State for DemoState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .with_reducer("items", reducers::append())
            .with_reducer(
                "total",
                Reducer::new(|current: &Value, incoming: &Value| {
                    let left = current.as_i64().unwrap_or(0);
                    let right = incoming.as_i64().unwrap_or(0);
                    Ok(json!(left.max(right)))
                }),
            )
            .last_value("note")
    }
}

fn build() -> langgraph_rs::CompiledGraph<DemoState> {
    StateGraph::<DemoState>::new()
        .add_node("left", |_state: &DemoState| {
            Ok(NodeOutput::new()
                .set("items", json!(["left-1", "left-2"]))
                .set("total", 3))
        })
        .add_node("right", |_state: &DemoState| {
            Ok(NodeOutput::new()
                .set("items", json!(["right-1", "right-2"]))
                .set("total", 7))
        })
        .add_node("summarize", |state: &DemoState| {
            Ok(NodeOutput::new().set(
                "note",
                format!("{} items, total={}", state.items.len(), state.total),
            ))
        })
        // Two edges out of START put both nodes in the same superstep.
        .add_edge(START, "left")
        .add_edge(START, "right")
        .add_edge("left", "summarize")
        .add_edge("right", "summarize")
        .set_finish_point("summarize")
        .compile()
        .expect("graph compiles")
}

fn run_once(graph: &langgraph_rs::CompiledGraph<DemoState>) -> DemoState {
    let input = DemoState::default();
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
            .unwrap_or(500);
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
    // Sorted for the same reason as the Python demos: parallel write order is an
    // implementation detail, the merged content is the contract.
    let mut items = state.items.clone();
    items.sort();
    println!(
        "{{\"items\":{},\"note\":{},\"total\":{}}}",
        serde_json::to_string(&items).expect("items serialize"),
        serde_json::to_string(&state.note).expect("note serialize"),
        state.total
    );
}