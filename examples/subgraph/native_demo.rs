//! Subgraphs and `Send` fan-out — Rust native API.
//!
//! Mirrors the Python demos with `JsonState`:
//!
//! * `dispatch` emits one `Send` per number (map),
//! * each `worker` task writes into the `results` reducer channel,
//! * `reduce` sorts and totals them,
//! * `scaled` is an embedded child graph (`add_subgraph`) that multiplies the total.
//!
//! ```text
//! cargo run --release --example subgraph_native
//! cargo run --release --example subgraph_native -- --bench 200
//! ```

use std::sync::Arc;
use std::time::Instant;

use langgraph_rs::{JsonState, NodeOutput, RunConfig, Send, StateGraph, StateSchema, reducers};
use serde_json::{Value, json};

const NUMBERS: [i64; 4] = [3, 8, 1, 6];

fn schema() -> StateSchema {
    StateSchema::dynamic().with_reducer("results", reducers::append())
}

fn child() -> Arc<langgraph_rs::CompiledGraph<JsonState>> {
    Arc::new(
        StateGraph::<JsonState>::with_schema(schema())
            .add_node("scale", |state: &JsonState| {
                let total: i64 = state.get_as("total").unwrap_or_default();
                let factor: i64 = state.get_as("factor").unwrap_or(1);
                Ok(NodeOutput::new().set("scaled", json!([total * factor])))
            })
            .set_entry_point("scale")
            .set_finish_point("scale")
            .compile()
            .expect("subgraph compiles"),
    )
}

fn build() -> langgraph_rs::CompiledGraph<JsonState> {
    StateGraph::<JsonState>::with_schema(schema())
        .add_node("dispatch", |_state: &JsonState| {
            let mut output = NodeOutput::new();
            for number in NUMBERS {
                output = output.send(Send::with_arg("worker", json!({ "number": number })));
            }
            Ok(output)
        })
        .add_node("worker", |state: &JsonState| {
            let number: i64 = state.get_as("number").unwrap_or_default();
            Ok(NodeOutput::new().set("results", json!([number * number])))
        })
        .add_node("reduce", |state: &JsonState| {
            let mut results: Vec<i64> = state.get_as("results").unwrap_or_default();
            results.sort_unstable();
            let total: i64 = results.iter().sum();
            Ok(NodeOutput::new()
                .set("sorted_results", json!(results))
                .set("total", json!(total)))
        })
        .add_subgraph("scaled", child())
        .set_entry_point("dispatch")
        .add_edge("worker", "reduce")
        .add_edge("reduce", "scaled")
        .set_finish_point("scaled")
        .compile()
        .expect("graph compiles")
}

fn run_once(graph: &langgraph_rs::CompiledGraph<JsonState>) -> JsonState {
    let input: JsonState = json!({
        "results": [],
        "sorted_results": [],
        "total": 0,
        "factor": 2,
        "scaled": [],
    })
    .as_object()
    .cloned()
    .expect("input is an object")
    .into();
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
            .unwrap_or(200);
        for _ in 0..20 {
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

    let output = run_once(&graph);
    let scaled: Vec<i64> = output.get_as("scaled").unwrap_or_default();
    let sorted: Vec<i64> = output.get_as("sorted_results").unwrap_or_default();
    let total: i64 = output.get_as("total").unwrap_or_default();
    let payload = json!({
        "scaled": scaled,
        "sorted_results": sorted,
        "total": total,
    });
    println!("{}", payload as Value);
}