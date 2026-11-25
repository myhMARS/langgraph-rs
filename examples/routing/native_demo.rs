//! Routing — Rust native API.
//!
//! Same graph as the Python demos:
//!
//! * `add_conditional_edges("classify", ..)` returns `Route::node(..)`,
//! * `polish` ends the run from inside the node with `NodeOutput::goto(END)` — the native
//!   equivalent of LangGraph's `Command(goto=...)`.
//!
//! ```text
//! cargo run --release --example routing_native
//! cargo run --release --example routing_native -- --bench 500
//! ```

use std::time::Instant;

use langgraph_rs::{
    END, NodeOutput, Route, RunConfig, START, State, StateGraph, StateSchema, reducers,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct DemoState {
    question: String,
    answer: String,
    route: String,
    trace: Vec<String>,
}

impl State for DemoState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("question")
            .last_value("answer")
            .last_value("route")
            .with_reducer("trace", reducers::append())
    }
}

fn build() -> langgraph_rs::CompiledGraph<DemoState> {
    StateGraph::<DemoState>::new()
        .add_node("classify", |state: &DemoState| {
            let route = if state.question.len() < 20 {
                "fast"
            } else {
                "slow"
            };
            Ok(NodeOutput::new()
                .set("route", route)
                .set("trace", json!(["classify"])))
        })
        .add_node("fast", |_state: &DemoState| {
            Ok(NodeOutput::new()
                .set("answer", "quick answer")
                .set("trace", json!(["fast"])))
        })
        .add_node("slow", |_state: &DemoState| {
            Ok(NodeOutput::new()
                .set("answer", "researched answer")
                .set("trace", json!(["slow"])))
        })
        .add_node("polish", |state: &DemoState| {
            Ok(NodeOutput::new()
                .set("answer", state.answer.to_uppercase())
                .set("trace", json!(["polish"]))
                .goto(END))
        })
        .add_edge(START, "classify")
        .add_conditional_edges("classify", |state: &DemoState| {
            Ok(Route::node(state.route.clone()))
        })
        .add_edge("fast", "polish")
        .add_edge("slow", "polish")
        .set_finish_point("polish")
        .compile()
        .expect("graph compiles")
}

fn run_once(graph: &langgraph_rs::CompiledGraph<DemoState>) -> DemoState {
    let input = DemoState {
        question: "What is a superstep?".to_string(),
        ..DemoState::default()
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
    let payload = json!({
        "answer": state.answer,
        "route": state.route,
        "trace": state.trace,
    });
    println!("{payload}");
}