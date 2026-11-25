//! Async execution — Rust native API.
//!
//! The same three-node chain as the Python demos, using `AsyncStateGraph` on Tokio. Nodes are
//! async closures, so a superstep with several of them runs concurrently.
//!
//! The JSON result goes to stdout (for `compare.py`); the per-node consumption report goes to
//! stderr, so it never disturbs the result comparison.
//!
//! ```text
//! cargo run --release --example async_native
//! cargo run --release --example async_native -- --bench 500
//! ```

use std::time::Instant;

use langgraph_rs::tokio_runtime::{AsyncContext, AsyncStateGraph};
use langgraph_rs::{
    NodeOutput, RunConfig, State, StateSchema, Usage, record_tokens, record_usage, reducers,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct AgentState {
    question: String,
    answer: String,
    trace: Vec<String>,
}

impl State for AgentState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("question")
            .last_value("answer")
            .with_reducer("trace", reducers::append())
    }
}

fn build() -> langgraph_rs::tokio_runtime::AsyncCompiledGraph<AgentState> {
    AsyncStateGraph::<AgentState>::new()
        .add_node("retrieve", |state: AgentState, _context: AsyncContext| async move {
            // Consumption is reported from inside the node, like a real retriever would.
            record_usage(
                Usage::new()
                    .with_model("text-embedding-3-small")
                    .with_input_tokens(64),
            )?;
            Ok(NodeOutput::new()
                .set("answer", format!("context for `{}`", state.question))
                .set("trace", json!(["retrieve"])))
        })
        .add_node("generate", |state: AgentState, _context: AsyncContext| async move {
            record_tokens(512, 96)?;
            Ok(NodeOutput::new()
                .set("answer", format!("answer based on: {}", state.answer))
                .set("trace", json!(["generate"])))
        })
        .add_node("judge", |_state: AgentState, _context: AsyncContext| async move {
            record_tokens(32, 4)?;
            Ok(NodeOutput::new().set("trace", json!(["judge"])))
        })
        .add_edge(langgraph_rs::START, "retrieve")
        .add_edge("retrieve", "generate")
        .add_edge("generate", "judge")
        .set_finish_point("judge")
        .compile()
        .expect("graph compiles")
}

async fn run_once(graph: &langgraph_rs::tokio_runtime::AsyncCompiledGraph<AgentState>) -> AgentState {
    let input = AgentState {
        question: "What is a superstep?".to_string(),
        ..AgentState::default()
    };
    graph
        .invoke(&input)
        .await
        .expect("graph runs")
}

#[tokio::main]
async fn main() {
    let graph = build();
    let args: Vec<String> = std::env::args().collect();
    if let Some(index) = args.iter().position(|arg| arg == "--bench") {
        let iterations: u32 = args
            .get(index + 1)
            .and_then(|value| value.parse().ok())
            .unwrap_or(500);
        for _ in 0..20 {
            let _ = run_once(&graph).await;
        }
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = run_once(&graph).await;
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

    // `run` returns the state plus the per-run report (consumption tracking).
    let input = AgentState {
        question: "What is a superstep?".to_string(),
        ..AgentState::default()
    };
    let outcome = graph.run(&input, &RunConfig::new()).await.expect("graph runs");
    let payload: Value = json!({
        "answer": outcome.state.answer,
        "trace": outcome.state.trace,
    });
    println!("{payload}");
    // Consumption tracking is a native-only extra here; it stays out of the compared JSON.
    eprintln!(
        "usage: {} tokens, ${:.6}, {} ms",
        outcome.report.total_tokens(),
        outcome.report.total_cost_usd(),
        outcome.report.total_duration_millis
    );
    for (node, usage) in outcome.report.usage_by_node() {
        eprintln!("  {node:<10} tokens={}", usage.total_tokens());
    }
}