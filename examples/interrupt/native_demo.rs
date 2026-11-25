//! Human-in-the-loop — Rust native API.
//!
//! Same review graph as the Python demos: `review` calls `interrupt(...)`, the engine checkpoints
//! and returns `Err(Error::Interrupt)`, and `graph.resume(thread, value)` re-runs the node with the
//! human's answer.
//!
//! ```text
//! cargo run --release --example interrupt_native
//! cargo run --release --example interrupt_native -- --bench 100
//! ```

use std::sync::Arc;
use std::time::Instant;

use langgraph_rs::{
    CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph, StateSchema, interrupt,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ReviewState {
    draft: String,
    approved: bool,
    feedback: String,
}

impl State for ReviewState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("draft")
            .last_value("approved")
            .last_value("feedback")
    }
}

fn build() -> langgraph_rs::CompiledGraph<ReviewState> {
    StateGraph::<ReviewState>::new()
        .add_node("draft", |_state: &ReviewState| {
            Ok(NodeOutput::new().set("draft", "Quarterly report v1"))
        })
        .add_node("review", |state: &ReviewState| {
            // First run: unwinds with an interrupt. After `resume`, the same call returns
            // the JSON value supplied there.
            let answer = interrupt(json!({
                "question": format!("Approve `{}`?", state.draft),
                "options": ["approve", "reject"],
            }))?;
            let approved = answer
                .get("approved")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let feedback = answer
                .get("comment")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            Ok(NodeOutput::new()
                .set("approved", approved)
                .set("feedback", feedback))
        })
        .add_node("publish", |state: &ReviewState| {
            let label = if state.approved {
                "published"
            } else {
                "sent back for revision"
            };
            Ok(NodeOutput::new().set("draft", format!("{} [{label}]", state.draft)))
        })
        .set_entry_point("draft")
        .add_edge("draft", "review")
        .add_edge("review", "publish")
        .set_finish_point("publish")
        .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
        .expect("graph compiles")
}

const ANSWER: &str = r#"{"approved": true, "comment": "Looks good, ship it!"}"#;

fn run_once(graph: &langgraph_rs::CompiledGraph<ReviewState>, thread: &str) -> Value {
    let config = RunConfig::new().with_thread(thread);
    let paused = match graph.invoke(&ReviewState::default(), &config) {
        Ok(state) => {
            return json!({
                "approved": state.approved,
                "draft": state.draft,
                "feedback": state.feedback,
                "paused_at": "",
                "payload": Value::Null,
            });
        }
        Err(error) if error.is_interrupt() => error.interrupt().expect("checked").clone(),
        Err(error) => panic!("unexpected error: {error}"),
    };
    let snapshot = graph
        .get_state(thread)
        .expect("state reads")
        .expect("a paused thread has a checkpoint");
    let paused_at = snapshot
        .next
        .first()
        .cloned()
        .unwrap_or_default();
    let answer: Value = serde_json::from_str(ANSWER).expect("answer parses");
    let final_state = graph.resume(thread, answer).expect("resume runs");
    json!({
        "approved": final_state.approved,
        "draft": final_state.draft,
        "feedback": final_state.feedback,
        "paused_at": paused_at,
        "payload": paused.value,
    })
}

fn main() {
    let graph = build();
    let args: Vec<String> = std::env::args().collect();
    if let Some(index) = args.iter().position(|arg| arg == "--bench") {
        let iterations: u32 = args
            .get(index + 1)
            .and_then(|value| value.parse().ok())
            .unwrap_or(100);
        for index in 0..20 {
            let _ = run_once(&graph, &format!("warm-{index}"));
        }
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            for index in 0..iterations {
                let _ = run_once(&graph, &format!("bench-{index}"));
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
    println!("{}", run_once(&graph, "demo"));
}