//! Per-node consumption tracking: record tokens/cost and read the run report.
//!
//! Run with:
//!
//! ```text
//! cargo run --example usage_tracking
//! ```

use langgraph_rs::{
    NodeOutput, RunConfig, State, StateGraph, StateSchema, Usage, record_tokens, record_usage,
    reducers,
};
use serde::{Deserialize, Serialize};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = StateGraph::<AgentState>::new()
        // A node that reports consumption piecemeal while it works.
        .add_node("retrieve", |state: &AgentState| {
            // Pretend the retriever called an embedding model.
            record_usage(
                Usage::new()
                    .with_model("text-embedding-3-small")
                    .with_input_tokens(64)
                    .with_cost_usd(0.000001),
            )?;
            record_usage(Usage::new().with_metadata("documents", 3))?;
            Ok(NodeOutput::new()
                .set("answer", format!("context for `{}`", state.question))
                .set("trace", vec!["retrieve".to_string()]))
        })
        // A node that attaches its usage to the returned output instead.
        .add_node("generate", |state: &AgentState| {
            let answer = format!("answer based on: {}", state.answer);
            Ok(NodeOutput::new()
                .set("answer", answer)
                .set("trace", vec!["generate".to_string()])
                .with_usage(
                    Usage::new()
                        .with_model("gpt-4o")
                        .with_input_tokens(512)
                        .with_output_tokens(96)
                        .with_cost_usd(0.006),
                ))
        })
        .add_node("judge", |_state: &AgentState| {
            record_tokens(32, 4)?;
            Ok(NodeOutput::new().set("trace", vec!["judge".to_string()]))
        })
        .add_edge(langgraph_rs::START, "retrieve")
        .add_edge("retrieve", "generate")
        .add_edge("generate", "judge")
        .set_finish_point("judge")
        .compile()?;

    let input = AgentState {
        question: "What is a superstep?".to_string(),
        ..AgentState::default()
    };
    let outcome = graph.run(&input, &RunConfig::new())?;

    println!("answer: {}", outcome.state.answer);
    println!("trace: {:?}", outcome.state.trace);
    println!();
    println!("per-node executions:");
    for execution in outcome.report.executions() {
        println!(
            "  {:<10} step={} duration={}ms usage={:?}",
            execution.node, execution.step, execution.duration_millis, execution.usage
        );
    }
    println!();
    println!("aggregated usage by node:");
    for (node, usage) in outcome.report.usage_by_node() {
        println!(
            "  {node:<10} tokens={} cost=${:.6}",
            usage.total_tokens(),
            usage.cost_usd.unwrap_or_default()
        );
    }
    println!(
        "total: {} tokens, ${:.6}, {} ms",
        outcome.report.total_tokens(),
        outcome.report.total_cost_usd(),
        outcome.report.total_duration_millis
    );

    Ok(())
}
