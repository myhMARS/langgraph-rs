//! Human-in-the-loop: pause a run with `interrupt()`, inspect the checkpoint, resume.
//!
//! The flow is the LangGraph "human approval" pattern:
//!
//! 1. a node calls `interrupt(payload)`;
//! 2. the engine checkpoints the state and returns `Err(Error::Interrupt)` — this is
//!    a pause, not a failure;
//! 3. the caller inspects the state with `graph.get_state(thread)`;
//! 4. the caller answers with `graph.resume(thread, value)` and the interrupted node
//!    is re-executed: its `interrupt()` call now returns the resume value.
//!
//! Run with:
//!
//! ```text
//! cargo run --example human_in_the_loop
//! ```

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph, StateSchema, interrupt,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// State for a small review workflow.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ReviewState {
    /// Text the graph wants to publish.
    draft: String,
    /// Decision produced by the human and written back by the resumed node.
    approved: bool,
    /// Optional note attached to the decision.
    feedback: Option<String>,
}

impl State for ReviewState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("draft")
            .last_value("approved")
            .last_value("feedback")
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `InMemorySaver` keeps checkpoints in memory. Any `Checkpointer` implementation
    // (file, Redis, Postgres, ...) can be used instead; it is required for interrupts
    // because the pending task has to be stored somewhere.
    let graph = StateGraph::<ReviewState>::new()
        // Node 1: produce a draft.
        .add_node("draft", |_state: &ReviewState| {
            Ok(NodeOutput::new().set("draft", "Quarterly report v1"))
        })
        // Node 2: ask a human and continue with their answer.
        .add_node("review", |state: &ReviewState| {
            // First execution: this returns `Err(Error::Interrupt)` and unwinds the
            // graph. After `resume`, the same call returns the value supplied there.
            let decision = interrupt(json!({
                "question": format!("Approve the draft `{}`?", state.draft),
                "options": ["approve", "reject"],
            }))?;

            let approved = decision
                .get("approved")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let feedback = decision
                .get("comment")
                .and_then(Value::as_str)
                .map(str::to_string);

            Ok(NodeOutput::new()
                .set("approved", approved)
                .set("feedback", feedback))
        })
        // Node 3: publish only after the human answered.
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
        .compile_with(
            // Waiting for a human only makes sense when the paused state is persisted.
            CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())),
        )?;

    let config = RunConfig::new().with_thread("demo");

    match graph.invoke(&ReviewState::default(), &config) {
        Ok(state) => {
            // Not expected here, but a graph that never interrupts finishes normally.
            println!(
                "graph finished without interrupting: approved = {}",
                state.approved
            );
        }
        Err(error) if error.is_interrupt() => {
            let raised = error.interrupt().expect("checked with is_interrupt()");
            println!("graph paused in node `{}`", raised.node);
            println!("interrupt payload: {}", raised.value);

            // Inspect the checkpoint written at the moment of the pause.
            let snapshot = graph
                .get_state("demo")?
                .expect("a paused thread always has a checkpoint");
            println!();
            println!("checkpoint snapshot:");
            println!("  interrupted = {}", snapshot.interrupted);
            println!("  next        = {:?}", snapshot.next);
            println!("  draft       = {:?}", snapshot.values.draft);
            println!("  approved    = {}", snapshot.values.approved);

            // Answer the interrupt. The `review` node runs again and `interrupt()`
            // returns exactly this JSON value.
            let final_state = graph.resume(
                "demo",
                json!({
                    "approved": true,
                    "comment": "Looks good, ship it!",
                }),
            )?;

            println!();
            println!("final state:");
            println!("  draft    = {}", final_state.draft);
            println!("  approved = {}", final_state.approved);
            println!("  feedback = {:?}", final_state.feedback);
        }
        Err(error) => return Err(error.into()),
    }

    Ok(())
}
