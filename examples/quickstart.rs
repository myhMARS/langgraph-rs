//! Quickstart: a typed state graph with three nodes and a conditional loop.
//!
//! This is the smallest useful LangGraph program:
//!
//! 1. define a state struct and declare its channels,
//! 2. register nodes that return *writes* (never mutate state directly),
//! 3. connect nodes with static and conditional edges,
//! 4. compile the graph and invoke it with a `RunConfig`.
//!
//! Run with:
//!
//! ```text
//! cargo run --example quickstart
//! ```

use langgraph_rs::{NodeOutput, Route, RunConfig, State, StateGraph, StateSchema, reducers};
use serde::{Deserialize, Serialize};

/// The graph state.
///
/// Every field is a *channel*. The `State::schema` implementation below decides how
/// writes to each channel are merged between supersteps.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct CounterState {
    /// Current counter value.
    count: i64,
    /// The loop stops once `count` reaches this limit.
    target: i64,
    /// Execution log. Backed by an appending reducer channel, so every node can
    /// contribute entries without overwriting the previous ones.
    steps: Vec<String>,
}

impl State for CounterState {
    fn schema() -> StateSchema {
        StateSchema::new()
            // Plain last-write-wins channels: only one writer per superstep.
            .last_value("count")
            .last_value("target")
            // `steps` uses `reducers::append()`: each write must be an array and the
            // arrays are concatenated, which is the usual "event log" pattern.
            .with_reducer("steps", reducers::append())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = StateGraph::<CounterState>::new()
        // Node 1: read the initial target from the input and initialize the loop.
        .add_node("seed", |state: &CounterState| {
            let target = if state.target > 0 { state.target } else { 3 };
            Ok(NodeOutput::new()
                .set("target", target)
                .set("steps", vec!["seed".to_string()]))
        })
        // Node 2: the loop body, executed once per superstep until the target is hit.
        .add_node("increment", |state: &CounterState| {
            Ok(NodeOutput::new()
                .set("count", state.count + 1)
                .set("steps", vec![format!("increment -> {}", state.count + 1)]))
        })
        // Node 3: terminal node, reached only after the loop exits.
        .add_node("finish", |state: &CounterState| {
            Ok(NodeOutput::new().set("steps", vec![format!("finish at count = {}", state.count)]))
        })
        .set_entry_point("seed")
        .add_edge("seed", "increment")
        // A conditional edge evaluates a router against the *updated* state after the
        // superstep and returns the next node(s). Routing back to `increment` is the loop.
        .add_conditional_edges("increment", |state: &CounterState| {
            Ok(if state.count < state.target {
                Route::node("increment") // keep looping
            } else {
                Route::node("finish") // exit the loop
            })
        })
        .set_finish_point("finish")
        .compile()?;

    // `RunConfig` carries per-run settings (thread id for checkpointing, recursion
    // limit, ...). This graph has no checkpointer, so a plain config is enough.
    let input = CounterState {
        target: 4,
        ..CounterState::default()
    };
    let config = RunConfig::new();

    let final_state = graph.invoke(&input, &config)?;

    // The engine ran `seed -> increment x4 -> finish`; `steps` accumulated one entry
    // per node execution thanks to the reducer channel.
    println!("final state:");
    println!("  count  = {}", final_state.count);
    println!("  target = {}", final_state.target);
    println!("  steps  = {:?}", final_state.steps);
    Ok(())
}
