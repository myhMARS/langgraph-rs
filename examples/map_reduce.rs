//! Map/reduce: dynamic fan-out with `Send`, per-task input and a reducer channel.
//!
//! `Send` is LangGraph's map primitive: a node schedules one task per item, each task
//! receives its own argument as state, and every task's write is folded into the shared
//! state by the channel reducer declared on the schema.
//!
//! Flow:
//!
//! ```text
//! dispatch --Send(n)--> worker x5 --(append)--> results --reduce--> sorted_results / total
//! ```
//!
//! Run with:
//!
//! ```text
//! cargo run --example map_reduce
//! ```

use langgraph_rs::{JsonState, NodeOutput, RunConfig, Send, StateGraph, StateSchema, reducers};
use serde_json::{Value, json};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `StateSchema::dynamic()` accepts channels that are not declared explicitly, so
    // ad-hoc JSON workflows stay convenient. Declare only what needs special semantics:
    // `results` is a reducer channel, so all worker writes are appended instead of
    // conflicting (a plain last-value channel rejects concurrent writes).
    let schema = StateSchema::dynamic().with_reducer("results", reducers::append());

    let graph = StateGraph::<JsonState>::with_schema(schema)
        // The "map" step: decide how many workers to run and what each one receives.
        // A node may emit any number of `Send` tasks; they all run in the next superstep.
        .add_node("dispatch", |_state: &JsonState| {
            let numbers = [3_i64, 8, 1, 6, 2];
            let mut output = NodeOutput::new();
            for number in numbers {
                // `with_arg` replaces the state the target node sees for this task only.
                output = output.send(Send::with_arg("worker", json!({ "number": number })));
            }
            Ok(output)
        })
        // The "map" body: one independent execution per `Send` task.
        .add_node("worker", |state: &JsonState| {
            let number = state
                .get("number")
                .and_then(Value::as_i64)
                .unwrap_or_default();
            // The reducer expects each write to be an array, so write a one-element
            // array; concurrent worker writes are concatenated by the runtime.
            Ok(NodeOutput::new().set("results", json!([number * number])))
        })
        // The "reduce" step: every worker has finished, so this runs exactly once.
        .add_node("reduce", |state: &JsonState| {
            let mut results: Vec<i64> = state.get_as("results").unwrap_or_default();
            results.sort_unstable();
            let total: i64 = results.iter().sum();
            // These channels are implicit last-value channels thanks to the dynamic schema.
            Ok(NodeOutput::new()
                .set("sorted_results", json!(results))
                .set("total", json!(total)))
        })
        .set_entry_point("dispatch")
        // The static edge is followed by each worker; the runtime de-duplicates the
        // five `worker -> reduce` requests into a single reduce run.
        .add_edge("worker", "reduce")
        .set_finish_point("reduce")
        .compile()?;

    let output = graph.invoke(&JsonState::default(), &RunConfig::new())?;

    // The dynamic schema keeps everything as JSON, so read values with `get_as`.
    let sorted: Vec<i64> = output.get_as("sorted_results").unwrap_or_default();
    let total: i64 = output.get_as("total").unwrap_or_default();

    println!("worker results (sorted): {sorted:?}");
    println!("total: {total}");
    Ok(())
}
