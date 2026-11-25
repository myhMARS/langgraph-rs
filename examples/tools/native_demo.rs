//! Tools — Rust native API.
//!
//! Same graph as the Python demos: one `ToolNode` holding two tools. `add` uses only its arguments;
//! `lookup` also reads the graph state (`ToolDefinition::with_state`), which is the native
//! equivalent of `InjectedState`.
//!
//! ```text
//! cargo run --release --example tools_native
//! cargo run --release --example tools_native -- --bench 500
//! ```

use std::time::Instant;

use langgraph_rs::prebuilt::{ToolDefinition, ToolNode};
use langgraph_rs::{JsonState, RunConfig, START, StateGraph, StateSchema, reducers};
use serde_json::{Value, json};

fn schema() -> StateSchema {
    StateSchema::dynamic()
        .with_reducer("messages", reducers::messages())
        .last_value("user")
}

fn build() -> langgraph_rs::CompiledGraph<JsonState> {
    let tools = ToolNode::new([
        ToolDefinition::new("add", |args: Value| {
            let a = args.get("a").and_then(Value::as_i64).unwrap_or_default();
            let b = args.get("b").and_then(Value::as_i64).unwrap_or_default();
            Ok(json!((a + b).to_string()))
        }),
        // State-aware tool: the native counterpart of `InjectedState`.
        ToolDefinition::with_state("lookup", |args: Value, state: &JsonState| {
            let query = args
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let user = state.get("user").and_then(Value::as_str).unwrap_or_default();
            Ok(json!(format!("{query}|{user}")))
        }),
    ]);
    StateGraph::<JsonState>::with_schema(schema())
        .add_node("tools", move |state: &JsonState| tools.execute(state))
        .add_edge(START, "tools")
        .set_finish_point("tools")
        .compile()
        .expect("graph compiles")
}

fn tool_call(name: &str, args: Value, call_id: &str) -> Value {
    json!({
        "role": "assistant",
        "content": "",
        "tool_calls": [{ "name": name, "args": args, "id": call_id }],
    })
}

/// Chat messages produced by the node before ours — used to build the next input.
fn previous_messages(state: &JsonState) -> Vec<Value> {
    state.get("messages").and_then(Value::as_array).cloned().unwrap_or_default()
}

fn collect_results(state: &JsonState) -> Vec<Value> {
    state
        .get("messages")
        .and_then(Value::as_array)
        .map(|messages| {
            messages
                .iter()
                .filter(|message| message.get("role").and_then(Value::as_str) == Some("tool"))
                .filter_map(|message| message.get("content").cloned())
                .collect()
        })
        .unwrap_or_default()
}

fn run_once(graph: &langgraph_rs::CompiledGraph<JsonState>) -> Value {
    let mut messages = vec![tool_call("add", json!({"a": 2, "b": 3}), "c1")];
    let mut state = graph
        .invoke(
            &json!({ "messages": messages, "user": "ada" })
                .as_object()
                .cloned()
                .expect("object")
                .into(),
            &RunConfig::new(),
        )
        .expect("graph runs");
    messages = previous_messages(&state);
    messages.push(tool_call("lookup", json!({"query": "q"}), "c2"));
    state = graph
        .invoke(
            &json!({ "messages": messages, "user": "ada" })
                .as_object()
                .cloned()
                .expect("object")
                .into(),
            &RunConfig::new(),
        )
        .expect("graph runs");
    json!({ "results": collect_results(&state) })
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
    println!("{}", run_once(&graph));
}