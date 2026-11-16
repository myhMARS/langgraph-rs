//! Prebuilt helpers: tools_condition, ToolNode and create_react_agent.

use langgraph_rs::prebuilt::{ToolDefinition, ToolNode, create_react_agent, tools_condition};
use langgraph_rs::{JsonState, NodeOutput, RunConfig, json};

fn tool_call_state() -> JsonState {
    let mut state = JsonState::new();
    state.insert(
        "messages".to_string(),
        json!([{
            "role": "assistant",
            "tool_calls": [{ "id": "call-1", "name": "add", "args": { "a": 1, "b": 2 } }]
        }]),
    );
    state
}

#[test]
fn tool_node_executes_calls_and_appends_results() {
    let node = ToolNode::new([ToolDefinition::new("add", |args| {
        let a = args
            .get("a")
            .and_then(|value| value.as_i64())
            .unwrap_or_default();
        let b = args
            .get("b")
            .and_then(|value| value.as_i64())
            .unwrap_or_default();
        Ok(json!(a + b))
    })]);

    let output = node.execute(&tool_call_state()).unwrap();
    let messages = output.updates_map().get("messages").unwrap();
    let results = messages.as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["role"], json!("tool"));
    assert_eq!(results[0]["tool_call_id"], json!("call-1"));
    assert_eq!(results[0]["content"], json!("3"));

    let mut idle = JsonState::new();
    idle.insert(
        "messages".to_string(),
        json!([{ "role": "user", "content": "hi" }]),
    );
    assert!(node.execute(&idle).unwrap().is_empty());
}

#[test]
fn tools_condition_routes_on_tool_calls() {
    let route = tools_condition(&tool_call_state());
    assert_eq!(route.targets(), [langgraph_rs::prebuilt::TOOLS]);
}

#[test]
fn create_react_agent_runs_a_tool_round_trip() {
    let model = std::sync::Arc::new(|state: &JsonState| {
        let last_role = state
            .get("messages")
            .and_then(|messages| messages.as_array())
            .and_then(|messages| messages.last())
            .and_then(|message| message.get("role"))
            .and_then(|role| role.as_str())
            .unwrap_or_default()
            .to_string();
        if last_role == "tool" {
            Ok(NodeOutput::new().update(
                "messages",
                json!([{ "role": "assistant", "content": "done" }]),
            ))
        } else {
            Ok(NodeOutput::new().update(
                "messages",
                json!([{
                    "role": "assistant",
                    "tool_calls": [{ "id": "call-1", "name": "add", "args": { "a": 1, "b": 2 } }]
                }]),
            ))
        }
    }) as langgraph_rs::NodeFn<JsonState>;

    let graph = create_react_agent(
        model,
        [ToolDefinition::new("add", |args| {
            let a = args
                .get("a")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            let b = args
                .get("b")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(json!(a + b))
        })],
    )
    .compile()
    .unwrap();

    let output = graph.invoke(&JsonState::new(), &RunConfig::new()).unwrap();
    let messages = output.get("messages").unwrap().as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1]["content"], json!("3"));
    assert_eq!(messages[2]["content"], json!("done"));
}
