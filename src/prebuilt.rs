//! Prebuilt helpers mirroring LangGraph's `langgraph.prebuilt` basics:
//! `tools_condition`, `ToolNode` and `create_react_agent`.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::kernel::node::{NodeFn, NodeOutput};
use crate::kernel::state::{JsonState, StateSchema};
use crate::{END, Result, Route, StateGraph, reducers};

/// Node name used by [`tools_condition`] and [`create_react_agent`] for the tool executor.
pub const TOOLS: &str = "tools";

/// Route to `"tools"` when the last message asks for a tool, to `END` otherwise.
pub fn tools_condition(state: &JsonState) -> Route {
    tools_condition_with(state, "messages")
}

/// Like [`tools_condition`] with a custom messages channel name.
pub fn tools_condition_with(state: &JsonState, messages_key: &str) -> Route {
    let calls = state
        .get(messages_key)
        .and_then(Value::as_array)
        .and_then(|messages| messages.last())
        .and_then(|message| message.get("tool_calls"))
        .and_then(Value::as_array);
    if calls.is_some_and(|calls| !calls.is_empty()) {
        Route::node(TOOLS)
    } else {
        Route::end()
    }
}

/// A named tool usable by [`ToolNode`].
#[derive(Clone)]
pub struct ToolDefinition {
    name: String,
    run: Arc<dyn Fn(Value) -> Result<Value> + Send + Sync>,
}

impl std::fmt::Debug for ToolDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolDefinition")
            .field("name", &self.name)
            .finish()
    }
}

impl ToolDefinition {
    /// Build a tool from a name and an argument handler.
    pub fn new<F>(name: impl Into<String>, run: F) -> Self
    where
        F: Fn(Value) -> Result<Value> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            run: Arc::new(run),
        }
    }

    /// Tool name used in `tool_calls`.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Executes the `tool_calls` of the last message and appends tool result messages.
#[derive(Clone)]
pub struct ToolNode {
    tools: Arc<HashMap<String, ToolDefinition>>,
    messages_key: String,
}

impl std::fmt::Debug for ToolNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolNode")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .field("messages_key", &self.messages_key)
            .finish()
    }
}

impl ToolNode {
    /// Build a tool node from tool definitions.
    pub fn new<I>(tools: I) -> Self
    where
        I: IntoIterator<Item = ToolDefinition>,
    {
        let tools = tools
            .into_iter()
            .map(|tool| (tool.name().to_string(), tool))
            .collect();
        Self {
            tools: Arc::new(tools),
            messages_key: "messages".to_string(),
        }
    }

    /// Channel that stores the message list.
    pub fn messages_key(mut self, key: impl Into<String>) -> Self {
        self.messages_key = key.into();
        self
    }

    /// Execute the tool calls for `state`.
    pub fn execute(&self, state: &JsonState) -> Result<NodeOutput> {
        let Some(messages) = state.get(&self.messages_key).and_then(Value::as_array) else {
            return Ok(NodeOutput::new());
        };
        let Some(calls) = messages
            .last()
            .and_then(|message| message.get("tool_calls"))
            .and_then(Value::as_array)
        else {
            return Ok(NodeOutput::new());
        };
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            let name = call
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let args = call.get("args").cloned().unwrap_or(Value::Null);
            let content = match self.tools.get(&name) {
                Some(tool) => match (tool.run)(args) {
                    Ok(value) => match value {
                        Value::String(text) => text,
                        other => other.to_string(),
                    },
                    Err(error) => format!("tool `{name}` failed: {error}"),
                },
                None => format!("unknown tool `{name}`"),
            };
            results.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "name": name,
                "content": content,
            }));
        }
        Ok(NodeOutput::new().update(&self.messages_key, Value::Array(results)))
    }

    /// Build a node function suitable for [`StateGraph::add_node`].
    pub fn node(&self) -> NodeFn<JsonState> {
        let node = self.clone();
        Arc::new(move |state: &JsonState| node.execute(state))
    }
}

/// Build a minimal ReAct agent: `agent` (the model node) ↔ `tools`.
pub fn create_react_agent<I>(model: NodeFn<JsonState>, tools: I) -> StateGraph<JsonState>
where
    I: IntoIterator<Item = ToolDefinition>,
{
    let tool_node = ToolNode::new(tools);
    StateGraph::<JsonState>::with_schema(
        StateSchema::dynamic()
            .with_reducer("messages", reducers::messages())
            .with_managed_values(),
    )
    .add_node("agent", move |state: &JsonState| {
        let output = model(state)?;
        if crate::remaining_steps().is_some_and(|remaining| remaining <= 2) {
            let calls_tools = output
                .updates_map()
                .get("messages")
                .and_then(Value::as_array)
                .and_then(|messages| messages.last())
                .and_then(|message| message.get("tool_calls"))
                .and_then(Value::as_array)
                .is_some_and(|calls| !calls.is_empty());
            if calls_tools {
                let mut guarded = NodeOutput::new();
                for (channel, value) in output.updates_map() {
                    if channel != "messages" {
                        guarded = guarded.update(channel.clone(), value.clone());
                    }
                }
                return Ok(guarded.update(
                    "messages",
                    json!([{
                        "role": "assistant",
                        "content": "Sorry, need more steps to process this request.",
                    }]),
                ));
            }
        }
        Ok(output)
    })
    .add_node(TOOLS, move |state: &JsonState| tool_node.execute(state))
    .set_entry_point("agent")
    .add_conditional_edges("agent", |state: &JsonState| Ok(tools_condition(state)))
    .add_edge(TOOLS, "agent")
}

/// LangGraph's `END` marker (re-exported for prebuilt users).
pub const END_MARKER: &str = END;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_condition_checks_the_last_message() {
        let mut state = JsonState::new();
        state.insert(
            "messages".to_string(),
            json!([{ "role": "assistant", "tool_calls": [{ "id": "1", "name": "add" }] }]),
        );
        let route = tools_condition(&state);
        assert_eq!(route.targets(), [TOOLS]);
        state.insert("messages".to_string(), json!([{ "role": "assistant" }]));
        assert!(tools_condition(&state).is_end());
    }
}
