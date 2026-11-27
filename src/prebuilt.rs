//! Prebuilt helpers mirroring LangGraph's `langgraph.prebuilt` basics:
//! `tools_condition`, `ToolNode` and `create_react_agent`.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::_internal::node::{NodeFn, NodeOutput};
use crate::_internal::state::{JsonState, StateSchema};
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
///
/// The handler receives the model-supplied arguments plus the current state, so bindings can
/// inject `InjectedState` / `InjectedStore` parameters the way LangGraph's `ToolNode` does.
#[derive(Clone)]
pub struct ToolDefinition {
    name: String,
    run: ToolHandler,
}

/// Argument handler of a [`ToolDefinition`]: model arguments plus the current state.
pub type ToolHandler = Arc<dyn Fn(Value, &JsonState) -> Result<Value> + Send + Sync>;

impl std::fmt::Debug for ToolDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolDefinition")
            .field("name", &self.name)
            .finish()
    }
}

impl ToolDefinition {
    /// Build a tool from a name and an argument handler that ignores the graph state.
    pub fn new<F>(name: impl Into<String>, run: F) -> Self
    where
        F: Fn(Value) -> Result<Value> + Send + Sync + 'static,
    {
        Self::with_state(name, move |args, _state| run(args))
    }

    /// Build a tool whose handler also sees the current state (for injected arguments).
    pub fn with_state<F>(name: impl Into<String>, run: F) -> Self
    where
        F: Fn(Value, &JsonState) -> Result<Value> + Send + Sync + 'static,
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
    /// Maps a tool failure message to the `ToolMessage` content; `None` propagates the error
    /// (LangGraph's `handle_tool_errors`).
    error_policy: Option<ToolErrorPolicy>,
}

/// Maps a tool failure message to the `ToolMessage` content (`None` propagates the error).
pub type ToolErrorPolicy = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// LangGraph's default error message for a failed tool call.
pub fn default_tool_error_policy() -> ToolErrorPolicy {
    Arc::new(|message: &str| {
        // `is_hidden` strips the invisible Python-exception marker some adapters append.
        Some(format!(
            "Error: {}\n Please fix your mistakes.",
            crate::_internal::error::strip_hidden_marker(message)
        ))
    })
}

impl std::fmt::Debug for ToolNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolNode")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .field("messages_key", &self.messages_key)
            .field("handles_tool_errors", &self.error_policy.is_some())
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
            error_policy: Some(default_tool_error_policy()),
        }
    }

    /// Channel that stores the message list.
    pub fn messages_key(mut self, key: impl Into<String>) -> Self {
        self.messages_key = key.into();
        self
    }

    /// Channel name holding the message list.
    pub fn messages_key_name(&self) -> &str {
        &self.messages_key
    }

    /// How tool failures are reported (LangGraph's `handle_tool_errors`).
    ///
    /// `None` propagates the exception; a policy maps the failure message to the content of an
    /// error `ToolMessage`.
    pub fn error_policy(mut self, policy: Option<ToolErrorPolicy>) -> Self {
        self.error_policy = policy;
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
            // `stream_mode="tools"` events, mirroring LangGraph's `StreamToolCallHandler`.
            let _ = crate::runtime::context::push_tool_event(json!({
                "event": "tool-started",
                "tool_call_id": id,
                "tool_name": name,
                "input": args,
            }));
            let (content, failed) = match self.tools.get(&name) {
                Some(tool) => match (tool.run)(args.clone(), state) {
                    Ok(value) => (
                        match value {
                            Value::String(text) => text,
                            other => other.to_string(),
                        },
                        None,
                    ),
                    Err(error) => {
                        // Interrupts (and other control-flow errors) are not tool failures: they
                        // must reach the engine so the graph can pause.
                        if matches!(error, crate::_internal::error::Error::Interrupt(_)) {
                            return Err(error);
                        }
                        let detail = error.to_string();
                        // A rejected `interrupt(response_schema=...)` resume is not a tool failure:
                        // LangGraph re-raises it so the graph stays paused.
                        #[cfg(feature = "python")]
                        if crate::adapters::python::convert::is_invalid_resume_error(&detail) {
                            return Err(error);
                        }
                        match &self.error_policy {
                            Some(policy) => match policy(&detail) {
                                Some(content) => (content, Some(detail)),
                                // The policy declined this error, so it propagates.
                                None => return Err(error),
                            },
                            None => return Err(error),
                        }
                    }
                },
                None => (
                    format!("unknown tool `{name}`"),
                    Some(format!("unknown tool `{name}`")),
                ),
            };
            let errored = failed.is_some();
            match failed {
                Some(message) => {
                    let _ = crate::runtime::context::push_tool_event(json!({
                        "event": "tool-error",
                        "tool_call_id": id,
                        "message": message,
                    }));
                }
                None => {
                    let _ = crate::runtime::context::push_tool_event(json!({
                        "event": "tool-finished",
                        "tool_call_id": id,
                        "output": content,
                    }));
                }
            }
            let mut message = json!({
                "role": "tool",
                "tool_call_id": id,
                "name": name,
                "content": content,
            });
            if errored {
                // LangGraph marks a failed tool call on the `ToolMessage`.
                message["status"] = Value::String("error".to_string());
            }
            results.push(message);
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
