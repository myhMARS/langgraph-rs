//! Node-local helper functions backed by the sync thread-local / async task-local context.

use serde::Serialize;
use serde_json::Value;

use super::*;

pub fn last_error() -> Option<NodeFailure> {
    #[cfg(feature = "async")]
    if !has_sync_task()
        && let Ok(ops) = ASYNC_TASK.try_with(|ops| ops.last_error())
    {
        return ops;
    }
    CURRENT_TASK.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|context| context.error.clone())
    })
}

/// Pause the graph and wait for human input.
///
/// On the first execution this returns [`Error::Interrupt`] carrying `value`; the
/// engine checkpoints the state and unwinds. When the graph is resumed with
/// [`CompiledGraph::resume`](crate::CompiledGraph::resume), the node is re-executed and
/// this same call returns the value supplied by the caller instead.
///
/// ```no_run
/// use langgraph_rs::{interrupt, NodeOutput};
/// # fn example(state: &()) -> langgraph_rs::Result<NodeOutput> {
/// let answer: String = interrupt(serde_json::json!({ "question": "approve?" }))?
///     .as_str()
///     .unwrap_or("no")
///     .to_string();
/// Ok(NodeOutput::new().set("answer", answer))
/// # }
/// ```
pub fn interrupt(value: Value) -> Result<Value> {
    interrupt_with_schema(value, None)
}

/// Like [`interrupt`], but validates the resume value against a JSON schema.
///
/// The schema is stored on the interrupt (visible in snapshots) and the resume value is
/// validated when the node re-runs; a mismatch fails the node with a descriptive error.
pub fn interrupt_with_schema(value: Value, response_schema: Option<Value>) -> Result<Value> {
    #[cfg(feature = "async")]
    if !has_sync_task()
        && let Ok(result) =
            ASYNC_TASK.try_with(|ops| ops.interrupt_value(value.clone(), response_schema.clone()))
    {
        return result;
    }
    CURRENT_TASK.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(context) = slot.as_mut() else {
            return Err(Error::other(
                "interrupt() can only be called from inside a running graph node",
            ));
        };
        let id = super::next_interrupt_id(context);
        if context.resume_index < context.resume_values.len() {
            let value = context.resume_values[context.resume_index].clone();
            context.resume_index += 1;
            context.resume_consumed = true;
            validate_response(&value, response_schema.as_ref())?;
            return Ok(value);
        }
        Err(Error::Interrupt(Interrupt::with_schema_and_id(
            context.node.clone(),
            value,
            response_schema,
            id,
        )))
    })
}

/// Validate a resume value against the supported JSON-schema subset:
/// `type` (string or array), `const`, `enum`, `anyOf`/`oneOf`/`allOf`, `not`,
/// `required`, `properties`, `additionalProperties`, `items`, numeric bounds,
/// string lengths, array size/unique items and object property counts.
pub(crate) fn validate_response(value: &Value, schema: Option<&Value>) -> Result<()> {
    let Some(schema) = schema else {
        return Ok(());
    };
    validate_schema(value, schema, schema, 0)
}

fn validate_schema(value: &Value, schema: &Value, root: &Value, depth: usize) -> Result<()> {
    if depth > 64 {
        return Err(Error::other("response_schema $ref nesting is too deep"));
    }
    let Some(object) = schema.as_object() else {
        return Ok(());
    };
    if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
        let resolved = resolve_schema_ref(root, reference)?;
        validate_schema(value, resolved, root, depth + 1)?;
    }
    if let Some(pattern) = object.get("pattern").and_then(Value::as_str) {
        let regex = regex::Regex::new(pattern).map_err(|error| {
            Error::other(format!(
                "invalid response_schema pattern `{pattern}`: {error}"
            ))
        })?;
        if let Some(text) = value.as_str()
            && !regex.is_match(text)
        {
            return Err(Error::other(format!(
                "resume value does not match response_schema pattern `{pattern}`: {value}"
            )));
        }
    }
    if let Some(kind) = object.get("type") {
        let types: Vec<&str> = match kind {
            Value::String(kind) => vec![kind.as_str()],
            Value::Array(kinds) => kinds.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        if !types.is_empty() && !types.iter().any(|kind| matches_type(value, kind)) {
            return Err(Error::other(format!(
                "resume value does not match response_schema type {types:?}: {value}"
            )));
        }
    }
    if let Some(expected) = object.get("const")
        && expected != value
    {
        return Err(Error::other(format!(
            "resume value does not match response_schema const `{expected}`: {value}"
        )));
    }
    if let Some(enum_values) = object.get("enum").and_then(Value::as_array)
        && !enum_values.iter().any(|candidate| candidate == value)
    {
        return Err(Error::other(format!(
            "resume value is not one of the response_schema enum values: {value}"
        )));
    }
    if let Some(any_of) = object.get("anyOf").and_then(Value::as_array)
        && !any_of
            .iter()
            .any(|candidate| validate_schema(value, candidate, root, depth + 1).is_ok())
    {
        return Err(Error::other(format!(
            "resume value does not match any response_schema anyOf branch: {value}"
        )));
    }
    if let Some(one_of) = object.get("oneOf").and_then(Value::as_array) {
        let matches = one_of
            .iter()
            .filter(|candidate| validate_schema(value, candidate, root, depth + 1).is_ok())
            .count();
        if matches != 1 {
            return Err(Error::other(format!(
                "resume value matches {matches} response_schema oneOf branches (expected 1): {value}"
            )));
        }
    }
    if let Some(all_of) = object.get("allOf").and_then(Value::as_array) {
        for candidate in all_of {
            validate_schema(value, candidate, root, depth + 1)?;
        }
    }
    if let Some(not) = object.get("not")
        && validate_schema(value, not, root, depth + 1).is_ok()
    {
        return Err(Error::other(format!(
            "resume value matches a response_schema `not` branch: {value}"
        )));
    }
    if let Some(number) = value.as_f64() {
        if let Some(minimum) = object.get("minimum").and_then(Value::as_f64)
            && number < minimum
        {
            return Err(Error::other(format!(
                "resume value is below response_schema minimum {minimum}: {value}"
            )));
        }
        if let Some(maximum) = object.get("maximum").and_then(Value::as_f64)
            && number > maximum
        {
            return Err(Error::other(format!(
                "resume value is above response_schema maximum {maximum}: {value}"
            )));
        }
        if let Some(minimum) = object.get("exclusiveMinimum").and_then(Value::as_f64)
            && number <= minimum
        {
            return Err(Error::other(format!(
                "resume value is not above response_schema exclusiveMinimum {minimum}: {value}"
            )));
        }
        if let Some(maximum) = object.get("exclusiveMaximum").and_then(Value::as_f64)
            && number >= maximum
        {
            return Err(Error::other(format!(
                "resume value is not below response_schema exclusiveMaximum {maximum}: {value}"
            )));
        }
        if let Some(step) = object.get("multipleOf").and_then(Value::as_f64)
            && step > 0.0
            && (number / step - (number / step).round()).abs() > 1e-9
        {
            return Err(Error::other(format!(
                "resume value is not a multiple of response_schema multipleOf {step}: {value}"
            )));
        }
    }
    if let Some(string) = value.as_str() {
        let length = string.chars().count();
        if let Some(min_length) = object.get("minLength").and_then(Value::as_u64)
            && (length as u64) < min_length
        {
            return Err(Error::other(format!(
                "resume value is shorter than response_schema minLength {min_length}: {value}"
            )));
        }
        if let Some(max_length) = object.get("maxLength").and_then(Value::as_u64)
            && (length as u64) > max_length
        {
            return Err(Error::other(format!(
                "resume value is longer than response_schema maxLength {max_length}: {value}"
            )));
        }
    }
    if let Some(array) = value.as_array() {
        if let Some(min_items) = object.get("minItems").and_then(Value::as_u64)
            && (array.len() as u64) < min_items
        {
            return Err(Error::other(format!(
                "resume value has fewer than response_schema minItems {min_items}: {value}"
            )));
        }
        if let Some(max_items) = object.get("maxItems").and_then(Value::as_u64)
            && (array.len() as u64) > max_items
        {
            return Err(Error::other(format!(
                "resume value has more than response_schema maxItems {max_items}: {value}"
            )));
        }
        if object
            .get("uniqueItems")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            for (index, item) in array.iter().enumerate() {
                if array[..index].iter().any(|candidate| candidate == item) {
                    return Err(Error::other(format!(
                        "resume value violates response_schema uniqueItems: {value}"
                    )));
                }
            }
        }
    }
    if let Some(object_value) = value.as_object() {
        if let Some(required) = object.get("required").and_then(Value::as_array) {
            for key in required.iter().filter_map(Value::as_str) {
                if object_value.get(key).is_none() {
                    return Err(Error::other(format!(
                        "resume value is missing required response_schema property `{key}`"
                    )));
                }
            }
        }
        if let Some(properties) = object.get("properties").and_then(Value::as_object) {
            for (key, property_schema) in properties {
                if let Some(property) = object_value.get(key) {
                    validate_schema(property, property_schema, root, depth + 1)?;
                }
            }
        }
        if object.get("additionalProperties").and_then(Value::as_bool) == Some(false)
            && let Some(properties) = object.get("properties").and_then(Value::as_object)
        {
            for key in object_value.keys() {
                if !properties.contains_key(key) {
                    return Err(Error::other(format!(
                        "resume value has response_schema additionalProperties=false but contains `{key}`"
                    )));
                }
            }
        }
        if let Some(min_properties) = object.get("minProperties").and_then(Value::as_u64)
            && (object_value.len() as u64) < min_properties
        {
            return Err(Error::other(format!(
                "resume value has fewer than response_schema minProperties {min_properties}: {value}"
            )));
        }
        if let Some(max_properties) = object.get("maxProperties").and_then(Value::as_u64)
            && (object_value.len() as u64) > max_properties
        {
            return Err(Error::other(format!(
                "resume value has more than response_schema maxProperties {max_properties}: {value}"
            )));
        }
    }
    if let Some(items) = object.get("items")
        && let Some(array) = value.as_array()
    {
        match items {
            Value::Array(tuple_schemas) => {
                for (index, item) in array.iter().enumerate() {
                    if let Some(item_schema) = tuple_schemas.get(index) {
                        validate_schema(item, item_schema, root, depth + 1)?;
                    }
                }
            }
            schema => {
                for item in array {
                    validate_schema(item, schema, root, depth + 1)?;
                }
            }
        }
    }
    Ok(())
}

fn resolve_schema_ref<'a>(root: &'a Value, reference: &str) -> Result<&'a Value> {
    let pointer = reference.strip_prefix('#').ok_or_else(|| {
        Error::other(format!(
            "response_schema $ref must be a local pointer, got `{reference}`"
        ))
    })?;
    if pointer.is_empty() {
        return Ok(root);
    }
    let pointer = pointer.strip_prefix('/').ok_or_else(|| {
        Error::other(format!(
            "response_schema $ref must start with `#/`, got `{reference}`"
        ))
    })?;
    root.pointer(&format!("/{pointer}")).ok_or_else(|| {
        Error::other(format!(
            "response_schema $ref `{reference}` cannot be resolved"
        ))
    })
}

fn matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
    }
}

/// LangGraph-style config of the current run (`configurable.thread_id` / `checkpoint_ns`).
pub fn get_config() -> Value {
    let (tags, mut metadata) = crate::runtime::context::user_config();
    // LangGraph merges the run metadata with the task's own `langgraph_*` keys.
    if let Some(thread_id) = current_thread_id() {
        metadata.insert("thread_id".into(), Value::String(thread_id));
    }
    if let Some(node) = crate::runtime::context::current_node() {
        metadata.insert("langgraph_node".into(), Value::String(node.clone()));
        let namespace = current_namespace();
        let mut path = vec![Value::String("__pregel_pull".into()), Value::String(node)];
        if !namespace.is_empty() {
            path.extend(
                namespace
                    .split('|')
                    .filter(|part| !part.is_empty())
                    .map(|part| Value::String(part.to_string())),
            );
        }
        metadata.insert("langgraph_path".into(), Value::Array(path));
    }
    if let Some(step) = crate::runtime::context::current_step() {
        // LangGraph counts supersteps from 1 in this metadata.
        metadata.insert("langgraph_step".into(), Value::from(step as u64 + 1));
    }
    let triggers = crate::runtime::context::current_triggers();
    if !triggers.is_empty() {
        metadata.insert(
            "langgraph_triggers".into(),
            Value::Array(triggers.into_iter().map(Value::String).collect()),
        );
    }
    // `<node>:<task id>` for the running task, the run's namespace otherwise.
    let checkpoint_ns = match crate::runtime::context::current_task_id() {
        Some(id) => match crate::runtime::context::current_node() {
            Some(node) if !node.is_empty() => format!("{node}:{id}"),
            _ => id,
        },
        None => current_namespace(),
    };
    if !checkpoint_ns.is_empty() {
        metadata.insert(
            "langgraph_checkpoint_ns".into(),
            Value::String(checkpoint_ns),
        );
    }
    let mut config = serde_json::json!({
        "callbacks": [],
        "configurable": {
            "thread_id": current_thread_id(),
            "checkpoint_ns": current_namespace(),
            "__pregel_task_id": crate::runtime::context::current_task_id(),
        },
        "metadata": metadata,
    });
    // LangGraph only reports `tags` when the run actually carries some.
    if !tags.is_empty()
        && let Some(object) = config.as_object_mut()
    {
        object.insert("tags".to_string(), serde_json::json!(tags));
    }
    config
}

/// Return a writer that emits custom stream values from helpers.
///
/// The writer fails outside a running node (the context is per node execution).
pub fn get_stream_writer() -> impl Fn(Value) -> Result<()> + Clone {
    |value| push_custom(value)
}

/// Emit a custom value into the graph stream from inside a node.
pub fn custom(value: impl Serialize) -> Result<()> {
    crate::runtime::context::bump_progress();
    push_custom(serde_json::to_value(value)?)
}

/// Emit an already serialized custom value into the graph stream.
/// Emit `chunk` to the run's live sink when one is installed.
///
/// Returns `true` when the chunk was handed to a live consumer; the caller then skips buffering so
/// the chunk is not delivered twice (once live, once with the task result).
fn push_live(chunk: StreamChunk) -> bool {
    let Some(sink) = crate::runtime::context::current_live_sink() else {
        return false;
    };
    // A chunk produced inside a subgraph carries that subgraph's namespace.
    let namespace = crate::runtime::context::current_namespace();
    let wrapped = if namespace.is_empty() {
        chunk
    } else {
        StreamChunk::Subgraph {
            namespace,
            chunk: Box::new(chunk),
        }
    };
    let mut sink = sink.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    sink(wrapped);
    true
}

/// Write `value` to `channel` from inside a node (LangGraph's `CONFIG_KEY_SEND`).
///
/// The write is applied with the node's own updates once the node returns, so the reducer sees it
/// exactly as if the node had returned it.
pub fn send(channel: impl Into<String>, value: Value) -> Result<()> {
    let channel = channel.into();
    CURRENT_TASK.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(context) = slot.as_mut() else {
            return Err(Error::other(
                "send() can only be called from inside a running graph node",
            ));
        };
        context.push_write(channel, value);
        Ok(())
    })
}

pub fn push_custom(value: Value) -> Result<()> {
    if push_live(StreamChunk::Custom {
        node: crate::runtime::context::current_node().unwrap_or_default(),
        value: value.clone(),
    }) {
        return Ok(());
    }
    #[cfg(feature = "async")]
    if !has_sync_task()
        && let Ok(result) = ASYNC_TASK.try_with(|ops| ops.push_custom(value.clone()))
    {
        return result;
    }
    CURRENT_TASK.with(|slot| match slot.borrow_mut().as_mut() {
        Some(context) => {
            context.custom.push(value);
            Ok(())
        }
        None => Err(Error::other(
            "custom() can only be called from inside a running graph node",
        )),
    })
}

/// Emit a tool-call lifecycle event (`stream_mode="tools"`) from inside a node.
///
/// `ToolNode` uses this to report `tool-started` / `tool-finished` / `tool-error`; the payload is
/// passed through unchanged.
pub fn push_tool_event(value: Value) -> Result<()> {
    if push_live(StreamChunk::Tool {
        node: crate::runtime::context::current_node().unwrap_or_default(),
        value: value.clone(),
    }) {
        return Ok(());
    }
    #[cfg(feature = "async")]
    if !has_sync_task()
        && let Ok(result) = ASYNC_TASK.try_with(|ops| ops.push_tool_event(value.clone()))
    {
        return result;
    }
    CURRENT_TASK.with(|slot| match slot.borrow_mut().as_mut() {
        Some(context) => {
            context.tools.push(value);
            Ok(())
        }
        None => Err(Error::other(
            "tool events can only be emitted from inside a running graph node",
        )),
    })
}
/// Emit a message (for example an LLM token) into the graph stream from inside a node.
///
/// This is the langgraph-rs analogue of LangGraph's `stream_mode="messages"` events;
/// collected chunks are exposed as [`StreamChunk::Message`](crate::StreamChunk::Message).
pub fn message(value: impl Serialize) -> Result<()> {
    push_message(serde_json::to_value(value)?)
}

/// Emit an already serialized message into the graph stream.
pub fn push_message(value: Value) -> Result<()> {
    if push_live(StreamChunk::Message {
        node: crate::runtime::context::current_node().unwrap_or_default(),
        message: value.clone(),
    }) {
        return Ok(());
    }
    #[cfg(feature = "async")]
    if !has_sync_task()
        && let Ok(result) = ASYNC_TASK.try_with(|ops| ops.push_message(value.clone()))
    {
        return result;
    }
    CURRENT_TASK.with(|slot| match slot.borrow_mut().as_mut() {
        Some(context) => {
            context.messages.push(value);
            Ok(())
        }
        None => Err(Error::other(
            "message() can only be called from inside a running graph node",
        )),
    })
}

/// Record what the current node consumed (tokens, model, cost, ...).
///
/// Records are attached to the node's [`NodeExecution`](crate::NodeExecution) and
/// aggregated into the run's [`RunReport`](crate::RunReport). Multiple calls from one
/// node are summed.
pub fn record_usage(usage: Usage) -> Result<()> {
    #[cfg(feature = "async")]
    if !has_sync_task()
        && let Ok(result) = ASYNC_TASK.try_with(|ops| ops.record_usage(usage.clone()))
    {
        return result;
    }
    CURRENT_TASK.with(|slot| match slot.borrow_mut().as_mut() {
        Some(context) => {
            context.usage.push(usage);
            Ok(())
        }
        None => Err(Error::other(
            "record_usage() can only be called from inside a running graph node",
        )),
    })
}

/// Convenience shorthand for [`record_usage`] with token counts.
pub fn record_tokens(input_tokens: u64, output_tokens: u64) -> Result<()> {
    record_usage(
        Usage::new()
            .with_input_tokens(input_tokens)
            .with_output_tokens(output_tokens),
    )
}
