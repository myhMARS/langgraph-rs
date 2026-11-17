//! Dynamic node context: human-in-the-loop interrupts and custom stream writes.
//!
//! The engine installs a thread-local task context while a node is running. Nodes can
//! then call [`interrupt`] or [`custom`] without threading a context argument through
//! every signature, which keeps node closures identical to their LangGraph counterparts.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde_json::{Map, Value};

use crate::kernel::error::NodeFailure;
use crate::kernel::store::{BaseStore, Runtime};
use crate::kernel::stream::StreamChunk;
use crate::kernel::usage::Usage;
use crate::{Error, Interrupt, Result};

thread_local! {
    static CURRENT_TASK: RefCell<Option<TaskContext>> = const { RefCell::new(None) };
    static CURRENT_RUN: RefCell<Option<RunContext>> = const { RefCell::new(None) };
    static CURRENT_STEP: RefCell<Option<(usize, usize)>> = const { RefCell::new(None) };
    static SUBGRAPH_COLLECTOR: RefCell<Option<Arc<Mutex<Vec<StreamChunk>>>>> =
        const { RefCell::new(None) };
    static PARENT_COMMAND: RefCell<Option<ParentCommand>> = const { RefCell::new(None) };
    static TASK_RESULTS: RefCell<Map<String, Value>> = RefCell::new(Map::new());
}

/// `Command(graph=PARENT)` collected from a subgraph node.
#[derive(Clone, Debug, Default)]
pub(crate) struct ParentCommand {
    /// Writes applied to the parent state.
    pub updates: Vec<(String, Value)>,
    /// Parent nodes to schedule.
    pub goto: Vec<String>,
}

pub(crate) fn clear_parent_command() {
    PARENT_COMMAND.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

pub(crate) fn record_parent_command(updates: Vec<(String, Value)>, goto: Vec<String>) {
    PARENT_COMMAND.with(|slot| {
        let mut slot = slot.borrow_mut();
        let command = slot.get_or_insert_with(ParentCommand::default);
        command.updates.extend(updates);
        command.goto.extend(goto);
    });
}

pub(crate) fn take_parent_command() -> Option<ParentCommand> {
    PARENT_COMMAND.with(|slot| slot.borrow_mut().take())
}

pub(crate) fn has_parent_command() -> bool {
    PARENT_COMMAND.with(|slot| slot.borrow().is_some())
}

/// Execution context of the current run.
#[derive(Clone, Default)]
pub(crate) struct RunContext {
    pub(crate) thread_id: Option<String>,
    pub(crate) namespace: String,
    pub(crate) store: Option<std::sync::Arc<dyn BaseStore>>,
    pub(crate) context: Option<Value>,
}

/// RAII guard restoring the previous run context when the engine returns.
pub(crate) struct RunGuard {
    previous: Option<RunContext>,
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CURRENT_RUN.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

/// Install the run context for the duration of a graph execution.
pub(crate) fn install_run(
    thread_id: Option<String>,
    namespace: String,
    store: Option<std::sync::Arc<dyn BaseStore>>,
    context: Option<Value>,
) -> RunGuard {
    let previous = CURRENT_RUN.with(|slot| {
        slot.borrow_mut().replace(RunContext {
            thread_id,
            namespace,
            store,
            context,
        })
    });
    RunGuard { previous }
}

/// Long-term memory store of the current run, when the graph was compiled with one.
pub fn get_store() -> Option<std::sync::Arc<dyn BaseStore>> {
    CURRENT_RUN.with(|slot| slot.borrow().as_ref().and_then(|run| run.store.clone()))
}

/// User context passed to the current run via `RunConfig::with_context`.
pub fn current_context() -> Option<Value> {
    CURRENT_RUN.with(|slot| slot.borrow().as_ref().and_then(|run| run.context.clone()))
}

/// Runtime information (thread, namespace, store, context) of the current run.
pub fn current_runtime() -> Option<Runtime> {
    CURRENT_RUN.with(|slot| {
        slot.borrow().as_ref().map(|run| Runtime {
            thread_id: run.thread_id.clone(),
            namespace: run.namespace.clone(),
            store: run.store.clone(),
            context: run.context.clone(),
        })
    })
}

/// RAII guard restoring the previous step context.
pub(crate) struct StepGuard {
    previous: Option<(usize, usize)>,
}

impl Drop for StepGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CURRENT_STEP.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

/// Install the current superstep number and recursion limit for managed values.
pub(crate) fn install_step(step: usize, recursion_limit: usize) -> StepGuard {
    let previous = CURRENT_STEP.with(|slot| slot.borrow_mut().replace((step, recursion_limit)));
    StepGuard { previous }
}

/// Remaining supersteps before the recursion limit (LangGraph `RemainingSteps`).
///
/// Returns `None` outside a run. The value is `recursion_limit - step`, so the first
/// superstep of a graph compiled with a limit of 25 reports 25.
pub fn remaining_steps() -> Option<usize> {
    CURRENT_STEP.with(|slot| {
        slot.borrow()
            .map(|(step, recursion_limit)| recursion_limit.saturating_sub(step + 1))
    })
}

/// Whether the current superstep is the last one before the recursion limit.
///
/// Mirrors LangGraph's `IsLastStep` managed value: `true` when at most one superstep is
/// left (`remaining_steps() <= 1`).
pub fn is_last_step() -> bool {
    remaining_steps()
        .map(|remaining| remaining <= 1)
        .unwrap_or(false)
}

/// Thread id of the current run, when it is bound to a checkpoint thread.
pub fn current_thread_id() -> Option<String> {
    CURRENT_RUN.with(|slot| slot.borrow().as_ref().and_then(|run| run.thread_id.clone()))
}

/// Checkpoint namespace of the current run (`""` for the root graph).
pub fn current_namespace() -> String {
    CURRENT_RUN.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|run| run.namespace.clone())
            .unwrap_or_default()
    })
}

/// Resume value stamped onto the current task, when replaying an interrupt.
pub fn current_resume() -> Option<Value> {
    CURRENT_TASK.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|context| context.resume_values.get(context.resume_index).cloned())
    })
}

#[derive(Debug, Default)]
pub(crate) struct TaskContext {
    node: String,
    resume_values: Vec<Value>,
    resume_index: usize,
    resume_consumed: bool,
    custom: Vec<Value>,
    messages: Vec<Value>,
    usage: Vec<Usage>,
    error: Option<NodeFailure>,
}

impl TaskContext {
    pub(crate) fn new(node: impl Into<String>, resume_values: Vec<Value>) -> Self {
        Self {
            node: node.into(),
            resume_values,
            resume_index: 0,
            resume_consumed: false,
            custom: Vec::new(),
            messages: Vec::new(),
            usage: Vec::new(),
            error: None,
        }
    }

    pub(crate) fn with_error(mut self, error: Option<NodeFailure>) -> Self {
        self.error = error;
        self
    }
}

/// Install `context` for the duration of `run`, returning its result, any custom
/// stream writes and any consumption records the node reported.
pub(crate) fn with_task<R>(
    context: TaskContext,
    run: impl FnOnce() -> R,
) -> (R, Vec<Value>, Vec<Value>, Vec<Usage>) {
    let previous = CURRENT_TASK.with(|slot| slot.borrow_mut().replace(context));
    let result = run();
    let context = CURRENT_TASK.with(|slot| {
        let mut slot = slot.borrow_mut();
        std::mem::replace(&mut *slot, previous)
    });
    let (custom, messages, usage) = context
        .map(|context| (context.custom, context.messages, context.usage))
        .unwrap_or_default();
    (result, custom, messages, usage)
}

/// Failure that caused the current node to be scheduled.
///
/// When a node fails permanently and the graph has an error edge
/// ([`StateGraph::add_error_edge`](crate::StateGraph::add_error_edge)), the handler node
/// runs and can inspect the failure with this function.
pub fn last_error() -> Option<NodeFailure> {
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
    CURRENT_TASK.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(context) = slot.as_mut() else {
            return Err(Error::other(
                "interrupt() can only be called from inside a running graph node",
            ));
        };
        if context.resume_index < context.resume_values.len() {
            let value = context.resume_values[context.resume_index].clone();
            context.resume_index += 1;
            context.resume_consumed = true;
            validate_response(&value, response_schema.as_ref())?;
            return Ok(value);
        }
        Err(Error::Interrupt(Interrupt::with_schema(
            context.node.clone(),
            value,
            response_schema,
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
    serde_json::json!({
        "configurable": {
            "thread_id": current_thread_id(),
            "checkpoint_ns": current_namespace(),
        }
    })
}

/// Return a writer that emits custom stream values from helpers.
///
/// The writer fails outside a running node (the context is per node execution).
pub fn get_stream_writer() -> impl Fn(Value) -> Result<()> + Clone {
    |value| push_custom(value)
}

/// Emit a custom value into the graph stream from inside a node.
pub fn custom(value: impl Serialize) -> Result<()> {
    push_custom(serde_json::to_value(value)?)
}

/// Emit an already serialized custom value into the graph stream.
pub fn push_custom(value: Value) -> Result<()> {
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

/// Emit a message (for example an LLM token) into the graph stream from inside a node.
///
/// This is the langgraph-rs analogue of LangGraph's `stream_mode="messages"` events;
/// collected chunks are exposed as [`StreamChunk::Message`](crate::StreamChunk::Message).
pub fn message(value: impl Serialize) -> Result<()> {
    push_message(serde_json::to_value(value)?)
}

/// Emit an already serialized message into the graph stream.
pub fn push_message(value: Value) -> Result<()> {
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

/// Name of the node that is currently executing, if any.
/// Superstep currently being executed, when called from a node.
pub fn current_step() -> Option<usize> {
    CURRENT_STEP.with(|slot| slot.borrow().map(|(step, _)| step))
}

pub fn current_node() -> Option<String> {
    CURRENT_TASK.with(|slot| slot.borrow().as_ref().map(|context| context.node.clone()))
}

/// RAII guard restoring the previous subgraph chunk collector.
pub(crate) struct CollectorGuard {
    previous: Option<Arc<Mutex<Vec<StreamChunk>>>>,
}

impl Drop for CollectorGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        SUBGRAPH_COLLECTOR.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

/// Install a collector that subgraph nodes push their chunks into.
pub(crate) fn install_subgraph_collector(
    collector: Arc<Mutex<Vec<StreamChunk>>>,
) -> CollectorGuard {
    let previous = SUBGRAPH_COLLECTOR.with(|slot| slot.borrow_mut().replace(collector));
    CollectorGuard { previous }
}

/// Collector installed by the running executor, when subgraph streaming is enabled.
pub(crate) fn current_subgraph_collector() -> Option<Arc<Mutex<Vec<StreamChunk>>>> {
    SUBGRAPH_COLLECTOR.with(|slot| slot.borrow().clone())
}

/// Read a memoized functional-API task result for `key`.
pub(crate) fn task_result_get(key: &str) -> Option<Value> {
    TASK_RESULTS.with(|slot| slot.borrow().get(key).cloned())
}

/// Store a functional-API task result under `key`.
pub(crate) fn task_result_put(key: String, value: Value) {
    TASK_RESULTS.with(|slot| {
        slot.borrow_mut().insert(key, value);
    });
}

/// Reset the task-result cache at the start of a fresh run.
pub(crate) fn clear_task_results() {
    TASK_RESULTS.with(|slot| slot.borrow_mut().clear());
}

/// Restore memoized task results from a resumed checkpoint.
pub(crate) fn load_task_results(results: &Map<String, Value>) {
    TASK_RESULTS.with(|slot| {
        let mut slot = slot.borrow_mut();
        slot.clear();
        slot.extend(results.clone());
    });
}

/// Snapshot of the memoized task results for checkpoint persistence.
pub(crate) fn snapshot_task_results() -> Map<String, Value> {
    TASK_RESULTS.with(|slot| slot.borrow().clone())
}

#[cfg(test)]
mod response_schema_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_rich_keywords() {
        let schema = json!({
            "type": "object",
            "required": ["choice"],
            "additionalProperties": false,
            "properties": {
                "choice": {
                    "anyOf": [
                        {"const": "yes"},
                        {"type": "integer", "minimum": 1}
                    ]
                },
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": true,
                    "items": {"type": "string", "minLength": 2}
                },
                "code": {"type": ["string", "null"], "maxLength": 4}
            }
        });
        validate_response(
            &json!({"choice": "yes", "tags": ["ab", "cd"], "code": null}),
            Some(&schema),
        )
        .unwrap();
        validate_response(&json!({"choice": 2, "tags": ["ab"]}), Some(&schema)).unwrap();
        assert!(validate_response(&json!({"choice": 0}), Some(&schema)).is_err());
        assert!(
            validate_response(&json!({"choice": "yes", "tags": ["a"]}), Some(&schema)).is_err()
        );
        assert!(
            validate_response(
                &json!({"choice": "yes", "tags": ["ab", "ab"]}),
                Some(&schema)
            )
            .is_err()
        );
        assert!(
            validate_response(
                &json!({"choice": "yes", "tags": ["ab"], "extra": 1}),
                Some(&schema)
            )
            .is_err()
        );
        assert!(
            validate_response(
                &json!({"choice": "yes", "tags": ["ab"], "code": "toolong"}),
                Some(&schema)
            )
            .is_err()
        );
    }

    #[test]
    fn validates_one_of_all_of_not_and_bounds() {
        let one_of = json!({"oneOf": [{"type": "integer"}, {"type": "string"}]});
        assert!(validate_response(&json!(true), Some(&one_of)).is_err());
        validate_response(&json!(1), Some(&one_of)).unwrap();

        let all_of = json!({"allOf": [{"type": "integer"}, {"minimum": 2}]});
        validate_response(&json!(3), Some(&all_of)).unwrap();
        assert!(validate_response(&json!(1), Some(&all_of)).is_err());

        let not = json!({"not": {"type": "string"}});
        validate_response(&json!(1), Some(&not)).unwrap();
        assert!(validate_response(&json!("x"), Some(&not)).is_err());

        let bounds =
            json!({"type": "number", "exclusiveMinimum": 0, "maximum": 10, "multipleOf": 2});
        validate_response(&json!(4), Some(&bounds)).unwrap();
        assert!(validate_response(&json!(0), Some(&bounds)).is_err());
        assert!(validate_response(&json!(3), Some(&bounds)).is_err());
    }

    #[test]
    fn validates_pattern_and_local_refs() {
        let schema = json!({
            "type": "object",
            "properties": {
                "code": {"$ref": "#/$defs/code"},
                "label": {"type": "string", "pattern": "^[a-z]+$"}
            },
            "required": ["code"],
            "$defs": {
                "code": {"type": "string", "pattern": "^[A-Z]{3}$"}
            }
        });
        validate_response(&json!({"code": "ABC", "label": "ok"}), Some(&schema)).unwrap();
        assert!(validate_response(&json!({"code": "abc"}), Some(&schema)).is_err());
        assert!(validate_response(&json!({"code": "ABC", "label": "NO"}), Some(&schema)).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn interrupt_without_resume_returns_the_payload() {
        let (result, _custom, _messages, _usage) =
            with_task(TaskContext::new("ask", Vec::new()), || {
                interrupt(json!({ "question": "approve?" }))
            });
        let error = result.unwrap_err();
        assert_eq!(
            error.interrupt().map(|interrupt| interrupt.node.as_str()),
            Some("ask")
        );
        assert_eq!(
            error.interrupt().map(|interrupt| interrupt.value.clone()),
            Some(json!({ "question": "approve?" }))
        );
    }

    #[test]
    fn interrupt_with_resume_value_returns_it() {
        let (result, _custom, _messages, _usage) =
            with_task(TaskContext::new("ask", vec![json!("yes")]), || {
                interrupt(json!({ "question": "approve?" }))
            });
        assert_eq!(result.unwrap(), json!("yes"));
    }

    #[test]
    fn custom_values_are_collected() {
        let (result, custom, messages, usage) =
            with_task(TaskContext::new("emit", Vec::new()), || {
                custom(json!({ "progress": 0.5 })).unwrap();
                push_custom(json!("done")).unwrap();
                record_usage(Usage::new().with_input_tokens(3)).unwrap();
                record_tokens(1, 2).unwrap();
                current_node()
            });
        assert_eq!(result.as_deref(), Some("emit"));
        assert_eq!(custom, vec![json!({ "progress": 0.5 }), json!("done")]);
        assert!(messages.is_empty());
        assert_eq!(usage.len(), 2);
        assert_eq!(usage[0].input_tokens, Some(3));
        assert_eq!(usage[1].total_tokens(), 3);
    }

    #[test]
    fn interrupt_outside_a_node_is_an_error() {
        assert!(interrupt(json!("nope")).is_err());
    }
}
