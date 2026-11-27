//! Dynamic node context: human-in-the-loop interrupts and custom stream writes.
//!
//! The engine installs a thread-local task context while a node is running. Nodes can
//! then call [`interrupt`] or [`custom`] without threading a context argument through
//! every signature, which keeps node closures identical to their LangGraph counterparts.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value};

use crate::_internal::error::NodeFailure;
use crate::_internal::stream::StreamChunk;
use crate::_internal::usage::Usage;
use crate::store::{BaseStore, Runtime};
use crate::{Error, Interrupt, Result};

thread_local! {
    static CURRENT_TASK: RefCell<Option<TaskContext>> = const { RefCell::new(None) };
    static CURRENT_RUN: RefCell<Option<RunContext>> = const { RefCell::new(None) };
    static CURRENT_STEP: RefCell<Option<(usize, usize, usize)>> = const { RefCell::new(None) };
    static SUBGRAPH_COLLECTOR: RefCell<Option<Arc<Mutex<Vec<StreamChunk>>>>> =
        const { RefCell::new(None) };
    static PARENT_COMMAND: RefCell<Option<ParentCommand>> = const { RefCell::new(None) };
    static TASK_RESULTS: RefCell<Map<String, Value>> = RefCell::new(Map::new());
    static TASK_WRITES: RefCell<Map<String, Value>> = RefCell::new(Map::new());
}

#[cfg(feature = "async")]
tokio::task_local! {
    /// Run context mirrored into the async task so tokio can migrate the future.
    pub(crate) static ASYNC_RUN: RunContext;
    static ASYNC_TASK_RESULTS: RefCell<Map<String, Value>>;
    static ASYNC_TASK_WRITES: RefCell<Map<String, Value>>;
    static ASYNC_NODE: (String, Option<String>);
    static ASYNC_TASK: std::sync::Arc<dyn AsyncTaskOps>;
}

/// Operations of the async node currently executing, exposed to the free helper
/// functions (`interrupt`, `custom`, `message`, `record_usage`, `last_error`).
#[cfg(feature = "async")]
pub(crate) trait AsyncTaskOps: Send + Sync {
    fn interrupt_value(&self, value: Value, schema: Option<Value>) -> Result<Value>;
    fn push_custom(&self, value: Value) -> Result<()>;
    fn push_tool_event(&self, value: Value) -> Result<()>;
    fn push_message(&self, value: Value) -> Result<()>;
    fn record_usage(&self, usage: crate::_internal::usage::Usage) -> Result<()>;
    fn last_error(&self) -> Option<crate::_internal::error::NodeFailure>;
    fn resume_value(&self) -> Option<Value>;
    fn resume_last(&self) -> Option<Value>;
}

/// Scope a node future with the async task operations used by the free helpers.
#[cfg(feature = "async")]
pub(crate) async fn with_async_task<F: std::future::Future>(
    ops: std::sync::Arc<dyn AsyncTaskOps>,
    future: F,
) -> F::Output {
    ASYNC_TASK.scope(ops, future).await
}

/// `Command(graph=PARENT)` collected from a subgraph node.
#[derive(Clone, Debug, Default)]
pub(crate) struct ParentCommand {
    /// Reducer writes applied to the parent state.
    pub updates: Vec<(String, Value)>,
    /// Direct replacements applied to the parent state.
    pub replaces: Vec<(String, Value)>,
    /// Parent nodes to schedule.
    pub goto: Vec<String>,
}

pub(crate) fn clear_parent_command() {
    PARENT_COMMAND.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

pub(crate) fn record_parent_command(
    updates: Vec<(String, Value)>,
    replaces: Vec<(String, Value)>,
    goto: Vec<String>,
) {
    PARENT_COMMAND.with(|slot| {
        let mut slot = slot.borrow_mut();
        let command = slot.get_or_insert_with(ParentCommand::default);
        command.updates.extend(updates);
        command.replaces.extend(replaces);
        command.goto.extend(goto);
    });
}

pub(crate) fn take_parent_command() -> Option<ParentCommand> {
    PARENT_COMMAND.with(|slot| slot.borrow_mut().take())
}

pub(crate) fn has_parent_command() -> bool {
    PARENT_COMMAND.with(|slot| slot.borrow().is_some())
}

#[path = "context_run.rs"]
mod context_run;
pub(crate) use context_run::*;
pub use context_run::{current_context, current_runtime, get_store, is_last_step, remaining_steps};

/// Thread id of the current run, when it is bound to a checkpoint thread.
pub fn current_thread_id() -> Option<String> {
    current_run().and_then(|run| run.thread_id.clone())
}

/// Checkpoint namespace of the current run (`""` for the root graph).
pub fn current_namespace() -> String {
    current_run().map(|run| run.namespace).unwrap_or_default()
}

/// LangGraph-compatible interrupt id for the current task and answered index.
///
/// LangGraph hashes `f"{checkpoint_ns}:{task_id}"`, appending `|{answered}` only when the
/// task already answered an interrupt, so parallel `Send` tasks get distinct ids.
pub(crate) fn next_interrupt_id(context: &mut TaskContext) -> String {
    let namespace = current_namespace();
    let task = context
        .task_id
        .clone()
        .unwrap_or_else(|| context.node.clone());
    // LangGraph's checkpoint namespace for a task includes the executing node.
    let node_ns = if namespace.is_empty() {
        context.node.clone()
    } else {
        format!("{namespace}|{}", context.node)
    };
    let task_namespace = format!("{node_ns}:{task}");
    let seed = if context.interrupt_calls == 0 {
        task_namespace
    } else {
        format!("{task_namespace}|{}", context.interrupt_calls)
    };
    context.interrupt_calls += 1;
    crate::_internal::error::deterministic_interrupt_id(&seed)
}

/// Resume value stamped onto the current task, when replaying an interrupt.
pub fn current_resume() -> Option<Value> {
    #[cfg(feature = "async")]
    if !has_sync_task()
        && let Ok(value) = ASYNC_TASK.try_with(|ops| ops.resume_value())
    {
        return value;
    }
    CURRENT_TASK.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|context| context.resume_values.get(context.resume_index).cloned())
    })
}

/// Latest resume value stamped onto the current task (subgraph propagation).
/// Resume value of the current task even before an `interrupt()` consumed it.
///
/// Subgraphs rely on this: the paused task is the parent node, so `current_resume_last()` is still
/// `None` while the child is being entered.
pub(crate) fn current_pending_resume() -> Option<Value> {
    CURRENT_TASK.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|context| context.pending_resume.clone())
    })
}

pub(crate) fn current_resume_last() -> Option<Value> {
    #[cfg(feature = "async")]
    if !has_sync_task()
        && let Ok(value) = ASYNC_TASK.try_with(|ops| ops.resume_last())
    {
        return value;
    }
    CURRENT_TASK.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|context| context.resume_values.last().cloned())
    })
}

#[derive(Debug, Default)]
pub(crate) struct TaskContext {
    node: String,
    task_id: Option<String>,
    resume_values: Vec<Value>,
    resume_index: usize,
    resume_consumed: bool,
    /// Resume value of the task itself, not yet consumed by an `interrupt()` call. A node that
    /// delegates to a subgraph needs it: the child pauses instead of the parent node.
    pending_resume: Option<Value>,
    /// Counts progress-reporting calls inside the node (custom writes, messages, usage).
    progress: std::sync::Arc<std::sync::atomic::AtomicU64>,
    interrupt_calls: usize,
    custom: Vec<Value>,
    tools: Vec<Value>,
    messages: Vec<Value>,
    /// Nodes (or `START`) that triggered this task, for LangGraph's `langgraph_triggers` metadata.
    triggers: Vec<String>,
    /// Channel writes the node performed itself (LangGraph's `CONFIG_KEY_SEND`).
    writes: Vec<(String, Value)>,
    usage: Vec<Usage>,
    error: Option<NodeFailure>,
}

impl TaskContext {
    /// Record a channel write the node performed itself.
    pub(crate) fn push_write(&mut self, channel: String, value: Value) {
        self.writes.push((channel, value));
    }

    pub(crate) fn new(node: impl Into<String>, resume_values: Vec<Value>) -> Self {
        Self {
            node: node.into(),
            task_id: None,
            resume_values,
            resume_index: 0,
            resume_consumed: false,
            pending_resume: None,
            progress: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            interrupt_calls: 0,
            custom: Vec::new(),
            tools: Vec::new(),
            messages: Vec::new(),
            writes: Vec::new(),
            triggers: Vec::new(),
            usage: Vec::new(),
            error: None,
        }
    }

    /// Record the nodes (or `START`) that triggered this task.
    pub(crate) fn with_triggers(mut self, triggers: Vec<String>) -> Self {
        self.triggers = triggers;
        self
    }

    pub(crate) fn with_task_id(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    pub(crate) fn with_error(mut self, error: Option<NodeFailure>) -> Self {
        self.error = error;
        self
    }

    /// Record the task's own (not yet consumed) resume value.
    pub(crate) fn with_pending_resume(mut self, resume: Option<Value>) -> Self {
        self.pending_resume = resume;
        self
    }
}

/// Install `context` for the duration of `run`, returning its result, any custom
/// stream writes and any consumption records the node reported.
/// Value returned by [`with_task`]: the node result plus everything its context collected.
pub(crate) type TaskOutcomeOf<R> =
    (R, Vec<Value>, Vec<Value>, Vec<Value>, Vec<(String, Value)>, Vec<Usage>);

/// Take (and clear) the channel writes the running node performed itself.
pub(crate) fn take_writes() -> Vec<(String, Value)> {
    CURRENT_TASK.with(|slot| {
        slot.borrow_mut()
            .as_mut()
            .map(|context| std::mem::take(&mut context.writes))
            .unwrap_or_default()
    })
}

/// The running task's progress counter, when a node is executing on this thread.
pub(crate) fn current_progress_counter() -> Option<std::sync::Arc<std::sync::atomic::AtomicU64>> {
    CURRENT_TASK.with(|slot| slot.borrow().as_ref().map(|context| context.progress.clone()))
}

/// Bump the running task's progress counter.
pub(crate) fn bump_progress() {
    if let Some(counter) = current_progress_counter() {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

pub(crate) fn with_task<R>(
    context: TaskContext,
    run: impl FnOnce() -> R,
) -> TaskOutcomeOf<R> {
    let previous = CURRENT_TASK.with(|slot| slot.borrow_mut().replace(context));
    let result = run();
    let context = CURRENT_TASK.with(|slot| {
        let mut slot = slot.borrow_mut();
        std::mem::replace(&mut *slot, previous)
    });
    let (custom, tools, messages, writes, usage) = context
        .map(|context| {
            (
                context.custom,
                context.tools,
                context.messages,
                context.writes,
                context.usage,
            )
        })
        .unwrap_or_default();
    (result, custom, tools, messages, writes, usage)
}

/// Failure that caused the current node to be scheduled.
///
/// When a node fails permanently and the graph has an error edge
/// Superstep currently being executed, when called from a node.
pub fn current_step() -> Option<usize> {
    if let Some(run) = current_run()
        && run.step_active.load(std::sync::atomic::Ordering::Relaxed)
    {
        return Some(run.step.load(std::sync::atomic::Ordering::Relaxed));
    }
    CURRENT_STEP.with(|slot| slot.borrow().map(|(step, _, _)| step))
}

/// Task id currently executing, when called from a node.
pub fn current_task_id() -> Option<String> {
    #[cfg(feature = "async")]
    if let Ok(task_id) = ASYNC_NODE.try_with(|(_, task_id)| task_id.clone()) {
        return task_id;
    }
    CURRENT_TASK.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|context| context.task_id.clone())
    })
}

/// Nodes (or `START`) that triggered the currently running task.
pub fn current_triggers() -> Vec<String> {
    CURRENT_TASK.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|context| context.triggers.clone())
            .unwrap_or_default()
    })
}

pub fn current_node() -> Option<String> {
    #[cfg(feature = "async")]
    if let Ok(node) = ASYNC_NODE.try_with(|(node, _)| node.clone()) {
        return Some(node);
    }
    CURRENT_TASK.with(|slot| slot.borrow().as_ref().map(|context| context.node.clone()))
}

#[path = "context_tasks.rs"]
mod context_tasks;
pub(crate) use context_tasks::*;

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
        let (result, _custom, _tools, _messages, _writes, _usage) =
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
        let (result, _custom, _tools, _messages, _writes, _usage) =
            with_task(TaskContext::new("ask", vec![json!("yes")]), || {
                interrupt(json!({ "question": "approve?" }))
            });
        assert_eq!(result.unwrap(), json!("yes"));
    }

    #[test]
    fn custom_values_are_collected() {
        let (result, custom, _tools, messages, _writes, usage) =
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

#[path = "context_task_helpers.rs"]
mod task_helpers;
pub(crate) use task_helpers::validate_response;
pub use task_helpers::*;
