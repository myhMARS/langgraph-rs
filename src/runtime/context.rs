//! Dynamic node context: human-in-the-loop interrupts and custom stream writes.
//!
//! The engine installs a thread-local task context while a node is running. Nodes can
//! then call [`interrupt`] or [`custom`] without threading a context argument through
//! every signature, which keeps node closures identical to their LangGraph counterparts.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

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

#[cfg(feature = "async")]
tokio::task_local! {
    /// Run context mirrored into the async task so tokio can migrate the future.
    pub(crate) static ASYNC_RUN: RunContext;
    static ASYNC_TASK_RESULTS: RefCell<Map<String, Value>>;
    static ASYNC_NODE: (String, Option<String>);
    static ASYNC_TASK: std::sync::Arc<dyn AsyncTaskOps>;
}

/// Operations of the async node currently executing, exposed to the free helper
/// functions (`interrupt`, `custom`, `message`, `record_usage`, `last_error`).
#[cfg(feature = "async")]
pub(crate) trait AsyncTaskOps: Send + Sync {
    fn interrupt_value(&self, value: Value, schema: Option<Value>) -> Result<Value>;
    fn push_custom(&self, value: Value) -> Result<()>;
    fn push_message(&self, value: Value) -> Result<()>;
    fn record_usage(&self, usage: crate::kernel::usage::Usage) -> Result<()>;
    fn last_error(&self) -> Option<crate::kernel::error::NodeFailure>;
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

/// Execution context of the current run.
#[derive(Clone, Default)]
pub(crate) struct RunContext {
    pub(crate) thread_id: Option<String>,
    pub(crate) namespace: String,
    pub(crate) store: Option<std::sync::Arc<dyn BaseStore>>,
    pub(crate) context: Option<Value>,
    step: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    managed_step: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    recursion_limit: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    step_active: std::sync::Arc<std::sync::atomic::AtomicBool>,
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
            step: Default::default(),
            managed_step: Default::default(),
            recursion_limit: Default::default(),
            step_active: Default::default(),
        })
    });
    RunGuard { previous }
}

/// Build a run context value for task-local scoping.
#[cfg(feature = "async")]
pub(crate) fn run_context(
    thread_id: Option<String>,
    namespace: String,
    store: Option<std::sync::Arc<dyn BaseStore>>,
    context: Option<Value>,
) -> RunContext {
    RunContext {
        thread_id,
        namespace,
        store,
        context,
        step: Default::default(),
        managed_step: Default::default(),
        recursion_limit: Default::default(),
        step_active: Default::default(),
    }
}

/// Scope an async run with a task-local copy of its run context.
#[cfg(feature = "async")]
pub(crate) async fn with_async_run<F: std::future::Future>(
    run: RunContext,
    future: F,
) -> F::Output {
    let results = RefCell::new(Map::new());
    ASYNC_TASK_RESULTS
        .scope(results, ASYNC_RUN.scope(run, future))
        .await
}

/// Scope an async node future with its task identity for context accessors.
#[cfg(feature = "async")]
pub(crate) async fn with_async_node<F: std::future::Future>(
    node: String,
    task_id: Option<String>,
    future: F,
) -> F::Output {
    ASYNC_NODE.scope((node, task_id), future).await
}

fn current_run() -> Option<RunContext> {
    #[cfg(feature = "async")]
    if let Ok(run) = ASYNC_RUN.try_with(|run| run.clone()) {
        return Some(run);
    }
    CURRENT_RUN.with(|slot| slot.borrow().clone())
}

/// Long-term memory store of the current run, when the graph was compiled with one.
pub fn get_store() -> Option<std::sync::Arc<dyn BaseStore>> {
    current_run().and_then(|run| run.store.clone())
}

/// User context passed to the current run via `RunConfig::with_context`.
pub fn current_context() -> Option<Value> {
    current_run().and_then(|run| run.context.clone())
}

/// Runtime information (thread, namespace, store, context) of the current run.
pub fn current_runtime() -> Option<Runtime> {
    current_run().map(|run| Runtime {
        thread_id: run.thread_id.clone(),
        namespace: run.namespace.clone(),
        store: run.store.clone(),
        context: run.context.clone(),
    })
}

fn has_sync_task() -> bool {
    CURRENT_TASK.with(|slot| slot.borrow().is_some())
}

/// Capture the current run context for a scoped worker thread.
pub(crate) fn capture_run_context() -> Option<RunContext> {
    current_run()
}

/// Install a captured run context on the current thread.
pub(crate) fn install_run_context(run: RunContext) -> RunGuard {
    let previous = CURRENT_RUN.with(|slot| slot.borrow_mut().replace(run));
    RunGuard { previous }
}

/// Current superstep context `(step, recursion_limit)`, when inside a step.
pub(crate) fn capture_step_context() -> Option<(usize, usize)> {
    CURRENT_STEP.with(|slot| *slot.borrow())
}

/// RAII guard restoring the previous step context.
type AsyncStepPrevious = (
    std::sync::Arc<std::sync::atomic::AtomicUsize>,
    std::sync::Arc<std::sync::atomic::AtomicUsize>,
    std::sync::Arc<std::sync::atomic::AtomicUsize>,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
    usize,
    usize,
    usize,
    bool,
);

pub(crate) struct StepGuard {
    previous: Option<(usize, usize)>,
    async_previous: Option<AsyncStepPrevious>,
}

impl Drop for StepGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CURRENT_STEP.with(|slot| {
            *slot.borrow_mut() = previous;
        });
        if let Some((step, managed, limit, active, old_step, old_managed, old_limit, old_active)) =
            self.async_previous.take()
        {
            step.store(old_step, std::sync::atomic::Ordering::Relaxed);
            managed.store(old_managed, std::sync::atomic::Ordering::Relaxed);
            limit.store(old_limit, std::sync::atomic::Ordering::Relaxed);
            active.store(old_active, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Install the current superstep number and recursion limit for managed values.
pub(crate) fn install_step(step: usize, managed_step: usize, recursion_limit: usize) -> StepGuard {
    let previous = CURRENT_STEP.with(|slot| slot.borrow_mut().replace((step, recursion_limit)));
    let async_previous = current_run().map(|run| {
        let old = (
            run.step.load(std::sync::atomic::Ordering::Relaxed),
            run.managed_step.load(std::sync::atomic::Ordering::Relaxed),
            run.recursion_limit
                .load(std::sync::atomic::Ordering::Relaxed),
            run.step_active.load(std::sync::atomic::Ordering::Relaxed),
        );
        run.step.store(step, std::sync::atomic::Ordering::Relaxed);
        run.managed_step
            .store(managed_step, std::sync::atomic::Ordering::Relaxed);
        run.recursion_limit
            .store(recursion_limit, std::sync::atomic::Ordering::Relaxed);
        run.step_active
            .store(true, std::sync::atomic::Ordering::Relaxed);
        (
            run.step,
            run.managed_step,
            run.recursion_limit,
            run.step_active,
            old.0,
            old.1,
            old.2,
            old.3,
        )
    });
    StepGuard {
        previous,
        async_previous,
    }
}

/// Remaining supersteps before the recursion limit (LangGraph `RemainingSteps`).
///
/// Returns `None` outside a run. The value is `recursion_limit - step`, so the first
/// superstep of a graph compiled with a limit of 25 reports 24.
pub fn remaining_steps() -> Option<usize> {
    if let Some(run) = current_run()
        && run.step_active.load(std::sync::atomic::Ordering::Relaxed)
    {
        let step = run.managed_step.load(std::sync::atomic::Ordering::Relaxed);
        let limit = run
            .recursion_limit
            .load(std::sync::atomic::Ordering::Relaxed);
        return Some(limit.saturating_add(1).saturating_sub(step));
    }
    CURRENT_STEP.with(|slot| {
        slot.borrow()
            .map(|(step, recursion_limit)| recursion_limit.saturating_add(1).saturating_sub(step))
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
    current_run().and_then(|run| run.thread_id.clone())
}

/// Checkpoint namespace of the current run (`""` for the root graph).
pub fn current_namespace() -> String {
    current_run().map(|run| run.namespace).unwrap_or_default()
}

/// Deterministic interrupt id for the current task/step/call index.
pub(crate) fn next_interrupt_id(context: &mut TaskContext) -> String {
    let step = CURRENT_STEP.with(|slot| slot.borrow().map(|(step, _)| step).unwrap_or(0));
    let namespace = current_namespace();
    let seed = format!(
        "{namespace}|{step}|{}|{}",
        context.node, context.interrupt_calls,
    );
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
    interrupt_calls: usize,
    custom: Vec<Value>,
    messages: Vec<Value>,
    usage: Vec<Usage>,
    error: Option<NodeFailure>,
}

impl TaskContext {
    pub(crate) fn new(node: impl Into<String>, resume_values: Vec<Value>) -> Self {
        Self {
            node: node.into(),
            task_id: None,
            resume_values,
            resume_index: 0,
            resume_consumed: false,
            interrupt_calls: 0,
            custom: Vec::new(),
            messages: Vec::new(),
            usage: Vec::new(),
            error: None,
        }
    }

    pub(crate) fn with_task_id(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
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
/// Superstep currently being executed, when called from a node.
pub fn current_step() -> Option<usize> {
    if let Some(run) = current_run()
        && run.step_active.load(std::sync::atomic::Ordering::Relaxed)
    {
        return Some(run.step.load(std::sync::atomic::Ordering::Relaxed));
    }
    CURRENT_STEP.with(|slot| slot.borrow().map(|(step, _)| step))
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

pub fn current_node() -> Option<String> {
    #[cfg(feature = "async")]
    if let Ok(node) = ASYNC_NODE.try_with(|(node, _)| node.clone()) {
        return Some(node);
    }
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
    #[cfg(feature = "async")]
    if let Ok(value) = ASYNC_TASK_RESULTS.try_with(|slot| slot.borrow().get(key).cloned()) {
        return value;
    }
    TASK_RESULTS.with(|slot| slot.borrow().get(key).cloned())
}

/// Store a functional-API task result under `key`.
pub(crate) fn task_result_put(key: String, value: Value) {
    #[cfg(feature = "async")]
    if ASYNC_TASK_RESULTS
        .try_with(|slot| {
            slot.borrow_mut().insert(key.clone(), value.clone());
        })
        .is_ok()
    {
        return;
    }
    TASK_RESULTS.with(|slot| {
        slot.borrow_mut().insert(key, value);
    });
}

/// Reset the task-result cache at the start of a fresh run.
pub(crate) fn clear_task_results() {
    #[cfg(feature = "async")]
    if ASYNC_TASK_RESULTS
        .try_with(|slot| slot.borrow_mut().clear())
        .is_ok()
    {
        return;
    }
    TASK_RESULTS.with(|slot| slot.borrow_mut().clear());
}

/// Restore memoized task results from a resumed checkpoint.
pub(crate) fn load_task_results(results: &Map<String, Value>) {
    #[cfg(feature = "async")]
    if ASYNC_TASK_RESULTS
        .try_with(|slot| {
            let mut slot = slot.borrow_mut();
            slot.clear();
            slot.extend(results.clone());
        })
        .is_ok()
    {
        return;
    }
    TASK_RESULTS.with(|slot| {
        let mut slot = slot.borrow_mut();
        slot.clear();
        slot.extend(results.clone());
    });
}

/// Snapshot of the memoized task results for checkpoint persistence.
pub(crate) fn snapshot_task_results() -> Map<String, Value> {
    #[cfg(feature = "async")]
    if let Ok(results) = ASYNC_TASK_RESULTS.try_with(|slot| slot.borrow().clone()) {
        return results;
    }
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

#[path = "context_task_helpers.rs"]
mod task_helpers;
pub(crate) use task_helpers::validate_response;
pub use task_helpers::*;
