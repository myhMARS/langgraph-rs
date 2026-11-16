//! Dynamic node context: human-in-the-loop interrupts and custom stream writes.
//!
//! The engine installs a thread-local task context while a node is running. Nodes can
//! then call [`interrupt`] or [`custom`] without threading a context argument through
//! every signature, which keeps node closures identical to their LangGraph counterparts.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde_json::Value;

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
            .map(|(step, recursion_limit)| recursion_limit.saturating_sub(step))
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
            return Ok(value);
        }
        Err(Error::Interrupt(Interrupt::new(
            context.node.clone(),
            value,
        )))
    })
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
