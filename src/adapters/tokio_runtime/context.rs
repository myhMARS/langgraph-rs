//! Per-node execution context for async nodes.

use std::cell::RefCell;
use std::future::Future;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde_json::Value;

use crate::application::context::ParentCommand;
use crate::kernel::error::{Error, Interrupt, NodeFailure, Result};
use crate::kernel::store::BaseStore;
use crate::kernel::stream::StreamChunk;
use crate::kernel::usage::Usage;

impl crate::application::context::AsyncTaskOps for AsyncContext {
    fn interrupt_value(&self, value: Value, schema: Option<Value>) -> Result<Value> {
        self.interrupt_with_schema(value, schema)
    }

    fn push_custom(&self, value: Value) -> Result<()> {
        self.custom(value)
    }

    fn push_message(&self, value: Value) -> Result<()> {
        self.message(value)
    }

    fn record_usage(&self, usage: Usage) -> Result<()> {
        self.record_usage(usage);
        Ok(())
    }

    fn last_error(&self) -> Option<NodeFailure> {
        self.last_error().cloned()
    }

    fn resume_value(&self) -> Option<Value> {
        let slot = self
            .resume
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        slot.0.get(slot.1).cloned()
    }

    fn resume_last(&self) -> Option<Value> {
        let slot = self
            .resume
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        slot.0.last().cloned()
    }
}

/// Per-node execution context for async nodes.
///
/// The context is cheap to clone; clones share the same capture buffer. Nodes use it to
/// report consumption (`record_usage`, `record_tokens`), emit custom stream values
/// (`custom`), pause the graph for human input (`interrupt`) and inspect a routed
/// failure (`last_error`).
#[derive(Clone)]
pub struct AsyncContext {
    node: String,
    step: usize,
    error: Option<NodeFailure>,
    resume: Arc<Mutex<(Vec<Value>, usize)>>,
    capture: Arc<Mutex<Capture>>,
    thread_id: Option<String>,
    namespace: String,
    store: Option<Arc<dyn BaseStore>>,
    run_timeout: Option<std::time::Duration>,
    context: Option<Value>,
    collector: Option<Arc<Mutex<Vec<StreamChunk>>>>,
    progress: Arc<std::sync::atomic::AtomicU64>,
}

#[derive(Default)]
struct Capture {
    usage: Vec<Usage>,
    custom: Vec<Value>,
    messages: Vec<Value>,
}

impl AsyncContext {
    pub(crate) fn new(
        node: String,
        step: usize,
        error: Option<NodeFailure>,
        resume: Vec<Value>,
    ) -> Self {
        Self {
            node,
            step,
            error,
            resume: Arc::new(Mutex::new((resume, 0))),
            capture: Arc::new(Mutex::new(Capture::default())),
            thread_id: None,
            namespace: String::new(),
            store: None,
            run_timeout: None,
            context: None,
            collector: None,
            progress: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Attach parent run information (thread, checkpoint namespace, store, context).
    pub(crate) fn with_run_info(
        mut self,
        thread_id: Option<String>,
        namespace: String,
        store: Option<Arc<dyn BaseStore>>,
        context: Option<Value>,
        collector: Option<Arc<Mutex<Vec<StreamChunk>>>>,
    ) -> Self {
        self.thread_id = thread_id;
        self.namespace = namespace;
        self.store = store;
        self.context = context;
        self.collector = collector;
        self
    }

    /// Thread id of the run this node belongs to.
    pub fn thread_id(&self) -> Option<&str> {
        self.thread_id.as_deref()
    }

    /// Checkpoint namespace of the run (`""` for the root graph).
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Long-term memory store attached to the run.
    /// Effective run timeout for this attempt, when one is configured.
    pub fn run_timeout(&self) -> Option<std::time::Duration> {
        self.run_timeout
    }

    pub(crate) fn with_run_timeout(mut self, timeout: Option<std::time::Duration>) -> Self {
        self.run_timeout = timeout;
        self
    }

    pub fn store(&self) -> Option<Arc<dyn BaseStore>> {
        self.store.clone()
    }

    /// User context passed to the run.
    pub fn context(&self) -> Option<&Value> {
        self.context.as_ref()
    }

    /// Subgraph chunk collector of the parent run (used by subgraph nodes).
    pub fn collector(&self) -> Option<Arc<Mutex<Vec<StreamChunk>>>> {
        self.collector.clone()
    }

    /// Take the remaining resume values stamped onto this task.
    pub fn take_resume(&self) -> Vec<Value> {
        let mut slot = self
            .resume
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let remaining = slot.0[slot.1..].to_vec();
        slot.1 = slot.0.len();
        remaining
    }

    /// Node currently executing.
    pub fn node(&self) -> &str {
        &self.node
    }

    /// Superstep the node runs in.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Failure that caused this node to be scheduled, when it is an error handler.
    pub fn last_error(&self) -> Option<&NodeFailure> {
        self.error.as_ref()
    }

    /// Pause the graph and wait for human input.
    ///
    /// On the first execution this returns [`Error::Interrupt`] carrying `value`; the
    /// engine checkpoints the state and returns. After
    /// [`AsyncCompiledGraph::resume`](crate::tokio_runtime::AsyncCompiledGraph::resume)
    /// the node runs again and this call returns the resume value.
    pub fn interrupt(&self, value: Value) -> Result<Value> {
        self.interrupt_with_schema(value, None)
    }

    /// Like [`AsyncContext::interrupt`], validating the resume value against a JSON schema.
    pub fn interrupt_with_schema(
        &self,
        value: Value,
        response_schema: Option<Value>,
    ) -> Result<Value> {
        let mut slot = self
            .resume
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot.1 < slot.0.len() {
            let resume = slot.0[slot.1].clone();
            slot.1 += 1;
            crate::application::context::validate_response(&resume, response_schema.as_ref())?;
            return Ok(resume);
        }
        Err(Error::Interrupt(Interrupt::with_schema(
            self.node.clone(),
            value,
            response_schema,
        )))
    }

    /// Serializing variant of [`AsyncContext::interrupt`].
    pub fn interrupt_with(&self, value: impl Serialize) -> Result<Value> {
        self.interrupt(serde_json::to_value(value)?)
    }

    /// Record consumption for this execution (multiple records are summed).
    pub fn record_usage(&self, usage: Usage) {
        self.bump_progress();
        self.capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .usage
            .push(usage);
    }

    /// Convenience shorthand for token usage.
    pub fn record_tokens(&self, input_tokens: u64, output_tokens: u64) {
        self.record_usage(
            Usage::new()
                .with_input_tokens(input_tokens)
                .with_output_tokens(output_tokens),
        );
    }

    /// Emit a streamed message (e.g. an LLM token) into the graph stream.
    pub fn message(&self, value: impl Serialize) -> Result<()> {
        self.bump_progress();
        let value = serde_json::to_value(value)?;
        self.capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .messages
            .push(value);
        Ok(())
    }

    /// Emit a custom value into the stream.
    pub fn custom(&self, value: impl Serialize) -> Result<()> {
        self.bump_progress();
        let value = serde_json::to_value(value)?;
        self.capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .custom
            .push(value);
        Ok(())
    }

    fn bump_progress(&self) {
        self.progress
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn progress_counter(&self) -> Arc<std::sync::atomic::AtomicU64> {
        self.progress.clone()
    }

    pub(crate) fn take(&self) -> (Vec<Usage>, Vec<Value>, Vec<Value>) {
        let mut capture = self
            .capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (
            std::mem::take(&mut capture.usage),
            std::mem::take(&mut capture.custom),
            std::mem::take(&mut capture.messages),
        )
    }
}

tokio::task_local! {
    static PARENT_COMMAND: RefCell<Option<ParentCommand>>;
}

/// Run `future` with a fresh parent-command slot and return what the child recorded.
pub(crate) async fn run_with_parent_scope<F>(future: F) -> (F::Output, Option<ParentCommand>)
where
    F: Future,
{
    PARENT_COMMAND
        .scope(RefCell::new(None), async move {
            let output = future.await;
            let command = PARENT_COMMAND.with(|slot| slot.borrow_mut().take());
            (output, command)
        })
        .await
}

/// Record a `Command(graph=PARENT)` inside the current async subgraph scope.
#[allow(dead_code)]
pub(crate) fn record_parent_command(
    updates: Vec<(String, Value)>,
    replaces: Vec<(String, Value)>,
    goto: Vec<String>,
) -> bool {
    PARENT_COMMAND
        .try_with(|slot| {
            let mut slot = slot.borrow_mut();
            let command = slot.get_or_insert_with(ParentCommand::default);
            command.updates.extend(updates);
            command.replaces.extend(replaces);
            command.goto.extend(goto);
        })
        .is_ok()
}

/// Returns `true` when a parent command was recorded in the current task scope.
#[allow(dead_code)]
pub(crate) fn has_parent_command() -> bool {
    PARENT_COMMAND
        .try_with(|slot| slot.borrow().is_some())
        .unwrap_or(false)
}
