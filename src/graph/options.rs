//! Compilation and per-run configuration.

use std::fmt;
use std::sync::Arc;

use crate::_internal::plugin::Kernel;
use crate::_internal::ports::Checkpointer;
use crate::store::BaseStore;
use serde_json::Value;

/// Default number of supersteps a single graph run may execute.
pub const DEFAULT_RECURSION_LIMIT: usize = 25;

/// One or more edge targets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Targets(Vec<String>);

impl Targets {
    /// Build a target list.
    pub fn new<I, T>(targets: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Self(targets.into_iter().map(Into::into).collect())
    }

    /// Consume the target list.
    pub fn into_vec(self) -> Vec<String> {
        self.0
    }

    /// Targets as a slice.
    pub fn as_slice(&self) -> &[String] {
        &self.0
    }
}

impl From<&str> for Targets {
    fn from(target: &str) -> Self {
        Targets(vec![target.to_string()])
    }
}

impl From<String> for Targets {
    fn from(target: String) -> Self {
        Targets(vec![target])
    }
}

impl From<Vec<String>> for Targets {
    fn from(targets: Vec<String>) -> Self {
        Targets(targets)
    }
}

impl From<Vec<&str>> for Targets {
    fn from(targets: Vec<&str>) -> Self {
        Targets(targets.into_iter().map(str::to_string).collect())
    }
}

impl<const N: usize> From<[&str; N]> for Targets {
    fn from(targets: [&str; N]) -> Self {
        Targets(targets.into_iter().map(str::to_string).collect())
    }
}

/// Options used when compiling a graph.
///
/// ```
/// use std::sync::Arc;
/// use langgraph_rs::{CompileOptions, InMemorySaver};
///
/// let options = CompileOptions::new()
///     .with_checkpointer(Arc::new(InMemorySaver::new()))
///     .with_recursion_limit(10)
///     .interrupt_before(["human_review"]);
/// assert_eq!(options.recursion_limit, Some(10));
/// ```
#[derive(Clone)]
pub struct CompileOptions {
    /// Enables checkpointing, thread history and human-in-the-loop resuming.
    pub checkpointer: Option<Arc<dyn Checkpointer>>,
    /// Microkernel registry: channel policies and trace sinks used by this graph.
    pub kernel: Kernel,
    /// Overrides [`DEFAULT_RECURSION_LIMIT`].
    pub recursion_limit: Option<usize>,
    /// Nodes that must be interrupted *before* they run.
    pub interrupt_before: Vec<String>,
    /// Nodes that must be interrupted *after* they run.
    pub interrupt_after: Vec<String>,
    /// Long-term memory store available through [`get_store`](crate::get_store).
    pub store: Option<Arc<dyn BaseStore>>,
    /// Optional graph name (LangGraph `compile(name=...)`).
    pub name: Option<String>,
    /// JSON schema validating `RunConfig::context` (LangGraph `context_schema`).
    pub context_schema: Option<Value>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            checkpointer: None,
            kernel: Kernel::builtin(),
            recursion_limit: None,
            interrupt_before: Vec::new(),
            interrupt_after: Vec::new(),
            store: None,
            name: None,
            context_schema: None,
        }
    }
}

impl fmt::Debug for CompileOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompileOptions")
            .field("checkpointer", &self.checkpointer.is_some())
            .field("kernel", &self.kernel)
            .field("recursion_limit", &self.recursion_limit)
            .field("interrupt_before", &self.interrupt_before)
            .field("interrupt_after", &self.interrupt_after)
            .field("store", &self.store.is_some())
            .field("name", &self.name)
            .field("context_schema", &self.context_schema.is_some())
            .finish()
    }
}

impl CompileOptions {
    /// Create default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a checkpointer.
    pub fn with_checkpointer(mut self, checkpointer: Arc<dyn Checkpointer>) -> Self {
        self.checkpointer = Some(checkpointer);
        self
    }

    /// Use a custom microkernel registry (channel plugins + trace sinks).
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Override the superstep budget.
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = Some(limit);
        self
    }

    /// Interrupt before the given nodes run.
    pub fn interrupt_before<I, T>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.interrupt_before = nodes.into_iter().map(Into::into).collect();
        self
    }

    /// Validate `RunConfig::context` against a JSON schema before each run.
    pub fn with_context_schema(mut self, schema: Value) -> Self {
        self.context_schema = Some(schema);
        self
    }

    /// Name the compiled graph (LangGraph `compile(name=...)`).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Attach a long-term memory store.
    pub fn with_store(mut self, store: Arc<dyn BaseStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Interrupt after the given nodes ran.
    pub fn interrupt_after<I, T>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.interrupt_after = nodes.into_iter().map(Into::into).collect();
        self
    }
}

/// Checkpoint durability policy for a run (LangGraph `durability`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Durability {
    /// Persist a checkpoint after every superstep (default).
    #[default]
    Sync,
    /// Queue checkpoints to a background writer thread (flushed when the run ends).
    Async,
    /// Only persist the initial, interrupt and final checkpoints.
    Exit,
}

impl Durability {
    /// Parse a LangGraph durability name.
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "sync" => Self::Sync,
            "async" => Self::Async,
            "exit" => Self::Exit,
            _ => return None,
        })
    }
}

/// Per-run configuration.
#[derive(Clone, Default)]
pub struct RunConfig {
    /// Thread id used for checkpointing. Required when the graph has a checkpointer.
    pub thread_id: Option<String>,
    /// Target checkpoint id (LangGraph `configurable.checkpoint_id`).
    pub checkpoint_id: Option<String>,
    /// Overrides the compiled recursion limit for this run.
    pub recursion_limit: Option<usize>,
    /// Checkpoint namespace (`""` for the root graph). LangGraph's `checkpoint_ns`.
    pub checkpoint_ns: Option<String>,
    /// Overrides the compiled long-term memory store for this run.
    pub store: Option<Arc<dyn BaseStore>>,
    /// User context visible to nodes through [`current_context`](crate::current_context).
    pub context: Option<Value>,
    /// Maximum number of concurrent graph invocations for `batch`.
    pub max_concurrency: Option<usize>,
    /// Checkpoint durability policy.
    pub durability: Durability,
    /// Forward chunks emitted inside subgraphs (wrapped in `StreamChunk::Subgraph`).
    pub stream_subgraphs: bool,
    /// Requested stream modes; when set, expensive chunk payloads are built only
    /// for the selected modes.
    pub stream_modes: Option<Vec<crate::_internal::stream::StreamMode>>,
    /// Cooperative abort flag checked between supersteps.
    pub control: Option<crate::RunControl>,
}

impl std::fmt::Debug for RunConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunConfig")
            .field("thread_id", &self.thread_id)
            .field("checkpoint_id", &self.checkpoint_id)
            .field("recursion_limit", &self.recursion_limit)
            .field("checkpoint_ns", &self.checkpoint_ns)
            .field("store", &self.store.is_some())
            .field("context", &self.context)
            .field("max_concurrency", &self.max_concurrency)
            .field("durability", &self.durability)
            .field("stream_subgraphs", &self.stream_subgraphs)
            .field(
                "control",
                &self.control.as_ref().map(crate::RunControl::is_aborted),
            )
            .finish()
    }
}

impl RunConfig {
    /// Create a config without a thread.
    pub fn new() -> Self {
        Self::default()
    }

    /// Restrict chunk construction to the requested stream modes.
    pub fn with_stream_modes(mut self, modes: Vec<crate::_internal::stream::StreamMode>) -> Self {
        self.stream_modes = Some(modes);
        self
    }

    /// Bind the run to a checkpoint thread.
    pub fn with_thread(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Override the superstep budget for this run.
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = Some(limit);
        self
    }

    /// Bind the run to a checkpoint namespace (`""` is the root graph).
    pub fn with_checkpoint_ns(mut self, namespace: impl Into<String>) -> Self {
        self.checkpoint_ns = Some(namespace.into());
        self
    }

    /// Override the long-term memory store for this run.
    pub fn with_store(mut self, store: Arc<dyn BaseStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Provide user context visible to nodes for this run.
    pub fn with_context(mut self, context: Value) -> Self {
        self.context = Some(context);
        self
    }

    /// Limit the number of concurrent invocations used by `batch`.
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = Some(max_concurrency.max(1));
        self
    }

    /// Attach a cooperative run-control handle.
    pub fn with_control(mut self, control: crate::RunControl) -> Self {
        self.control = Some(control);
        self
    }

    /// Select the checkpoint durability policy.
    pub fn with_durability(mut self, durability: Durability) -> Self {
        self.durability = durability;
        self
    }

    /// Forward subgraph stream chunks, wrapped with their checkpoint namespace.
    pub fn with_stream_subgraphs(mut self, enabled: bool) -> Self {
        self.stream_subgraphs = enabled;
        self
    }
}
