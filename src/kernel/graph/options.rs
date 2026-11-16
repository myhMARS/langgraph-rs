//! Compilation and per-run configuration.

use std::fmt;
use std::sync::Arc;

use crate::kernel::plugin::Kernel;
use crate::kernel::ports::Checkpointer;

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
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            checkpointer: None,
            kernel: Kernel::builtin(),
            recursion_limit: None,
            interrupt_before: Vec::new(),
            interrupt_after: Vec::new(),
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

/// Per-run configuration.
#[derive(Clone, Debug, Default)]
pub struct RunConfig {
    /// Thread id used for checkpointing. Required when the graph has a checkpointer.
    pub thread_id: Option<String>,
    /// Overrides the compiled recursion limit for this run.
    pub recursion_limit: Option<usize>,
}

impl RunConfig {
    /// Create a config without a thread.
    pub fn new() -> Self {
        Self::default()
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
}
