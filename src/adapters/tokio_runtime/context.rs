//! Per-node execution context for async nodes.

use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde_json::Value;

use crate::kernel::error::{Error, Interrupt, NodeFailure, Result};
use crate::kernel::usage::Usage;

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
    resume: Arc<Mutex<Option<Value>>>,
    capture: Arc<Mutex<Capture>>,
}

#[derive(Default)]
struct Capture {
    usage: Vec<Usage>,
    custom: Vec<Value>,
}

impl AsyncContext {
    pub(crate) fn new(
        node: String,
        step: usize,
        error: Option<NodeFailure>,
        resume: Option<Value>,
    ) -> Self {
        Self {
            node,
            step,
            error,
            resume: Arc::new(Mutex::new(resume)),
            capture: Arc::new(Mutex::new(Capture::default())),
        }
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
        let mut slot = self
            .resume
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(resume) = slot.take() {
            return Ok(resume);
        }
        Err(Error::Interrupt(Interrupt {
            node: self.node.clone(),
            value,
        }))
    }

    /// Serializing variant of [`AsyncContext::interrupt`].
    pub fn interrupt_with(&self, value: impl Serialize) -> Result<Value> {
        self.interrupt(serde_json::to_value(value)?)
    }

    /// Record consumption for this execution (multiple records are summed).
    pub fn record_usage(&self, usage: Usage) {
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

    /// Emit a custom value into the stream.
    pub fn custom(&self, value: impl Serialize) -> Result<()> {
        let value = serde_json::to_value(value)?;
        self.capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .custom
            .push(value);
        Ok(())
    }

    pub(crate) fn take(&self) -> (Vec<Usage>, Vec<Value>) {
        let mut capture = self
            .capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (
            std::mem::take(&mut capture.usage),
            std::mem::take(&mut capture.custom),
        )
    }
}
