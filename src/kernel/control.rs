//! Cooperative run control (LangGraph `RunControl` basics).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Abort flag shared with a running graph.
///
/// The engine checks it between supersteps; `abort()` therefore stops the run at the next
/// superstep boundary (it does not cancel an in-flight node).
#[derive(Clone, Debug, Default)]
pub struct RunControl {
    aborted: Arc<AtomicBool>,
}

impl RunControl {
    /// Create an unaborted control handle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request that the run stops at the next superstep boundary.
    pub fn abort(&self) {
        self.aborted.store(true, Ordering::SeqCst);
    }

    /// Returns `true` once [`RunControl::abort`] was called.
    pub fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::SeqCst)
    }
}
