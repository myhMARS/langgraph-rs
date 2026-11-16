//! Per-node execution timeouts (`TimeoutPolicy`).

use std::time::Duration;

/// Timeouts applied to node execution.
///
/// `run_timeout` bounds one attempt of a node. `idle_timeout` is accepted for LangGraph
/// API compatibility; the current engine enforces `run_timeout` only.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TimeoutPolicy {
    /// Maximum duration of one node attempt.
    pub run_timeout: Option<Duration>,
    /// Maximum idle duration (stored, not enforced yet).
    pub idle_timeout: Option<Duration>,
}

impl TimeoutPolicy {
    /// An empty policy (no timeout).
    pub fn new() -> Self {
        Self::default()
    }

    /// Bound a single node attempt.
    pub fn with_run_timeout(mut self, timeout: Duration) -> Self {
        self.run_timeout = Some(timeout);
        self
    }

    /// Record an idle timeout (not enforced yet).
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = Some(timeout);
        self
    }

    /// Configured run timeout.
    pub fn run_timeout(&self) -> Option<Duration> {
        self.run_timeout
    }

    /// Timeout applied to one attempt: `run_timeout`, falling back to `idle_timeout`.
    pub fn effective_timeout(&self) -> Option<Duration> {
        self.run_timeout.or(self.idle_timeout)
    }

    /// Configured idle timeout.
    pub fn idle_timeout(&self) -> Option<Duration> {
        self.idle_timeout
    }
}
