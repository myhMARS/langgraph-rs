//! Ports (hexagonal architecture): the traits the kernel depends on.
//!
//! Adapters in [`crate::adapters`] (and in downstream crates) implement these traits.
//! The kernel never constructs an adapter itself.

use super::checkpoint::Checkpoint;
use super::error::Result;
use super::usage::NodeExecution;

/// Storage backend for checkpoints.
///
/// Implementations can persist to memory, a file, Redis, Postgres, ... The engine only
/// needs to append checkpoints and read back the latest one for a thread.
pub trait Checkpointer: Send + Sync {
    /// Persist a checkpoint.
    fn put(&self, checkpoint: Checkpoint) -> Result<()>;

    /// Read a specific checkpoint.
    fn get(&self, thread_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>>;

    /// Read the newest checkpoint of a thread.
    fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>>;

    /// Read the full history of a thread, newest first.
    fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>>;

    /// List all threads that have at least one checkpoint.
    fn threads(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
}

/// Output port for per-node execution records (duration + consumption).
///
/// Adapters forward these records to logs, metrics, OpenTelemetry, a database, ...
/// A blanket implementation covers plain closures: `Arc::new(|execution| ...)`.
pub trait TraceSink: Send + Sync {
    /// Record one finished node execution.
    fn record(&self, execution: &NodeExecution);
}

impl<F> TraceSink for F
where
    F: Fn(&NodeExecution) + Send + Sync,
{
    fn record(&self, execution: &NodeExecution) {
        self(execution);
    }
}
