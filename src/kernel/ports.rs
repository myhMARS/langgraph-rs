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

    /// Newest checkpoint of a thread in a namespace (`""` is the root graph).
    ///
    /// The default implementation ignores namespaces; adapters that store the
    /// namespace (all bundled adapters do) override it.
    fn latest_in(&self, thread_id: &str, namespace: &str) -> Result<Option<Checkpoint>> {
        let _ = namespace;
        self.latest(thread_id)
    }

    /// Full history of a thread in a namespace, newest first.
    fn history_in(&self, thread_id: &str, namespace: &str) -> Result<Vec<Checkpoint>> {
        let _ = namespace;
        self.history(thread_id)
    }

    /// Delete every checkpoint of a thread.
    fn delete_thread(&self, thread_id: &str) -> Result<()> {
        let _ = thread_id;
        Err(crate::kernel::error::Error::Checkpoint(
            "delete_thread is not supported by this checkpointer".to_string(),
        ))
    }

    /// Copy every checkpoint of one thread to another thread id.
    fn copy_thread(&self, source_thread: &str, target_thread: &str) -> Result<()> {
        let _ = (source_thread, target_thread);
        Err(crate::kernel::error::Error::Checkpoint(
            "copy_thread is not supported by this checkpointer".to_string(),
        ))
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
