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

    /// Keep only the newest `keep_last` checkpoints of a thread.
    fn prune_thread(&self, thread_id: &str, keep_last: usize) -> Result<()> {
        let _ = (thread_id, keep_last);
        Err(crate::kernel::error::Error::Checkpoint(
            "prune_thread is not supported by this checkpointer".to_string(),
        ))
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

/// Boxed future returned by [`AsyncCheckpointer`] methods.
#[cfg(feature = "async")]
pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// Async storage backend for checkpoints.
///
/// Mirrors [`Checkpointer`] with awaitable methods so adapters can use async database
/// drivers or offload blocking clients to a blocking pool. The async engine uses this
/// trait when a graph is compiled with [`with_async_checkpointer`].
///
/// [`with_async_checkpointer`]: crate::tokio_runtime::AsyncStateGraph::with_async_checkpointer
#[cfg(feature = "async")]
pub trait AsyncCheckpointer: Send + Sync {
    /// Persist a checkpoint.
    fn put(&self, checkpoint: Checkpoint) -> BoxFuture<'_, Result<()>>;
    /// Read a specific checkpoint.
    fn get(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> BoxFuture<'_, Result<Option<Checkpoint>>>;
    /// Read the newest checkpoint of a thread.
    fn latest(&self, thread_id: &str) -> BoxFuture<'_, Result<Option<Checkpoint>>>;
    /// Read the full history of a thread, newest first.
    fn history(&self, thread_id: &str) -> BoxFuture<'_, Result<Vec<Checkpoint>>>;
    /// Newest checkpoint of a thread in a namespace.
    fn latest_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> BoxFuture<'_, Result<Option<Checkpoint>>> {
        let _ = namespace;
        self.latest(thread_id)
    }
    /// Full history of a thread in a namespace, newest first.
    fn history_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> BoxFuture<'_, Result<Vec<Checkpoint>>> {
        let _ = namespace;
        self.history(thread_id)
    }
    /// List all threads with at least one checkpoint.
    fn threads(&self) -> BoxFuture<'_, Result<Vec<String>>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    /// Keep only the newest `keep_last` checkpoints of a thread.
    fn prune_thread(&self, thread_id: &str, keep_last: usize) -> BoxFuture<'_, Result<()>> {
        let _ = (thread_id, keep_last);
        Box::pin(async {
            Err(crate::kernel::error::Error::Checkpoint(
                "prune_thread is not supported by this async checkpointer".to_string(),
            ))
        })
    }

    /// Delete every checkpoint of a thread.
    fn delete_thread(&self, thread_id: &str) -> BoxFuture<'_, Result<()>> {
        let _ = thread_id;
        Box::pin(async {
            Err(crate::kernel::error::Error::Checkpoint(
                "delete_thread is not supported by this async checkpointer".to_string(),
            ))
        })
    }
    /// Copy every checkpoint of one thread to another thread id.
    fn copy_thread(&self, source: &str, target: &str) -> BoxFuture<'_, Result<()>> {
        let _ = (source, target);
        Box::pin(async {
            Err(crate::kernel::error::Error::Checkpoint(
                "copy_thread is not supported by this async checkpointer".to_string(),
            ))
        })
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
