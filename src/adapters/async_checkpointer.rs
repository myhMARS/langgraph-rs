//! Async checkpointer adapter that offloads a synchronous [`Checkpointer`] to a blocking pool.

use std::sync::Arc;

use crate::_internal::error::{Error, Result};
use crate::_internal::ports::{AsyncCheckpointer, BoxFuture, Checkpointer};
use crate::checkpoint::Checkpoint;

/// Wraps any synchronous checkpointer into an [`AsyncCheckpointer`].
///
/// Persistence calls run on a dedicated blocking pool, so a Tokio worker is never blocked
/// by SQLite/Redis/Postgres I/O. The wrapper owns a small private runtime for that pool.
pub struct BlockingSaver {
    inner: Arc<dyn Checkpointer>,
    runtime: Option<tokio::runtime::Runtime>,
}

impl std::fmt::Debug for BlockingSaver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockingSaver").finish_non_exhaustive()
    }
}

impl BlockingSaver {
    /// Wrap a synchronous checkpointer.
    pub fn new(inner: Arc<dyn Checkpointer>) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .map_err(|error| {
                Error::Checkpoint(format!("failed to start checkpoint runtime: {error}"))
            })?;
        Ok(Self {
            inner,
            runtime: Some(runtime),
        })
    }
}

fn join_error(error: tokio::task::JoinError) -> Error {
    Error::Checkpoint(format!("checkpoint task failed: {error}"))
}

impl BlockingSaver {
    fn handle(&self) -> tokio::runtime::Handle {
        self.runtime
            .as_ref()
            .expect("checkpoint runtime is present")
            .handle()
            .clone()
    }
}

impl Drop for BlockingSaver {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            // `Runtime::drop` blocks; shut down in the background so dropping from an async
            // context (or a test) never panics.
            runtime.shutdown_background();
        }
    }
}

impl AsyncCheckpointer for BlockingSaver {
    fn put(&self, checkpoint: Checkpoint) -> BoxFuture<'_, Result<()>> {
        let inner = self.inner.clone();
        Box::pin(async move {
            let handle = self.handle();
            handle
                .spawn_blocking(move || inner.put(checkpoint))
                .await
                .map_err(join_error)?
        })
    }

    fn get(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> BoxFuture<'_, Result<Option<Checkpoint>>> {
        let inner = self.inner.clone();
        let thread_id = thread_id.to_string();
        let checkpoint_id = checkpoint_id.to_string();
        Box::pin(async move {
            let handle = self.handle();
            handle
                .spawn_blocking(move || inner.get(&thread_id, &checkpoint_id))
                .await
                .map_err(join_error)?
        })
    }

    fn latest(&self, thread_id: &str) -> BoxFuture<'_, Result<Option<Checkpoint>>> {
        let inner = self.inner.clone();
        let thread_id = thread_id.to_string();
        Box::pin(async move {
            let handle = self.handle();
            handle
                .spawn_blocking(move || inner.latest(&thread_id))
                .await
                .map_err(join_error)?
        })
    }

    fn history(&self, thread_id: &str) -> BoxFuture<'_, Result<Vec<Checkpoint>>> {
        let inner = self.inner.clone();
        let thread_id = thread_id.to_string();
        Box::pin(async move {
            let handle = self.handle();
            handle
                .spawn_blocking(move || inner.history(&thread_id))
                .await
                .map_err(join_error)?
        })
    }

    fn latest_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> BoxFuture<'_, Result<Option<Checkpoint>>> {
        let inner = self.inner.clone();
        let thread_id = thread_id.to_string();
        let namespace = namespace.to_string();
        Box::pin(async move {
            let handle = self.handle();
            handle
                .spawn_blocking(move || inner.latest_in(&thread_id, &namespace))
                .await
                .map_err(join_error)?
        })
    }

    fn history_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> BoxFuture<'_, Result<Vec<Checkpoint>>> {
        let inner = self.inner.clone();
        let thread_id = thread_id.to_string();
        let namespace = namespace.to_string();
        Box::pin(async move {
            let handle = self.handle();
            handle
                .spawn_blocking(move || inner.history_in(&thread_id, &namespace))
                .await
                .map_err(join_error)?
        })
    }

    fn threads(&self) -> BoxFuture<'_, Result<Vec<String>>> {
        let inner = self.inner.clone();
        Box::pin(async move {
            let handle = self.handle();
            handle
                .spawn_blocking(move || inner.threads())
                .await
                .map_err(join_error)?
        })
    }

    fn prune_thread(&self, thread_id: &str, keep_last: usize) -> BoxFuture<'_, Result<()>> {
        let inner = self.inner.clone();
        let thread_id = thread_id.to_string();
        Box::pin(async move {
            let handle = self.handle();
            handle
                .spawn_blocking(move || inner.prune_thread(&thread_id, keep_last))
                .await
                .map_err(join_error)?
        })
    }

    fn delete_thread(&self, thread_id: &str) -> BoxFuture<'_, Result<()>> {
        let inner = self.inner.clone();
        let thread_id = thread_id.to_string();
        Box::pin(async move {
            let handle = self.handle();
            handle
                .spawn_blocking(move || inner.delete_thread(&thread_id))
                .await
                .map_err(join_error)?
        })
    }

    fn copy_thread(&self, source: &str, target: &str) -> BoxFuture<'_, Result<()>> {
        let inner = self.inner.clone();
        let source = source.to_string();
        let target = target.to_string();
        Box::pin(async move {
            let handle = self.handle();
            handle
                .spawn_blocking(move || inner.copy_thread(&source, &target))
                .await
                .map_err(join_error)?
        })
    }
}
