//! Thread management APIs (list, prune, delete, copy).
#![allow(clippy::too_many_arguments)]

use super::compiled::CompiledGraph;
use crate::_internal::error::{Error, Result};
use crate::_internal::state::State;

impl<S: State> CompiledGraph<S> {
    pub fn threads(&self) -> Result<Vec<String>> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or(crate::Error::MissingCheckpointer("threads"))?;
        checkpointer.threads()
    }

    /// Keep only the newest `keep_last` checkpoints of a thread.
    pub fn prune_thread(&self, thread_id: &str, keep_last: usize) -> Result<()> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or(crate::Error::MissingCheckpointer("prune_thread"))?;
        checkpointer.prune_thread(thread_id, keep_last)
    }

    /// Delete every checkpoint of a thread.
    pub fn delete_thread(&self, thread_id: &str) -> Result<()> {
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("delete_thread"))?;
        checkpointer.delete_thread(thread_id)
    }

    /// Copy every checkpoint of one thread to another thread id.
    pub fn copy_thread(&self, source_thread: &str, target_thread: &str) -> Result<()> {
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("copy_thread"))?;
        checkpointer.copy_thread(source_thread, target_thread)
    }
}
