//! Async checkpoint saver dispatch (sync and async checkpointer adapters).
#![allow(clippy::too_many_arguments)]

use super::compiled::AsyncCompiledGraph;
use crate::kernel::checkpoint::Checkpoint;
use crate::kernel::error::{Error, Result};
use crate::kernel::state::State;

impl<S: State> AsyncCompiledGraph<S> {
    pub(crate) async fn saver_put(&self, checkpoint: Checkpoint) -> Result<()> {
        if let Some(saver) = &self.async_checkpointer {
            saver.put(checkpoint).await
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.put(checkpoint)
        } else {
            Err(Error::MissingCheckpointer("save_checkpoint"))
        }
    }

    pub(crate) async fn saver_get(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<Option<Checkpoint>> {
        if let Some(saver) = &self.async_checkpointer {
            saver.get(thread_id, checkpoint_id).await
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.get(thread_id, checkpoint_id)
        } else {
            Err(Error::MissingCheckpointer("get"))
        }
    }

    pub(crate) async fn saver_latest(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        if let Some(saver) = &self.async_checkpointer {
            saver.latest(thread_id).await
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.latest(thread_id)
        } else {
            Err(Error::MissingCheckpointer("latest"))
        }
    }

    pub(crate) async fn saver_latest_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> Result<Option<Checkpoint>> {
        if let Some(saver) = &self.async_checkpointer {
            saver.latest_in(thread_id, namespace).await
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.latest_in(thread_id, namespace)
        } else {
            Err(Error::MissingCheckpointer("latest_in"))
        }
    }

    pub(crate) async fn saver_history(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        if let Some(saver) = &self.async_checkpointer {
            saver.history(thread_id).await
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.history(thread_id)
        } else {
            Err(Error::MissingCheckpointer("history"))
        }
    }

    pub(crate) async fn saver_history_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> Result<Vec<Checkpoint>> {
        if let Some(saver) = &self.async_checkpointer {
            saver.history_in(thread_id, namespace).await
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.history_in(thread_id, namespace)
        } else {
            Err(Error::MissingCheckpointer("history_in"))
        }
    }

    pub(crate) fn saver_history_in_sync(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> Result<Vec<Checkpoint>> {
        futures::executor::block_on(self.saver_history_in(thread_id, namespace))
    }

    pub(crate) fn saver_threads_sync(&self) -> Result<Vec<String>> {
        if let Some(saver) = &self.async_checkpointer {
            futures::executor::block_on(saver.threads())
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.threads()
        } else {
            Err(Error::MissingCheckpointer("threads"))
        }
    }

    pub(crate) fn saver_latest_in_sync(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> Result<Option<Checkpoint>> {
        futures::executor::block_on(self.saver_latest_in(thread_id, namespace))
    }

    pub(crate) fn saver_history_sync(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        futures::executor::block_on(self.saver_history(thread_id))
    }
}
