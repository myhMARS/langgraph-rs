//! In-memory checkpoint adapter.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::kernel::checkpoint::Checkpoint;
use crate::kernel::error::Result;
use crate::kernel::ports::Checkpointer;

/// In-memory checkpointer, useful for tests, notebooks and short-lived services.
#[derive(Debug, Default)]
pub struct InMemorySaver {
    inner: Mutex<HashMap<String, Vec<Checkpoint>>>,
}

impl InMemorySaver {
    /// Create an empty saver.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Checkpointer for InMemorySaver {
    fn put(&self, checkpoint: Checkpoint) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        inner
            .entry(checkpoint.thread_id.clone())
            .or_default()
            .push(checkpoint);
        Ok(())
    }

    fn get(&self, thread_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner
            .get(thread_id)
            .and_then(|checkpoints| checkpoints.iter().find(|cp| cp.id == checkpoint_id))
            .cloned())
    }

    fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner
            .get(thread_id)
            .and_then(|checkpoints| checkpoints.last())
            .cloned())
    }

    fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner
            .get(thread_id)
            .map(|checkpoints| checkpoints.iter().rev().cloned().collect())
            .unwrap_or_default())
    }

    fn threads(&self) -> Result<Vec<String>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner.keys().cloned().collect())
    }
}

/// LangGraph-compatible alias for [`InMemorySaver`].
pub type MemorySaver = InMemorySaver;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Map, json};

    fn checkpoint(id: &str, thread: &str, step: usize, value: i64) -> Checkpoint {
        Checkpoint::new(
            id,
            thread,
            step,
            Map::from_iter([("value".to_string(), json!(value))]),
        )
    }

    #[test]
    fn saves_latest_and_history() {
        let saver = InMemorySaver::new();
        saver.put(checkpoint("c1", "thread", 0, 1)).unwrap();
        saver.put(checkpoint("c2", "thread", 1, 2)).unwrap();

        let latest = saver.latest("thread").unwrap().unwrap();
        assert_eq!(latest.id, "c2");

        let history = saver.history("thread").unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].id, "c2");
        assert_eq!(history[1].id, "c1");
        assert_eq!(saver.threads().unwrap(), vec!["thread".to_string()]);
    }

    #[test]
    fn unknown_thread_returns_nothing() {
        let saver = InMemorySaver::new();
        assert!(saver.latest("missing").unwrap().is_none());
        assert!(saver.get("missing", "nope").unwrap().is_none());
    }
}
