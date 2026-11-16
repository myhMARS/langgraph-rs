//! Checkpointing: durable snapshots of graph state between supersteps.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Result;
use crate::node::Task;

/// A snapshot of a graph run, stored between supersteps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Unique checkpoint id.
    pub id: String,
    /// Thread this checkpoint belongs to.
    pub thread_id: String,
    /// Number of supersteps completed when the checkpoint was written.
    pub step: usize,
    /// Channel values at this point in time.
    pub values: Map<String, Value>,
    /// Work that is still scheduled.
    #[serde(default)]
    pub next: Vec<Task>,
    /// `true` when the graph stopped because of a human-in-the-loop interrupt.
    #[serde(default)]
    pub interrupted: bool,
    /// `true` when the checkpoint was written by an `interrupt_before` hook.
    ///
    /// Resuming such a checkpoint must run the pending node instead of pausing on the
    /// same hook again.
    #[serde(default)]
    pub paused_before: bool,
    /// Previous checkpoint in the thread history.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// Wall clock creation time in milliseconds since the Unix epoch.
    #[serde(default)]
    pub created_at_millis: u64,
}

impl Checkpoint {
    /// Create a checkpoint.
    pub fn new(
        id: impl Into<String>,
        thread_id: impl Into<String>,
        step: usize,
        values: Map<String, Value>,
    ) -> Self {
        Self {
            id: id.into(),
            thread_id: thread_id.into(),
            step,
            values,
            next: Vec::new(),
            interrupted: false,
            paused_before: false,
            parent_id: None,
            created_at_millis: now_millis(),
        }
    }

    /// Node names that are scheduled after this checkpoint.
    pub fn next_nodes(&self) -> Vec<String> {
        self.next.iter().map(|task| task.node.clone()).collect()
    }
}

/// Storage backend for checkpoints.
///
/// Implementations can persist to memory, a file, Redis, Postgres, ... The engine only
/// needs to write a checkpoint and read back the latest one for a thread.
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

/// Monotonic counter that keeps checkpoint ids unique within a process.
static CHECKPOINT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a checkpoint id for a thread and superstep.
pub(crate) fn new_checkpoint_id(thread_id: &str, step: usize) -> String {
    let sequence = CHECKPOINT_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{thread_id}-step{step}-{sequence:06}")
}

/// Current wall clock time in milliseconds.
pub(crate) fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
