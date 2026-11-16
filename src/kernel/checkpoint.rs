//! Checkpointing: durable snapshots of graph state between supersteps.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::error::Interrupt;
use super::node::Task;

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
    /// Interrupt payloads captured when the checkpoint was written.
    #[serde(default)]
    pub interrupts: Vec<Interrupt>,
    /// Free-form checkpoint metadata (`source`, ...).
    #[serde(default)]
    pub metadata: Map<String, Value>,
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
            interrupts: Vec::new(),
            metadata: Map::new(),
        }
    }

    /// Node names that are scheduled after this checkpoint.
    pub fn next_nodes(&self) -> Vec<String> {
        self.next.iter().map(|task| task.node.clone()).collect()
    }
}

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

    #[test]
    fn checkpoint_reports_scheduled_nodes() {
        let mut checkpoint = Checkpoint::new(
            "cp-1",
            "thread",
            3,
            Map::from_iter([("value".to_string(), json!(1))]),
        );
        checkpoint.next.push(Task::new("a"));
        checkpoint.next.push(Task::new("b"));
        assert_eq!(
            checkpoint.next_nodes(),
            vec!["a".to_string(), "b".to_string()]
        );
        assert!(!checkpoint.interrupted);
        assert!(!checkpoint.paused_before);
    }

    #[test]
    fn checkpoint_ids_are_unique() {
        let first = new_checkpoint_id("thread", 0);
        let second = new_checkpoint_id("thread", 0);
        assert_ne!(first, second);
        assert!(first.starts_with("thread-step0-"));
    }
}
