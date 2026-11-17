//! Checkpointing: durable snapshots of graph state between supersteps.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::error::Interrupt;
use super::node::Task;

/// A task write that was produced before an interrupt and must be replayed on resume.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PendingWrite {
    /// Id of the task that produced the write.
    pub task_id: String,
    /// Node that produced the write.
    pub node: String,
    /// Channel written.
    pub channel: String,
    /// Written value.
    pub value: Value,
    /// `true` when the write replaces the channel instead of merging it.
    #[serde(default)]
    pub replace: bool,
}

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
    /// Checkpoint namespace (`""` for the root graph), LangGraph's `checkpoint_ns`.
    #[serde(default)]
    pub namespace: String,
    /// Memoized functional-API task results (task key -> JSON value).
    #[serde(default)]
    pub task_results: Map<String, Value>,
    /// Per-channel version map (LangGraph `channel_versions`).
    #[serde(default)]
    pub channel_versions: Map<String, Value>,
    /// Per-node seen versions (LangGraph `versions_seen`, simplified).
    #[serde(default)]
    pub versions_seen: Map<String, Value>,
    /// Channels whose values changed when this checkpoint was written.
    #[serde(default)]
    pub updated_channels: Vec<String>,
    /// Successful task writes recorded when a superstep paused on an interrupt.
    #[serde(default)]
    pub pending_writes: Vec<PendingWrite>,
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
            namespace: String::new(),
            task_results: Map::new(),
            channel_versions: Map::new(),
            versions_seen: Map::new(),
            updated_channels: Vec::new(),
            pending_writes: Vec::new(),
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

/// LangGraph-compatible channel version string.
static VERSION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a LangGraph-style version: 32-digit decimal counter, `.0.`, random suffix.
pub(crate) fn version_string(_step: usize) -> String {
    let sequence = VERSION_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    let mut hasher = RandomState::new().build_hasher();
    sequence.hash(&mut hasher);
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
        .hash(&mut hasher);
    format!("{:032}.0.{}", sequence, hasher.finish())
}

/// Compute LangGraph-style version metadata for a new checkpoint.
pub(crate) fn versions_for(
    previous: Option<&Checkpoint>,
    values: &Map<String, Value>,
    step: usize,
    executed: &[String],
) -> (Map<String, Value>, Map<String, Value>, Vec<String>) {
    let mut channel_versions = previous
        .map(|checkpoint| checkpoint.channel_versions.clone())
        .unwrap_or_default();
    let mut versions_seen = previous
        .map(|checkpoint| checkpoint.versions_seen.clone())
        .unwrap_or_default();
    let mut updated = Vec::new();
    for (channel, value) in values {
        let changed = previous.and_then(|checkpoint| checkpoint.values.get(channel)) != Some(value);
        if changed {
            let version = Value::String(version_string(step));
            channel_versions.insert(channel.clone(), version);
            updated.push(channel.clone());
        }
    }
    // LangGraph records the channel versions each executed node has observed.
    if !executed.is_empty() {
        let seen: Map<String, Value> = values
            .keys()
            .filter_map(|channel| {
                channel_versions
                    .get(channel)
                    .map(|version| (channel.clone(), version.clone()))
            })
            .collect();
        for node in executed {
            versions_seen.insert(node.clone(), Value::Object(seen.clone()));
        }
    }
    (channel_versions, versions_seen, updated)
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
