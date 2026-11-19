//! Checkpointing: durable snapshots of graph state between supersteps.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::_internal::error::{Interrupt, Result};
use crate::_internal::node::Task;

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
    /// Routing seeds (node names) of successful siblings of an interrupted superstep.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deferred_routes: Vec<Task>,
    /// `goto`/`Send` targets the successful siblings scheduled before the interrupt.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resume_tasks: Vec<Task>,
    /// Sibling nodes that chose `goto`; their static/conditional edges must not be
    /// re-derived when the pause is replayed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub goto_sources: Vec<String>,
    /// Tasks of the interrupted superstep that already completed (LangGraph reports them
    /// in `get_state().tasks` next to the pending ones).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completed_tasks: Vec<Task>,
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
    /// Deferred tasks parked until a superstep has no regular work.
    #[serde(default)]
    pub deferred: Vec<Task>,
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
            deferred_routes: Vec::new(),
            resume_tasks: Vec::new(),
            goto_sources: Vec::new(),
            completed_tasks: Vec::new(),
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
            deferred: Vec::new(),
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
    // LangGraph checkpoint ids are UUID-shaped hex: `bytes.fromhex(id.replace("-", ""))`
    // must yield 16 bytes, and v4 task ids hash those bytes.
    let sequence = CHECKPOINT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let millis = now_millis();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};
    thread_id.hash(&mut hasher);
    step.hash(&mut hasher);
    sequence.hash(&mut hasher);
    let tail = hasher.finish();
    let hex = format!(
        "{:012x}{:08x}{:08x}{:04x}",
        millis & 0xffff_ffff_ffff,
        sequence & 0xffff_ffff,
        step & 0xffff,
        tail & 0xffff
    );
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// LangGraph v4 task id: `xxh3_128(checkpoint_id_bytes + ns + step + node + kind + parts)`.
///
/// `namespace` is the task's checkpoint namespace including the node (`parent_ns|node` or
/// just `node`), matching `prepare_next_tasks` in `langgraph/pregel/_algo.py`.
pub(crate) fn v4_task_id(
    parent_checkpoint_id: &str,
    namespace: &str,
    node: &str,
    step: usize,
    push_index: Option<usize>,
) -> Option<String> {
    let cleaned: String = parent_checkpoint_id
        .chars()
        .filter(|ch| *ch != '-')
        .collect();
    if cleaned.len() != 32 || !cleaned.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return None;
    }
    let mut bytes = Vec::with_capacity(32);
    let digits = cleaned.as_bytes();
    let mut index = 0;
    while index + 1 < digits.len() {
        let pair = std::str::from_utf8(&digits[index..index + 2]).ok()?;
        bytes.push(u8::from_str_radix(pair, 16).ok()?);
        index += 2;
    }
    let mut payload = bytes;
    payload.extend_from_slice(namespace.as_bytes());
    payload.extend_from_slice(step.to_string().as_bytes());
    payload.extend_from_slice(node.as_bytes());
    match push_index {
        Some(position) => {
            payload.extend_from_slice(b"__pregel_push");
            payload.extend_from_slice(position.to_string().as_bytes());
        }
        None => {
            payload.extend_from_slice(b"__pregel_pull");
            payload.extend_from_slice(format!("branch:to:{node}").as_bytes());
        }
    }
    let digest = format!("{:032x}", xxhash_rust::xxh3::xxh3_128(&payload));
    Some(format!(
        "{}-{}-{}-{}-{}",
        &digest[0..8],
        &digest[8..12],
        &digest[12..16],
        &digest[16..20],
        &digest[20..32]
    ))
}

/// Current wall clock time in milliseconds.
/// ISO-8601 UTC timestamp for an epoch-millisecond value.
pub(crate) fn iso_timestamp(millis: u64) -> String {
    let seconds = (millis / 1000) as i64;
    let millis_part = millis % 1000;
    let days = seconds.div_euclid(86_400);
    let day_seconds = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = day_seconds / 3_600;
    let minute = (day_seconds % 3_600) / 60;
    let second = day_seconds % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis_part:03}Z")
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    if month <= 2 {
        year += 1;
    }
    (year, month, day)
}

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
    format!(
        "{:032}.0.{:016}",
        sequence,
        hasher.finish() % 10_000_000_000_000_000
    )
}

/// Compute LangGraph-style version metadata for a new checkpoint.
pub(crate) fn versions_for(
    previous: Option<&Checkpoint>,
    values: &Map<String, Value>,
    step: usize,
    executed: &[String],
    written_by_node: &std::collections::HashMap<String, Vec<String>>,
) -> (Map<String, Value>, Map<String, Value>, Vec<String>) {
    // The previous checkpoint is shared with the checkpointer (`Arc`), so the version maps
    // are copied rather than taken; they are small next to the values map and this is far
    // cheaper than cloning the whole checkpoint before persisting it.
    let mut channel_versions = previous
        .map(|checkpoint| checkpoint.channel_versions.clone())
        .unwrap_or_default();
    let mut versions_seen = previous
        .map(|checkpoint| checkpoint.versions_seen.clone())
        .unwrap_or_default();
    let mut updated = Vec::new();
    for (channel, value) in values {
        if previous.and_then(|checkpoint| checkpoint.values.get(channel)) != Some(value) {
            updated.push(channel.clone());
        }
    }
    // LangGraph records, per executed node, the versions of the channels it read *before*
    // this step. Reads are not tracked, so every executed node records the pre-step map.
    if !executed.is_empty() {
        let _ = written_by_node;
        let pre_step = previous
            .map(|checkpoint| checkpoint.channel_versions.clone())
            .unwrap_or_default();
        let seen = Value::Object(pre_step);
        for node in executed {
            versions_seen.insert(node.clone(), seen.clone());
        }
    }
    let version = Value::String(version_string(step));
    for channel in &updated {
        channel_versions.insert(channel.clone(), version.clone());
    }
    (channel_versions, versions_seen, updated)
}
/// Infer the node that last wrote, matching LangGraph's `_infer_as_node`.
///
/// LangGraph picks the node with the highest observed channel version and treats a tie at
/// the top as ambiguous. Our `versions_seen` records all current versions per executed
/// node (an approximation of per-trigger reads), so a tie is common; `None` therefore means
/// "cannot infer" and callers keep the pending tasks instead of failing.
pub(crate) fn infer_as_node(
    checkpoint: &Checkpoint,
    single_node: Option<&str>,
    is_node: impl Fn(&str) -> bool,
) -> Result<Option<String>> {
    // LangGraph short-circuits a one-node graph, where the author is unambiguous.
    if let Some(node) = single_node {
        return Ok(Some(node.to_string()));
    }
    let mut pairs: Vec<(String, String)> = Vec::new();
    for (node, seen) in &checkpoint.versions_seen {
        // LangGraph skips its internal snapshot-bump bookkeeping and nodes that no longer
        // exist in this graph.
        if node == "__snapshot_bumps__" || !is_node(node) {
            continue;
        }
        let Some(seen) = seen.as_object() else {
            continue;
        };
        for version in seen.values().filter_map(Value::as_str) {
            pairs.push((version.to_string(), node.clone()));
        }
    }
    if pairs.is_empty() {
        return Ok(None);
    }
    pairs.sort();
    let Some((version, node)) = pairs.pop() else {
        return Ok(None);
    };
    if pairs
        .last()
        .is_some_and(|(previous, _)| previous == &version)
    {
        return Ok(None);
    }
    Ok(Some(node))
}

/// Apply the `before`/`limit` window to a newest-first checkpoint list.
/// Checkpoints are shared between the run loop and the checkpointer so a save does not
/// have to deep-clone the whole checkpoint before handing it to the writer.
pub(crate) type SharedCheckpoint = std::sync::Arc<Checkpoint>;

pub(crate) fn slice_history(
    mut checkpoints: Vec<Checkpoint>,
    before: Option<&str>,
    limit: Option<usize>,
) -> Vec<Checkpoint> {
    if let Some(before) = before {
        match checkpoints.iter().position(|checkpoint| checkpoint.id == before) {
            Some(index) => {
                checkpoints.drain(0..=index);
            }
            None => checkpoints.clear(),
        };
    }
    if let Some(limit) = limit {
        checkpoints.truncate(limit);
    }
    checkpoints
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
    fn v4_task_ids_match_langgraph() {
        // Values captured from Python LangGraph 1.2.22 for a two-step graph: the `b` task
        // stored in the checkpoint after `a` (metadata step 0, so task step 1... 2 for the
        // next tick) and the `a` task stored in the post-input checkpoint.
        assert_eq!(
            v4_task_id("1f1e35d4-e33b-6e8d-8001-25c431728cdf", "b", "b", 2, None),
            Some("62d6e473-cf27-c2a6-3df4-4c2fde5aa4a8".to_string())
        );
        assert_eq!(
            v4_task_id("1f1e35d4-e33b-6e8c-8000-ef5e29b5e742", "a", "a", 1, None),
            Some("0df6e4c8-0330-3c00-e775-77561adb21d1".to_string())
        );
    }

    #[test]
    fn checkpoint_ids_are_unique_and_uuid_shaped() {
        let first = new_checkpoint_id("thread", 0);
        let second = new_checkpoint_id("thread", 0);
        assert_ne!(first, second);
        // LangGraph hashes the un-hexed id, so it must be 32 hex digits in UUID form.
        let cleaned: String = first.chars().filter(|ch| *ch != '-').collect();
        assert_eq!(cleaned.len(), 32);
        assert!(cleaned.chars().all(|ch| ch.is_ascii_hexdigit()));
        assert_eq!(first.len(), 36);
    }
}
