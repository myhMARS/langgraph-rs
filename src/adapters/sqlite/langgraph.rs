//! Read-only access to LangGraph's SQLite checkpoint tables (feature `sqlite`).
//!
//! LangGraph stores checkpoints as MessagePack blobs and metadata as JSON bytes in
//! `checkpoints(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type,
//! checkpoint, metadata)`; pending task writes live in `writes(...)`. This reader converts
//! those rows into the crate's [`Checkpoint`] so histories can be inspected or exported.
//! LangGraph's internal channels (`__root__`, `__start__`, `branch:to:*`) are normalized.

use std::sync::{Mutex, MutexGuard};

use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde_json::{Map, Value};

use crate::_internal::error::Interrupt;
use crate::_internal::error::{Error, Result};
use crate::_internal::node::Task;
use crate::checkpoint::{Checkpoint, PendingWrite};

pub(crate) type CheckpointRow = (
    String,
    String,
    String,
    Option<String>,
    Option<Vec<u8>>,
    Option<Vec<u8>>,
);

/// Read-only reader for a LangGraph SQLite checkpointer database.
pub struct LangGraphSqliteReader {
    connection: Mutex<Connection>,
}

impl std::fmt::Debug for LangGraphSqliteReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangGraphSqliteReader")
            .finish_non_exhaustive()
    }
}

impl LangGraphSqliteReader {
    /// Open an existing LangGraph SQLite database read-only.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let connection = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
        )
        .map_err(|error| Error::Checkpoint(format!("langgraph sqlite: open: {error}")))?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Connection> {
        self.connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// All threads with at least one checkpoint.
    pub fn threads(&self) -> Result<Vec<String>> {
        let connection = self.lock();
        let mut statement = connection
            .prepare("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite: threads: {error}")))?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite: threads: {error}")))?;
        let mut threads = Vec::new();
        for thread in rows {
            threads
                .push(thread.map_err(|error| {
                    Error::Checkpoint(format!("langgraph sqlite row: {error}"))
                })?);
        }
        Ok(threads)
    }

    /// Newest checkpoint of a thread in a namespace.
    pub fn latest(&self, thread_id: &str, namespace: &str) -> Result<Option<Checkpoint>> {
        let connection = self.lock();
        let row = connection
            .query_row(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                        checkpoint, metadata
                 FROM checkpoints
                 WHERE thread_id = ?1 AND checkpoint_ns = ?2
                 ORDER BY REPLACE(checkpoint_id, '-', '') DESC
                 LIMIT 1",
                params![thread_id, namespace],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<Vec<u8>>>(4)?,
                        row.get::<_, Option<Vec<u8>>>(5)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite: latest: {error}")))?;
        drop(connection);
        let Some(mut checkpoint) = row.map(Self::convert_row).transpose()? else {
            return Ok(None);
        };
        self.enrich(&mut checkpoint)?;
        Ok(Some(checkpoint))
    }

    /// Full history of a thread in a namespace, newest first.
    pub fn history(&self, thread_id: &str, namespace: &str) -> Result<Vec<Checkpoint>> {
        let connection = self.lock();
        let mut statement = connection
            .prepare(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                        checkpoint, metadata
                 FROM checkpoints
                 WHERE thread_id = ?1 AND checkpoint_ns = ?2
                 ORDER BY REPLACE(checkpoint_id, '-', '') DESC",
            )
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite: history: {error}")))?;
        let rows = statement
            .query_map(params![thread_id, namespace], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<Vec<u8>>>(4)?,
                    row.get::<_, Option<Vec<u8>>>(5)?,
                ))
            })
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite: history: {error}")))?;
        let mut collected: Vec<CheckpointRow> = Vec::new();
        for row in rows {
            collected.push(
                row.map_err(|error| Error::Checkpoint(format!("langgraph sqlite row: {error}")))?,
            );
        }
        drop(statement);
        drop(connection);
        let mut checkpoints = Vec::with_capacity(collected.len());
        for row in collected {
            let mut checkpoint = Self::convert_row(row)?;
            self.enrich(&mut checkpoint)?;
            checkpoints.push(checkpoint);
        }
        Ok(checkpoints)
    }

    /// A specific checkpoint in a namespace.
    pub fn get(
        &self,
        thread_id: &str,
        namespace: &str,
        checkpoint_id: &str,
    ) -> Result<Option<Checkpoint>> {
        let connection = self.lock();
        let row = connection
            .query_row(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                        checkpoint, metadata
                 FROM checkpoints
                 WHERE thread_id = ?1 AND checkpoint_ns = ?2 AND checkpoint_id = ?3",
                params![thread_id, namespace, checkpoint_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<Vec<u8>>>(4)?,
                        row.get::<_, Option<Vec<u8>>>(5)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite: get: {error}")))?;
        drop(connection);
        let Some(mut checkpoint) = row.map(Self::convert_row).transpose()? else {
            return Ok(None);
        };
        self.enrich(&mut checkpoint)?;
        Ok(Some(checkpoint))
    }

    /// Pending writes recorded for a checkpoint.
    pub fn pending_writes(
        &self,
        thread_id: &str,
        namespace: &str,
        checkpoint_id: &str,
    ) -> Result<Vec<PendingWrite>> {
        let connection = self.lock();
        let mut statement = connection
            .prepare(
                "SELECT task_id, task_path, channel, value FROM writes
                 WHERE thread_id = ?1 AND checkpoint_ns = ?2 AND checkpoint_id = ?3
                 ORDER BY task_id, idx",
            )
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite: writes: {error}")))?;
        let rows = statement
            .query_map(params![thread_id, namespace, checkpoint_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<Vec<u8>>>(3)?,
                ))
            })
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite: writes: {error}")))?;
        let mut writes = Vec::new();
        for row in rows {
            let (task_id, task_path, channel, value) = row.map_err(|error| {
                Error::Checkpoint(format!("langgraph sqlite writes row: {error}"))
            })?;
            let Some(value) = value.filter(|value| !value.is_empty()) else {
                continue;
            };
            let Ok(value) = crate::interop::msgpack::decode(&value) else {
                continue;
            };
            let node = task_path
                .rsplit(',')
                .next()
                .unwrap_or_default()
                .trim()
                .to_string();
            writes.push(PendingWrite {
                task_id,
                node,
                channel,
                value,
                replace: false,
            });
        }
        Ok(writes)
    }

    /// Fill LangGraph pending tasks from the `writes` table.
    fn enrich(&self, checkpoint: &mut Checkpoint) -> Result<()> {
        let connection = self.lock();
        enrich_with(&connection, checkpoint)
    }

    pub(crate) fn convert_row(row: CheckpointRow) -> Result<Checkpoint> {
        let (thread_id, namespace, checkpoint_id, parent_id, checkpoint_blob, metadata_blob) = row;
        let checkpoint_blob = checkpoint_blob.ok_or_else(|| {
            Error::Checkpoint("langgraph sqlite: checkpoint blob is missing".to_string())
        })?;
        let document = crate::interop::msgpack::decode(&checkpoint_blob)?;
        let object = document.as_object().ok_or_else(|| {
            Error::Checkpoint("langgraph sqlite: checkpoint is not a map".to_string())
        })?;
        let channel_values = object
            .get("channel_values")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let interrupts = extract_interrupts(&channel_values);
        let values = normalize_channel_values(channel_values);
        let channel_versions = object
            .get("channel_versions")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let versions_seen: crate::checkpoint::VersionsSeen = object
            .get("versions_seen")
            .and_then(Value::as_object)
            .map(|seen| {
                seen.iter()
                    .map(|(node, versions)| (node.clone(), std::sync::Arc::new(versions.clone())))
                    .collect()
            })
            .unwrap_or_default();
        let updated_channels = object
            .get("updated_channels")
            .and_then(Value::as_array)
            .map(|channels| {
                channels
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let metadata = metadata_blob
            .as_deref()
            .and_then(|blob| serde_json::from_slice::<Map<String, Value>>(blob).ok())
            .unwrap_or_default();
        if let Some(extension) = metadata.get("langgraph_rs") {
            if let Ok(mut checkpoint) = serde_json::from_value::<Checkpoint>(extension.clone()) {
                checkpoint.thread_id = thread_id;
                checkpoint.namespace = namespace;
                // The row's hex id is the public checkpoint id; the sidecar may carry an
                // older internal id, so the database column always wins.
                checkpoint.id = checkpoint_id;
                return Ok(checkpoint);
            }
        }
        // Rows carry LangGraph's metadata step, one below our internal counter.
        let step = metadata
            .get("step")
            .and_then(Value::as_i64)
            .map(|step| step.max(0) as usize)
            .unwrap_or_default();
        let created_at_millis = object
            .get("ts")
            .and_then(Value::as_str)
            .and_then(parse_rfc3339_millis)
            .unwrap_or_default();
        let mut checkpoint = Checkpoint::new(checkpoint_id, thread_id, step, values);
        checkpoint.namespace = namespace;
        checkpoint.parent_id = parent_id;
        checkpoint.created_at_millis = created_at_millis;
        checkpoint.metadata = metadata;
        checkpoint.channel_versions = channel_versions;
        checkpoint.versions_seen = versions_seen;
        checkpoint.updated_channels = updated_channels;
        checkpoint.interrupted = !interrupts.is_empty();
        checkpoint.interrupts = interrupts;
        Ok(checkpoint)
    }
}

fn normalize_channel_values(channel_values: Map<String, Value>) -> Map<String, Value> {
    let mut values = Map::new();
    for channel in ["__root__", "__start__"] {
        if let Some(root) = channel_values.get(channel).and_then(Value::as_object) {
            for (key, value) in root {
                if key != "__interrupt__" {
                    values.insert(key.clone(), value.clone());
                }
            }
        }
    }
    for (channel, value) in channel_values {
        if channel == "__root__"
            || channel == "__start__"
            || channel == "__end__"
            || channel == "__interrupt__"
            || channel.starts_with("branch:to:")
            || value.is_null()
        {
            continue;
        }
        values.insert(channel, value);
    }
    values
}

pub(crate) fn enrich_with(connection: &Connection, checkpoint: &mut Checkpoint) -> Result<()> {
    if !checkpoint.next.is_empty() {
        return Ok(());
    }
    let mut statement = connection
        .prepare(
            "SELECT task_id, task_path, channel, value FROM writes
             WHERE thread_id = ?1 AND checkpoint_ns = ?2 AND checkpoint_id = ?3
             ORDER BY task_id, idx",
        )
        .map_err(|error| Error::Checkpoint(format!("langgraph sqlite: writes: {error}")))?;
    let rows = statement
        .query_map(
            params![checkpoint.thread_id, checkpoint.namespace, checkpoint.id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, rusqlite::types::Value>(3)?,
                ))
            },
        )
        .map_err(|error| Error::Checkpoint(format!("langgraph sqlite: writes: {error}")))?;
    let mut collected: Vec<(String, String, String, rusqlite::types::Value)> = Vec::new();
    for row in rows {
        collected.push(
            row.map_err(|error| {
                Error::Checkpoint(format!("langgraph sqlite writes row: {error}"))
            })?,
        );
    }
    drop(statement);
    // LangGraph persists interrupts as `__interrupt__` writes; decode them so a foreign
    // database reports the same pending interrupts as `get_state`.
    let mut decoded_interrupts: Vec<Interrupt> = Vec::new();
    for (task_id, task_path, channel, value) in &collected {
        if channel != "__interrupt__" {
            continue;
        }
        let node = node_from_task_path(task_path);
        let Some(blob) = blob_bytes(value) else {
            continue;
        };
        let Ok(document) = crate::interop::msgpack::decode(&blob) else {
            continue;
        };
        let entries: Vec<&Value> = match &document {
            Value::Array(items) => items.iter().collect(),
            other => vec![other],
        };
        for entry in entries {
            let Some(entry) = entry.as_object() else {
                continue;
            };
            let Some(id) = entry.get("id").and_then(Value::as_str) else {
                continue;
            };
            if decoded_interrupts.iter().any(|known| known.id == id) {
                continue;
            }
            decoded_interrupts.push(Interrupt {
                node: node.clone(),
                value: entry.get("value").cloned().unwrap_or(Value::Null),
                id: id.to_string(),
                response_schema: entry
                    .get("response_schema")
                    .filter(|schema| !schema.is_null())
                    .cloned(),
                task_id: Some(task_id.clone().into_boxed_str()),
                static_pause: false,
                values: None,
            });
        }
    }
    if checkpoint.interrupts.is_empty() {
        checkpoint.interrupts = decoded_interrupts;
        checkpoint.interrupted = !checkpoint.interrupts.is_empty();
    }
    // Only interrupted tasks are still pending; a task that merely wrote state finished.
    let mut seen = std::collections::HashSet::new();
    for (task_id, task_path, channel, value) in collected {
        if channel != "__interrupt__" && channel != "__error__" {
            continue;
        }
        if !seen.insert(task_id.clone()) {
            continue;
        }
        let node = task_path
            .rsplit(',')
            .next()
            .unwrap_or_default()
            .trim()
            .to_string();
        if node.is_empty()
            || node == "__start__"
            || node == "__input__"
            || node.starts_with("~__pregel")
            || node.starts_with("__pregel")
        {
            continue;
        }
        let error = if channel == "__error__" {
            blob_bytes(&value)
                .and_then(|bytes| crate::interop::msgpack::decode(&bytes).ok())
                .map(|document| crate::_internal::error::NodeFailure {
                    node: node.clone(),
                        message: match &document {
                        Value::Object(map) => map
                            .get("message")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                            .unwrap_or_else(|| document.to_string()),
                        Value::String(text) => text.clone(),
                        other => other.to_string(),
                    },
                    attempts: 1,
                })
        } else {
            None
        };
        checkpoint.next.push(Task {
            id: task_id,
            node,
            input: None,
            resume: None,
            resume_history: Vec::new(),
            error,
            timeout_millis: None,
            interrupted: !checkpoint.interrupts.is_empty(),
            triggers: Vec::new(),
            push_index: None,
            schedule_index: None,
            push: false,
        });
    }
    // Attach node/task identity to interrupts decoded from `__interrupt__` payloads so
    // resume maps keyed by interrupt id can match their pending task.
    for (index, interrupt) in checkpoint.interrupts.iter_mut().enumerate() {
        if !interrupt.node.is_empty() {
            continue;
        }
        let task = checkpoint
            .next
            .get(index)
            .or_else(|| checkpoint.next.first());
        if let Some(task) = task {
            interrupt.node = task.node.clone();
            if interrupt.task_id.is_none() {
                interrupt.task_id = Some(task.id.clone().into_boxed_str());
            }
        }
    }
    Ok(())
}

fn extract_interrupts(channel_values: &Map<String, Value>) -> Vec<Interrupt> {
    let mut raw: Vec<Value> = Vec::new();
    if let Some(root) = channel_values.get("__root__").and_then(Value::as_object) {
        if let Some(values) = root.get("__interrupt__").and_then(Value::as_array) {
            raw.extend(values.clone());
        }
    }
    if let Some(values) = channel_values
        .get("__interrupt__")
        .and_then(Value::as_array)
    {
        raw.extend(values.clone());
    }
    raw.iter()
        .filter_map(|value| {
            let object = value.as_object()?;
            let payload = object.get("value").cloned().unwrap_or(Value::Null);
            let id = object
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let response_schema = object
                .get("response_schema")
                .filter(|schema| !schema.is_null())
                .cloned();
            Some(Interrupt {
                node: String::new(),
                value: payload,
                id,
                response_schema,
                task_id: None,
                static_pause: false,
                values: None,
            })
        })
        .collect()
}

fn parse_rfc3339_millis(value: &str) -> Option<u64> {
    // LangGraph emits RFC3339 with microseconds; parse the common `YYYY-MM-DDTHH:MM:SS` prefix.
    let date = &value.get(0..10)?;
    let time = value.get(11..19)?;
    let mut parts = date.split('-').map(|part| part.parse::<u32>().ok());
    let year = parts.next()??;
    let month = parts.next()??;
    let day = parts.next()??;
    let mut time_parts = time.split(':').map(|part| part.parse::<u32>().ok());
    let hour = time_parts.next()??;
    let minute = time_parts.next()??;
    let second = time_parts.next()??;
    let days = days_from_civil(year as i64, month as i64, day as i64);
    let seconds = days * 86_400 + hour as i64 * 3_600 + minute as i64 * 60 + second as i64;
    if seconds < 0 {
        return None;
    }
    Some(seconds as u64 * 1000)
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// Node name encoded in a LangGraph task path (`~__pregel_pull, <node>`).
fn node_from_task_path(path: &str) -> String {
    path.rsplit(',')
        .next()
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// Bytes of a `writes.value` cell, which may be stored as a blob or text.
fn blob_bytes(value: &rusqlite::types::Value) -> Option<Vec<u8>> {
    match value {
        rusqlite::types::Value::Blob(bytes) => Some(bytes.clone()),
        rusqlite::types::Value::Text(text) => Some(text.clone().into_bytes()),
        _ => None,
    }
}
