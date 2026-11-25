//! LangGraph-schema SQLite saver (feature `sqlite`).
//!
//! Writes checkpoints into LangGraph's `checkpoints` table (MessagePack checkpoint +
//! JSON metadata) so LangGraph tooling can inspect the database, and stores the full
//! internal checkpoint under `metadata.langgraph_rs` so this crate can restore pending
//! tasks/pending writes and resume exactly.

use std::sync::{Mutex, MutexGuard};

use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Map, Value, json};

use crate::_internal::error::{Error, Result};
use crate::_internal::ports::Checkpointer;
use crate::checkpoint::Checkpoint;

/// A [`Checkpointer`] backed by LangGraph's SQLite table layout.
pub struct LangGraphSqliteSaver {
    connection: Mutex<Connection>,
}

impl std::fmt::Debug for LangGraphSqliteSaver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangGraphSqliteSaver")
            .finish_non_exhaustive()
    }
}

impl LangGraphSqliteSaver {
    /// Open (or create) a LangGraph-schema SQLite database.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let connection = Connection::open(path)
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite saver: open: {error}")))?;
        Self::from_connection(connection)
    }

    /// Create an in-memory LangGraph-schema database.
    pub fn in_memory() -> Result<Self> {
        let connection = Connection::open_in_memory().map_err(|error| {
            Error::Checkpoint(format!("langgraph sqlite saver: open memory: {error}"))
        })?;
        Self::from_connection(connection)
    }

    fn from_connection(connection: Connection) -> Result<Self> {
        let _ = connection.execute_batch(
            "PRAGMA journal_mode=DELETE; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;",
        );
        connection
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint BLOB,
                    metadata BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                );
                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    task_path TEXT NOT NULL DEFAULT '',
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT,
                    value BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                );",
            )
            .map_err(|error| {
                Error::Checkpoint(format!("langgraph sqlite saver: schema: {error}"))
            })?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Connection> {
        self.connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn row(
        connection: &Connection,
        sql: &str,
        parameters: &[&dyn rusqlite::ToSql],
    ) -> Result<Option<super::langgraph::CheckpointRow>> {
        connection
            .query_row(sql, parameters, |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<Vec<u8>>>(4)?,
                    row.get::<_, Option<Vec<u8>>>(5)?,
                ))
            })
            .optional()
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite saver: read: {error}")))
    }

    fn rows(
        connection: &Connection,
        sql: &str,
        parameters: &[&dyn rusqlite::ToSql],
    ) -> Result<Vec<super::langgraph::CheckpointRow>> {
        let mut statement = connection.prepare(sql).map_err(|error| {
            Error::Checkpoint(format!("langgraph sqlite saver: history: {error}"))
        })?;
        let rows = statement
            .query_map(parameters, |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<Vec<u8>>>(4)?,
                    row.get::<_, Option<Vec<u8>>>(5)?,
                ))
            })
            .map_err(|error| {
                Error::Checkpoint(format!("langgraph sqlite saver: history: {error}"))
            })?;
        let mut collected = Vec::new();
        for row in rows {
            collected.push(row.map_err(|error| {
                Error::Checkpoint(format!("langgraph sqlite saver: row: {error}"))
            })?);
        }
        Ok(collected)
    }

    fn read(
        connection: &Connection,
        sql: &str,
        parameters: &[&dyn rusqlite::ToSql],
    ) -> Result<Option<Checkpoint>> {
        Self::row(connection, sql, parameters)?
            .map(super::langgraph::LangGraphSqliteReader::convert_row)
            .transpose()
    }

    fn langgraph_id(checkpoint: &Checkpoint) -> String {
        // UUIDv6-style id: LangGraph orders checkpoints by id string, so both engines must emit
        // time-ordered ids for a thread written by both. LangGraph normalises the id with
        // `unhexlify(id.replace("-", ""))`, so the dashes are harmless.
        let ticks = checkpoint
            .created_at_millis
            .saturating_mul(10_000)
            .saturating_add((checkpoint.step as u64).min(9_999))
            .wrapping_add(0x01b2_1dd2_1381_4000);
        let time_high = ((ticks >> 28) & 0xffff_ffff) as u32;
        let time_mid = ((ticks >> 12) & 0xffff) as u16;
        let time_low = (ticks & 0x0fff) as u16;
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in checkpoint.id.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let clock = (((hash >> 48) & 0x3fff) as u16) | 0x8000;
        let node = hash & 0xffff_ffff_ffff;
        format!("{time_high:08x}{time_mid:04x}6{time_low:03x}{clock:04x}{node:012x}")
    }

    fn langgraph_payload(checkpoint: &Checkpoint) -> Result<(Vec<u8>, Vec<u8>)> {
        let document = crate::interop::checkpoint_to_langgraph(checkpoint);
        let mut object = document
            .get("checkpoint")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| {
                Error::Checkpoint("langgraph sqlite saver: checkpoint object missing".to_string())
            })?;
        object.insert(
            "id".to_string(),
            Value::String(Self::langgraph_id(checkpoint)),
        );
        // LangGraph's root (dict) state lives in `__root__`; pending branch channels
        // and interrupts must be visible for its scheduler and resume logic.
        let mut root = checkpoint.values.clone();
        let interrupt_entries =
            langgraph_interrupt_entries(&Self::langgraph_id(checkpoint), checkpoint);
        if !checkpoint.interrupts.is_empty() {
            let interrupts = checkpoint
                .interrupts
                .iter()
                .zip(&interrupt_entries)
                .map(|(interrupt, (id, _, _))| {
                    json!({
                        "__langgraph_type__": "Interrupt",
                        "value": interrupt.value,
                        "id": id,
                        "response_schema": interrupt.response_schema,
                    })
                })
                .collect::<Vec<_>>();
            root.insert("__interrupt__".to_string(), Value::Array(interrupts));
        }
        let mut channel_values = Map::new();
        channel_values.insert("__root__".to_string(), Value::Object(root.clone()));
        // Typed LangGraph schemas read individual channels, dict schemas read `__root__`.
        for (channel, value) in &root {
            channel_values.insert(channel.clone(), value.clone());
        }
        let mut versions = checkpoint.channel_versions.clone();
        let version = Value::String(crate::checkpoint::version_string(checkpoint.step));
        for channel in root.keys() {
            versions
                .entry(channel.clone())
                .or_insert_with(|| version.clone());
        }
        let mut sends = Vec::new();
        for task in &checkpoint.next {
            if let Some(input) = &task.input {
                sends.push(json!({
                    "__langgraph_type__": "Send",
                    "node": task.node,
                    "input": input,
                }));
            } else {
                channel_values.insert(format!("branch:to:{}", task.node), Value::Null);
                versions.insert(format!("branch:to:{}", task.node), version.clone());
            }
        }
        if !sends.is_empty() {
            channel_values.insert("__pregel_tasks".to_string(), Value::Array(sends));
            versions.insert("__pregel_tasks".to_string(), version.clone());
        }
        versions.insert("__root__".to_string(), version);
        object.insert("channel_values".to_string(), Value::Object(channel_values));
        object.insert("channel_versions".to_string(), Value::Object(versions));
        let checkpoint_blob = crate::interop::msgpack::encode(&Value::Object(object));

        let mut metadata = checkpoint.metadata.clone();
        metadata
            .entry("source".to_string())
            .or_insert_with(|| Value::String("loop".to_string()));
        // LangGraph metadata steps are one below our internal counter; task ids are
        // recomputed as `metadata.step + 1`, which is our internal step.
        metadata.insert("step".to_string(), json!(checkpoint.step as i64));
        metadata
            .entry("parents".to_string())
            .or_insert_with(|| json!({}));
        let mut sidecar = checkpoint.clone();
        sidecar.id = Self::langgraph_id(checkpoint);
        metadata.insert(
            "langgraph_rs".to_string(),
            serde_json::to_value(sidecar).map_err(|error| {
                Error::Checkpoint(format!("langgraph sqlite saver: metadata: {error}"))
            })?,
        );
        let metadata_blob = serde_json::to_vec(&Value::Object(metadata)).map_err(|error| {
            Error::Checkpoint(format!("langgraph sqlite saver: metadata encode: {error}"))
        })?;
        Ok((checkpoint_blob, metadata_blob))
    }
}

fn send_task_id(checkpoint_id: &str, node: &str, step: usize, index: usize) -> String {
    // Dashes are not part of the value; strip them before grouping nibbles.
    let hex: String = checkpoint_id
        .chars()
        .filter(char::is_ascii_hexdigit)
        .collect();
    let mut bytes: Vec<u8> = hex
        .as_bytes()
        .chunks(2)
        .filter_map(|pair| {
            std::str::from_utf8(pair)
                .ok()
                .and_then(|pair| u8::from_str_radix(pair, 16).ok())
        })
        .collect();
    bytes.extend_from_slice(node.as_bytes());
    bytes.extend_from_slice((step + 1).to_string().as_bytes());
    bytes.extend_from_slice(node.as_bytes());
    bytes.extend_from_slice(b"__pregel_push");
    bytes.extend_from_slice(index.to_string().as_bytes());
    use sha1::{Digest, Sha1};
    let mut hasher = Sha1::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

fn langgraph_interrupt_entries(
    checkpoint_id: &str,
    checkpoint: &crate::checkpoint::Checkpoint,
) -> Vec<(String, String, String)> {
    let mut send_queues: std::collections::BTreeMap<
        String,
        std::collections::VecDeque<(usize, String)>,
    > = std::collections::BTreeMap::new();
    for (index, task) in checkpoint
        .next
        .iter()
        .filter(|task| task.input.is_some())
        .enumerate()
    {
        let task_id = send_task_id(checkpoint_id, &task.node, checkpoint.step, index);
        send_queues
            .entry(task.node.clone())
            .or_default()
            .push_back((index, task_id));
    }
    let mut answered: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    checkpoint
        .interrupts
        .iter()
        .map(|interrupt| {
            if let Some(queue) = send_queues.get_mut(&interrupt.node) {
                if let Some((index, task_id)) = queue.pop_front() {
                    let key = format!("{}:{}", interrupt.node, task_id);
                    let id = format!("{:032x}", xxhash_rust::xxh3::xxh3_128(key.as_bytes()));
                    let path = format!("~__pregel_push, {index:010}, 0000000000");
                    return (id, task_id, path);
                }
            }
            let count = *answered.entry(interrupt.node.clone()).or_insert(0);
            *answered.get_mut(&interrupt.node).expect("counter") += 1;
            let task_id = langgraph_task_id(checkpoint_id, &interrupt.node, checkpoint.step);
            let namespace = format!("{}:{}", interrupt.node, task_id);
            let key = if count == 0 {
                namespace
            } else {
                // LangGraph separates the answered index with `|` (NS_SEP), not `:`.
                format!("{namespace}|{count}")
            };
            let id = format!("{:032x}", xxhash_rust::xxh3::xxh3_128(key.as_bytes()));
            let path = format!("~__pregel_pull, {}", interrupt.node);
            (id, task_id, path)
        })
        .collect()
}

fn encode_interrupt_ext(interrupt: &crate::_internal::error::Interrupt, id: &str) -> Vec<u8> {
    let mut payload = Map::new();
    payload.insert("value".to_string(), interrupt.value.clone());
    payload.insert("id".to_string(), Value::String(id.to_string()));
    if let Some(schema) = &interrupt.response_schema {
        payload.insert("response_schema".to_string(), schema.clone());
    }
    let body = crate::interop::msgpack::encode(&Value::Array(vec![
        Value::String("langgraph.types".to_string()),
        Value::String("Interrupt".to_string()),
        Value::Object(payload),
    ]));
    // LangGraph's serializer wraps types in msgpack ext type 2.
    let mut bytes = Vec::with_capacity(body.len() + 4);
    if body.len() <= u8::MAX as usize {
        bytes.extend_from_slice(&[0xc7, body.len() as u8, 0x02]);
    } else if body.len() <= u16::MAX as usize {
        bytes.push(0xc8);
        bytes.extend_from_slice(&(body.len() as u16).to_be_bytes());
        bytes.push(0x02);
    } else {
        bytes.push(0xc9);
        bytes.extend_from_slice(&(body.len() as u32).to_be_bytes());
        bytes.push(0x02);
    }
    bytes.extend_from_slice(&body);
    bytes
}

fn langgraph_task_id(checkpoint_id: &str, node: &str, step: usize) -> String {
    use sha1::{Digest, Sha1};
    let bytes: Vec<u8> = checkpoint_id
        .as_bytes()
        .chunks(2)
        .filter_map(|pair| {
            std::str::from_utf8(pair)
                .ok()
                .and_then(|pair| u8::from_str_radix(pair, 16).ok())
        })
        .collect();
    let mut hasher = Sha1::new();
    hasher.update(&bytes);
    let trigger = format!("branch:to:{node}");
    hasher.update(node.as_bytes());
    hasher.update((step + 1).to_string().as_bytes());
    hasher.update(node.as_bytes());
    hasher.update(b"__pregel_pull");
    hasher.update(trigger.as_bytes());
    let digest = hasher.finalize();
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

impl Checkpointer for LangGraphSqliteSaver {
    fn put(&self, checkpoint: Checkpoint) -> Result<()> {
        let (checkpoint_blob, metadata_blob) = Self::langgraph_payload(&checkpoint)?;
        let checkpoint_id = Self::langgraph_id(&checkpoint);
        let connection = self.lock();
        connection
            .execute(
                "INSERT OR REPLACE INTO checkpoints
                    (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata)
                 VALUES (?1, ?2, ?3, ?4, 'msgpack', ?5, ?6)",
                params![
                    checkpoint.thread_id,
                    checkpoint.namespace,
                    checkpoint_id,
                    checkpoint.parent_id,
                    checkpoint_blob,
                    metadata_blob,
                ],
            )
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite saver: put: {error}")))?;
        connection
            .execute(
                "DELETE FROM writes WHERE thread_id = ?1 AND checkpoint_ns = ?2 AND checkpoint_id = ?3",
                params![checkpoint.thread_id, checkpoint.namespace, checkpoint_id],
            )
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite saver: writes: {error}")))?;
        let mut statement = connection
            .prepare(
                "INSERT OR REPLACE INTO writes
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, value)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite saver: writes: {error}")))?;
        // Group writes per task and publish them as one LangGraph `__root__` write.
        let mut grouped: std::collections::BTreeMap<String, (String, Map<String, Value>)> =
            std::collections::BTreeMap::new();
        for write in &checkpoint.pending_writes {
            let entry = grouped
                .entry(write.task_id.clone())
                .or_insert_with(|| (write.node.clone(), Map::new()));
            entry.1.insert(write.channel.clone(), write.value.clone());
        }
        for (task_id, (node, writes)) in grouped {
            let value = crate::interop::msgpack::encode(&Value::Object(writes));
            statement
                .execute(params![
                    checkpoint.thread_id,
                    checkpoint.namespace,
                    checkpoint_id,
                    task_id,
                    format!("~__pregel_pull, {node}"),
                    0i64,
                    "__root__",
                    "msgpack",
                    value,
                ])
                .map_err(|error| {
                    Error::Checkpoint(format!("langgraph sqlite saver: write row: {error}"))
                })?;
        }
        for (index, task) in checkpoint
            .next
            .iter()
            .enumerate()
            .filter(|(_, task)| task.input.is_none())
        {
            statement
                .execute(params![
                    checkpoint.thread_id,
                    checkpoint.namespace,
                    checkpoint_id,
                    "00000000-0000-0000-0000-000000000000",
                    format!("~__pregel_pull, {}", task.node),
                    index as i64 + 1,
                    format!("branch:to:{}", task.node),
                    "null",
                    Vec::<u8>::new(),
                ])
                .map_err(|error| {
                    Error::Checkpoint(format!("langgraph sqlite saver: branch write: {error}"))
                })?;
        }
        let interrupt_entries = langgraph_interrupt_entries(&checkpoint_id, &checkpoint);
        for (interrupt, (interrupt_id, task_id, task_path)) in
            checkpoint.interrupts.iter().zip(interrupt_entries)
        {
            let value = encode_interrupt_ext(interrupt, &interrupt_id);
            statement
                .execute(params![
                    checkpoint.thread_id,
                    checkpoint.namespace,
                    checkpoint_id,
                    task_id,
                    task_path,
                    -3i64,
                    "__interrupt__",
                    "msgpack",
                    value,
                ])
                .map_err(|error| {
                    Error::Checkpoint(format!("langgraph sqlite saver: interrupt write: {error}"))
                })?;
        }
        Ok(())
    }

    fn get(&self, thread_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        let connection = self.lock();
        Self::read(
            &connection,
            "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                    checkpoint, metadata
             FROM checkpoints WHERE thread_id = ?1 AND checkpoint_id = ?2",
            &[&thread_id, &checkpoint_id],
        )
    }

    fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        self.latest_in(thread_id, "")
    }

    fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        self.history_in(thread_id, "")
    }

    fn latest_in(&self, thread_id: &str, namespace: &str) -> Result<Option<Checkpoint>> {
        let connection = self.lock();
        let mut checkpoint = Self::read(
            &connection,
            "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                    checkpoint, metadata
             FROM checkpoints WHERE thread_id = ?1 AND checkpoint_ns = ?2
             ORDER BY REPLACE(checkpoint_id, '-', '') DESC LIMIT 1",
            &[&thread_id, &namespace],
        )?;
        if let Some(checkpoint) = &mut checkpoint {
            let _ = super::langgraph::enrich_with(&connection, checkpoint);
        }
        Ok(checkpoint)
    }

    fn history_in(&self, thread_id: &str, namespace: &str) -> Result<Vec<Checkpoint>> {
        let connection = self.lock();
        let rows = Self::rows(
            &connection,
            "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                    checkpoint, metadata
             FROM checkpoints WHERE thread_id = ?1 AND checkpoint_ns = ?2
             ORDER BY REPLACE(checkpoint_id, '-', '') DESC",
            &[&thread_id, &namespace],
        )?;
        let mut checkpoints = Vec::with_capacity(rows.len());
        for row in rows {
            let mut checkpoint = super::langgraph::LangGraphSqliteReader::convert_row(row)?;
            let _ = super::langgraph::enrich_with(&connection, &mut checkpoint);
            checkpoints.push(checkpoint);
        }
        Ok(checkpoints)
    }

    fn threads(&self) -> Result<Vec<String>> {
        let connection = self.lock();
        let mut statement = connection
            .prepare("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")
            .map_err(|error| {
                Error::Checkpoint(format!("langgraph sqlite saver: threads: {error}"))
            })?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|error| {
                Error::Checkpoint(format!("langgraph sqlite saver: threads: {error}"))
            })?;
        let mut threads = Vec::new();
        for thread in rows {
            threads.push(thread.map_err(|error| {
                Error::Checkpoint(format!("langgraph sqlite saver: thread: {error}"))
            })?);
        }
        Ok(threads)
    }

    fn prune_thread(&self, thread_id: &str, keep_last: usize) -> Result<()> {
        let connection = self.lock();
        connection
            .execute(
                "DELETE FROM checkpoints WHERE thread_id = ?1 AND checkpoint_id NOT IN (
                    SELECT checkpoint_id FROM checkpoints WHERE thread_id = ?1
                    ORDER BY REPLACE(checkpoint_id, '-', '') DESC LIMIT ?2
                 )",
                params![thread_id, keep_last as i64],
            )
            .map_err(|error| {
                Error::Checkpoint(format!("langgraph sqlite saver: prune: {error}"))
            })?;
        Ok(())
    }

    fn delete_thread(&self, thread_id: &str) -> Result<()> {
        let connection = self.lock();
        connection
            .execute(
                "DELETE FROM checkpoints WHERE thread_id = ?1",
                params![thread_id],
            )
            .map_err(|error| {
                Error::Checkpoint(format!("langgraph sqlite saver: delete: {error}"))
            })?;
        connection
            .execute(
                "DELETE FROM writes WHERE thread_id = ?1",
                params![thread_id],
            )
            .map_err(|error| {
                Error::Checkpoint(format!("langgraph sqlite saver: delete: {error}"))
            })?;
        Ok(())
    }

    fn copy_thread(&self, source: &str, target: &str) -> Result<()> {
        let checkpoints = self.history_in(source, "")?;
        for mut checkpoint in checkpoints {
            checkpoint.thread_id = target.to_string();
            self.put(checkpoint)?;
        }
        Ok(())
    }
}

#[allow(dead_code)]
fn metadata_map(checkpoint: &Checkpoint) -> Map<String, Value> {
    checkpoint.metadata.clone()
}
