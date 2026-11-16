//! LangGraph-schema SQLite saver (feature `sqlite`).
//!
//! Writes checkpoints into LangGraph's `checkpoints` table (MessagePack checkpoint +
//! JSON metadata) so LangGraph tooling can inspect the database, and stores the full
//! internal checkpoint under `metadata.langgraph_rs` so this crate can restore pending
//! tasks/pending writes and resume exactly.

use std::sync::{Mutex, MutexGuard};

use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Map, Value, json};

use crate::kernel::checkpoint::Checkpoint;
use crate::kernel::error::{Error, Result};
use crate::kernel::ports::Checkpointer;

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
    ) -> Result<Option<crate::adapters::langgraph_sqlite::CheckpointRow>> {
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
    ) -> Result<Vec<crate::adapters::langgraph_sqlite::CheckpointRow>> {
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
            .map(crate::adapters::langgraph_sqlite::LangGraphSqliteReader::convert_row)
            .transpose()
    }

    fn langgraph_id(checkpoint: &Checkpoint) -> String {
        // LangGraph expects a hex checkpoint id. Keep a time-ordered prefix and derive the
        // suffix from the internal id so the mapping is deterministic.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in checkpoint.id.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{:016x}{hash:016x}", checkpoint.created_at_millis)
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
        // LangGraph's root (dict) state lives in `__root__`.
        object.insert(
            "channel_values".to_string(),
            json!({ "__root__": checkpoint.values }),
        );
        let mut versions = checkpoint.channel_versions.clone();
        versions.insert(
            "__root__".to_string(),
            Value::String(crate::kernel::checkpoint::version_string(checkpoint.step)),
        );
        object.insert("channel_versions".to_string(), Value::Object(versions));
        let checkpoint_blob = crate::interop::msgpack::encode(&Value::Object(object));

        let mut metadata = checkpoint.metadata.clone();
        metadata
            .entry("source".to_string())
            .or_insert_with(|| Value::String("loop".to_string()));
        metadata.insert("step".to_string(), json!(checkpoint.step as i64));
        metadata
            .entry("parents".to_string())
            .or_insert_with(|| json!({}));
        metadata.insert(
            "langgraph_rs".to_string(),
            serde_json::to_value(checkpoint).map_err(|error| {
                Error::Checkpoint(format!("langgraph sqlite saver: metadata: {error}"))
            })?,
        );
        let metadata_blob = serde_json::to_vec(&Value::Object(metadata)).map_err(|error| {
            Error::Checkpoint(format!("langgraph sqlite saver: metadata encode: {error}"))
        })?;
        Ok((checkpoint_blob, metadata_blob))
    }
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
                    Option::<String>::None,
                    checkpoint_blob,
                    metadata_blob,
                ],
            )
            .map_err(|error| Error::Checkpoint(format!("langgraph sqlite saver: put: {error}")))?;
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
             ORDER BY checkpoint_id DESC LIMIT 1",
            &[&thread_id, &namespace],
        )?;
        if let Some(checkpoint) = &mut checkpoint {
            let _ = crate::adapters::langgraph_sqlite::enrich_with(&connection, checkpoint);
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
             ORDER BY checkpoint_id DESC",
            &[&thread_id, &namespace],
        )?;
        let mut checkpoints = Vec::with_capacity(rows.len());
        for row in rows {
            let mut checkpoint =
                crate::adapters::langgraph_sqlite::LangGraphSqliteReader::convert_row(row)?;
            let _ = crate::adapters::langgraph_sqlite::enrich_with(&connection, &mut checkpoint);
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
                    ORDER BY checkpoint_id DESC LIMIT ?2
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
