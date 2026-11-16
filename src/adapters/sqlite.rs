//! SQLite checkpoint persistence (feature `sqlite`).
//!
//! Checkpoints are stored as JSON payloads in a single table, ordered by superstep and
//! creation time. The adapter is a plain [`Checkpointer`] implementation and needs no
//! runtime or connection pool.

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use crate::kernel::checkpoint::Checkpoint;
use crate::kernel::error::{Error, Result};
use crate::kernel::ports::Checkpointer;
use rusqlite::{Connection, OptionalExtension, params};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS langgraph_rs_checkpoints (
    thread_id TEXT NOT NULL,
    id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT '',
    step INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (thread_id, id)
);
CREATE INDEX IF NOT EXISTS langgraph_rs_checkpoints_thread
    ON langgraph_rs_checkpoints (thread_id, namespace, step DESC, created_at DESC);
";

/// Checkpointer backed by a SQLite database file.
pub struct SqliteSaver {
    connection: Mutex<Connection>,
}

impl std::fmt::Debug for SqliteSaver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteSaver").finish_non_exhaustive()
    }
}

impl SqliteSaver {
    /// Open (or create) a database file and ensure the schema exists.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let connection = Connection::open(path)
            .map_err(|error| Error::Checkpoint(format!("sqlite: open database: {error}")))?;
        Self::from_connection(connection)
    }

    /// Open an in-memory database (useful for tests and short-lived processes).
    pub fn in_memory() -> Result<Self> {
        let connection = Connection::open_in_memory()
            .map_err(|error| Error::Checkpoint(format!("sqlite: open memory database: {error}")))?;
        Self::from_connection(connection)
    }

    fn from_connection(connection: Connection) -> Result<Self> {
        connection
            .execute_batch(SCHEMA)
            .map_err(|error| Error::Checkpoint(format!("sqlite: create schema: {error}")))?;
        // Migration for databases created before namespace support.
        let _ = connection.execute(
            "ALTER TABLE langgraph_rs_checkpoints ADD COLUMN namespace TEXT NOT NULL DEFAULT ''",
            [],
        );
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Connection> {
        self.connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn decode(row: &rusqlite::Row<'_>) -> rusqlite::Result<Checkpoint> {
        let payload: String = row.get("payload")?;
        serde_json::from_str(&payload).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })
    }
}

fn encode(checkpoint: &Checkpoint) -> Result<String> {
    serde_json::to_string(checkpoint)
        .map_err(|error| Error::Checkpoint(format!("sqlite: serialize checkpoint: {error}")))
}

impl Checkpointer for SqliteSaver {
    fn put(&self, checkpoint: Checkpoint) -> Result<()> {
        let payload = encode(&checkpoint)?;
        self.lock()
            .execute(
                "INSERT INTO langgraph_rs_checkpoints
                     (thread_id, id, namespace, step, created_at, payload)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT (thread_id, id) DO UPDATE SET
                     namespace = excluded.namespace,
                     step = excluded.step,
                     created_at = excluded.created_at,
                     payload = excluded.payload",
                params![
                    checkpoint.thread_id,
                    checkpoint.id,
                    checkpoint.namespace,
                    checkpoint.step as i64,
                    checkpoint.created_at_millis as i64,
                    payload
                ],
            )
            .map_err(|error| Error::Checkpoint(format!("sqlite: store checkpoint: {error}")))?;
        Ok(())
    }

    fn get(&self, thread_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        self.lock()
            .query_row(
                "SELECT payload FROM langgraph_rs_checkpoints WHERE thread_id = ?1 AND id = ?2",
                params![thread_id, checkpoint_id],
                SqliteSaver::decode,
            )
            .optional()
            .map_err(|error| Error::Checkpoint(format!("sqlite: read checkpoint: {error}")))
    }

    fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        self.lock()
            .query_row(
                "SELECT payload FROM langgraph_rs_checkpoints
                 WHERE thread_id = ?1
                 ORDER BY step DESC, created_at DESC, id DESC
                 LIMIT 1",
                params![thread_id],
                SqliteSaver::decode,
            )
            .optional()
            .map_err(|error| Error::Checkpoint(format!("sqlite: read latest checkpoint: {error}")))
    }

    fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        let connection = self.lock();
        let mut statement = connection
            .prepare(
                "SELECT payload FROM langgraph_rs_checkpoints
                 WHERE thread_id = ?1
                 ORDER BY step DESC, created_at DESC, id DESC",
            )
            .map_err(|error| Error::Checkpoint(format!("sqlite: prepare history: {error}")))?;
        let rows = statement
            .query_map(params![thread_id], SqliteSaver::decode)
            .map_err(|error| Error::Checkpoint(format!("sqlite: read history: {error}")))?;
        let mut checkpoints = Vec::new();
        for row in rows {
            checkpoints.push(
                row.map_err(|error| Error::Checkpoint(format!("sqlite: decode row: {error}")))?,
            );
        }
        Ok(checkpoints)
    }

    fn latest_in(&self, thread_id: &str, namespace: &str) -> Result<Option<Checkpoint>> {
        self.lock()
            .query_row(
                "SELECT payload FROM langgraph_rs_checkpoints
                 WHERE thread_id = ?1 AND namespace = ?2
                 ORDER BY step DESC, created_at DESC, id DESC
                 LIMIT 1",
                params![thread_id, namespace],
                SqliteSaver::decode,
            )
            .optional()
            .map_err(|error| Error::Checkpoint(format!("sqlite: read latest checkpoint: {error}")))
    }

    fn history_in(&self, thread_id: &str, namespace: &str) -> Result<Vec<Checkpoint>> {
        let connection = self.lock();
        let mut statement = connection
            .prepare(
                "SELECT payload FROM langgraph_rs_checkpoints
                 WHERE thread_id = ?1 AND namespace = ?2
                 ORDER BY step DESC, created_at DESC, id DESC",
            )
            .map_err(|error| Error::Checkpoint(format!("sqlite: prepare history: {error}")))?;
        let rows = statement
            .query_map(params![thread_id, namespace], |row| {
                SqliteSaver::decode(row)
            })
            .map_err(|error| Error::Checkpoint(format!("sqlite: read history: {error}")))?;
        let mut checkpoints = Vec::new();
        for row in rows {
            checkpoints.push(
                row.map_err(|error| Error::Checkpoint(format!("sqlite: decode row: {error}")))?,
            );
        }
        Ok(checkpoints)
    }

    fn delete_thread(&self, thread_id: &str) -> Result<()> {
        self.lock()
            .execute(
                "DELETE FROM langgraph_rs_checkpoints WHERE thread_id = ?1",
                params![thread_id],
            )
            .map_err(|error| Error::Checkpoint(format!("sqlite: delete thread: {error}")))?;
        Ok(())
    }

    fn copy_thread(&self, source_thread: &str, target_thread: &str) -> Result<()> {
        let checkpoints = self.history(source_thread)?;
        for mut checkpoint in checkpoints.into_iter().rev() {
            checkpoint.thread_id = target_thread.to_string();
            self.put(checkpoint)?;
        }
        Ok(())
    }

    fn threads(&self) -> Result<Vec<String>> {
        let connection = self.lock();
        let mut statement = connection
            .prepare(
                "SELECT DISTINCT thread_id FROM langgraph_rs_checkpoints ORDER BY thread_id ASC",
            )
            .map_err(|error| Error::Checkpoint(format!("sqlite: prepare threads: {error}")))?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|error| Error::Checkpoint(format!("sqlite: read threads: {error}")))?;
        let mut threads = Vec::new();
        for row in rows {
            threads.push(
                row.map_err(|error| Error::Checkpoint(format!("sqlite: decode thread: {error}")))?,
            );
        }
        Ok(threads)
    }
}
