//! Postgres checkpoint persistence (feature `postgres`).
//!
//! Checkpoints are stored as JSON payloads in `langgraph_rs_checkpoints`, ordered by
//! superstep and creation time.

use std::sync::{Mutex, MutexGuard};

use crate::kernel::checkpoint::Checkpoint;
use crate::kernel::error::{Error, Result};
use crate::kernel::ports::Checkpointer;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS langgraph_rs_checkpoints (
    thread_id TEXT NOT NULL,
    id TEXT NOT NULL,
    step BIGINT NOT NULL,
    created_at BIGINT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (thread_id, id)
);
CREATE INDEX IF NOT EXISTS langgraph_rs_checkpoints_thread
    ON langgraph_rs_checkpoints (thread_id, step DESC, created_at DESC);
";

/// Checkpointer backed by Postgres.
pub struct PostgresSaver {
    client: Mutex<postgres::Client>,
}

impl std::fmt::Debug for PostgresSaver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresSaver").finish_non_exhaustive()
    }
}

impl PostgresSaver {
    /// Connect to Postgres (`postgres://...` URL) and ensure the schema exists.
    pub fn connect(url: &str) -> Result<Self> {
        let client = postgres::Client::connect(url, postgres::NoTls)
            .map_err(|error| Error::Checkpoint(format!("postgres: connect: {error}")))?;
        Self::from_client(client)
    }

    fn from_client(mut client: postgres::Client) -> Result<Self> {
        client
            .batch_execute(SCHEMA)
            .map_err(|error| Error::Checkpoint(format!("postgres: create schema: {error}")))?;
        Ok(Self {
            client: Mutex::new(client),
        })
    }

    fn lock(&self) -> MutexGuard<'_, postgres::Client> {
        self.client
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn decode(row: &postgres::Row) -> Result<Checkpoint> {
        let payload: String = row
            .try_get("payload")
            .map_err(|error| Error::Checkpoint(format!("postgres: read payload: {error}")))?;
        serde_json::from_str(&payload)
            .map_err(|error| Error::Checkpoint(format!("postgres: decode checkpoint: {error}")))
    }

    fn collect(rows: Vec<postgres::Row>) -> Result<Vec<Checkpoint>> {
        rows.iter().map(Self::decode).collect()
    }
}

impl Checkpointer for PostgresSaver {
    fn put(&self, checkpoint: Checkpoint) -> Result<()> {
        let payload = serde_json::to_string(&checkpoint).map_err(|error| {
            Error::Checkpoint(format!("postgres: serialize checkpoint: {error}"))
        })?;
        self.lock()
            .execute(
                "INSERT INTO langgraph_rs_checkpoints (thread_id, id, step, created_at, payload)
                 VALUES ($1, $2, $3, $4, $5)
                 ON CONFLICT (thread_id, id) DO UPDATE SET
                     step = EXCLUDED.step,
                     created_at = EXCLUDED.created_at,
                     payload = EXCLUDED.payload",
                &[
                    &checkpoint.thread_id,
                    &checkpoint.id,
                    &(checkpoint.step as i64),
                    &(checkpoint.created_at_millis as i64),
                    &payload,
                ],
            )
            .map_err(|error| Error::Checkpoint(format!("postgres: store checkpoint: {error}")))?;
        Ok(())
    }

    fn get(&self, thread_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        let mut client = self.lock();
        let row = client
            .query_opt(
                "SELECT payload FROM langgraph_rs_checkpoints WHERE thread_id = $1 AND id = $2",
                &[&thread_id, &checkpoint_id],
            )
            .map_err(|error| Error::Checkpoint(format!("postgres: read checkpoint: {error}")))?;
        row.as_ref().map(Self::decode).transpose()
    }

    fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        let mut client = self.lock();
        let row = client
            .query_opt(
                "SELECT payload FROM langgraph_rs_checkpoints
                 WHERE thread_id = $1
                 ORDER BY step DESC, created_at DESC, id DESC
                 LIMIT 1",
                &[&thread_id],
            )
            .map_err(|error| {
                Error::Checkpoint(format!("postgres: read latest checkpoint: {error}"))
            })?;
        row.as_ref().map(Self::decode).transpose()
    }

    fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        let mut client = self.lock();
        let rows = client
            .query(
                "SELECT payload FROM langgraph_rs_checkpoints
                 WHERE thread_id = $1
                 ORDER BY step DESC, created_at DESC, id DESC",
                &[&thread_id],
            )
            .map_err(|error| Error::Checkpoint(format!("postgres: read history: {error}")))?;
        Self::collect(rows)
    }

    fn threads(&self) -> Result<Vec<String>> {
        let mut client = self.lock();
        let rows = client
            .query(
                "SELECT DISTINCT thread_id FROM langgraph_rs_checkpoints ORDER BY thread_id ASC",
                &[],
            )
            .map_err(|error| Error::Checkpoint(format!("postgres: read threads: {error}")))?;
        rows.iter()
            .map(|row| {
                row.try_get::<_, String>(0)
                    .map_err(|error| Error::Checkpoint(format!("postgres: read thread: {error}")))
            })
            .collect()
    }
}
