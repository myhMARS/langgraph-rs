//! SQLite-backed node cache (feature `sqlite`).

use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use rusqlite::{Connection, OptionalExtension, params};

use crate::kernel::cache::{CachedOutput, NodeCache};
use crate::kernel::error::{Error, Result};

/// Node cache persisted in SQLite.
pub struct SqliteCache {
    connection: Mutex<Connection>,
    max_entries: Option<usize>,
    sequence: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for SqliteCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteCache")
            .field("max_entries", &self.max_entries)
            .finish_non_exhaustive()
    }
}

impl SqliteCache {
    /// Open (or create) a cache at `path`.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let connection = Connection::open(path)
            .map_err(|error| Error::Checkpoint(format!("sqlite cache: open: {error}")))?;
        Self::from_connection(connection, None)
    }

    /// Create an in-memory cache.
    pub fn in_memory() -> Result<Self> {
        let connection = Connection::open_in_memory()
            .map_err(|error| Error::Checkpoint(format!("sqlite cache: open memory: {error}")))?;
        Self::from_connection(connection, None)
    }

    fn from_connection(connection: Connection, max_entries: Option<usize>) -> Result<Self> {
        connection
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS langgraph_rs_cache (
                    key TEXT PRIMARY KEY,
                    output TEXT NOT NULL,
                    expires_at INTEGER,
                    last_used INTEGER NOT NULL
                );",
            )
            .map_err(|error| Error::Checkpoint(format!("sqlite cache: init: {error}")))?;
        Ok(Self {
            connection: Mutex::new(connection),
            max_entries,
            sequence: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Bound the cache to `max_entries`, evicting the least-recently-used row.
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = Some(max_entries.max(1));
        self
    }

    fn next_sequence(&self) -> u64 {
        self.sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    fn lock(&self) -> MutexGuard<'_, Connection> {
        self.connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn encode(output: &CachedOutput) -> Result<String> {
    serde_json::to_string(output)
        .map_err(|error| Error::Checkpoint(format!("sqlite cache: encode: {error}")))
}

fn decode(payload: &str) -> Result<CachedOutput> {
    serde_json::from_str(payload)
        .map_err(|error| Error::Checkpoint(format!("sqlite cache: decode: {error}")))
}

fn now_millis() -> i64 {
    crate::kernel::checkpoint::now_millis() as i64
}

impl NodeCache for SqliteCache {
    fn get(&self, key: &str) -> Result<Option<CachedOutput>> {
        let connection = self.lock();
        let row: Option<(String, Option<i64>)> = connection
            .query_row(
                "SELECT output, expires_at FROM langgraph_rs_cache WHERE key = ?1",
                params![key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|error| Error::Checkpoint(format!("sqlite cache: get: {error}")))?;
        let Some((payload, expires_at)) = row else {
            return Ok(None);
        };
        if let Some(expires_at) = expires_at
            && now_millis() >= expires_at
        {
            connection
                .execute(
                    "DELETE FROM langgraph_rs_cache WHERE key = ?1",
                    params![key],
                )
                .map_err(|error| Error::Checkpoint(format!("sqlite cache: expire: {error}")))?;
            return Ok(None);
        }
        let sequence = self.next_sequence() as i64;
        connection
            .execute(
                "UPDATE langgraph_rs_cache SET last_used = ?2 WHERE key = ?1",
                params![key, sequence],
            )
            .map_err(|error| Error::Checkpoint(format!("sqlite cache: touch: {error}")))?;
        Ok(Some(decode(&payload)?))
    }

    fn put(&self, key: &str, output: CachedOutput, ttl: Option<Duration>) -> Result<()> {
        let payload = encode(&output)?;
        let expires_at = ttl.map(|ttl| now_millis() + ttl.as_millis() as i64);
        let sequence = self.next_sequence() as i64;
        let connection = self.lock();
        if let Some(max_entries) = self.max_entries {
            let count: i64 = connection
                .query_row("SELECT COUNT(*) FROM langgraph_rs_cache", [], |row| {
                    row.get(0)
                })
                .map_err(|error| Error::Checkpoint(format!("sqlite cache: count: {error}")))?;
            if count as usize >= max_entries {
                connection
                    .execute(
                        "DELETE FROM langgraph_rs_cache WHERE key IN (
                            SELECT key FROM langgraph_rs_cache ORDER BY last_used ASC LIMIT ?1
                        )",
                        params![(count as usize - max_entries + 1) as i64],
                    )
                    .map_err(|error| Error::Checkpoint(format!("sqlite cache: evict: {error}")))?;
            }
        }
        connection
            .execute(
                "INSERT INTO langgraph_rs_cache (key, output, expires_at, last_used)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(key) DO UPDATE SET
                     output = excluded.output,
                     expires_at = excluded.expires_at,
                     last_used = excluded.last_used",
                params![key, payload, expires_at, sequence],
            )
            .map_err(|error| Error::Checkpoint(format!("sqlite cache: put: {error}")))?;
        Ok(())
    }

    fn clear(&self) -> Result<()> {
        self.lock()
            .execute("DELETE FROM langgraph_rs_cache", [])
            .map_err(|error| Error::Checkpoint(format!("sqlite cache: clear: {error}")))?;
        Ok(())
    }
}
