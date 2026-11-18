//! Postgres-backed node cache (feature `postgres`).

use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use postgres::NoTls;

use crate::_internal::cache::{CachedOutput, NodeCache};
use crate::_internal::error::{Error, Result};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS langgraph_rs_cache (
    key TEXT PRIMARY KEY,
    output TEXT NOT NULL,
    expires_at BIGINT,
    last_used BIGINT NOT NULL
);";

/// `NodeCache` persisted in Postgres.
pub struct PostgresCache {
    client: Mutex<postgres::Client>,
    max_entries: Option<usize>,
    sequence: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for PostgresCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresCache")
            .field("max_entries", &self.max_entries)
            .finish_non_exhaustive()
    }
}

impl PostgresCache {
    /// Connect using a libpq URL/DSN and create the cache table.
    pub fn connect(url: &str) -> Result<Self> {
        let client = postgres::Client::connect(url, NoTls)
            .map_err(|error| Error::Checkpoint(format!("postgres cache: connect: {error}")))?;
        Self::from_client(client)
    }

    fn from_client(mut client: postgres::Client) -> Result<Self> {
        client
            .batch_execute(SCHEMA)
            .map_err(|error| Error::Checkpoint(format!("postgres cache: schema: {error}")))?;
        Ok(Self {
            client: Mutex::new(client),
            max_entries: None,
            sequence: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Bound the cache to `max_entries`, evicting the least-recently-used row.
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = Some(max_entries.max(1));
        self
    }

    fn lock(&self) -> MutexGuard<'_, postgres::Client> {
        self.client
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn next_sequence(&self) -> i64 {
        self.sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed) as i64
    }
}

fn now_millis() -> i64 {
    crate::checkpoint::now_millis() as i64
}

impl NodeCache for PostgresCache {
    fn get(&self, key: &str) -> Result<Option<CachedOutput>> {
        let mut client = self.lock();
        let row = client
            .query_opt(
                "SELECT output, expires_at FROM langgraph_rs_cache WHERE key = $1",
                &[&key],
            )
            .map_err(|error| Error::Checkpoint(format!("postgres cache: get: {error}")))?;
        let Some(row) = row else {
            return Ok(None);
        };
        let payload: String = row
            .try_get(0)
            .map_err(|error| Error::Checkpoint(format!("postgres cache: row: {error}")))?;
        let expires_at: Option<i64> = row
            .try_get(1)
            .map_err(|error| Error::Checkpoint(format!("postgres cache: row: {error}")))?;
        if let Some(expires_at) = expires_at
            && now_millis() >= expires_at
        {
            client
                .execute("DELETE FROM langgraph_rs_cache WHERE key = $1", &[&key])
                .map_err(|error| Error::Checkpoint(format!("postgres cache: expire: {error}")))?;
            return Ok(None);
        }
        let sequence = self.next_sequence();
        client
            .execute(
                "UPDATE langgraph_rs_cache SET last_used = $2 WHERE key = $1",
                &[&key, &sequence],
            )
            .map_err(|error| Error::Checkpoint(format!("postgres cache: touch: {error}")))?;
        serde_json::from_str(&payload)
            .map(Some)
            .map_err(|error| Error::Checkpoint(format!("postgres cache: decode: {error}")))
    }

    fn put(&self, key: &str, output: CachedOutput, ttl: Option<Duration>) -> Result<()> {
        let payload = serde_json::to_string(&output)
            .map_err(|error| Error::Checkpoint(format!("postgres cache: encode: {error}")))?;
        let expires_at = ttl.map(|ttl| now_millis() + ttl.as_millis() as i64);
        let sequence = self.next_sequence();
        let mut client = self.lock();
        if let Some(max_entries) = self.max_entries {
            let count: i64 = client
                .query_one("SELECT COUNT(*) FROM langgraph_rs_cache", &[])
                .map_err(|error| Error::Checkpoint(format!("postgres cache: count: {error}")))?
                .get(0);
            if count as usize >= max_entries {
                client
                    .execute(
                        "DELETE FROM langgraph_rs_cache WHERE key IN (
                            SELECT key FROM langgraph_rs_cache ORDER BY last_used ASC LIMIT $1
                        )",
                        &[&((count as usize - max_entries + 1) as i64)],
                    )
                    .map_err(|error| {
                        Error::Checkpoint(format!("postgres cache: evict: {error}"))
                    })?;
            }
        }
        client
            .execute(
                "INSERT INTO langgraph_rs_cache (key, output, expires_at, last_used)
                 VALUES ($1, $2, $3, $4)
                 ON CONFLICT (key) DO UPDATE SET
                     output = excluded.output,
                     expires_at = excluded.expires_at,
                     last_used = excluded.last_used",
                &[&key, &payload, &expires_at, &sequence],
            )
            .map_err(|error| Error::Checkpoint(format!("postgres cache: put: {error}")))?;
        Ok(())
    }

    fn clear(&self) -> Result<()> {
        self.lock()
            .execute("DELETE FROM langgraph_rs_cache", &[])
            .map_err(|error| Error::Checkpoint(format!("postgres cache: clear: {error}")))?;
        Ok(())
    }
}
