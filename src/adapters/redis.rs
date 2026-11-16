//! Redis checkpoint persistence (feature `redis`).
//!
//! Each checkpoint is a JSON string under `langgraph_rs:checkpoint:{thread}:{id}` and a
//! thread index is kept in a sorted set scored by creation time, so `latest` and
//! `history` are cheap range lookups.

use std::sync::{Mutex, MutexGuard};

use redis::Commands;

use crate::kernel::checkpoint::Checkpoint;
use crate::kernel::error::{Error, Result};
use crate::kernel::ports::Checkpointer;

const THREADS_KEY: &str = "langgraph_rs:threads";

fn checkpoint_key(thread_id: &str, checkpoint_id: &str) -> String {
    format!("langgraph_rs:checkpoint:{thread_id}:{checkpoint_id}")
}

fn index_key(thread_id: &str) -> String {
    format!("langgraph_rs:thread:{thread_id}:index")
}

/// Checkpointer backed by Redis.
pub struct RedisSaver {
    connection: Mutex<redis::Connection>,
}

impl std::fmt::Debug for RedisSaver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisSaver").finish_non_exhaustive()
    }
}

impl RedisSaver {
    /// Connect to Redis (`redis://...` URL) and return the adapter.
    pub fn connect(url: &str) -> Result<Self> {
        let client = redis::Client::open(url)
            .map_err(|error| Error::Checkpoint(format!("redis: open client: {error}")))?;
        let connection = client
            .get_connection()
            .map_err(|error| Error::Checkpoint(format!("redis: connect: {error}")))?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    fn lock(&self) -> MutexGuard<'_, redis::Connection> {
        self.connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn decode(payload: String) -> Result<Checkpoint> {
        serde_json::from_str(&payload)
            .map_err(|error| Error::Checkpoint(format!("redis: decode checkpoint: {error}")))
    }

    fn load(connection: &mut redis::Connection, key: &str) -> Result<Option<Checkpoint>> {
        let payload: Option<String> = connection
            .get(key)
            .map_err(|error| Error::Checkpoint(format!("redis: get checkpoint: {error}")))?;
        payload.map(Self::decode).transpose()
    }
}

impl Checkpointer for RedisSaver {
    fn put(&self, checkpoint: Checkpoint) -> Result<()> {
        let payload = serde_json::to_string(&checkpoint)
            .map_err(|error| Error::Checkpoint(format!("redis: serialize checkpoint: {error}")))?;
        let mut connection = self.lock();
        let key = checkpoint_key(&checkpoint.thread_id, &checkpoint.id);
        let index = index_key(&checkpoint.thread_id);
        let score = if checkpoint.created_at_millis > 0 {
            checkpoint.created_at_millis as f64
        } else {
            checkpoint.step as f64
        };
        redis::pipe()
            .set(&key, payload)
            .ignore()
            .zadd(&index, &checkpoint.id, score)
            .ignore()
            .sadd(THREADS_KEY, &checkpoint.thread_id)
            .ignore()
            .query::<()>(&mut *connection)
            .map_err(|error| Error::Checkpoint(format!("redis: store checkpoint: {error}")))
    }

    fn get(&self, thread_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        let mut connection = self.lock();
        Self::load(&mut connection, &checkpoint_key(thread_id, checkpoint_id))
    }

    fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        let mut connection = self.lock();
        let ids: Vec<String> = connection
            .zrevrange(index_key(thread_id), 0, 0)
            .map_err(|error| Error::Checkpoint(format!("redis: read latest id: {error}")))?;
        let Some(id) = ids.first() else {
            return Ok(None);
        };
        Self::load(&mut connection, &checkpoint_key(thread_id, id))
    }

    fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        let mut connection = self.lock();
        let ids: Vec<String> = connection
            .zrevrange(index_key(thread_id), 0, -1)
            .map_err(|error| Error::Checkpoint(format!("redis: read history ids: {error}")))?;
        let mut checkpoints = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(checkpoint) = Self::load(&mut connection, &checkpoint_key(thread_id, &id))?
            {
                checkpoints.push(checkpoint);
            }
        }
        Ok(checkpoints)
    }

    fn threads(&self) -> Result<Vec<String>> {
        let mut connection = self.lock();
        connection
            .smembers(THREADS_KEY)
            .map_err(|error| Error::Checkpoint(format!("redis: read threads: {error}")))
    }
}
