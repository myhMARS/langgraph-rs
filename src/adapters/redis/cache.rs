//! Redis-backed node cache (feature `redis`).

use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use redis::Commands;

use crate::_internal::cache::{CachedOutput, NodeCache};
use crate::_internal::error::{Error, Result};

const PREFIX: &str = "langgraph_rs:cache:";

fn cache_key(key: &str) -> String {
    format!("{PREFIX}{key}")
}

/// `NodeCache` persisted in Redis.
pub struct RedisCache {
    connection: Mutex<redis::Connection>,
}

impl std::fmt::Debug for RedisCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisCache").finish_non_exhaustive()
    }
}

impl RedisCache {
    /// Connect to Redis (`redis://...` URL).
    pub fn connect(url: &str) -> Result<Self> {
        let client = redis::Client::open(url)
            .map_err(|error| Error::Checkpoint(format!("redis cache: open client: {error}")))?;
        let connection = client
            .get_connection()
            .map_err(|error| Error::Checkpoint(format!("redis cache: connect: {error}")))?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    fn lock(&self) -> MutexGuard<'_, redis::Connection> {
        self.connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl NodeCache for RedisCache {
    fn get(&self, key: &str) -> Result<Option<CachedOutput>> {
        let payload: Option<String> = self
            .lock()
            .get(cache_key(key))
            .map_err(|error| Error::Checkpoint(format!("redis cache: get: {error}")))?;
        payload
            .map(|payload| {
                serde_json::from_str(&payload)
                    .map_err(|error| Error::Checkpoint(format!("redis cache: decode: {error}")))
            })
            .transpose()
    }

    fn put(&self, key: &str, output: CachedOutput, ttl: Option<Duration>) -> Result<()> {
        let payload = serde_json::to_string(&output)
            .map_err(|error| Error::Checkpoint(format!("redis cache: encode: {error}")))?;
        let mut connection = self.lock();
        let result: redis::RedisResult<()> = match ttl {
            Some(ttl) => redis::cmd("SETEX")
                .arg(cache_key(key))
                .arg(ttl.as_secs().max(1))
                .arg(payload)
                .query(&mut *connection),
            None => redis::cmd("SET")
                .arg(cache_key(key))
                .arg(payload)
                .query(&mut *connection),
        };
        result.map_err(|error| Error::Checkpoint(format!("redis cache: put: {error}")))
    }

    fn clear(&self) -> Result<()> {
        let mut connection = self.lock();
        let keys: Vec<String> = connection
            .keys(format!("{PREFIX}*"))
            .map_err(|error| Error::Checkpoint(format!("redis cache: keys: {error}")))?;
        if !keys.is_empty() {
            let _: () = connection
                .del(keys)
                .map_err(|error| Error::Checkpoint(format!("redis cache: clear: {error}")))?;
        }
        Ok(())
    }
}
