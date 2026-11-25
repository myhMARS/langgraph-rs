//! Redis-backed long-term memory store (feature `redis`).

use std::sync::{Mutex, MutexGuard};

use redis::Commands;
use serde_json::Value;

use crate::_internal::error::{Error, Result};
use crate::store::{BaseStore, StoreItem, item_matches, namespace_matches};

const NAMESPACES_KEY: &str = "langgraph_rs:store:namespaces";

fn item_key(namespace: &[String], key: &str) -> String {
    let namespace = namespace.join("/");
    if namespace.is_empty() {
        format!("langgraph_rs:store:{key}")
    } else {
        format!("langgraph_rs:store:{namespace}/{key}")
    }
}

fn redis_namespace_prefix(namespace: &[String]) -> String {
    if namespace.is_empty() {
        "langgraph_rs:store:".to_string()
    } else {
        format!("langgraph_rs:store:{}/", namespace.join("/"))
    }
}

/// `BaseStore` persisted in Redis (JSON values, one string per item).
pub struct RedisStore {
    connection: Mutex<redis::Connection>,
}

impl std::fmt::Debug for RedisStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisStore").finish_non_exhaustive()
    }
}

impl RedisStore {
    /// Connect to Redis (`redis://...` URL).
    pub fn connect(url: &str) -> Result<Self> {
        let client = redis::Client::open(url)
            .map_err(|error| Error::Checkpoint(format!("redis store: open client: {error}")))?;
        let connection = client
            .get_connection()
            .map_err(|error| Error::Checkpoint(format!("redis store: connect: {error}")))?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    fn lock(&self) -> MutexGuard<'_, redis::Connection> {
        self.connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn decode(payload: String) -> Result<(Value, u64, u64, Vec<String>, String)> {
        let mut stored: Value = serde_json::from_str(&payload)
            .map_err(|error| Error::Checkpoint(format!("redis store: decode: {error}")))?;
        let created = stored
            .get("_created_at")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let updated = stored
            .get("_updated_at")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let namespace = stored
            .get("_namespace")
            .and_then(Value::as_str)
            .map(|namespace| {
                namespace
                    .split('/')
                    .filter(|segment| !segment.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let key = stored
            .get("_key")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let value = stored
            .get_mut("value")
            .map(Value::take)
            .unwrap_or(Value::Null);
        Ok((value, created, updated, namespace, key))
    }
}

impl BaseStore for RedisStore {
    fn get(&self, namespace: &[String], key: &str) -> Result<Option<Value>> {
        let payload: Option<String> = self
            .lock()
            .get(item_key(namespace, key))
            .map_err(|error| Error::Checkpoint(format!("redis store: get: {error}")))?;
        payload
            .map(Self::decode)
            .transpose()
            .map(|entry| entry.map(|(value, _, _, _, _)| value))
    }

    fn put(&self, namespace: &[String], key: &str, value: Value) -> Result<()> {
        let now = crate::checkpoint::now_millis();
        let stored = serde_json::json!({
            "value": value,
            "_created_at": now,
            "_updated_at": now,
            "_namespace": namespace.join("/"),
            "_key": key,
        });
        let payload = serde_json::to_string(&stored)
            .map_err(|error| Error::Checkpoint(format!("redis store: encode: {error}")))?;
        let mut connection = self.lock();
        redis::pipe()
            .set(item_key(namespace, key), payload)
            .ignore()
            .sadd(NAMESPACES_KEY, namespace.join("/"))
            .ignore()
            .query::<()>(&mut *connection)
            .map_err(|error| Error::Checkpoint(format!("redis store: put: {error}")))
    }

    fn delete(&self, namespace: &[String], key: &str) -> Result<()> {
        self.lock()
            .del(item_key(namespace, key))
            .map_err(|error| Error::Checkpoint(format!("redis store: delete: {error}")))
    }

    fn search(
        &self,
        namespace_prefix: &[String],
        query: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StoreItem>> {
        let mut connection = self.lock();
        let pattern = format!("{}*", redis_namespace_prefix(namespace_prefix));
        let keys: Vec<String> = connection
            .keys(pattern)
            .map_err(|error| Error::Checkpoint(format!("redis store: keys: {error}")))?;
        let mut items = Vec::new();
        for key in keys {
            let Some(payload) = connection
                .get::<_, Option<String>>(&key)
                .map_err(|error| Error::Checkpoint(format!("redis store: read: {error}")))?
            else {
                continue;
            };
            let (value, created, updated, namespace, item_key) = Self::decode(payload)?;
            if namespace.is_empty() && item_key.is_empty() {
                continue;
            }
            if !namespace_matches(namespace_prefix, &namespace) {
                continue;
            }
            let item = StoreItem {
                namespace,
                key: item_key,
                value,
                created_at_millis: created,
                updated_at_millis: updated,
                expires_at_millis: None,
            };
            if item_matches(&item, query) {
                items.push(item);
            }
        }
        items.sort_by(|left, right| {
            right
                .updated_at_millis
                .cmp(&left.updated_at_millis)
                .then_with(|| left.key.cmp(&right.key))
        });
        if let Some(limit) = limit {
            items.truncate(limit);
        }
        Ok(items)
    }

    fn namespaces(&self) -> Result<Vec<Vec<String>>> {
        let namespaces: Vec<String> = self
            .lock()
            .smembers(NAMESPACES_KEY)
            .map_err(|error| Error::Checkpoint(format!("redis store: namespaces: {error}")))?;
        let mut result = Vec::new();
        for namespace in namespaces {
            result.push(
                namespace
                    .split('/')
                    .filter(|segment| !segment.is_empty())
                    .map(str::to_owned)
                    .collect(),
            );
        }
        result.sort();
        result.dedup();
        Ok(result)
    }
}
