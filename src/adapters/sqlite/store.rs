//! SQLite-backed long-term memory store (feature `sqlite`).

use std::sync::{Mutex, MutexGuard};

use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;

use crate::_internal::error::{Error, Result};
use crate::store::{BaseStore, StoreItem, item_matches, namespace_matches};

/// Long-term memory store persisted in SQLite.
pub struct SqliteStore {
    connection: Mutex<Connection>,
}

impl std::fmt::Debug for SqliteStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteStore").finish_non_exhaustive()
    }
}

impl SqliteStore {
    /// Open (or create) a store at `path`.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let connection = Connection::open(path)
            .map_err(|error| Error::Checkpoint(format!("sqlite store: open: {error}")))?;
        Self::from_connection(connection)
    }

    /// Create an in-memory store.
    pub fn in_memory() -> Result<Self> {
        let connection = Connection::open_in_memory()
            .map_err(|error| Error::Checkpoint(format!("sqlite store: open memory: {error}")))?;
        Self::from_connection(connection)
    }

    fn from_connection(connection: Connection) -> Result<Self> {
        connection
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS langgraph_rs_store (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (namespace, key)
                );",
            )
            .map_err(|error| Error::Checkpoint(format!("sqlite store: init: {error}")))?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Connection> {
        self.connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn encode_namespace(namespace: &[String]) -> String {
    namespace.join("/")
}

fn decode_namespace(namespace: &str) -> Vec<String> {
    if namespace.is_empty() {
        Vec::new()
    } else {
        namespace.split('/').map(str::to_owned).collect()
    }
}

impl BaseStore for SqliteStore {
    fn get(&self, namespace: &[String], key: &str) -> Result<Option<Value>> {
        let payload: Option<String> = self
            .lock()
            .query_row(
                "SELECT value FROM langgraph_rs_store WHERE namespace = ?1 AND key = ?2",
                params![encode_namespace(namespace), key],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| Error::Checkpoint(format!("sqlite store: get: {error}")))?;
        payload
            .map(|payload| {
                serde_json::from_str(&payload)
                    .map_err(|error| Error::Checkpoint(format!("sqlite store: decode: {error}")))
            })
            .transpose()
    }

    fn put(&self, namespace: &[String], key: &str, value: Value) -> Result<()> {
        let payload = serde_json::to_string(&value)
            .map_err(|error| Error::Checkpoint(format!("sqlite store: encode: {error}")))?;
        let now = crate::store::store_now() as i64;
        self.lock()
            .execute(
                "INSERT INTO langgraph_rs_store (namespace, key, value, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?4)
                 ON CONFLICT(namespace, key) DO UPDATE SET
                     value = excluded.value,
                     updated_at = excluded.updated_at",
                params![encode_namespace(namespace), key, payload, now],
            )
            .map_err(|error| Error::Checkpoint(format!("sqlite store: put: {error}")))?;
        Ok(())
    }

    fn delete(&self, namespace: &[String], key: &str) -> Result<()> {
        self.lock()
            .execute(
                "DELETE FROM langgraph_rs_store WHERE namespace = ?1 AND key = ?2",
                params![encode_namespace(namespace), key],
            )
            .map_err(|error| Error::Checkpoint(format!("sqlite store: delete: {error}")))?;
        Ok(())
    }

    fn search(
        &self,
        namespace_prefix: &[String],
        query: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StoreItem>> {
        let connection = self.lock();
        let mut statement = connection
            .prepare(
                "SELECT namespace, key, value, created_at, updated_at
                 FROM langgraph_rs_store
                 WHERE namespace = ?1 OR namespace LIKE ?2
                 ORDER BY updated_at DESC, key ASC",
            )
            .map_err(|error| Error::Checkpoint(format!("sqlite store: search prepare: {error}")))?;
        let prefix = encode_namespace(namespace_prefix);
        let like = if prefix.is_empty() {
            "%".to_string()
        } else {
            format!("{prefix}/%")
        };
        let rows = statement
            .query_map(params![prefix, like], |row| {
                let namespace: String = row.get(0)?;
                let key: String = row.get(1)?;
                let payload: String = row.get(2)?;
                let created_at: i64 = row.get(3)?;
                let updated_at: i64 = row.get(4)?;
                Ok((namespace, key, payload, created_at, updated_at))
            })
            .map_err(|error| Error::Checkpoint(format!("sqlite store: search: {error}")))?;
        let mut items = Vec::new();
        for row in rows {
            let (namespace, key, payload, created_at, updated_at) = row
                .map_err(|error| Error::Checkpoint(format!("sqlite store: search row: {error}")))?;
            let value: Value = match serde_json::from_str(&payload) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let namespace = decode_namespace(&namespace);
            if !namespace_matches(namespace_prefix, &namespace) {
                continue;
            }
            let item = StoreItem {
                namespace,
                key,
                value,
                created_at_millis: created_at as u64,
                updated_at_millis: updated_at as u64,
            };
            if item_matches(&item, query) {
                items.push(item);
                if let Some(limit) = limit {
                    if items.len() >= limit {
                        break;
                    }
                }
            }
        }
        Ok(items)
    }

    fn namespaces(&self) -> Result<Vec<Vec<String>>> {
        let connection = self.lock();
        let mut statement = connection
            .prepare("SELECT DISTINCT namespace FROM langgraph_rs_store ORDER BY namespace")
            .map_err(|error| {
                Error::Checkpoint(format!("sqlite store: namespaces prepare: {error}"))
            })?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|error| Error::Checkpoint(format!("sqlite store: namespaces: {error}")))?;
        let mut namespaces = Vec::new();
        for namespace in rows {
            let namespace = namespace
                .map_err(|error| Error::Checkpoint(format!("sqlite store row: {error}")))?;
            namespaces.push(decode_namespace(&namespace));
        }
        Ok(namespaces)
    }
}
