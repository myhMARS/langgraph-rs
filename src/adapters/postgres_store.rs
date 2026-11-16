//! Postgres-backed long-term memory store (feature `postgres`).

use std::sync::{Mutex, MutexGuard};

use postgres::NoTls;
use serde_json::Value;

use crate::kernel::error::{Error, Result};
use crate::kernel::store::{BaseStore, StoreItem, item_matches, namespace_matches};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS langgraph_rs_store (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    PRIMARY KEY (namespace, key)
);";

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

/// `BaseStore` persisted in Postgres.
pub struct PostgresStore {
    client: Mutex<postgres::Client>,
}

impl std::fmt::Debug for PostgresStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresStore").finish_non_exhaustive()
    }
}

impl PostgresStore {
    /// Connect using a libpq URL/DSN and create the store table.
    pub fn connect(url: &str) -> Result<Self> {
        let client = postgres::Client::connect(url, NoTls)
            .map_err(|error| Error::Checkpoint(format!("postgres store: connect: {error}")))?;
        Self::from_client(client)
    }

    fn from_client(mut client: postgres::Client) -> Result<Self> {
        client
            .batch_execute(SCHEMA)
            .map_err(|error| Error::Checkpoint(format!("postgres store: schema: {error}")))?;
        Ok(Self {
            client: Mutex::new(client),
        })
    }

    fn lock(&self) -> MutexGuard<'_, postgres::Client> {
        self.client
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl BaseStore for PostgresStore {
    fn get(&self, namespace: &[String], key: &str) -> Result<Option<Value>> {
        let row = self
            .lock()
            .query_opt(
                "SELECT value FROM langgraph_rs_store WHERE namespace = $1 AND key = $2",
                &[&encode_namespace(namespace), &key],
            )
            .map_err(|error| Error::Checkpoint(format!("postgres store: get: {error}")))?;
        row.map(|row| {
            let payload: String = row
                .try_get(0)
                .map_err(|error| Error::Checkpoint(format!("postgres store: row: {error}")))?;
            serde_json::from_str(&payload)
                .map_err(|error| Error::Checkpoint(format!("postgres store: decode: {error}")))
        })
        .transpose()
    }

    fn put(&self, namespace: &[String], key: &str, value: Value) -> Result<()> {
        let payload = serde_json::to_string(&value)
            .map_err(|error| Error::Checkpoint(format!("postgres store: encode: {error}")))?;
        let now = crate::kernel::checkpoint::now_millis() as i64;
        self.lock()
            .execute(
                "INSERT INTO langgraph_rs_store (namespace, key, value, created_at, updated_at)
                 VALUES ($1, $2, $3, $4, $4)
                 ON CONFLICT (namespace, key) DO UPDATE SET
                     value = excluded.value,
                     updated_at = excluded.updated_at",
                &[&encode_namespace(namespace), &key, &payload, &now],
            )
            .map_err(|error| Error::Checkpoint(format!("postgres store: put: {error}")))?;
        Ok(())
    }

    fn delete(&self, namespace: &[String], key: &str) -> Result<()> {
        self.lock()
            .execute(
                "DELETE FROM langgraph_rs_store WHERE namespace = $1 AND key = $2",
                &[&encode_namespace(namespace), &key],
            )
            .map_err(|error| Error::Checkpoint(format!("postgres store: delete: {error}")))?;
        Ok(())
    }

    fn search(
        &self,
        namespace_prefix: &[String],
        query: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StoreItem>> {
        let prefix = encode_namespace(namespace_prefix);
        let like = if prefix.is_empty() {
            "%".to_string()
        } else {
            format!("{prefix}/%")
        };
        let rows = self
            .lock()
            .query(
                "SELECT namespace, key, value, created_at, updated_at
                 FROM langgraph_rs_store
                 WHERE namespace = $1 OR namespace LIKE $2
                 ORDER BY updated_at DESC, key ASC",
                &[&prefix, &like],
            )
            .map_err(|error| Error::Checkpoint(format!("postgres store: search: {error}")))?;
        let mut items = Vec::new();
        for row in rows {
            let namespace: String = row
                .try_get(0)
                .map_err(|error| Error::Checkpoint(format!("postgres store: row: {error}")))?;
            let key: String = row
                .try_get(1)
                .map_err(|error| Error::Checkpoint(format!("postgres store: row: {error}")))?;
            let payload: String = row
                .try_get(2)
                .map_err(|error| Error::Checkpoint(format!("postgres store: row: {error}")))?;
            let created_at: i64 = row
                .try_get(3)
                .map_err(|error| Error::Checkpoint(format!("postgres store: row: {error}")))?;
            let updated_at: i64 = row
                .try_get(4)
                .map_err(|error| Error::Checkpoint(format!("postgres store: row: {error}")))?;
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
        let rows = self
            .lock()
            .query(
                "SELECT DISTINCT namespace FROM langgraph_rs_store ORDER BY namespace",
                &[],
            )
            .map_err(|error| Error::Checkpoint(format!("postgres store: namespaces: {error}")))?;
        let mut namespaces = Vec::new();
        for row in rows {
            let namespace: String = row
                .try_get(0)
                .map_err(|error| Error::Checkpoint(format!("postgres store: row: {error}")))?;
            namespaces.push(decode_namespace(&namespace));
        }
        Ok(namespaces)
    }
}
