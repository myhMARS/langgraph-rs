//! Long-term memory (`BaseStore`) and run-time context.
//!
//! A store persists values across threads and runs (LangGraph's `BaseStore`), while the
//! runtime context carries per-run user data and the store into node code.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::kernel::checkpoint::now_millis;
use crate::kernel::error::Result;

/// Hierarchical namespace of a stored item (`("users", "42")`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Namespace(pub Vec<String>);

impl Namespace {
    /// Build a namespace from string segments.
    pub fn new<I, T>(segments: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Self(segments.into_iter().map(Into::into).collect())
    }

    /// Namespace segments.
    pub fn as_slice(&self) -> &[String] {
        &self.0
    }

    /// Render as `a/b/c`.
    pub fn joined(&self) -> String {
        self.0.join("/")
    }
}

/// A stored value with its namespace and timestamps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoreItem {
    /// Namespace of the item.
    pub namespace: Vec<String>,
    /// Item key.
    pub key: String,
    /// Stored value.
    pub value: Value,
    /// Creation time in milliseconds since the Unix epoch.
    pub created_at_millis: u64,
    /// Last update time in milliseconds since the Unix epoch.
    pub updated_at_millis: u64,
}

/// Long-term memory backend (port).
///
/// Implementations can store values in memory, SQLite, Postgres, Redis, ...
pub trait BaseStore: Send + Sync {
    /// Read a value.
    fn get(&self, namespace: &[String], key: &str) -> Result<Option<Value>>;

    /// Insert or replace a value.
    fn put(&self, namespace: &[String], key: &str, value: Value) -> Result<()>;

    /// Delete a value.
    fn delete(&self, namespace: &[String], key: &str) -> Result<()>;

    /// List values under a namespace prefix, optionally filtered by a substring query.
    fn search(
        &self,
        namespace_prefix: &[String],
        query: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StoreItem>>;

    /// List every namespace that contains at least one item.
    fn namespaces(&self) -> Result<Vec<Vec<String>>>;
}

/// Per-run runtime information available from node code.
#[derive(Clone, Default)]
pub struct Runtime {
    /// Thread id of the run, if checkpoint-bound.
    pub thread_id: Option<String>,
    /// Checkpoint namespace of the run (`""` for the root graph).
    pub namespace: String,
    /// Long-term memory store, when the graph was compiled with one.
    pub store: Option<Arc<dyn BaseStore>>,
    /// User-provided context for this run.
    pub context: Option<Value>,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("thread_id", &self.thread_id)
            .field("namespace", &self.namespace)
            .field("store", &self.store.is_some())
            .field("context", &self.context)
            .finish()
    }
}

/// `true` when `namespace` starts with `prefix`.
pub(crate) fn namespace_matches(prefix: &[String], namespace: &[String]) -> bool {
    namespace.len() >= prefix.len() && namespace.iter().zip(prefix).all(|(a, b)| a == b)
}

/// `true` when the item passes the optional search query (substring over key + value).
pub(crate) fn item_matches(item: &StoreItem, query: Option<&str>) -> bool {
    let Some(query) = query else {
        return true;
    };
    let query = query.to_lowercase();
    item.key.to_lowercase().contains(&query)
        || item.value.to_string().to_lowercase().contains(&query)
}

/// Current timestamp helper for store items.
pub(crate) fn store_now() -> u64 {
    now_millis()
}
