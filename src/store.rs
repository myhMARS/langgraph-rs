//! Long-term memory (`BaseStore`) and run-time context.
//!
//! A store persists values across threads and runs (LangGraph's `BaseStore`), while the
//! runtime context carries per-run user data and the store into node code.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::_internal::error::Result;
use crate::checkpoint::now_millis;

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
    /// Expiry time in milliseconds since the Unix epoch (`None` = no TTL).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at_millis: Option<u64>,
}

impl StoreItem {
    /// Whether a TTL expired this item.
    pub fn is_expired(&self) -> bool {
        self.expires_at_millis
            .is_some_and(|expires| now_millis() >= expires)
    }
}

/// Options of a `search` call (LangGraph's `query` / `filter` / `limit` / `offset`).
#[derive(Clone, Debug, Default)]
pub struct SearchOptions {
    /// Text or semantic query.
    pub query: Option<String>,
    /// Value filter (LangGraph's JSONB-like matching).
    pub filter: Option<serde_json::Map<String, Value>>,
    /// Maximum number of hits (`0` = every match).
    pub limit: usize,
    /// Number of matching items to skip.
    pub offset: usize,
}

/// One search result with an optional semantic score.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    /// Stored item.
    pub item: StoreItem,
    /// Cosine similarity when the store indexes embeddings.
    pub score: Option<f64>,
}

impl SearchHit {
    /// Build an unscored hit.
    pub fn new(item: StoreItem) -> Self {
        Self { item, score: None }
    }
}

/// Options of a `list_namespaces` call (LangGraph's namespace listing).
#[derive(Clone, Debug, Default)]
pub struct NamespaceOptions {
    /// Keep namespaces that start with these segments.
    pub prefix: Option<Vec<String>>,
    /// Keep namespaces that end with these segments.
    pub suffix: Option<Vec<String>>,
    /// Cut namespaces to this many segments.
    pub max_depth: Option<usize>,
    /// Maximum number of namespaces (`0` = every match).
    pub limit: usize,
    /// Number of namespaces to skip.
    pub offset: usize,
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

    /// Insert or replace a value with an optional TTL in minutes.
    ///
    /// Backends without TTL support ignore the TTL and store the value permanently.
    fn put_with_ttl(
        &self,
        namespace: &[String],
        key: &str,
        value: Value,
        _ttl_minutes: Option<f64>,
    ) -> Result<()> {
        self.put(namespace, key, value)
    }

    /// Search with LangGraph's `filter` / `limit` / `offset` semantics.
    ///
    /// The default implementation reuses [`BaseStore::search`] for the text query, drops expired
    /// items, applies the filter and finally paginates — the same order LangGraph uses.
    fn search_with(&self, namespace_prefix: &[String], options: &SearchOptions) -> Result<Vec<SearchHit>> {
        let items = self.search(namespace_prefix, options.query.as_deref(), None)?;
        let filtered = items
            .into_iter()
            .filter(|item| !item.is_expired())
            .filter(|item| match &options.filter {
                Some(filter) => item_matches_filter(&item.value, filter),
                None => true,
            })
            .skip(options.offset)
            .take(if options.limit == 0 {
                usize::MAX
            } else {
                options.limit
            })
            .map(SearchHit::new)
            .collect();
        Ok(filtered)
    }

    /// List namespaces with prefix / suffix / depth filters and pagination.
    fn list_namespaces(&self, options: &NamespaceOptions) -> Result<Vec<Vec<String>>> {
        let mut namespaces = self.namespaces()?;
        if let Some(prefix) = &options.prefix {
            namespaces.retain(|namespace| namespace_matches(prefix, namespace));
        }
        if let Some(suffix) = &options.suffix {
            namespaces.retain(|namespace| {
                namespace.len() >= suffix.len() && namespace[namespace.len() - suffix.len()..] == suffix[..]
            });
        }
        if let Some(depth) = options.max_depth {
            for namespace in &mut namespaces {
                namespace.truncate(depth);
            }
            namespaces.sort();
            namespaces.dedup();
        }
        let start = options.offset.min(namespaces.len());
        let end = if options.limit == 0 {
            namespaces.len()
        } else {
            (start + options.limit).min(namespaces.len())
        };
        Ok(namespaces[start..end].to_vec())
    }
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

/// LangGraph's JSONB-like filter matching (`_compare_values` in `langgraph.store.memory`).
///
/// A filter object compares recursively; a nested object whose keys start with `$` applies
/// comparison operators (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`).
pub(crate) fn item_matches_filter(value: &Value, filter: &serde_json::Map<String, Value>) -> bool {
    filter
        .iter()
        .all(|(key, expected)| compare_values(value.get(key), expected))
}

fn compare_values(value: Option<&Value>, expected: &Value) -> bool {
    if let Some(filter) = expected.as_object() {
        if filter.keys().any(|key| key.starts_with('$')) {
            return filter
                .iter()
                .all(|(operator, operand)| apply_operator(value, operator, operand));
        }
        let Some(value) = value.and_then(Value::as_object) else {
            return false;
        };
        return filter.iter().all(|(key, expected)| {
            compare_values(value.get(key), expected)
        });
    }
    if let Some(expected_items) = expected.as_array() {
        let Some(value_items) = value.and_then(Value::as_array) else {
            return false;
        };
        return value_items.len() == expected_items.len()
            && value_items
                .iter()
                .zip(expected_items)
                .all(|(value, expected)| compare_values(Some(value), expected));
    }
    matches!(value, Some(actual) if actual == expected)
}

fn apply_operator(value: Option<&Value>, operator: &str, operand: &Value) -> bool {
    let number = |value: Option<&Value>| value.and_then(Value::as_f64);
    match operator {
        "$eq" => matches!(value, Some(actual) if actual == operand),
        "$ne" => !matches!(value, Some(actual) if actual == operand),
        "$gt" => number(value).is_some_and(|actual| actual > number(Some(operand)).unwrap_or(f64::NAN)),
        "$gte" => number(value).is_some_and(|actual| actual >= number(Some(operand)).unwrap_or(f64::NAN)),
        "$lt" => number(value).is_some_and(|actual| actual < number(Some(operand)).unwrap_or(f64::NAN)),
        "$lte" => number(value).is_some_and(|actual| actual <= number(Some(operand)).unwrap_or(f64::NAN)),
        _ => false,
    }
}
/// Current timestamp helper for store items.
pub(crate) fn store_now() -> u64 {
    now_millis()
}
