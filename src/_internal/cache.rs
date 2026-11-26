//! Node-level caching: a port plus a cache policy for individual nodes.
//!
//! Caching is opt-in per node. When a graph has a [`NodeCache`] adapter and a node has a
//! [`CachePolicy`], the executor looks up a cached node output before running the node and
//! stores successful outputs afterwards. Interrupts and failures are never cached.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::_internal::error::Result;
use crate::_internal::node::Send as GraphSend;
use crate::_internal::usage::Usage;

/// A node output as stored in a cache.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CachedOutput {
    /// Channel writes of the node.
    pub updates: IndexMap<String, Value>,
    /// Direct channel replacements of the node.
    #[serde(default)]
    pub replaces: IndexMap<String, Value>,
    /// Explicit jumps requested by the node.
    #[serde(default)]
    pub goto: Vec<String>,
    /// Dynamic fan-out tasks requested by the node.
    #[serde(default)]
    pub sends: Vec<GraphSend>,
    /// Usage reported by the node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Cache backend for node outputs (port).
///
/// Adapters can store entries in memory, on disk, in Redis, ... Durations are advisory
/// from the engine's point of view: the backend is responsible for expiry.
pub trait NodeCache: Send + Sync {
    /// Look up a cached node output.
    fn get(&self, key: &str) -> Result<Option<CachedOutput>>;

    /// Store a node output, optionally with a time-to-live.
    fn put(&self, key: &str, output: CachedOutput, ttl: Option<Duration>) -> Result<()>;

    /// Remove every entry.
    fn clear(&self) -> Result<()>;
}

/// Custom cache key function over the node input state.
pub type CacheKeyFn = Arc<dyn Fn(&Value) -> String + Send + Sync>;

/// Cache configuration of one node.
///
/// The default cache key hashes the node name and the serialized node input state, so a
/// node is only reused when it sees exactly the same state. Use [`CachePolicy::with_key`]
/// to provide a coarser or custom key (for example, one that ignores timestamps).
#[derive(Clone, Default)]
pub struct CachePolicy {
    ttl: Option<Duration>,
    key_fn: Option<CacheKeyFn>,
}

impl std::fmt::Debug for CachePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachePolicy")
            .field("ttl", &self.ttl)
            .field("custom_key", &self.key_fn.is_some())
            .finish()
    }
}

impl CachePolicy {
    /// Create a policy with the default key and no expiry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Expire cached outputs after `ttl`.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Use a custom key function over the node input state.
    pub fn with_key<F>(mut self, key_fn: F) -> Self
    where
        F: Fn(&Value) -> String + Send + Sync + 'static,
    {
        self.key_fn = Some(Arc::new(key_fn));
        self
    }

    /// Configured time-to-live.
    /// Whether a custom cache key function is configured.
    pub fn has_custom_key(&self) -> bool {
        self.key_fn.is_some()
    }

    pub fn ttl(&self) -> Option<Duration> {
        self.ttl
    }

    /// Compute the cache key for `node` given its input channel values.
    pub(crate) fn cache_key(&self, node: &str, values: &Map<String, Value>) -> String {
        if let Some(key_fn) = &self.key_fn {
            return format!("{node}:{}", key_fn(&Value::Object(values.clone())));
        }
        let mut hasher = DefaultHasher::new();
        node.hash(&mut hasher);
        serde_json::to_vec(values)
            .unwrap_or_default()
            .hash(&mut hasher);
        format!("{node}:{:016x}", hasher.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn default_key_depends_on_node_and_state() {
        let policy = CachePolicy::new();
        let first = policy.cache_key("a", &Map::from_iter([("x".to_string(), json!(1))]));
        let same = policy.cache_key("a", &Map::from_iter([("x".to_string(), json!(1))]));
        let other_state = policy.cache_key("a", &Map::from_iter([("x".to_string(), json!(2))]));
        let other_node = policy.cache_key("b", &Map::from_iter([("x".to_string(), json!(1))]));
        assert_eq!(first, same);
        assert_ne!(first, other_state);
        assert_ne!(first, other_node);
    }

    #[test]
    fn custom_key_overrides_hashing() {
        let policy = CachePolicy::new().with_key(|state| {
            state
                .get("x")
                .and_then(Value::as_i64)
                .unwrap_or_default()
                .to_string()
        });
        let first = policy.cache_key("a", &Map::from_iter([("x".to_string(), json!(1))]));
        let same = policy.cache_key(
            "a",
            &Map::from_iter([
                ("x".to_string(), json!(1)),
                ("y".to_string(), json!("ignored")),
            ]),
        );
        assert_eq!(first, same);
    }
}
