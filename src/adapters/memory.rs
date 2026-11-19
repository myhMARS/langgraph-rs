//! In-memory checkpoint adapter.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::_internal::cache::{CachedOutput, NodeCache};
use crate::_internal::error::Result;
use crate::_internal::ports::Checkpointer;
use crate::checkpoint::Checkpoint;
use crate::store::{BaseStore, StoreItem, item_matches, namespace_matches, store_now};

/// In-memory checkpointer, useful for tests, notebooks and short-lived services.
#[derive(Debug, Default)]
pub struct InMemorySaver {
    inner: Mutex<HashMap<String, Vec<std::sync::Arc<Checkpoint>>>>,
}

impl InMemorySaver {
    /// Create an empty saver.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Checkpointer for InMemorySaver {
    fn is_in_memory(&self) -> bool {
        true
    }

    fn put(&self, checkpoint: Checkpoint) -> Result<()> {
        self.put_shared(std::sync::Arc::new(checkpoint))
    }

    fn put_shared(&self, checkpoint: std::sync::Arc<Checkpoint>) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let checkpoints = inner.entry(checkpoint.thread_id.clone()).or_default();
        // Writes are keyed by checkpoint id: an interrupt attaches to the checkpoint it
        // paused on, so re-putting the same id updates it instead of duplicating history.
        // The re-put is virtually always the newest entry, so check that first: scanning the
        // whole thread on every save made long threads quadratic.
        if let Some(last) = checkpoints.last_mut()
            && last.id == checkpoint.id
        {
            *last = checkpoint;
            return Ok(());
        }
        if checkpoint.interrupted || checkpoint.paused_before {
            if let Some(existing) = checkpoints
                .iter_mut()
                .find(|existing| existing.id == checkpoint.id)
            {
                *existing = checkpoint;
                return Ok(());
            }
        }
        checkpoints.push(checkpoint);
        Ok(())
    }

    fn get(&self, thread_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner
            .get(thread_id)
            .and_then(|checkpoints| checkpoints.iter().find(|cp| cp.id == checkpoint_id))
            .map(|arc| (**arc).clone()))
    }

    fn has_thread(&self, thread_id: &str) -> Result<bool> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner.contains_key(thread_id))
    }

    fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner
            .get(thread_id)
            .and_then(|checkpoints| checkpoints.last())
            .map(|arc| (**arc).clone()))
    }

    fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner
            .get(thread_id)
            .map(|checkpoints| checkpoints.iter().rev().map(|arc| (**arc).clone()).collect())
            .unwrap_or_default())
    }

    fn threads(&self) -> Result<Vec<String>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner.keys().cloned().collect())
    }

    fn latest_in(&self, thread_id: &str, namespace: &str) -> Result<Option<Checkpoint>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner.get(thread_id).and_then(|checkpoints| {
            checkpoints
                .iter()
                .rev()
                .find(|checkpoint| checkpoint.namespace == namespace)
                .map(|arc| (**arc).clone())
        }))
    }

    fn history_page_in(
        &self,
        thread_id: &str,
        namespace: &str,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Checkpoint>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(checkpoints) = inner.get(thread_id) else {
            return Ok(Vec::new());
        };
        // Newest first, namespace-filtered, then windowed: only the selected checkpoints
        // are cloned, so a long thread is not copied for a five-entry page.
        let mut selected: Vec<&std::sync::Arc<Checkpoint>> = checkpoints
            .iter()
            .rev()
            .filter(|checkpoint| checkpoint.namespace == namespace)
            .collect();
        if let Some(before) = before {
            match selected.iter().position(|checkpoint| checkpoint.id == before) {
                Some(index) => {
                    selected.drain(0..=index);
                }
                None => selected.clear(),
            };
        }
        if let Some(limit) = limit {
            selected.truncate(limit);
        }
        Ok(selected.into_iter().map(|arc| (**arc).clone()).collect())
    }

    fn history_in(&self, thread_id: &str, namespace: &str) -> Result<Vec<Checkpoint>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner
            .get(thread_id)
            .map(|checkpoints| {
                checkpoints
                    .iter()
                    .rev()
                    .filter(|checkpoint| checkpoint.namespace == namespace)
                    .map(|arc| (**arc).clone())
                    .collect()
            })
            .unwrap_or_default())
    }

    fn prune_thread(&self, thread_id: &str, keep_last: usize) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(checkpoints) = inner.get_mut(thread_id) {
            let excess = checkpoints.len().saturating_sub(keep_last);
            checkpoints.drain(..excess);
        }
        Ok(())
    }

    fn delete_thread(&self, thread_id: &str) -> Result<()> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(thread_id);
        Ok(())
    }

    fn copy_thread(&self, source_thread: &str, target_thread: &str) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(source) = inner.get(source_thread).cloned() else {
            return Ok(());
        };
        let copies = source
            .into_iter()
            .map(|arc| {
                let mut checkpoint = (*arc).clone();
                checkpoint.thread_id = target_thread.to_string();
                std::sync::Arc::new(checkpoint)
            })
            .collect::<Vec<_>>();
        inner.insert(target_thread.to_string(), copies);
        Ok(())
    }
}

/// LangGraph-compatible alias for [`InMemorySaver`].
pub type MemorySaver = InMemorySaver;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Map, json};

    fn checkpoint(id: &str, thread: &str, step: usize, value: i64) -> Checkpoint {
        Checkpoint::new(
            id,
            thread,
            step,
            Map::from_iter([("value".to_string(), json!(value))]),
        )
    }

    #[test]
    fn saves_latest_and_history() {
        let saver = InMemorySaver::new();
        saver.put(checkpoint("c1", "thread", 0, 1)).unwrap();
        saver.put(checkpoint("c2", "thread", 1, 2)).unwrap();

        let latest = saver.latest("thread").unwrap().unwrap();
        assert_eq!(latest.id, "c2");

        let history = saver.history("thread").unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].id, "c2");
        assert_eq!(history[1].id, "c1");
        assert_eq!(saver.threads().unwrap(), vec!["thread".to_string()]);
    }

    #[test]
    fn unknown_thread_returns_nothing() {
        let saver = InMemorySaver::new();
        assert!(saver.latest("missing").unwrap().is_none());
        assert!(saver.get("missing", "nope").unwrap().is_none());
    }
}

/// Entry stored by the in-memory cache: output, expiry and LRU sequence.
type CacheEntry = (CachedOutput, Option<Instant>, u64);

/// In-memory node cache adapter with optional per-entry TTL and a bounded LRU size.
#[derive(Debug, Default)]
pub struct InMemoryCache {
    entries: Mutex<HashMap<String, CacheEntry>>,
    sequence: std::sync::atomic::AtomicU64,
    max_entries: Option<usize>,
}

impl InMemoryCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bound the cache to `max_entries`, evicting the least-recently-used entry.
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = Some(max_entries.max(1));
        self
    }

    /// Number of entries (expired entries are not pruned eagerly).
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Returns `true` when the cache holds no entry.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn next_sequence(&self) -> u64 {
        self.sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

impl NodeCache for InMemoryCache {
    fn get(&self, key: &str) -> Result<Option<CachedOutput>> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let now = Instant::now();
        let Some((output, expires_at, last_used)) = entries.get_mut(key) else {
            return Ok(None);
        };
        if let Some(expires_at) = expires_at {
            if now >= *expires_at {
                entries.remove(key);
                return Ok(None);
            }
        }
        *last_used = self.next_sequence();
        Ok(Some(output.clone()))
    }

    fn put(&self, key: &str, output: CachedOutput, ttl: Option<Duration>) -> Result<()> {
        let expires_at = ttl.map(|ttl| Instant::now() + ttl);
        let sequence = self.next_sequence();
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(max_entries) = self.max_entries
            && entries.len() >= max_entries
            && !entries.contains_key(key)
            && let Some(oldest) = entries
                .iter()
                .min_by_key(|(_, (_, _, last_used))| *last_used)
                .map(|(key, _)| key.clone())
        {
            entries.remove(&oldest);
        }
        entries.insert(key.to_string(), (output, expires_at, sequence));
        Ok(())
    }

    fn clear(&self) -> Result<()> {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        Ok(())
    }
}

type StoreKey = (Vec<String>, String);
type StoreEntry = (Value, u64, u64);

/// In-memory long-term memory store (LangGraph `BaseStore` adapter).
#[derive(Debug, Default)]
pub struct InMemoryStore {
    items: Mutex<HashMap<StoreKey, StoreEntry>>,
}

impl InMemoryStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored items.
    pub fn len(&self) -> usize {
        self.items
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Returns `true` when the store holds no item.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl BaseStore for InMemoryStore {
    fn get(&self, namespace: &[String], key: &str) -> Result<Option<Value>> {
        Ok(self
            .items
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&(namespace.to_vec(), key.to_string()))
            .map(|(value, _, _)| value.clone()))
    }

    fn put(&self, namespace: &[String], key: &str, value: Value) -> Result<()> {
        let now = store_now();
        let mut items = self
            .items
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = items
            .entry((namespace.to_vec(), key.to_string()))
            .or_insert_with(|| (value.clone(), now, now));
        entry.0 = value;
        entry.2 = now;
        Ok(())
    }

    fn delete(&self, namespace: &[String], key: &str) -> Result<()> {
        self.items
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&(namespace.to_vec(), key.to_string()));
        Ok(())
    }

    fn search(
        &self,
        namespace_prefix: &[String],
        query: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StoreItem>> {
        let items = self
            .items
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut results: Vec<StoreItem> = items
            .iter()
            .filter(|((namespace, _), _)| namespace_matches(namespace_prefix, namespace))
            .map(
                |((namespace, key), (value, created_at, updated_at))| StoreItem {
                    namespace: namespace.clone(),
                    key: key.clone(),
                    value: value.clone(),
                    created_at_millis: *created_at,
                    updated_at_millis: *updated_at,
                },
            )
            .filter(|item| item_matches(item, query))
            .collect();
        results.sort_by(|left, right| {
            left.namespace
                .cmp(&right.namespace)
                .then_with(|| left.key.cmp(&right.key))
        });
        if let Some(limit) = limit {
            results.truncate(limit);
        }
        Ok(results)
    }

    fn namespaces(&self) -> Result<Vec<Vec<String>>> {
        let items = self
            .items
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut namespaces: Vec<Vec<String>> = items
            .keys()
            .map(|(namespace, _)| namespace.clone())
            .collect();
        namespaces.sort();
        namespaces.dedup();
        Ok(namespaces)
    }
}
