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

    fn history_shared(&self, thread_id: &str) -> Result<Vec<std::sync::Arc<Checkpoint>>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner
            .get(thread_id)
            .map(|checkpoints| checkpoints.iter().rev().cloned().collect())
            .unwrap_or_default())
    }

    fn history_in_shared(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> Result<Vec<std::sync::Arc<Checkpoint>>> {
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
                    .cloned()
                    .collect()
            })
            .unwrap_or_default())
    }

    fn history_page_in_shared(
        &self,
        thread_id: &str,
        namespace: &str,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<std::sync::Arc<Checkpoint>>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(checkpoints) = inner.get(thread_id) else {
            return Ok(Vec::new());
        };
        let mut selected: Vec<&std::sync::Arc<Checkpoint>> = checkpoints
            .iter()
            .rev()
            .filter(|checkpoint| checkpoint.namespace == namespace)
            .collect();
        if let Some(before) = before {
            match selected
                .iter()
                .position(|checkpoint| checkpoint.id == before)
            {
                Some(index) => selected.drain(..=index),
                None => return Ok(Vec::new()),
            };
        }
        if let Some(limit) = limit {
            selected.truncate(limit);
        }
        Ok(selected.into_iter().cloned().collect())
    }

    fn latest_shared(&self, thread_id: &str) -> Result<Option<std::sync::Arc<Checkpoint>>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner.get(thread_id).and_then(|checkpoints| checkpoints.last()).cloned())
    }

    fn latest_in_shared(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> Result<Option<std::sync::Arc<Checkpoint>>> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(inner.get(thread_id).and_then(|checkpoints| {
            checkpoints
                .iter()
                .rev()
                .find(|checkpoint| checkpoint.namespace == namespace)
                .cloned()
        }))
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
type StoreEntry = (Value, u64, u64, Option<u64>);

/// In-memory long-term memory store (LangGraph `BaseStore` adapter).
#[derive(Debug, Default)]
pub struct InMemoryStore {
    items: Mutex<HashMap<StoreKey, StoreEntry>>,
    /// `(namespace, key, field path)` -> embedding vector.
    vectors: Mutex<HashMap<(StoreKey, String), Vec<f32>>>,
    index: Option<crate::store::IndexConfig>,
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

    /// Enable vector search with this configuration.
    pub fn with_index(mut self, index: crate::store::IndexConfig) -> Self {
        self.index = Some(index);
        self
    }

    /// Whether the store has a vector index.
    pub fn has_index(&self) -> bool {
        self.index.is_some()
    }

    /// Fields indexed by default (`None` = the whole value).
    pub fn indexed_fields(&self) -> Option<Vec<String>> {
        self.index.as_ref().and_then(|index| index.fields.clone())
    }

    /// Store a value and index the requested fields.
    ///
    /// `index`: `None` = the configured fields (or the whole value), `Some([])` = skip indexing
    /// (LangGraph's `index=False`), `Some(paths)` = index exactly those fields.
    pub fn put_indexed(
        &self,
        namespace: &[String],
        key: &str,
        value: Value,
        index: Option<&[String]>,
        ttl_minutes: Option<f64>,
    ) -> Result<()> {
        let paths: Vec<String> = match (&self.index, index) {
            (None, _) => {
                return self.put_with_ttl(namespace, key, value, ttl_minutes);
            }
            (Some(_), Some(paths)) => paths.to_vec(),
            (Some(config), None) => config.fields.clone().unwrap_or_else(|| vec!["$".into()]),
        };
        let mut texts = Vec::new();
        let mut slots = Vec::new();
        for path in &paths {
            for (position, text) in crate::store::indexed_texts(&value, path).iter().enumerate() {
                let slot = if paths.len() == 1 && position == 0 {
                    path.clone()
                } else {
                    format!("{path}.{position}")
                };
                texts.push(text.clone());
                slots.push(slot);
            }
        }
        if let Some(config) = &self.index
            && !texts.is_empty()
        {
            let vectors = config.embed.embed_documents(texts)?;
            let mut stored = self
                .vectors
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let store_key = (namespace.to_vec(), key.to_string());
            stored.retain(|(owner, _), _| owner != &store_key);
            for (slot, vector) in slots.iter().zip(vectors) {
                stored.insert((store_key.clone(), slot.clone()), vector);
            }
        }
        self.insert_entry(namespace, key, value);
        if let Some(minutes) = ttl_minutes {
            let expires = store_now() + (minutes * 60_000.0) as u64;
            let mut items = self
                .items
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(entry) = items.get_mut(&(namespace.to_vec(), key.to_string())) {
                entry.3 = Some(expires);
            }
        }
        Ok(())
    }

    /// Write the value without touching the vector index.
    fn insert_entry(&self, namespace: &[String], key: &str, value: Value) {
        let now = store_now();
        let mut items = self
            .items
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = items
            .entry((namespace.to_vec(), key.to_string()))
            .or_insert_with(|| (value.clone(), now, now, None));
        entry.0 = value;
        entry.2 = now;
        entry.3 = None;
    }

    /// Embedding vectors of one item (`field path` -> vector).
    fn item_vectors(&self, namespace: &[String], key: &str) -> Vec<Vec<f32>> {
        self.vectors
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|((owner, _), _)| owner.0 == namespace && owner.1 == key)
            .map(|(_, vector)| vector.clone())
            .collect()
    }
}

impl BaseStore for InMemoryStore {
    fn get(&self, namespace: &[String], key: &str) -> Result<Option<Value>> {
        Ok(self
            .items
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&(namespace.to_vec(), key.to_string()))
            .filter(|(_, _, _, expires)| !expires.is_some_and(|expires| store_now() >= expires))
            .map(|(value, _, _, _)| value.clone()))
    }

    fn put(&self, namespace: &[String], key: &str, value: Value) -> Result<()> {
        if self.index.is_some() {
            return self.put_indexed(namespace, key, value, None, None);
        }
        self.insert_entry(namespace, key, value);
        Ok(())
    }

    fn put_with_ttl(
        &self,
        namespace: &[String],
        key: &str,
        value: Value,
        ttl_minutes: Option<f64>,
    ) -> Result<()> {
        self.put(namespace, key, value)?;
        if let Some(minutes) = ttl_minutes {
            let expires = store_now() + (minutes * 60_000.0) as u64;
            let mut items = self
                .items
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(entry) = items.get_mut(&(namespace.to_vec(), key.to_string())) {
                entry.3 = Some(expires);
            }
        }
        Ok(())
    }

    fn delete(&self, namespace: &[String], key: &str) -> Result<()> {
        self.items
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&(namespace.to_vec(), key.to_string()));
        Ok(())
    }

    fn search_with(
        &self,
        namespace_prefix: &[String],
        options: &crate::store::SearchOptions,
    ) -> Result<Vec<crate::store::SearchHit>> {
        let Some(config) = &self.index else {
            // Without a vector index LangGraph ignores `query` entirely (it only ranks items that
            // have embeddings), so the text query becomes a no-op here as well.
            let mut options = options.clone();
            options.query = None;
            return crate::store::default_search_with(self, namespace_prefix, &options);
        };
        let candidates: Vec<StoreItem> = self
            .search(namespace_prefix, None, None)?
            .into_iter()
            .filter(|item| !item.is_expired())
            .filter(|item| match &options.filter {
                Some(filter) => crate::store::item_matches_filter(&item.value, filter),
                None => true,
            })
            .collect();
        let limit = if options.limit == 0 {
            usize::MAX
        } else {
            options.limit
        };
        let Some(query) = options.query.as_deref().filter(|query| !query.is_empty()) else {
            return Ok(candidates
                .into_iter()
                .skip(options.offset)
                .take(limit)
                .map(crate::store::SearchHit::new)
                .collect());
        };
        let embedding = config.embed.embed_query(query)?;
        let mut scored: Vec<(f64, StoreItem)> = Vec::new();
        let mut scoreless: Vec<StoreItem> = Vec::new();
        for item in candidates {
            let vectors = self.item_vectors(&item.namespace, &item.key);
            if vectors.is_empty() {
                scoreless.push(item);
                continue;
            }
            for vector in vectors {
                scored.push((crate::store::cosine_similarity(&embedding, &vector), item.clone()));
            }
        }
        scored.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut seen = std::collections::HashSet::new();
        let mut kept: Vec<crate::store::SearchHit> = Vec::new();
        for (score, item) in scored {
            if !seen.insert((item.namespace.clone(), item.key.clone())) {
                continue;
            }
            let position = seen.len() - 1;
            if position >= options.offset + limit {
                break;
            }
            if position < options.offset {
                continue;
            }
            kept.push(crate::store::SearchHit { item, score: Some(score) });
        }
        if !scoreless.is_empty() && kept.len() < limit {
            let room = limit - kept.len();
            kept.extend(scoreless.into_iter().take(room).map(crate::store::SearchHit::new));
        }
        Ok(kept)
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
                |((namespace, key), (value, created_at, updated_at, expires))| StoreItem {
                    namespace: namespace.clone(),
                    key: key.clone(),
                    value: value.clone(),
                    created_at_millis: *created_at,
                    updated_at_millis: *updated_at,
                    expires_at_millis: *expires,
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
