#![cfg(feature = "sqlite")]

//! SQLite node cache.

use langgraph_rs::{CachedOutput, IndexMap, NodeCache, SqliteCache};

fn output() -> CachedOutput {
    CachedOutput {
        updates: IndexMap::new(),
        replaces: IndexMap::new(),
        goto: Vec::new(),
        sends: Vec::new(),
        usage: None,
    }
}

#[test]
fn sqlite_cache_roundtrip_and_clear() {
    let cache = SqliteCache::in_memory().unwrap();
    assert!(cache.get("a").unwrap().is_none());
    cache.put("a", output(), None).unwrap();
    assert!(cache.get("a").unwrap().is_some());
    cache.clear().unwrap();
    assert!(cache.get("a").unwrap().is_none());
}

#[test]
fn sqlite_cache_evicts_least_recently_used() {
    let cache = SqliteCache::in_memory().unwrap().with_max_entries(2);
    cache.put("a", output(), None).unwrap();
    cache.put("b", output(), None).unwrap();
    cache.get("a").unwrap();
    cache.put("c", output(), None).unwrap();
    assert!(cache.get("b").unwrap().is_none());
    assert!(cache.get("a").unwrap().is_some());
    assert!(cache.get("c").unwrap().is_some());
}
