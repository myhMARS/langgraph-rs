//! Live Redis/Postgres adapter tests, enabled through environment variables.

#[cfg(feature = "redis")]
#[test]
fn redis_store_and_cache_roundtrip() {
    use langgraph_rs::{
        BaseStore, CachedOutput, IndexMap, NodeCache, RedisCache, RedisStore, json,
    };

    let Ok(url) = std::env::var("REDIS_URL") else {
        return;
    };
    let store = RedisStore::connect(&url).unwrap();
    let namespace = vec![
        "lgrs".to_string(),
        format!("test-{}", std::process::id()),
        "users".to_string(),
    ];
    store.get(&namespace, "name").unwrap();
    store
        .put(&namespace, "name", json!({"first": "Ada"}))
        .unwrap();
    store
        .put(&namespace, "email", json!("ada@example.com"))
        .unwrap();
    assert_eq!(
        store.get(&namespace, "name").unwrap(),
        Some(json!({"first": "Ada"}))
    );
    let items = store.search(&namespace[..2], Some("ada"), None).unwrap();
    assert!(items.iter().any(|item| item.key == "name"));
    assert!(
        store
            .namespaces()
            .unwrap()
            .iter()
            .any(|candidate| candidate == &namespace)
    );
    store.delete(&namespace, "name").unwrap();
    assert!(store.get(&namespace, "name").unwrap().is_none());

    let cache = RedisCache::connect(&url).unwrap();
    let output = CachedOutput {
        updates: IndexMap::new(),
        replaces: IndexMap::new(),
        goto: Vec::new(),
        sends: Vec::new(),
        usage: None,
    };
    cache.put("lgrs-test-cache", output.clone(), None).unwrap();
    assert!(cache.get("lgrs-test-cache").unwrap().is_some());
    cache.clear().unwrap();
    assert!(cache.get("lgrs-test-cache").unwrap().is_none());
}

#[cfg(feature = "postgres")]
#[test]
fn postgres_store_and_cache_roundtrip() {
    use langgraph_rs::{
        BaseStore, CachedOutput, IndexMap, NodeCache, PostgresCache, PostgresStore, json,
    };

    let Ok(url) = std::env::var("POSTGRES_URL") else {
        return;
    };
    let store = PostgresStore::connect(&url).unwrap();
    let namespace = vec![
        "lgrs".to_string(),
        format!("test-{}", std::process::id()),
        "users".to_string(),
    ];
    store.get(&namespace, "name").unwrap();
    store
        .put(&namespace, "name", json!({"first": "Ada"}))
        .unwrap();
    store
        .put(&namespace, "email", json!("ada@example.com"))
        .unwrap();
    assert_eq!(
        store.get(&namespace, "name").unwrap(),
        Some(json!({"first": "Ada"}))
    );
    let items = store.search(&namespace[..2], Some("ada"), None).unwrap();
    assert!(items.iter().any(|item| item.key == "name"));
    assert!(
        store
            .namespaces()
            .unwrap()
            .iter()
            .any(|candidate| candidate == &namespace)
    );
    store.delete(&namespace, "name").unwrap();
    assert!(store.get(&namespace, "name").unwrap().is_none());

    let cache = PostgresCache::connect(&url).unwrap();
    let output = CachedOutput {
        updates: IndexMap::new(),
        replaces: IndexMap::new(),
        goto: Vec::new(),
        sends: Vec::new(),
        usage: None,
    };
    cache.put("lgrs-test-cache", output.clone(), None).unwrap();
    assert!(cache.get("lgrs-test-cache").unwrap().is_some());
    cache.clear().unwrap();
    assert!(cache.get("lgrs-test-cache").unwrap().is_none());
}
