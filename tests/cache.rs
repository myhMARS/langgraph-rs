//! Node-level cache tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use langgraph_rs::{
    CachePolicy, CompiledGraph, InMemoryCache, NodeCache, NodeOutput, RunConfig, State, StateGraph,
    StateSchema,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
struct CacheState {
    value: i64,
    calls: i64,
}

impl State for CacheState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value").last_value("calls")
    }
}

fn counting_graph(
    cache: Arc<InMemoryCache>,
    calls: Arc<AtomicUsize>,
    policy: CachePolicy,
) -> CompiledGraph<CacheState> {
    StateGraph::<CacheState>::new()
        .add_node("work", move |state: &CacheState| {
            let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(NodeOutput::new()
                .set("value", state.value + call as i64)
                .set("calls", call as i64))
        })
        .set_entry_point("work")
        .set_finish_point("work")
        .with_cache(cache)
        .set_cache("work", policy)
        .compile()
        .unwrap()
}

#[test]
fn identical_inputs_reuse_the_cached_output() {
    let cache = Arc::new(InMemoryCache::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let graph = counting_graph(cache, calls.clone(), CachePolicy::new());

    let first = graph
        .run(&CacheState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(first.state.calls, 1);
    assert!(!first.report.executions()[0].cached);

    let second = graph
        .run(&CacheState::default(), &RunConfig::new())
        .unwrap();
    // The node did not run again: the cached writes were replayed.
    assert_eq!(second.state.calls, 1);
    assert_eq!(second.state.value, 1);
    assert!(second.report.executions()[0].cached);
    assert_eq!(second.report.executions()[0].attempts, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn different_inputs_miss_the_cache() {
    let cache = Arc::new(InMemoryCache::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let graph = counting_graph(cache, calls.clone(), CachePolicy::new());

    let first = graph
        .run(&CacheState::default(), &RunConfig::new())
        .unwrap();
    let second = graph
        .run(&CacheState { value: 5, calls: 0 }, &RunConfig::new())
        .unwrap();
    assert_eq!(first.state.value, 1);
    assert_eq!(second.state.value, 7);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn ttl_expiry_reruns_the_node() {
    let cache = Arc::new(InMemoryCache::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let graph = counting_graph(
        cache,
        calls.clone(),
        CachePolicy::new().with_ttl(Duration::from_millis(10)),
    );

    graph
        .run(&CacheState::default(), &RunConfig::new())
        .unwrap();
    std::thread::sleep(Duration::from_millis(25));
    let second = graph
        .run(&CacheState::default(), &RunConfig::new())
        .unwrap();
    assert!(!second.report.executions()[0].cached);
    assert_eq!(second.state.calls, 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn clear_forces_a_miss() {
    let cache = Arc::new(InMemoryCache::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let graph = counting_graph(cache.clone(), calls.clone(), CachePolicy::new());

    graph
        .run(&CacheState::default(), &RunConfig::new())
        .unwrap();
    assert!(!cache.is_empty());
    cache.clear().unwrap();
    assert!(cache.is_empty());

    let second = graph
        .run(&CacheState::default(), &RunConfig::new())
        .unwrap();
    assert!(!second.report.executions()[0].cached);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn cache_can_expose_node_usage() {
    let cache = Arc::new(InMemoryCache::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let graph = StateGraph::<CacheState>::new()
        .add_node("work", move |_state: &CacheState| {
            let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(NodeOutput::new()
                .set("calls", call as i64)
                .with_usage(langgraph_rs::Usage::new().with_input_tokens(4)))
        })
        .set_entry_point("work")
        .set_finish_point("work")
        .with_cache(cache)
        .set_cache("work", CachePolicy::new())
        .compile()
        .unwrap();

    let first = graph
        .run(&CacheState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(first.report.total_tokens(), 4);
    let second = graph
        .run(&CacheState::default(), &RunConfig::new())
        .unwrap();
    // Cached replays keep the recorded usage without re-running the node.
    assert_eq!(second.report.total_tokens(), 4);
    assert!(second.report.executions()[0].cached);
}

#[test]
fn in_memory_cache_evicts_least_recently_used() {
    use langgraph_rs::CachedOutput;
    use langgraph_rs::InMemoryCache;
    use langgraph_rs::IndexMap;
    use langgraph_rs::NodeCache;

    let cache = InMemoryCache::new().with_max_entries(2);
    let output = || CachedOutput {
        updates: IndexMap::new(),
        replaces: IndexMap::new(),
        goto: Vec::new(),
        sends: Vec::new(),
        usage: None,
    };
    cache.put("a", output(), None).unwrap();
    cache.put("b", output(), None).unwrap();
    cache.get("a").unwrap();
    cache.put("c", output(), None).unwrap();
    assert_eq!(cache.len(), 2);
    assert!(cache.get("a").unwrap().is_some());
    assert!(cache.get("b").unwrap().is_none());
    assert!(cache.get("c").unwrap().is_some());
}
