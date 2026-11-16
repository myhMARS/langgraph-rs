//! Closeout checks: snapshot config, get_config, threads, retry predicates.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use langgraph_rs::{
    CompileOptions, InMemorySaver, JsonState, NodeOutput, RetryPolicy, RunConfig, StateGraph,
    StateSchema, get_config, json,
};

#[test]
fn snapshot_exposes_config_and_threads() {
    let graph = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("cfg", |_state: &JsonState| {
            Ok(NodeOutput::new().update("cfg", get_config()))
        })
        .set_entry_point("cfg")
        .set_finish_point("cfg")
        .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
        .unwrap();

    graph
        .run(&JsonState::new(), &RunConfig::new().with_thread("closeout"))
        .unwrap();
    let snapshot = graph.get_state("closeout").unwrap().unwrap();
    assert_eq!(
        snapshot.values.get("cfg").unwrap()["configurable"]["thread_id"],
        json!("closeout")
    );
    assert_eq!(
        snapshot.config["configurable"]["thread_id"],
        json!("closeout")
    );
    assert_eq!(
        snapshot.config["configurable"]["checkpoint_id"],
        json!(snapshot.checkpoint_id)
    );
    assert!(graph.threads().unwrap().contains(&"closeout".to_string()));
}

fn failing_graph(
    policy: RetryPolicy,
    calls: Arc<AtomicUsize>,
) -> langgraph_rs::CompiledGraph<JsonState> {
    StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node_with_retry("fail", policy, move |_state: &JsonState| {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(langgraph_rs::Error::other("boom"))
        })
        .set_entry_point("fail")
        .compile()
        .unwrap()
}

#[test]
fn retry_on_predicates_stop_unmatched_retries() {
    let calls = Arc::new(AtomicUsize::new(0));
    let graph = failing_graph(
        RetryPolicy::new(3).with_retry_on(["connect"]),
        calls.clone(),
    );
    assert!(graph.run(&JsonState::new(), &RunConfig::new()).is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let calls = Arc::new(AtomicUsize::new(0));
    let graph = failing_graph(RetryPolicy::new(3).with_retry_on(["boom"]), calls.clone());
    assert!(graph.run(&JsonState::new(), &RunConfig::new()).is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[test]
fn jitter_shortens_but_never_extends_backoff() {
    let base = RetryPolicy::new(3).with_backoff(std::time::Duration::from_millis(100));
    let jittered = base.clone().with_jitter(0.5);
    for attempt in 1..=3 {
        assert!(jittered.backoff_for(attempt) <= base.backoff_for(attempt));
    }
    assert!(jittered.backoff_for(1) > std::time::Duration::ZERO);
}
