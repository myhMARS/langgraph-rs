#![cfg(feature = "async")]

//! Async parity APIs: update_state, history filters, checkpoint export/import.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{
    Durability, InMemorySaver, JsonState, NodeOutput, RunConfig, StateSchema, json,
};

fn graph(saver: Arc<InMemorySaver>) -> langgraph_rs::tokio_runtime::AsyncCompiledGraph<JsonState> {
    AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("inc", |state: JsonState, _ctx| async move {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or(0);
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("inc")
        .set_finish_point("inc")
        .with_checkpointer(saver)
        .compile()
        .unwrap()
}

#[tokio::test]
async fn update_state_and_filtered_history() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = graph(saver);
    graph
        .run(
            &JsonState::from(json!({"value": 0}).as_object().unwrap().clone()),
            &RunConfig::new().with_thread("async-update"),
        )
        .await
        .unwrap();

    let snapshot = graph
        .update_state("async-update", [("value", json!(42))], None)
        .await
        .unwrap();
    assert_eq!(snapshot.values.get("value"), Some(&json!(42)));

    let history = graph
        .get_state_history_filtered("async-update", None, Some(1))
        .unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].values.get("value"), Some(&json!(42)));

    let latest_id = snapshot.config["configurable"]["checkpoint_id"]
        .as_str()
        .unwrap()
        .to_string();
    let after = graph
        .get_state_history_filtered("async-update", Some(&latest_id), Some(10))
        .unwrap();
    assert!(
        after
            .iter()
            .all(|snapshot| snapshot.config["configurable"]["checkpoint_id"] != json!(latest_id))
    );
}

#[tokio::test]
async fn checkpoint_export_and_import_roundtrip() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = graph(saver);
    graph
        .run(
            &JsonState::from(json!({"value": 0}).as_object().unwrap().clone()),
            &RunConfig::new().with_thread("async-export"),
        )
        .await
        .unwrap();
    let document = graph
        .export_checkpoint("async-export")
        .unwrap()
        .expect("checkpoint");
    assert_eq!(document["checkpoint"]["channel_values"]["value"], json!(1));
    let version = document["checkpoint"]["channel_versions"]["value"]
        .as_str()
        .expect("string version");
    assert_eq!(version.split('.').next().unwrap().len(), 32);
    assert!(version.ends_with(".0.1"));

    graph
        .import_checkpoint("async-imported", &document)
        .unwrap();
    let snapshot = graph
        .get_state("async-imported")
        .unwrap()
        .expect("imported checkpoint");
    assert_eq!(snapshot.values.get("value"), Some(&json!(1)));
}

#[tokio::test]
async fn durability_exit_writes_the_final_checkpoint() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = graph(saver);
    let output = graph
        .run(
            &JsonState::from(json!({"value": 0}).as_object().unwrap().clone()),
            &RunConfig::new()
                .with_thread("async-exit")
                .with_durability(Durability::Exit),
        )
        .await
        .unwrap();
    assert_eq!(output.state.get("value"), Some(&json!(1)));
    let latest = graph
        .get_state("async-exit")
        .unwrap()
        .expect("final checkpoint");
    assert_eq!(latest.values.get("value"), Some(&json!(1)));
}

#[tokio::test]
async fn builder_conveniences_and_name() {
    let graph = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .with_name("async-demo")
        .add_node("a", |_state: JsonState, _ctx| async move {
            Ok(NodeOutput::new().update("value", json!(1)))
        })
        .add_node("b", |_state: JsonState, _ctx| async move {
            Ok(NodeOutput::new().update("value", json!(2)))
        })
        .set_entry_point("a")
        .add_edges_from(["a"], "b")
        .set_finish_point("b")
        .compile()
        .unwrap();
    assert_eq!(graph.name(), Some("async-demo"));
    let output = graph.invoke(&JsonState::new()).await.unwrap();
    assert_eq!(output.get("value"), Some(&json!(2)));
}

#[tokio::test]
async fn subgraph_states_are_listed() {
    let saver = Arc::new(InMemorySaver::new());
    let child = Arc::new(
        AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
            .add_node("inc", |state: JsonState, _ctx| async move {
                let value = state
                    .get("value")
                    .and_then(|value| value.as_i64())
                    .unwrap_or_default();
                Ok(NodeOutput::new().update("value", json!(value + 1)))
            })
            .set_entry_point("inc")
            .set_finish_point("inc")
            .with_checkpointer(saver.clone())
            .compile()
            .unwrap(),
    );
    let parent = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_subgraph("child", child)
        .set_entry_point("child")
        .set_finish_point("child")
        .with_checkpointer(saver)
        .compile()
        .unwrap();
    parent
        .run(
            &JsonState::new(),
            &RunConfig::new().with_thread("async-subgraph-states"),
        )
        .await
        .unwrap();
    let (root, subgraphs) = parent
        .get_state_subgraphs("async-subgraph-states")
        .unwrap()
        .expect("root state");
    assert_eq!(root.values.get("value"), Some(&json!(1)));
    assert_eq!(subgraphs.len(), 1);
}

struct SlowSaver {
    inner: InMemorySaver,
    puts: AtomicUsize,
}

impl SlowSaver {
    fn new() -> Self {
        Self {
            inner: InMemorySaver::new(),
            puts: AtomicUsize::new(0),
        }
    }
}

impl langgraph_rs::Checkpointer for SlowSaver {
    fn put(&self, checkpoint: langgraph_rs::Checkpoint) -> langgraph_rs::Result<()> {
        self.puts.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(30));
        self.inner.put(checkpoint)
    }

    fn get(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> langgraph_rs::Result<Option<langgraph_rs::Checkpoint>> {
        self.inner.get(thread_id, checkpoint_id)
    }

    fn latest(&self, thread_id: &str) -> langgraph_rs::Result<Option<langgraph_rs::Checkpoint>> {
        self.inner.latest(thread_id)
    }

    fn latest_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> langgraph_rs::Result<Option<langgraph_rs::Checkpoint>> {
        self.inner.latest_in(thread_id, namespace)
    }

    fn history(&self, thread_id: &str) -> langgraph_rs::Result<Vec<langgraph_rs::Checkpoint>> {
        self.inner.history(thread_id)
    }

    fn history_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> langgraph_rs::Result<Vec<langgraph_rs::Checkpoint>> {
        self.inner.history_in(thread_id, namespace)
    }
}

#[tokio::test]
async fn durability_async_writes_in_the_background() {
    let saver = Arc::new(SlowSaver::new());
    let started = Arc::new(std::sync::Mutex::new(None));
    let started_in_node = started.clone();
    let graph = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("a", move |_state: JsonState, _ctx| {
            let started = started_in_node.clone();
            async move {
                if let Ok(mut slot) = started.lock() {
                    *slot = Some(Instant::now());
                }
                Ok(NodeOutput::new().update("value", json!(1)))
            }
        })
        .add_node("b", |state: JsonState, _ctx| async move {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .with_checkpointer(saver.clone())
        .compile()
        .unwrap();
    let before_run = Instant::now();
    let output = graph
        .run(
            &JsonState::new(),
            &RunConfig::new()
                .with_thread("async-durability-bg")
                .with_durability(Durability::Async),
        )
        .await
        .unwrap();
    assert_eq!(output.state.get("value"), Some(&json!(2)));
    let first_node_start = started.lock().unwrap().expect("node started");
    assert!(
        first_node_start.duration_since(before_run) < Duration::from_millis(20),
        "checkpoint writes blocked the first node"
    );
    assert!(saver.puts.load(Ordering::SeqCst) >= 2);
    let history = graph.get_state_history("async-durability-bg").unwrap();
    assert!(!history.is_empty());
    assert_eq!(history[0].values.get("value"), Some(&json!(2)));
}
