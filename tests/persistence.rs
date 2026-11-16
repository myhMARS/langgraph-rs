//! Persistent checkpointer adapter tests.

#![cfg(any(feature = "sqlite", feature = "redis", feature = "postgres"))]

//!
//! SQLite runs everywhere (bundled SQLite). Redis and Postgres tests are opt-in through
//! `LANGGRAPH_RS_REDIS_URL` / `LANGGRAPH_RS_POSTGRES_URL`; without those variables the
//! tests print a skip notice and pass.

use std::sync::Arc;

use langgraph_rs::{CompileOptions, NodeOutput, RunConfig, State, StateGraph, StateSchema};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct PersistState {
    value: i64,
}

impl State for PersistState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

fn graph_with(
    checkpointer: Arc<dyn langgraph_rs::Checkpointer>,
) -> langgraph_rs::CompiledGraph<PersistState> {
    StateGraph::<PersistState>::new()
        .add_node("increment", |state: &PersistState| {
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .set_entry_point("increment")
        .set_finish_point("increment")
        .compile_with(CompileOptions::new().with_checkpointer(checkpointer))
        .unwrap()
}

#[cfg(feature = "sqlite")]
mod sqlite_tests {
    use super::*;
    use langgraph_rs::{Checkpointer, SqliteSaver};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::current_dir().unwrap_or_else(|_| std::env::temp_dir());
        path.push("target");
        path.push(format!(
            "langgraph-rs-{name}-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    #[test]
    fn sqlite_in_memory_smoke() {
        SqliteSaver::in_memory().unwrap();
    }

    #[test]
    fn sqlite_round_trip_latest_history_and_threads() {
        let path = temp_path("roundtrip");
        let saver = Arc::new(SqliteSaver::open(&path).unwrap());
        let graph = graph_with(saver.clone());
        let config = RunConfig::new().with_thread("sqlite-thread");
        let outcome = graph.run(&PersistState::default(), &config).unwrap();
        assert_eq!(outcome.state.value, 1);

        let latest = saver.latest("sqlite-thread").unwrap().unwrap();
        assert_eq!(latest.thread_id, "sqlite-thread");
        assert!(!latest.interrupted);
        assert_eq!(latest.values.get("value"), Some(&serde_json::json!(1)));

        let by_id = saver
            .get("sqlite-thread", &latest.id)
            .unwrap()
            .expect("checkpoint by id");
        assert_eq!(by_id, latest);

        let history = saver.history("sqlite-thread").unwrap();
        assert!(history.len() >= 2);
        assert_eq!(history[0].id, latest.id);
        assert_eq!(saver.threads().unwrap(), vec!["sqlite-thread".to_string()]);

        // State queries through the compiled graph use the same adapter.
        let state = graph.get_state("sqlite-thread").unwrap().unwrap();
        assert_eq!(state.values.value, 1);
    }

    #[test]
    fn sqlite_namespaces_and_thread_management() {
        let path = temp_path("namespaces");
        let saver = Arc::new(SqliteSaver::open(&path).unwrap());
        let graph = graph_with(saver.clone());
        let thread = "sqlite-ns";
        graph
            .run(
                &PersistState::default(),
                &RunConfig::new()
                    .with_thread(thread)
                    .with_checkpoint_ns("sub/one"),
            )
            .unwrap();
        graph
            .run(
                &PersistState { value: 10 },
                &RunConfig::new()
                    .with_thread(thread)
                    .with_checkpoint_ns("sub/two"),
            )
            .unwrap();

        let one = saver.latest_in(thread, "sub/one").unwrap().unwrap();
        let two = saver.latest_in(thread, "sub/two").unwrap().unwrap();
        assert_eq!(one.values.get("value"), Some(&serde_json::json!(1)));
        assert_eq!(two.values.get("value"), Some(&serde_json::json!(11)));
        assert!(saver.latest_in(thread, "sub/missing").unwrap().is_none());
        assert_eq!(saver.history_in(thread, "sub/one").unwrap().len(), 3);

        saver.copy_thread(thread, "sqlite-copy").unwrap();
        let copied = saver.latest_in("sqlite-copy", "sub/two").unwrap().unwrap();
        assert_eq!(copied.values.get("value"), Some(&serde_json::json!(11)));

        saver.delete_thread(thread).unwrap();
        assert!(saver.latest_in(thread, "sub/one").unwrap().is_none());
        assert!(saver.latest_in("sqlite-copy", "sub/two").unwrap().is_some());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn sqlite_resume_after_reopen() {
        let path = temp_path("resume");
        {
            let saver = Arc::new(SqliteSaver::open(&path).unwrap());
            let graph = graph_with(saver);
            graph
                .run(
                    &PersistState::default(),
                    &RunConfig::new().with_thread("sqlite-resume"),
                )
                .unwrap();
        }
        // A fresh process/connection reads the persisted checkpoint and continues.
        let saver = Arc::new(SqliteSaver::open(&path).unwrap());
        let graph = graph_with(saver);
        let continued = graph.continue_thread("sqlite-resume").unwrap();
        assert_eq!(continued.value, 1);
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(feature = "redis")]
mod redis_tests {
    use super::*;
    use langgraph_rs::{Checkpointer, RedisSaver};

    #[test]
    fn redis_round_trip() {
        let Ok(url) = std::env::var("LANGGRAPH_RS_REDIS_URL") else {
            eprintln!("skipping redis test: set LANGGRAPH_RS_REDIS_URL to run it");
            return;
        };
        let saver = Arc::new(RedisSaver::connect(&url).unwrap());
        let graph = graph_with(saver.clone());
        let thread = format!("redis-{}", std::process::id());
        graph
            .run(
                &PersistState::default(),
                &RunConfig::new().with_thread(&thread),
            )
            .unwrap();
        let latest = saver.latest(&thread).unwrap().unwrap();
        assert_eq!(latest.values.get("value"), Some(&serde_json::json!(1)));
        assert!(!saver.history(&thread).unwrap().is_empty());
    }
}

#[cfg(feature = "postgres")]
mod postgres_tests {
    use super::*;
    use langgraph_rs::{Checkpointer, PostgresSaver};

    #[test]
    fn postgres_round_trip() {
        let Ok(url) = std::env::var("LANGGRAPH_RS_POSTGRES_URL") else {
            eprintln!("skipping postgres test: set LANGGRAPH_RS_POSTGRES_URL to run it");
            return;
        };
        let saver = Arc::new(PostgresSaver::connect(&url).unwrap());
        let graph = graph_with(saver.clone());
        let thread = format!("postgres-{}", std::process::id());
        graph
            .run(
                &PersistState::default(),
                &RunConfig::new().with_thread(&thread),
            )
            .unwrap();
        let latest = saver.latest(&thread).unwrap().unwrap();
        assert_eq!(latest.values.get("value"), Some(&serde_json::json!(1)));
        assert!(!saver.history(&thread).unwrap().is_empty());
    }
}
