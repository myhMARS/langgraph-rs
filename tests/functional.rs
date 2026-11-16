//! Functional API approximation: `entrypoint` + memoized `task`s.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use langgraph_rs::functional::{entrypoint, task};
use langgraph_rs::{
    BaseStore, CompileOptions, InMemorySaver, InMemoryStore, RunConfig, State, StateSchema,
    interrupt, json,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct Demo {
    value: i64,
}

impl State for Demo {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

#[test]
fn entrypoint_tasks_are_memoized_in_the_store() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let add = task("add-one", move |input: i64| {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(input + 1)
    });
    let store = Arc::new(InMemoryStore::new());

    let graph = entrypoint(move |state: Demo| {
        let first = add.call(state.value)?;
        let second = add.call(first)?;
        Ok(Demo { value: second })
    })
    .with_store(store.clone())
    .compile()
    .unwrap();

    let first = graph.invoke(&Demo { value: 1 }, &RunConfig::new()).unwrap();
    assert_eq!(first.value, 3);
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    let second = graph.invoke(&Demo { value: 1 }, &RunConfig::new()).unwrap();
    assert_eq!(second, first);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "cached tasks must not run again"
    );

    let namespaces = store.namespaces().unwrap();
    assert!(
        namespaces
            .iter()
            .any(|namespace| namespace == &vec!["functional".to_string(), "add-one".to_string()])
    );
}

#[test]
fn completed_tasks_are_not_rerun_after_interrupt_resume() {
    let first_calls = Arc::new(AtomicUsize::new(0));
    let second_calls = Arc::new(AtomicUsize::new(0));
    let first_counter = first_calls.clone();
    let second_counter = second_calls.clone();
    let first_task = task("first", move |input: i64| {
        first_counter.fetch_add(1, Ordering::SeqCst);
        Ok(input * 10)
    });
    let second_task = task("second", move |input: i64| {
        second_counter.fetch_add(1, Ordering::SeqCst);
        Ok(input + 1)
    });
    let store = Arc::new(InMemoryStore::new());
    let saver = Arc::new(InMemorySaver::new());

    let graph = entrypoint(move |state: Demo| {
        let first = first_task.call(state.value)?;
        let answer = interrupt(json!({ "question": "delta" }))?;
        let delta = answer.as_i64().unwrap_or_default();
        let second = second_task.call(delta)?;
        Ok(Demo {
            value: first + second,
        })
    })
    .with_store(store)
    .compile_with(CompileOptions::new().with_checkpointer(saver))
    .unwrap();

    let error = graph
        .run(
            &Demo { value: 1 },
            &RunConfig::new().with_thread("functional-hitl"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());
    assert_eq!(first_calls.load(Ordering::SeqCst), 1);
    assert_eq!(second_calls.load(Ordering::SeqCst), 0);

    let output = graph.resume("functional-hitl", json!(4)).unwrap();
    assert_eq!(output.value, 15);
    assert_eq!(
        first_calls.load(Ordering::SeqCst),
        1,
        "the completed task must be served from the store"
    );
    assert_eq!(second_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn tasks_run_every_time_without_a_store() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let add = task("add", move |input: i64| {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(input + 1)
    });
    let graph = entrypoint(move |state: Demo| {
        let next = add.call(state.value)?;
        Ok(Demo { value: next })
    })
    .compile()
    .unwrap();

    graph.invoke(&Demo { value: 0 }, &RunConfig::new()).unwrap();
    graph.invoke(&Demo { value: 0 }, &RunConfig::new()).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn completed_tasks_persist_in_checkpoints_without_a_store() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let first = task("first", move |input: i64| {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(input * 10)
    });
    let saver = Arc::new(InMemorySaver::new());

    let graph = entrypoint(move |state: Demo| {
        let first = first.call(state.value)?;
        let delta = interrupt(json!({ "question": "delta" }))?;
        Ok(Demo {
            value: first + delta.as_i64().unwrap_or_default(),
        })
    })
    .compile_with(CompileOptions::new().with_checkpointer(saver))
    .unwrap();

    let _ = graph
        .run(
            &Demo { value: 1 },
            &RunConfig::new().with_thread("functional-checkpoint"),
        )
        .unwrap_err();
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let output = graph.resume("functional-checkpoint", json!(4)).unwrap();
    assert_eq!(output.value, 14);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
