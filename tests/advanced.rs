//! Graph construction and execution extras: sequences, multi-source edges, batch,
//! durability and concurrency.

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, Durability, InMemorySaver, NodeOutput, Route, RunConfig, Send, State,
    StateGraph, StateSchema, json, reducers,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct FlowState {
    value: i64,
    log: Vec<String>,
}

impl State for FlowState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("value")
            .with_reducer("log", reducers::append())
    }
}

#[test]
fn add_sequence_chains_nodes() {
    let graph = StateGraph::<FlowState>::new()
        .add_sequence([
            (
                "a",
                std::sync::Arc::new(|state: &FlowState| {
                    Ok(NodeOutput::new()
                        .set("value", state.value + 1)
                        .set("log", vec!["a".to_string()]))
                }) as langgraph_rs::NodeFn<FlowState>,
            ),
            (
                "b",
                std::sync::Arc::new(|state: &FlowState| {
                    Ok(NodeOutput::new()
                        .set("value", state.value + 1)
                        .set("log", vec!["b".to_string()]))
                }),
            ),
            (
                "c",
                std::sync::Arc::new(|state: &FlowState| {
                    Ok(NodeOutput::new()
                        .set("value", state.value + 1)
                        .set("log", vec!["c".to_string()]))
                }),
            ),
        ])
        .compile()
        .unwrap();

    let output = graph
        .invoke(&FlowState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.value, 3);
    assert_eq!(output.log, vec!["a", "b", "c"]);
}

#[test]
fn add_edges_from_fans_in_to_one_target() {
    let graph = StateGraph::<FlowState>::new()
        .add_node("a", |_state: &FlowState| {
            Ok(NodeOutput::new().set("log", vec!["a".to_string()]))
        })
        .add_node("b", |_state: &FlowState| {
            Ok(NodeOutput::new().set("log", vec!["b".to_string()]))
        })
        .add_node("join", |state: &FlowState| {
            Ok(NodeOutput::new().set("value", state.log.len() as i64))
        })
        .set_entry_point("a")
        .add_edge(langgraph_rs::START, "b")
        .add_edges_from(["a", "b"], "join")
        .set_finish_point("join")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&FlowState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.value, 2);
    let mut log = output.log;
    log.sort();
    assert_eq!(log, vec!["a", "b"]);
}

#[test]
fn batch_runs_inputs_and_preserves_order() {
    let graph = StateGraph::<FlowState>::new()
        .add_node("inc", |state: &FlowState| {
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .set_entry_point("inc")
        .set_finish_point("inc")
        .compile()
        .unwrap();

    let inputs = vec![-1, 0, 1, 5].into_iter().map(|value| FlowState {
        value,
        log: Vec::new(),
    });
    let results = graph.batch(inputs, &RunConfig::new().with_max_concurrency(2));
    let values: Vec<i64> = results
        .into_iter()
        .map(|result| result.unwrap().value)
        .collect();
    assert_eq!(values, vec![0, 1, 2, 6]);
}

#[test]
fn batch_reports_per_input_failures() {
    let graph = StateGraph::<FlowState>::new()
        .add_node("work", |state: &FlowState| {
            if state.value < 0 {
                return Err(langgraph_rs::Error::other("negative input"));
            }
            Ok(NodeOutput::new().set("value", state.value * 2))
        })
        .set_entry_point("work")
        .set_finish_point("work")
        .compile()
        .unwrap();

    let inputs = vec![
        FlowState {
            value: 2,
            log: Vec::new(),
        },
        FlowState {
            value: -1,
            log: Vec::new(),
        },
        FlowState {
            value: 4,
            log: Vec::new(),
        },
    ];
    let results = graph.batch(inputs, &RunConfig::new().with_max_concurrency(3));
    assert_eq!(results[0].as_ref().unwrap().value, 4);
    assert!(results[1].is_err());
    assert_eq!(results[2].as_ref().unwrap().value, 8);
}

#[test]
fn durability_exit_writes_fewer_checkpoints_than_sync() {
    let build = |thread: &str, durability: Durability| {
        let saver = Arc::new(InMemorySaver::new());
        let graph = StateGraph::<FlowState>::new()
            .add_sequence([
                (
                    "a",
                    std::sync::Arc::new(|state: &FlowState| {
                        Ok(NodeOutput::new().set("value", state.value + 1))
                    }) as langgraph_rs::NodeFn<FlowState>,
                ),
                (
                    "b",
                    std::sync::Arc::new(|state: &FlowState| {
                        Ok(NodeOutput::new().set("value", state.value + 1))
                    }),
                ),
                (
                    "c",
                    std::sync::Arc::new(|state: &FlowState| {
                        Ok(NodeOutput::new().set("value", state.value + 1))
                    }),
                ),
            ])
            .compile_with(CompileOptions::new().with_checkpointer(saver))
            .unwrap();
        graph
            .run(
                &FlowState::default(),
                &RunConfig::new()
                    .with_thread(thread)
                    .with_durability(durability),
            )
            .unwrap();
        graph.get_state_history(thread).unwrap().len()
    };

    let sync = build("advanced-sync", Durability::Sync);
    let exit = build("advanced-exit", Durability::Exit);
    assert!(sync > exit, "sync={sync} exit={exit}");
    assert!(exit >= 1);

    // The final state is still persisted with the exit policy.
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<FlowState>::new()
        .add_sequence([(
            "only",
            std::sync::Arc::new(|state: &FlowState| {
                Ok(NodeOutput::new().set("value", state.value + 7))
            }) as langgraph_rs::NodeFn<FlowState>,
        )])
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();
    graph
        .run(
            &FlowState::default(),
            &RunConfig::new()
                .with_thread("advanced-final")
                .with_durability(Durability::Exit),
        )
        .unwrap();
    assert_eq!(
        graph
            .get_state("advanced-final")
            .unwrap()
            .unwrap()
            .values
            .value,
        7
    );
}

#[test]
fn durability_names_parse() {
    assert_eq!(Durability::parse("sync"), Some(Durability::Sync));
    assert_eq!(Durability::parse("async"), Some(Durability::Async));
    assert_eq!(Durability::parse("exit"), Some(Durability::Exit));
    assert_eq!(Durability::parse("nope"), None);
}

#[test]
fn conditional_router_can_dispatch_send_tasks() {
    let graph = StateGraph::<FlowState>::new()
        .add_node("start", |_state: &FlowState| Ok(NodeOutput::new()))
        .add_node("worker", |state: &FlowState| {
            Ok(NodeOutput::new().set("log", vec![state.value.to_string()]))
        })
        .add_conditional_edges("start", |_state: &FlowState| {
            Ok(Route::sends([
                Send::with_arg("worker", json!({ "value": 1 })),
                Send::with_arg("worker", json!({ "value": 2 })),
            ]))
        })
        .set_entry_point("start")
        .set_finish_point("worker")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&FlowState::default(), &RunConfig::new())
        .unwrap();
    let mut log = output.log;
    log.sort();
    assert_eq!(log, vec!["1", "2"]);
}

#[test]
fn router_can_mix_static_targets_and_sends() {
    let graph = StateGraph::<FlowState>::new()
        .add_node("a", |_state: &FlowState| {
            Ok(NodeOutput::new().set("log", vec!["a".to_string()]))
        })
        .add_node("worker", |state: &FlowState| {
            Ok(NodeOutput::new().set("log", vec![state.value.to_string()]))
        })
        .add_conditional_edges("start", |_state: &FlowState| {
            Ok(Route::nodes(["a"]).with_sends([Send::with_arg("worker", json!({ "value": 7 }))]))
        })
        .add_node("start", |_state: &FlowState| Ok(NodeOutput::new()))
        .set_entry_point("start")
        .set_finish_point("a")
        .set_finish_point("worker")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&FlowState::default(), &RunConfig::new())
        .unwrap();
    let mut log = output.log;
    log.sort();
    assert_eq!(log, vec!["7", "a"]);
}

#[test]
fn conditional_entry_point_can_dispatch_sends() {
    let graph = StateGraph::<FlowState>::new()
        .add_node("worker", |state: &FlowState| {
            Ok(NodeOutput::new().set("log", vec![state.value.to_string()]))
        })
        .set_conditional_entry_point(|_state: &FlowState| {
            Ok(Route::sends([Send::with_arg(
                "worker",
                json!({ "value": 3 }),
            )]))
        })
        .set_finish_point("worker")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&FlowState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.log, vec!["3"]);
}

#[test]
fn node_metadata_is_stored_on_the_compiled_graph() {
    let mut metadata = serde_json::Map::new();
    metadata.insert("owner".to_string(), json!("team-graph"));
    metadata.insert("tier".to_string(), json!(1));
    let graph = StateGraph::<FlowState>::new()
        .add_node_with_metadata("work", metadata, |_state: &FlowState| Ok(NodeOutput::new()))
        .set_entry_point("work")
        .set_finish_point("work")
        .compile()
        .unwrap();

    let stored = graph.node_metadata_for("work").unwrap();
    assert_eq!(stored.get("owner"), Some(&json!("team-graph")));
    assert_eq!(graph.node_metadata().len(), 1);
}
