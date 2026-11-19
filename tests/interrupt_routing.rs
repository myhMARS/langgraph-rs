//! Successful siblings' routes must survive an interrupted superstep.

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph, StateSchema,
    interrupt, json, reducers,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct RouteState {
    log: Vec<String>,
}

impl State for RouteState {
    fn schema() -> StateSchema {
        StateSchema::new().with_reducer("log", reducers::append())
    }
}

#[test]
fn successful_sibling_routes_are_scheduled_after_a_single_interrupt() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<RouteState>::new()
        .add_node("ok", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["ok"])))
        })
        .add_node("ask", |_state: &RouteState| {
            interrupt(json!({ "q": "ask" }))?;
            Ok(NodeOutput::new().update("log", json!(["ask"])))
        })
        .add_node("down", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["down"])))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask")
        .add_edge("ok", "down")
        .set_finish_point("down")
        .set_finish_point("ask")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &RouteState::default(),
            &RunConfig::new().with_thread("interrupt-route"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("interrupt-route").unwrap().unwrap();
    let resume = json!({ snapshot.interrupts[0].id.clone(): "resumed" });
    let output = graph.resume("interrupt-route", resume).unwrap();
    assert_eq!(output.log, vec!["ok", "ask", "down"]);
}

#[test]
fn successful_sibling_routes_are_scheduled_after_multiple_interrupts() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<RouteState>::new()
        .add_node("ok", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["ok"])))
        })
        .add_node("ask_a", |_state: &RouteState| {
            interrupt(json!({ "q": "a" }))?;
            Ok(NodeOutput::new().update("log", json!(["a"])))
        })
        .add_node("ask_b", |_state: &RouteState| {
            interrupt(json!({ "q": "b" }))?;
            Ok(NodeOutput::new().update("log", json!(["b"])))
        })
        .add_node("down", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["down"])))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask_a")
        .add_edge(langgraph_rs::START, "ask_b")
        .add_edge("ok", "down")
        .set_finish_point("down")
        .set_finish_point("ask_a")
        .set_finish_point("ask_b")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &RouteState::default(),
            &RunConfig::new().with_thread("interrupt-route-multi"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("interrupt-route-multi").unwrap().unwrap();
    let mut resume = serde_json::Map::new();
    for interrupt in &snapshot.interrupts {
        resume.insert(interrupt.id.clone(), json!("ok"));
    }
    let output = graph
        .resume("interrupt-route-multi", serde_json::Value::Object(resume))
        .unwrap();
    assert_eq!(output.log, vec!["ok", "a", "b", "down"]);
}

#[test]
fn routed_sibling_interrupt_does_not_consume_the_previous_answer() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<RouteState>::new()
        .add_node("ok", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["ok"])))
        })
        .add_node("ask1", |_state: &RouteState| {
            let answer = interrupt(json!({ "q": 1 }))?;
            Ok(NodeOutput::new().update(
                "log",
                json!([format!("ask1:{}", answer.as_str().unwrap_or("?"))]),
            ))
        })
        .add_node("ask2", |_state: &RouteState| {
            let answer = interrupt(json!({ "q": 2 }))?;
            Ok(NodeOutput::new().update(
                "log",
                json!([format!("ask2:{}", answer.as_str().unwrap_or("?"))]),
            ))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask1")
        .add_edge("ok", "ask2")
        .set_finish_point("ask1")
        .set_finish_point("ask2")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &RouteState::default(),
            &RunConfig::new().with_thread("routed-interrupt"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    // ask1 consumes "first"; ask2 must raise its own interrupt, not reuse that answer.
    let error = graph
        .resume("routed-interrupt", json!("first"))
        .unwrap_err();
    assert!(error.is_interrupt());
    let snapshot = graph.get_state("routed-interrupt").unwrap().unwrap();
    assert_eq!(snapshot.interrupts[0].node, "ask2");

    let output = graph.resume("routed-interrupt", json!("second")).unwrap();
    assert!(
        output.log.contains(&"ask1:first".to_string()),
        "log={:?}",
        output.log
    );
    assert!(output.log.contains(&"ask2:second".to_string()));
}

#[test]
fn deferred_tasks_survive_an_interrupt() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<RouteState>::new()
        .add_node("ok", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["ok"])))
        })
        .add_node("ask", |_state: &RouteState| {
            interrupt(json!({ "q": "ask" }))?;
            Ok(NodeOutput::new().update("log", json!(["ask"])))
        })
        .add_node("regular", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["regular"])))
        })
        .add_node_deferred("late", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["late"])))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask")
        .add_edge("ok", "regular")
        .add_edge("ok", "late")
        .set_finish_point("regular")
        .set_finish_point("ask")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &RouteState::default(),
            &RunConfig::new().with_thread("deferred-interrupt"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("deferred-interrupt").unwrap().unwrap();
    let resume = json!({ snapshot.interrupts[0].id.clone(): "answer" });
    let output = graph.resume("deferred-interrupt", resume).unwrap();
    assert_eq!(output.log, vec!["ok", "ask", "regular", "late"]);
}

#[test]
fn deferred_only_route_stays_deferred_across_an_interrupt() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<RouteState>::new()
        .add_node("ok", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["ok"])))
        })
        .add_node("ask", |_state: &RouteState| {
            interrupt(json!({ "q": "ask" }))?;
            Ok(NodeOutput::new().update("log", json!(["ask"])))
        })
        .add_node_deferred("late", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["late"])))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask")
        .add_edge("ok", "late")
        .set_finish_point("ask")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &RouteState::default(),
            &RunConfig::new().with_thread("deferred-only"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("deferred-only").unwrap().unwrap();
    let resume = json!({ snapshot.interrupts[0].id.clone(): "answer" });
    let output = graph.resume("deferred-only", resume).unwrap();
    assert_eq!(output.log, vec!["ok", "ask", "late"]);
}

#[test]
fn sibling_without_persisted_write_still_routes_after_resume() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<RouteState>::new()
        .add_node("ok", |_state: &RouteState| Ok(NodeOutput::new()))
        .add_node("ask", |_state: &RouteState| {
            interrupt(json!({ "q": "ask" }))?;
            Ok(NodeOutput::new().update("log", json!(["ask"])))
        })
        .add_node("down", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["down"])))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask")
        .add_edge("ok", "down")
        .set_finish_point("down")
        .set_finish_point("ask")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &RouteState::default(),
            &RunConfig::new().with_thread("interrupt-route-silent"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("interrupt-route-silent").unwrap().unwrap();
    let resume = json!({ snapshot.interrupts[0].id.clone(): "resumed" });
    let output = graph.resume("interrupt-route-silent", resume).unwrap();
    assert_eq!(output.log, vec!["ask", "down"]);
}

#[test]
fn sibling_goto_target_survives_a_pause() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<RouteState>::new()
        .add_node("ok", |_state: &RouteState| {
            Ok(NodeOutput::new().goto("detour"))
        })
        .add_node("ask", |_state: &RouteState| {
            interrupt(json!({ "q": "ask" }))?;
            Ok(NodeOutput::new().update("log", json!(["ask"])))
        })
        .add_node("detour", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["detour"])))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask")
        .set_finish_point("detour")
        .set_finish_point("ask")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &RouteState::default(),
            &RunConfig::new().with_thread("interrupt-route-goto"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("interrupt-route-goto").unwrap().unwrap();
    let resume = json!({ snapshot.interrupts[0].id.clone(): "resumed" });
    let output = graph.resume("interrupt-route-goto", resume).unwrap();
    assert_eq!(output.log, vec!["ask", "detour"]);
}

#[test]
fn sibling_goto_end_does_not_schedule_a_pseudo_task() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<RouteState>::new()
        .add_node("ok", |_state: &RouteState| {
            Ok(NodeOutput::new().goto(langgraph_rs::END))
        })
        .add_node("ask", |_state: &RouteState| {
            interrupt(json!({ "q": "ask" }))?;
            Ok(NodeOutput::new().update("log", json!(["ask"])))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask")
        .set_finish_point("ask")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &RouteState::default(),
            &RunConfig::new().with_thread("interrupt-route-end"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("interrupt-route-end").unwrap().unwrap();
    let resume = json!({ snapshot.interrupts[0].id.clone(): "resumed" });
    let output = graph.resume("interrupt-route-end", resume).unwrap();
    assert_eq!(output.log, vec!["ask"]);
}

#[test]
fn sibling_routes_survive_a_second_interrupt() {
    let saver = Arc::new(InMemorySaver::new());

    let graph = StateGraph::<RouteState>::new()
        .add_node("ok", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["ok"])))
        })
        .add_node("ask", |_state: &RouteState| {
            // Two sequential interrupts: the first resume advances to the second one.
            let first = interrupt(json!({ "q": "first" }))?;
            interrupt(json!({ "q": "second", "first": first }))?;
            Ok(NodeOutput::new().update("log", json!(["ask"])))
        })
        .add_node("down", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["down"])))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask")
        .add_edge("ok", "down")
        .set_finish_point("down")
        .set_finish_point("ask")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &RouteState::default(),
            &RunConfig::new().with_thread("interrupt-route-twice"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("interrupt-route-twice").unwrap().unwrap();
    let resume = json!({ snapshot.interrupts[0].id.clone(): "resumed" });
    let error = graph.resume("interrupt-route-twice", resume).unwrap_err();
    assert!(error.is_interrupt(), "second pause expected");

    let snapshot = graph.get_state("interrupt-route-twice").unwrap().unwrap();
    let resume = json!({ snapshot.interrupts[0].id.clone(): "again" });
    let output = graph.resume("interrupt-route-twice", resume).unwrap();
    assert_eq!(output.log, vec!["ok", "ask", "down"]);
}

#[test]
fn goto_sibling_does_not_also_run_static_targets_after_resume() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<RouteState>::new()
        .add_node("ok", |_state: &RouteState| {
            Ok(NodeOutput::new()
                .update("log", json!(["ok"]))
                .goto("detour"))
        })
        .add_node("ask", |_state: &RouteState| {
            interrupt(json!({ "q": "ask" }))?;
            Ok(NodeOutput::new().update("log", json!(["ask"])))
        })
        .add_node("detour", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["detour"])))
        })
        .add_node("down", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["down"])))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask")
        .add_edge("ok", "down")
        .set_finish_point("detour")
        .set_finish_point("down")
        .set_finish_point("ask")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &RouteState::default(),
            &RunConfig::new().with_thread("interrupt-route-goto-override"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph
        .get_state("interrupt-route-goto-override")
        .unwrap()
        .unwrap();
    let resume = json!({ snapshot.interrupts[0].id.clone(): "resumed" });
    let output = graph
        .resume("interrupt-route-goto-override", resume)
        .unwrap();
    assert_eq!(output.log, vec!["ok", "ask", "detour"]);
}

#[test]
fn exit_durability_does_not_replay_consumed_routes() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<RouteState>::new()
        .add_node("ok", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["ok"])))
        })
        .add_node("ask", |_state: &RouteState| {
            interrupt(json!({ "q": "ask" }))?;
            Ok(NodeOutput::new().update("log", json!(["ask"])))
        })
        .add_node("down", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["down"])))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask")
        .add_edge("ok", "down")
        .set_finish_point("down")
        .set_finish_point("ask")
        .compile_with(
            CompileOptions::new()
                .with_checkpointer(saver)
                .interrupt_before(["down"]),
        )
        .unwrap();
    let config = RunConfig::new()
        .with_thread("exit-routes")
        .with_durability(langgraph_rs::Durability::Exit);

    let error = graph.run(&RouteState::default(), &config).unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("exit-routes").unwrap().unwrap();
    let resume = json!({ snapshot.interrupts[0].id.clone(): "resumed" });
    let error = graph
        .resume_with_config("exit-routes", resume, &config)
        .unwrap_err();
    assert!(error.is_interrupt(), "the static hook pauses before `down`");

    let output = graph
        .resume_with_config("exit-routes", json!("continue"), &config)
        .unwrap();
    assert_eq!(output.log, vec!["ok", "ask", "down"]);
}

#[test]
fn nested_parallel_runs_do_not_deadlock_on_the_worker_pool() {
    // A node that invokes a child graph with its own fan-out: the child must not queue on
    // the same pool the parent is occupying (the guard falls back to scoped threads).
    let child = StateGraph::<RouteState>::new()
        .add_node("spawn", |_state: &RouteState| {
            let mut output = NodeOutput::new();
            for index in 0..4 {
                output = output.send(langgraph_rs::Send::with_arg("leaf", json!({ "n": index })));
            }
            Ok(output)
        })
        .add_node("leaf", |state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["leaf"])))
        })
        .add_edge(langgraph_rs::START, "spawn")
        .set_finish_point("leaf")
        .compile()
        .unwrap();

    let parent = StateGraph::<RouteState>::new()
        .add_node("outer", move |_state: &RouteState| {
            let mut output = NodeOutput::new();
            for index in 0..4 {
                output = output.send(langgraph_rs::Send::with_arg("inner", json!({ "n": index })));
            }
            Ok(output)
        })
        .add_node("inner", move |_state: &RouteState| {
            let nested = child
                .run(&RouteState::default(), &RunConfig::new())
                .unwrap();
            Ok(NodeOutput::new().update("log", json!(["inner"])))
        })
        .add_edge(langgraph_rs::START, "outer")
        .set_finish_point("inner")
        .compile()
        .unwrap();

    let output = parent
        .run(&RouteState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.state.log.len(), 4);
}