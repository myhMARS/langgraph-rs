//! Temporary performance probe (not part of the public examples surface).

use std::sync::Arc;
use std::time::Instant;

use langgraph_rs::{
    CompileOptions, InMemorySaver, JsonState, NodeOutput, RunConfig, StateGraph, StateSchema, json,
};

fn build(nodes: usize, saver: bool) -> langgraph_rs::CompiledGraph<JsonState> {
    let mut graph = StateGraph::<JsonState>::with_schema(
        StateSchema::dynamic().with_reducer("log", langgraph_rs::reducers::append()),
    );
    for index in 0..nodes {
        graph = graph.add_node(format!("n{index}"), move |_state: &JsonState| {
            Ok(NodeOutput::new().update("log", json!(["x"])))
        });
    }
    graph = graph.add_edge(langgraph_rs::START, "n0");
    for index in 0..nodes.saturating_sub(1) {
        graph = graph.add_edge(format!("n{index}"), format!("n{}", index + 1));
    }
    if nodes > 0 {
        graph = graph.set_finish_point(format!("n{}", nodes - 1));
    }
    let options = if saver {
        CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new()))
    } else {
        CompileOptions::new()
    };
    graph.compile_with(options).unwrap()
}

fn input() -> JsonState {
    JsonState::new().with("log", json!([]))
}

fn time_invoke(graph: &langgraph_rs::CompiledGraph<JsonState>, iterations: usize, thread: bool) -> f64 {
    let started = Instant::now();
    for index in 0..iterations {
        let config = if thread {
            RunConfig::new().with_thread(format!("t{index}"))
        } else {
            RunConfig::new()
        };
        graph.run(&input(), &config).unwrap();
    }
    started.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0
}

fn main() {
    for nodes in [1usize, 10, 30] {
        let plain = time_invoke(&build(nodes, false), 300, false);
        let saver = build(nodes, true);
        let started = Instant::now();
        for index in 0..300 {
            let config = RunConfig::new()
                .with_thread(format!("s{index}"))
                .with_durability(langgraph_rs::Durability::Sync);
            saver.run(&input(), &config).unwrap();
        }
        let sync = started.elapsed().as_nanos() as f64 / 300.0 / 1000.0;
        let started = Instant::now();
        for index in 0..300 {
            let config = RunConfig::new()
                .with_thread(format!("e{index}"))
                .with_durability(langgraph_rs::Durability::Exit);
            saver.run(&input(), &config).unwrap();
        }
        let exit = started.elapsed().as_nanos() as f64 / 300.0 / 1000.0;
        let saved = time_invoke(&saver, 300, true);
        let history = saver.get_state_history("t0").unwrap().len();
        println!(
            "nodes={nodes:>2}  invoke={plain:7.1} us  async={saved:7.1} us  sync={sync:7.1} us  exit={exit:7.1} us  history={history}  per-save={:.1} us",
            saved / history as f64
        );
    }
    // same-thread vs fresh-thread cost with a long history
    let graph = build(1, true);
    for _ in 0..2000 {
        graph
            .run(&input(), &RunConfig::new().with_thread("long"))
            .unwrap();
    }
    let started = Instant::now();
    for _ in 0..300 {
        graph
            .run(&input(), &RunConfig::new().with_thread("long"))
            .unwrap();
    }
    let same = started.elapsed().as_nanos() as f64 / 300.0 / 1000.0;
    let started = Instant::now();
    for index in 0..300 {
        graph
            .run(&input(), &RunConfig::new().with_thread(format!("fresh{index}")))
            .unwrap();
    }
    let fresh = started.elapsed().as_nanos() as f64 / 300.0 / 1000.0;
    println!("long-thread same={same:.1} us  fresh={fresh:.1} us  history={}", graph.get_state_history("long").unwrap().len());
    for k in [1usize, 3, 8] {
        println!("fanout k={k}: {:.1} us", time_fanout(k, 300));
    }
    // fixed overhead of state/history reads on a thread with history
    let graph = build(10, true);
    graph
        .run(&input(), &RunConfig::new().with_thread("hist"))
        .unwrap();
    let started = Instant::now();
    for _ in 0..300 {
        std::hint::black_box(graph.get_state("hist").unwrap());
    }
    println!(
        "get_state={:.1} us",
        started.elapsed().as_nanos() as f64 / 300.0 / 1000.0
    );
    let started = Instant::now();
    for _ in 0..300 {
        std::hint::black_box(graph.get_state_history_filtered("hist", None, Some(5)).unwrap());
    }
    println!(
        "history(5)={:.1} us",
        started.elapsed().as_nanos() as f64 / 300.0 / 1000.0
    );
}
fn build_fanout(k: usize) -> langgraph_rs::CompiledGraph<JsonState> {
    let mut graph = StateGraph::<JsonState>::with_schema(
        StateSchema::dynamic().with_reducer("log", langgraph_rs::reducers::append()),
    );
    graph = graph.add_node("fan", move |_state: &JsonState| {
        let mut output = langgraph_rs::NodeOutput::new();
        for index in 0..k {
            output = output.send(langgraph_rs::Send::with_arg("worker", json!({ "n": index })));
        }
        Ok(output)
    });
    graph = graph.add_node("worker", |state: &JsonState| {
        let n = state.get("n").and_then(|value| value.as_i64()).unwrap_or(0);
        Ok(NodeOutput::new().update("log", json!([n])))
    });
    graph = graph
        .add_edge(langgraph_rs::START, "fan")
        .set_finish_point("worker");
    graph
        .compile_with(CompileOptions::new().with_checkpointer(std::sync::Arc::new(InMemorySaver::new())))
        .unwrap()
}

fn time_fanout(k: usize, iterations: usize) -> f64 {
    let graph = build_fanout(k);
    // warm up
    for index in 0..20 {
        graph
            .run(&input(), &RunConfig::new().with_thread(format!("w{index}")))
            .unwrap();
    }
    let started = Instant::now();
    for index in 0..iterations {
        graph
            .run(&input(), &RunConfig::new().with_thread(format!("f{index}")))
            .unwrap();
    }
    started.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0
}