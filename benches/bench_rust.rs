//! Native Rust benchmarks for `langgraph-rs` (no external harness).
//!
//! Run with `cargo bench --bench bench_rust` (or through `benchmarks/run.py`).
//! One JSON object per scenario is printed to stdout; a human summary goes to stderr.
//!
//! Scenarios intentionally mirror `python/benchmarks/bench.py` so the results are
//! comparable with Python LangGraph and with the PyO3 bindings.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use langgraph_rs::{
    CompileOptions, InMemorySaver, JsonState, NodeOutput, Route, RunConfig, State, StateGraph,
    StateSchema,
};
use serde::{Deserialize, Serialize};

const LINEAR_NODES: usize = 10;
const LINEAR_ITERATIONS: usize = 1000;
const LOOP_TARGET: i64 = 50;
const LOOP_ITERATIONS: usize = 200;
const FANOUT_ITERATIONS: usize = 1000;
const CHECKPOINT_ITERATIONS: usize = 200;
const CHECKPOINT_NODES: usize = 5;
const BUILD_ITERATIONS: usize = 500;

#[derive(Clone, Default, Serialize, Deserialize)]
struct CounterState {
    value: i64,
}

impl State for CounterState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

fn increment(state: &CounterState) -> langgraph_rs::Result<NodeOutput> {
    Ok(NodeOutput::new().set("value", state.value + 1))
}

fn linear_graph() -> langgraph_rs::CompiledGraph<CounterState> {
    let mut builder = StateGraph::<CounterState>::new();
    for index in 0..LINEAR_NODES {
        let name = format!("n{index:02}");
        builder = builder.add_node(name.clone(), increment);
        if index == 0 {
            builder = builder.set_entry_point(name);
        } else {
            builder = builder.add_edge(format!("n{:02}", index - 1), name);
        }
    }
    builder
        .set_finish_point(format!("n{:02}", LINEAR_NODES - 1))
        .compile()
        .expect("linear graph compiles")
}

fn loop_graph() -> langgraph_rs::CompiledGraph<CounterState> {
    StateGraph::<CounterState>::new()
        .add_node("increment", increment)
        .set_entry_point("increment")
        .add_conditional_edges("increment", |state: &CounterState| {
            Ok(if state.value < LOOP_TARGET {
                Route::node("increment")
            } else {
                Route::end()
            })
        })
        .compile_with(CompileOptions::new().with_recursion_limit(LOOP_TARGET as usize + 10))
        .expect("loop graph compiles")
}

fn fanout_graph() -> langgraph_rs::CompiledGraph<JsonState> {
    let schema = StateSchema::dynamic()
        .last_value("value")
        .last_value("a")
        .last_value("b")
        .last_value("c")
        .last_value("d");

    let mut builder = StateGraph::<JsonState>::with_schema(schema)
        .add_node("dispatch", |_state: &JsonState| Ok(NodeOutput::new()))
        .set_entry_point("dispatch");
    for key in ["a", "b", "c", "d"] {
        builder = builder.add_node(key, move |_state: &JsonState| {
            Ok(NodeOutput::new().set(key, 1))
        });
        builder = builder.add_edge("dispatch", key);
        builder = builder.add_edge(key, "join");
    }
    builder
        .add_node("join", |state: &JsonState| {
            let sum: i64 = ["a", "b", "c", "d"]
                .iter()
                .filter_map(|key| state.get(key).and_then(serde_json::Value::as_i64))
                .sum();
            Ok(NodeOutput::new().set("value", sum))
        })
        .set_finish_point("join")
        .compile()
        .expect("fan-out graph compiles")
}

fn checkpoint_graph(saver: Arc<InMemorySaver>) -> langgraph_rs::CompiledGraph<CounterState> {
    let mut builder = StateGraph::<CounterState>::new();
    for index in 0..CHECKPOINT_NODES {
        let name = format!("n{index:02}");
        builder = builder.add_node(name.clone(), increment);
        if index == 0 {
            builder = builder.set_entry_point(name);
        } else {
            builder = builder.add_edge(format!("n{:02}", index - 1), name);
        }
    }
    builder
        .set_finish_point(format!("n{:02}", CHECKPOINT_NODES - 1))
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .expect("checkpoint graph compiles")
}

fn measure<F: FnMut()>(warmup: usize, iterations: usize, mut operation: F) -> f64 {
    for _ in 0..warmup {
        operation();
    }
    let started = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    started.elapsed().as_secs_f64()
}

fn emit(scenario: &str, iterations: usize, seconds: f64) {
    let ns_per_op = seconds * 1e9 / iterations as f64;
    let ops_per_sec = iterations as f64 / seconds;
    println!(
        "{{\"engine\":\"rust\",\"scenario\":\"{scenario}\",\"iterations\":{iterations},\"total_ms\":{:.3},\"ns_per_op\":{:.1},\"ops_per_sec\":{:.1}}}",
        seconds * 1000.0,
        ns_per_op,
        ops_per_sec
    );
    eprintln!("rust        {scenario:<18} {ns_per_op:>12.0} ns/op {ops_per_sec:>12.0} ops/s");
}

fn main() {
    let linear = linear_graph();
    let input = CounterState::default();
    let config = RunConfig::new();
    let seconds = measure(LINEAR_ITERATIONS / 10, LINEAR_ITERATIONS, || {
        black_box(linear.invoke(&input, &config).expect("linear run"));
    });
    emit("linear_10", LINEAR_ITERATIONS, seconds);

    let looping = loop_graph();
    let seconds = measure(LOOP_ITERATIONS / 10, LOOP_ITERATIONS, || {
        black_box(looping.invoke(&input, &config).expect("loop run"));
    });
    emit("loop_50", LOOP_ITERATIONS, seconds);

    let fanout = fanout_graph();
    let dynamic_input = JsonState::default();
    let seconds = measure(FANOUT_ITERATIONS / 10, FANOUT_ITERATIONS, || {
        black_box(fanout.invoke(&dynamic_input, &config).expect("fan-out run"));
    });
    emit("fanout_4", FANOUT_ITERATIONS, seconds);

    let saver = Arc::new(InMemorySaver::new());
    let checkpointed = checkpoint_graph(saver);
    let mut run = 0usize;
    let seconds = measure(CHECKPOINT_ITERATIONS / 10, CHECKPOINT_ITERATIONS, || {
        run += 1;
        let config = RunConfig::new().with_thread(format!("bench-{run}"));
        black_box(
            checkpointed
                .invoke(&input, &config)
                .expect("checkpoint run"),
        );
    });
    emit("checkpoint_5", CHECKPOINT_ITERATIONS, seconds);

    let seconds = measure(BUILD_ITERATIONS / 10, BUILD_ITERATIONS, || {
        black_box(linear_graph());
    });
    emit("build_compile", BUILD_ITERATIONS, seconds);
}
