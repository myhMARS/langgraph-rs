//! Native Rust benchmarks for `langgraph-rs` (no external harness).
//!
//! Run with `cargo bench --bench bench_rust` (or through `benchmarks/run.py`).
//! One JSON object per scenario is printed to stdout; a human summary goes to stderr.
//!
//! Scenarios intentionally mirror `python/benchmarks/bench.py` so the results are
//! comparable with Python LangGraph and with the PyO3 bindings. Besides the latency
//! scenarios there are:
//!
//! - `scale_*`: linear chains of increasing size;
//! - `mt_8x_linear_10`: eight threads sharing one compiled graph;
//! - `mem_*`: allocation churn using the counting global allocator below.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
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
const MULTITHREAD_THREADS: usize = 8;
const MULTITHREAD_ITERATIONS_PER_THREAD: usize = 125;
const MEMORY_ITERATIONS: usize = 200;
const SCALE_SIZES: &[usize] = &[5, 10, 25, 50, 100];

// --------------------------------------------------------------------------- #
// Counting allocator (memory scenarios)
// --------------------------------------------------------------------------- #

static COUNTING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static TOTAL_BYTES: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.load(Ordering::Relaxed) {
            let current = CURRENT_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            TOTAL_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            PEAK_BYTES.fetch_max(current, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        if COUNTING.load(Ordering::Relaxed) {
            CURRENT_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() && COUNTING.load(Ordering::Relaxed) {
            if new_size >= layout.size() {
                let difference = new_size - layout.size();
                let current = CURRENT_BYTES.fetch_add(difference, Ordering::Relaxed) + difference;
                TOTAL_BYTES.fetch_add(difference, Ordering::Relaxed);
                PEAK_BYTES.fetch_max(current, Ordering::Relaxed);
            } else {
                CURRENT_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() && COUNTING.load(Ordering::Relaxed) {
            let current = CURRENT_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            TOTAL_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            PEAK_BYTES.fetch_max(current, Ordering::Relaxed);
        }
        pointer
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

// --------------------------------------------------------------------------- #
// Graph fixtures
// --------------------------------------------------------------------------- #

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

fn linear_graph_n(nodes: usize) -> langgraph_rs::CompiledGraph<CounterState> {
    let mut builder = StateGraph::<CounterState>::new();
    for index in 0..nodes {
        let name = format!("n{index:02}");
        builder = builder.add_node(name.clone(), increment);
        if index == 0 {
            builder = builder.set_entry_point(name);
        } else {
            builder = builder.add_edge(format!("n{:02}", index - 1), name);
        }
    }
    builder
        .set_finish_point(format!("n{:02}", nodes - 1))
        .compile_with(CompileOptions::new().with_recursion_limit(nodes + 10))
        .expect("linear graph compiles")
}

fn linear_graph() -> langgraph_rs::CompiledGraph<CounterState> {
    linear_graph_n(LINEAR_NODES)
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

// --------------------------------------------------------------------------- #
// Measurement helpers
// --------------------------------------------------------------------------- #

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

fn emit_threaded(scenario: &str, threads: usize, iterations: usize, seconds: f64) {
    let ns_per_op = seconds * 1e9 / iterations as f64;
    let ops_per_sec = iterations as f64 / seconds;
    println!(
        "{{\"engine\":\"rust\",\"scenario\":\"{scenario}\",\"threads\":{threads},\"iterations\":{iterations},\"total_ms\":{:.3},\"ns_per_op\":{:.1},\"ops_per_sec\":{:.1}}}",
        seconds * 1000.0,
        ns_per_op,
        ops_per_sec
    );
    eprintln!(
        "rust        {scenario:<18} {ns_per_op:>12.0} ns/op {ops_per_sec:>12.0} ops/s ({threads} threads)"
    );
}

fn emit_memory(scenario: &str, iterations: usize, bytes_per_op: f64, peak_bytes: usize) {
    println!(
        "{{\"engine\":\"rust\",\"scenario\":\"{scenario}\",\"iterations\":{iterations},\"bytes_per_op\":{:.1},\"peak_bytes\":{peak_bytes}}}",
        bytes_per_op
    );
    eprintln!("rust        {scenario:<18} {bytes_per_op:>12.0} B/op {peak_bytes:>12} B peak");
}

/// Measure total allocation churn and high-water mark of `iterations` operations.
fn measure_memory<F: FnMut()>(warmup: usize, iterations: usize, mut operation: F) -> (f64, usize) {
    COUNTING.store(true, Ordering::Relaxed);
    for _ in 0..warmup {
        operation();
    }
    let baseline_total = TOTAL_BYTES.load(Ordering::Relaxed);
    let baseline_current = CURRENT_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(baseline_current, Ordering::Relaxed);
    for _ in 0..iterations {
        operation();
    }
    COUNTING.store(false, Ordering::Relaxed);
    let allocated = TOTAL_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(baseline_total);
    let peak = PEAK_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(baseline_current);
    (allocated as f64 / iterations as f64, peak)
}

fn run_multithreaded(graph: Arc<langgraph_rs::CompiledGraph<CounterState>>) -> f64 {
    let input = CounterState::default();
    let warmup = MULTITHREAD_ITERATIONS_PER_THREAD / 5;
    // Warm up the graph once so the measurement does not include first-use setup.
    for _ in 0..warmup {
        graph.invoke(&input, &RunConfig::new()).expect("warmup run");
    }
    let iterations = MULTITHREAD_THREADS * MULTITHREAD_ITERATIONS_PER_THREAD;
    let started = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..MULTITHREAD_THREADS {
            let graph = Arc::clone(&graph);
            scope.spawn(move || {
                let input = CounterState::default();
                for _ in 0..MULTITHREAD_ITERATIONS_PER_THREAD {
                    black_box(
                        graph
                            .invoke(&input, &RunConfig::new())
                            .expect("threaded run"),
                    );
                }
            });
        }
    });
    started.elapsed().as_secs_f64() * iterations as f64 / iterations as f64
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

    // Graph-size scaling: fixed linear chains of 5/10/25/50/100 nodes.
    for &nodes in SCALE_SIZES {
        let graph = linear_graph_n(nodes);
        let iterations = match nodes {
            0..=10 => LINEAR_ITERATIONS,
            11..=25 => LINEAR_ITERATIONS / 2,
            26..=50 => LINEAR_ITERATIONS / 4,
            _ => LINEAR_ITERATIONS / 10,
        };
        let seconds = measure(iterations / 10, iterations, || {
            black_box(graph.invoke(&input, &config).expect("scale run"));
        });
        emit(&format!("scale_{nodes:03}"), iterations, seconds);
    }

    // Multi-threaded throughput: eight threads share one compiled graph.
    let shared = Arc::new(linear_graph());
    let seconds = run_multithreaded(shared);
    emit_threaded(
        "mt_8x_linear_10",
        MULTITHREAD_THREADS,
        MULTITHREAD_THREADS * MULTITHREAD_ITERATIONS_PER_THREAD,
        seconds,
    );

    // Memory: allocation churn and high-water mark per scenario.
    let (bytes_per_op, peak_bytes) =
        measure_memory(MEMORY_ITERATIONS / 10, MEMORY_ITERATIONS, || {
            black_box(linear.invoke(&input, &config).expect("memory linear run"));
        });
    emit_memory("mem_linear_10", MEMORY_ITERATIONS, bytes_per_op, peak_bytes);

    let (bytes_per_op, peak_bytes) =
        measure_memory(MEMORY_ITERATIONS / 10, MEMORY_ITERATIONS, || {
            black_box(looping.invoke(&input, &config).expect("memory loop run"));
        });
    emit_memory("mem_loop_50", MEMORY_ITERATIONS, bytes_per_op, peak_bytes);
}
