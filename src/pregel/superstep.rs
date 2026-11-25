//! Node execution for one superstep.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::_internal::cache::CachedOutput;
use crate::_internal::error::{Error, Interrupt, NodeFailure};
use crate::_internal::node::{NodeFn, NodeOutput, Task};
use crate::_internal::state::{State, state_to_values, values_to_state};
use crate::_internal::usage::Usage;
use crate::graph::CompiledGraph;
use crate::runtime::context::TaskContext;

use super::outcome::{TaskFailure, TaskOutcome, TaskResult};

/// Execute every task of a superstep.
///
/// A single task runs inline; multiple tasks run on scoped threads and their results
/// are collected in task order, which keeps merging deterministic. Failures are
/// returned per task so the executor can decide between error routing and aborting.
pub(super) fn run_superstep<S: State>(
    graph: &std::sync::Arc<CompiledGraph<S>>,
    state: &S,
    tasks: &[Task],
    max_concurrency: Option<usize>,
) -> Vec<TaskOutcome> {
    if tasks.is_empty() {
        return Vec::new();
    }
    let limit = max_concurrency
        .filter(|limit| *limit > 0)
        .unwrap_or(usize::MAX);
    let workers = limit.min(tasks.len()).max(1);
    if workers == 1 {
        return tasks
            .iter()
            .map(|task| run_task(graph, state, task))
            .collect();
    }

    if let Some(outcomes) = run_superstep_pooled(graph, state, tasks, workers) {
        return outcomes;
    }

    let run = crate::runtime::context::capture_run_context();
    let step = crate::runtime::context::capture_step_context();
    let collector = crate::runtime::context::current_subgraph_collector();
    let next = AtomicUsize::new(0);
    let outcomes: Mutex<Vec<Option<TaskOutcome>>> =
        Mutex::new((0..tasks.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let run = run.clone();
            let collector = collector.clone();
            let next = &next;
            let outcomes = &outcomes;
            scope.spawn(move || {
                let _run_guard = run.map(crate::runtime::context::install_run_context);
                let _step_guard = step.map(|(step, managed, limit)| {
                    crate::runtime::context::install_step(step, managed, limit)
                });
                let _collector_guard =
                    collector.map(crate::runtime::context::install_subgraph_collector);
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= tasks.len() {
                        break;
                    }
                    let outcome = run_task(graph, state, &tasks[index]);
                    if let Ok(mut outcomes) = outcomes.lock() {
                        outcomes[index] = Some(outcome);
                    }
                }
            });
        }
    });
    outcomes
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .into_iter()
        .map(|outcome| outcome.expect("every task has an outcome"))
        .collect()
}

/// Run the workers on the process-wide thread pool.
///
/// Spawning a scoped thread per task costs 50-80 us on Windows, so a fan-out of three
/// trivial nodes spent more time in `CreateThread` than in the graph. Jobs are `'static`,
/// so each worker receives an `Arc` graph, an owned task list and a cloned state; the
/// completion channel keeps the "results in task order" contract. Returns `None` when the
/// pool is unavailable so the caller can fall back to scoped threads.
fn run_superstep_pooled<S: State>(
    graph: &std::sync::Arc<CompiledGraph<S>>,
    state: &S,
    tasks: &[Task],
    workers: usize,
) -> Option<Vec<TaskOutcome>> {
    if crate::pregel::pool::on_worker() {
        return None;
    }
    let pool = crate::pregel::pool::ThreadPool::global().filter(|pool| pool.size() >= workers)?;
    let run = crate::runtime::context::capture_run_context();
    let step = crate::runtime::context::capture_step_context();
    let collector = crate::runtime::context::current_subgraph_collector();
    let next = std::sync::Arc::new(AtomicUsize::new(0));
    let outcomes = std::sync::Arc::new(Mutex::new(
        (0..tasks.len()).map(|_| None).collect::<Vec<_>>(),
    ));
    let graph = std::sync::Arc::clone(graph);
    let state = std::sync::Arc::new(state.clone());
    let tasks = std::sync::Arc::new(tasks.to_vec());
    let (sender, receiver) = std::sync::mpsc::channel::<()>();

    let mut submitted = 0usize;
    for _ in 0..workers {
        let run = run.clone();
        let collector = collector.clone();
        let next = std::sync::Arc::clone(&next);
        let outcomes = std::sync::Arc::clone(&outcomes);
        let graph = std::sync::Arc::clone(&graph);
        let state = std::sync::Arc::clone(&state);
        let tasks = std::sync::Arc::clone(&tasks);
        let sender = sender.clone();
        let accepted = pool.submit(Box::new(move || {
            let _worker_guard = crate::pregel::pool::enter_worker();
            let _run_guard = run.map(crate::runtime::context::install_run_context);
            let _step_guard = step.map(|(step, managed, limit)| {
                crate::runtime::context::install_step(step, managed, limit)
            });
            let _collector_guard =
                collector.map(crate::runtime::context::install_subgraph_collector);
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= tasks.len() {
                    break;
                }
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_task(&graph, &state, &tasks[index])
                }))
                .unwrap_or_else(|_| {
                    failed(
                        &tasks[index],
                        format!("node `{}` panicked", tasks[index].node),
                        1,
                        Vec::new(),
                        Instant::now(),
                    )
                });
                if let Ok(mut outcomes) = outcomes.lock() {
                    outcomes[index] = Some(outcome);
                }
            }
            let _ = sender.send(());
        }));
        if !accepted {
            break;
        }
        submitted += 1;
    }
    drop(sender);

    // Wait for the jobs that were accepted (a closed pool is the only failure mode).
    for _ in 0..submitted {
        if receiver.recv().is_err() {
            break;
        }
    }
    // Drain anything the pool did not take so no task is skipped or run twice.
    loop {
        let index = next.fetch_add(1, Ordering::Relaxed);
        if index >= tasks.len() {
            break;
        }
        let outcome = run_task(&graph, &state, &tasks[index]);
        if let Ok(mut outcomes) = outcomes.lock() {
            outcomes[index] = Some(outcome);
        }
    }
    let mut guarded = outcomes
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let collected: Vec<Option<TaskOutcome>> = std::mem::take(&mut *guarded);
    drop(guarded);
    Some(
        collected
            .into_iter()
            .map(|outcome| outcome.expect("every task has an outcome"))
            .collect(),
    )
}

fn run_task<S: State>(graph: &CompiledGraph<S>, state: &S, task: &Task) -> TaskOutcome {
    let Some(node) = graph.node(&task.node) else {
        return failed(
            task,
            format!("node `{}` is not registered in this graph", task.node),
            1,
            Vec::new(),
            Instant::now(),
        );
    };

    // Borrow the run state when the task carries no custom input instead of cloning it:
    // a node only needs `&S`, and the clone is the dominant per-node cost for large states.
    let owned_state;
    let node_state: &S = match &task.input {
        Some(input) => match values_to_state::<S>(input) {
            Ok(state) => {
                owned_state = state;
                &owned_state
            }
            Err(error) => {
                return failed(task, error.to_string(), 1, Vec::new(), Instant::now());
            }
        },
        None => state,
    };

    let cache = graph.cache().cloned();
    let cache_policy = graph.cache_policy(&task.node).cloned();
    let cache_key = match (&cache, &cache_policy) {
        (Some(_), Some(policy)) => {
            let values = match state_to_values(node_state) {
                Ok(values) => values,
                Err(error) => {
                    return failed(task, error.to_string(), 1, Vec::new(), Instant::now());
                }
            };
            Some(policy.cache_key(&task.node, &values))
        }
        _ => None,
    };
    if let (Some(cache), Some(key)) = (&cache, &cache_key) {
        match cache.get(key) {
            Ok(Some(cached)) => {
                let usage = cached
                    .usage
                    .clone()
                    .map(|usage| vec![usage])
                    .unwrap_or_default();
                return TaskOutcome::Ok(Box::new(TaskResult {
                    id: task.id.clone(),
                    node: task.node.clone(),
                    output: NodeOutput::from_cached(cached),
                    custom: Vec::new(),
                    tools: Vec::new(),
                    messages: Vec::new(),
                    usage,
                    duration_millis: 0,
                    attempts: 0,
                    cached: true,
                    input: task.input.clone().map(Value::Object),
                }));
            }
            Ok(None) => {}
            Err(error) => {
                return failed(
                    task,
                    format!("node cache lookup failed: {error}"),
                    1,
                    Vec::new(),
                    Instant::now(),
                );
            }
        }
    }

    let policy = graph.retry_policy(&task.node).cloned().unwrap_or_default();
    let max_attempts = policy.max_attempts.max(1);
    let started = Instant::now();
    let mut usage: Vec<Usage> = Vec::new();
    let mut last_error: Option<String> = None;
    let mut attempts_used = 0u32;

    for attempt in 1..=max_attempts {
        attempts_used = attempt as u32;
        let timeout = task.timeout_millis.map(Duration::from_millis).or_else(|| {
            graph
                .timeout_of(&task.node)
                .and_then(|policy| policy.effective_timeout())
        });
        let (result, custom, tools, messages, mut reported) = match timeout {
            Some(timeout) => run_node_timed(node.clone(), (*node_state).clone(), task, timeout),
            None => {
                let context = TaskContext::new(task.node.clone(), task.resume_history.clone())
                    .with_task_id(task.id.clone())
                    .with_error(task.error.clone())
                    .with_pending_resume(task.resume.clone());
                crate::runtime::context::with_task(context, || node(node_state))
            }
        };
        usage.append(&mut reported);

        match result {
            Ok(output) => {
                usage.extend(output.usage().cloned());
                if let (Some(cache), Some(cache_policy), Some(key)) =
                    (&cache, &cache_policy, &cache_key)
                {
                    let mut combined: Option<Usage> = None;
                    for item in &usage {
                        combined.get_or_insert_with(Usage::new).merge(item);
                    }
                    let cached_output = CachedOutput {
                        updates: output.updates_map().clone(),
                        replaces: output.replaces_map().clone(),
                        goto: output.goto_targets().to_vec(),
                        sends: output.sends().to_vec(),
                        usage: combined,
                    };
                    if let Err(error) = cache.put(key, cached_output, cache_policy.ttl()) {
                        return failed(
                            task,
                            format!("node cache store failed: {error}"),
                            attempt as u32,
                            usage,
                            started,
                        );
                    }
                }
                return TaskOutcome::Ok(Box::new(TaskResult {
                    id: task.id.clone(),
                    node: task.node.clone(),
                    output,
                    custom,
                    tools,
                    messages,
                    usage,
                    duration_millis: started.elapsed().as_millis() as u64,
                    attempts: attempt as u32,
                    cached: false,
                    input: task.input.clone().map(Value::Object),
                }));
            }
            Err(Error::Interrupt(interrupt)) => {
                return TaskOutcome::Interrupted(Interrupt {
                    id: interrupt.id,
                    node: task.node.clone(),
                    value: interrupt.value,
                    response_schema: interrupt.response_schema,
                    task_id: Some(task.id.clone().into_boxed_str()),
                    static_pause: interrupt.static_pause,
                    values: interrupt.values,
                });
            }
            Err(error) => {
                last_error = Some(error.to_string());
                if !policy.should_retry(&error.to_string()) {
                    break;
                }
                if attempt < max_attempts {
                    let backoff = policy.backoff_for(attempt);
                    if !backoff.is_zero() {
                        std::thread::sleep(backoff);
                    }
                }
            }
        }
    }

    failed(
        task,
        last_error.unwrap_or_else(|| "node failed without an error message".to_string()),
        attempts_used,
        usage,
        started,
    )
}

fn failed(
    task: &Task,
    message: impl Into<String>,
    attempts: u32,
    usage: Vec<Usage>,
    started: Instant,
) -> TaskOutcome {
    TaskOutcome::Failed(Box::new(TaskFailure {
        id: task.id.clone(),
        failure: NodeFailure::new(task.node.clone(), message, attempts),
        usage,
        duration_millis: started.elapsed().as_millis() as u64,
        input: task.input.clone().map(Value::Object),
    }))
}

/// Run a node attempt on a helper thread so a configured `run_timeout` can expire.
///
/// The thread keeps running when the timeout fires (Rust cannot cancel a blocking
/// closure); the run fails with [`Error::Timeout`] and the late result is discarded.
/// Result of a timed node run: output, custom values, tool events, messages and usage.
type TimedNodeOutcome = (
    crate::Result<NodeOutput>,
    Vec<Value>,
    Vec<Value>,
    Vec<Value>,
    Vec<Usage>,
);
fn run_node_timed<S: State>(
    node: NodeFn<S>,
    node_state: S,
    task: &Task,
    timeout: Duration,
) -> TimedNodeOutcome {
    let (sender, receiver) = std::sync::mpsc::channel();
    let node_name = task.node.clone();
    let resume = task.resume_history.clone();
    let task_id = task.id.clone();
    let error = task.error.clone();
    let run = crate::runtime::context::capture_run_context();
    let step = crate::runtime::context::capture_step_context();
    let collector = crate::runtime::context::current_subgraph_collector();
    std::thread::spawn(move || {
        let _run_guard = run.map(crate::runtime::context::install_run_context);
        let _step_guard = step.map(|(step, managed, limit)| {
            crate::runtime::context::install_step(step, managed, limit)
        });
        let _collector_guard = collector.map(crate::runtime::context::install_subgraph_collector);
        let context = TaskContext::new(node_name.clone(), resume.clone())
            .with_task_id(task_id)
            .with_error(error)
            .with_pending_resume(resume.last().cloned());
        let outcome = crate::runtime::context::with_task(context, || node(&node_state));
        let _ = sender.send(outcome);
    });
    match receiver.recv_timeout(timeout) {
        Ok(outcome) => outcome,
        Err(_) => (
            Err(Error::Timeout {
                node: task.node.clone(),
                millis: timeout.as_millis() as u64,
            }),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ),
    }
}
