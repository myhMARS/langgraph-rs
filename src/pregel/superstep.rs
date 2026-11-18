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
    graph: &CompiledGraph<S>,
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

    let node_state = match &task.input {
        Some(input) => match values_to_state::<S>(input) {
            Ok(state) => state,
            Err(error) => {
                return failed(task, error.to_string(), 1, Vec::new(), Instant::now());
            }
        },
        None => state.clone(),
    };

    let cache = graph.cache().cloned();
    let cache_policy = graph.cache_policy(&task.node).cloned();
    let cache_key = match (&cache, &cache_policy) {
        (Some(_), Some(policy)) => {
            let values = match state_to_values(&node_state) {
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
        let (result, custom, messages, mut reported) = match timeout {
            Some(timeout) => run_node_timed(node.clone(), node_state.clone(), task, timeout),
            None => {
                let context = TaskContext::new(task.node.clone(), task.resume_history.clone())
                    .with_task_id(task.id.clone())
                    .with_error(task.error.clone());
                crate::runtime::context::with_task(context, || node(&node_state))
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
fn run_node_timed<S: State>(
    node: NodeFn<S>,
    node_state: S,
    task: &Task,
    timeout: Duration,
) -> (
    crate::Result<NodeOutput>,
    Vec<Value>,
    Vec<Value>,
    Vec<Usage>,
) {
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
        let context = TaskContext::new(node_name.clone(), resume)
            .with_task_id(task_id)
            .with_error(error);
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
        ),
    }
}
