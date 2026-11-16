//! Node execution for one superstep.

use std::panic;
use std::time::Instant;

use serde_json::Value;

use crate::kernel::context::TaskContext;
use crate::kernel::error::{Error, Interrupt, NodeFailure};
use crate::kernel::graph::CompiledGraph;
use crate::kernel::node::{NodeOutput, Task};
use crate::kernel::state::{State, values_to_state};
use crate::kernel::usage::Usage;

/// Successful result of one node execution, before the superstep is merged.
pub(super) struct TaskResult {
    pub(super) node: String,
    pub(super) output: NodeOutput,
    pub(super) custom: Vec<Value>,
    pub(super) usage: Vec<Usage>,
    pub(super) duration_millis: u64,
    pub(super) attempts: u32,
}

/// Permanent failure of one node execution.
pub(super) struct TaskFailure {
    pub(super) failure: NodeFailure,
    pub(super) usage: Vec<Usage>,
    pub(super) duration_millis: u64,
}

/// Outcome of one scheduled task.
pub(super) enum TaskOutcome {
    Ok(Box<TaskResult>),
    Interrupted(Interrupt),
    Failed(Box<TaskFailure>),
}

/// Execute every task of a superstep.
///
/// A single task runs inline; multiple tasks run on scoped threads and their results
/// are collected in task order, which keeps merging deterministic. Failures are
/// returned per task so the executor can decide between error routing and aborting.
pub(super) fn run_superstep<S: State>(
    graph: &CompiledGraph<S>,
    state: &S,
    tasks: &[Task],
) -> Vec<TaskOutcome> {
    if tasks.len() <= 1 {
        return tasks
            .iter()
            .map(|task| run_task(graph, state, task))
            .collect();
    }

    std::thread::scope(|scope| {
        let handles: Vec<_> = tasks
            .iter()
            .map(|task| scope.spawn(move || run_task(graph, state, task)))
            .collect();
        handles
            .into_iter()
            .map(|handle| match handle.join() {
                Ok(outcome) => outcome,
                Err(panic) => panic::resume_unwind(panic),
            })
            .collect()
    })
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

    let policy = graph.retry_policy(&task.node).cloned().unwrap_or_default();
    let max_attempts = policy.max_attempts.max(1);
    let started = Instant::now();
    let mut usage: Vec<Usage> = Vec::new();
    let mut last_error: Option<String> = None;

    for attempt in 1..=max_attempts {
        let context =
            TaskContext::new(task.node.clone(), task.resume.clone()).with_error(task.error.clone());
        let (result, custom, mut reported) =
            crate::kernel::context::with_task(context, || node(&node_state));
        usage.append(&mut reported);

        match result {
            Ok(output) => {
                usage.extend(output.usage().cloned());
                return TaskOutcome::Ok(Box::new(TaskResult {
                    node: task.node.clone(),
                    output,
                    custom,
                    usage,
                    duration_millis: started.elapsed().as_millis() as u64,
                    attempts: attempt as u32,
                }));
            }
            Err(Error::Interrupt(interrupt)) => {
                return TaskOutcome::Interrupted(Interrupt {
                    node: task.node.clone(),
                    value: interrupt.value,
                });
            }
            Err(error) => {
                last_error = Some(error.to_string());
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
        max_attempts as u32,
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
        failure: NodeFailure::new(task.node.clone(), message, attempts),
        usage,
        duration_millis: started.elapsed().as_millis() as u64,
    }))
}
