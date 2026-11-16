//! Node execution for one superstep.

use std::panic;
use std::time::Instant;

use serde_json::Value;

use crate::context::TaskContext;
use crate::error::{Error, Interrupt, Result};
use crate::graph::CompiledGraph;
use crate::node::{NodeOutput, Task};
use crate::state::{State, values_to_state};
use crate::usage::Usage;

/// Result of one node execution, before the superstep is merged.
pub(super) struct TaskResult {
    pub(super) node: String,
    pub(super) output: NodeOutput,
    pub(super) custom: Vec<Value>,
    pub(super) usage: Vec<Usage>,
    pub(super) duration_millis: u64,
}

/// Execute every task of a superstep.
///
/// A single task runs inline; multiple tasks run on scoped threads and their results
/// are collected in task order, which keeps merging deterministic.
pub(super) fn run_superstep<S: State>(
    graph: &CompiledGraph<S>,
    state: &S,
    tasks: &[Task],
) -> Result<Vec<TaskResult>> {
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
                Ok(result) => result,
                Err(panic) => panic::resume_unwind(panic),
            })
            .collect()
    })
}

fn run_task<S: State>(graph: &CompiledGraph<S>, state: &S, task: &Task) -> Result<TaskResult> {
    let node = graph.node(&task.node).ok_or_else(|| {
        Error::other(format!(
            "node `{}` is not registered in this graph",
            task.node
        ))
    })?;

    let node_state = match &task.input {
        Some(input) => values_to_state::<S>(input)?,
        None => state.clone(),
    };

    let context = TaskContext::new(task.node.clone(), task.resume.clone());
    let started = Instant::now();
    let (result, custom, mut usage) = crate::context::with_task(context, || node(&node_state));
    let duration_millis = started.elapsed().as_millis() as u64;

    match result {
        Ok(output) => {
            usage.extend(output.usage().cloned());
            Ok(TaskResult {
                node: task.node.clone(),
                output,
                custom,
                usage,
                duration_millis,
            })
        }
        Err(Error::Interrupt(interrupt)) => Err(Error::Interrupt(Interrupt {
            node: task.node.clone(),
            value: interrupt.value,
        })),
        Err(error) => Err(Error::Node {
            node: task.node.clone(),
            message: error.to_string(),
        }),
    }
}
