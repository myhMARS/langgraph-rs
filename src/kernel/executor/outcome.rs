//! Shared result types for the sync and async Pregel loops.
//!
//! Both runtimes compute the same kind of transition after a superstep. Keeping the
//! shape in one place is the first step towards one loop with pluggable execution
//! backends.

use std::collections::HashSet;

use serde_json::Value;

use crate::kernel::error::{Error, Result};
use crate::kernel::error::{Interrupt, NodeFailure};
use crate::kernel::node::{NodeOutput, Task};
use crate::kernel::ports::TraceSink;
use crate::kernel::state::ChannelUpdate;
use crate::kernel::stream::StreamChunk;
use crate::kernel::usage::{NodeExecution, RunOutcome, RunReport, Usage};

/// Everything one superstep's node outcomes contribute to the run.
#[derive(Debug, Default)]
pub(crate) struct ProcessedResults {
    pub(crate) channel_updates: Vec<ChannelUpdate>,
    pub(crate) replacements: Vec<(String, Value)>,
    pub(crate) next_tasks: Vec<Task>,
    pub(crate) goto_targets: Vec<(String, String)>,
    pub(crate) executed: Vec<String>,
    pub(crate) goto_nodes: HashSet<String>,
}

/// Merge per-attempt usage into one entry for a node execution.
pub(crate) fn merge_usage(usage: Vec<Usage>) -> Option<Usage> {
    let mut combined: Option<Usage> = None;
    for item in usage {
        combined.get_or_insert_with(Usage::new).merge(&item);
    }
    combined
}

/// Successful result of one node execution.
#[derive(Debug)]
pub(crate) struct TaskResult {
    pub(crate) node: String,
    pub(crate) output: NodeOutput,
    pub(crate) custom: Vec<Value>,
    pub(crate) messages: Vec<Value>,
    pub(crate) usage: Vec<Usage>,
    pub(crate) duration_millis: u64,
    pub(crate) attempts: u32,
    pub(crate) cached: bool,
    pub(crate) input: Option<Value>,
}

/// Permanent failure of one node execution.
#[derive(Debug)]
pub(crate) struct TaskFailure {
    pub(crate) failure: NodeFailure,
    pub(crate) usage: Vec<Usage>,
    pub(crate) duration_millis: u64,
    pub(crate) input: Option<Value>,
}

/// Outcome of one scheduled task.
pub(crate) enum TaskOutcome {
    Ok(Box<TaskResult>),
    Interrupted(Interrupt),
    Failed(Box<TaskFailure>),
}

/// Read-only graph information the shared outcome processing needs.
pub(crate) trait OutcomeGraph {
    fn trace_sinks(&self) -> &[std::sync::Arc<dyn TraceSink>];
    fn error_handler(&self, node: &str) -> Option<&str>;
    fn has_node(&self, node: &str) -> bool;
    fn destinations_of(&self, node: &str) -> Option<&[String]>;
    fn record_parent_command(
        &self,
        updates: Vec<(String, Value)>,
        replaces: Vec<(String, Value)>,
        goto: Vec<String>,
    );
    fn has_parent_command(&self) -> bool;
}

/// Process a superstep's outcomes into channel writes, scheduled tasks and stream events.
pub(crate) fn process_outcomes<G: OutcomeGraph + ?Sized>(
    graph: &G,
    outcomes: Vec<TaskOutcome>,
    step: usize,
    report: &mut RunReport,
    sink: &mut dyn FnMut(StreamChunk),
) -> Result<ProcessedResults> {
    let mut channel_updates: Vec<ChannelUpdate> = Vec::new();
    let mut replacements: Vec<(String, Value)> = Vec::new();
    let mut next_tasks: Vec<Task> = Vec::new();
    let mut goto_targets: Vec<(String, String)> = Vec::new();
    let mut executed: Vec<String> = Vec::new();
    let mut goto_nodes: HashSet<String> = HashSet::new();
    let mut routed_handlers: HashSet<String> = HashSet::new();

    for outcome in outcomes {
        match outcome {
            TaskOutcome::Interrupted(_) => {
                unreachable!("interrupts are handled before processing outcomes");
            }
            TaskOutcome::Failed(task_failure) => {
                let TaskFailure {
                    failure,
                    usage,
                    duration_millis,
                    input,
                } = *task_failure;
                let execution = NodeExecution {
                    node: failure.node.clone(),
                    step,
                    duration_millis,
                    attempts: failure.attempts,
                    cached: false,
                    error: Some(failure.message.clone()),
                    usage: merge_usage(usage),
                };
                sink(StreamChunk::Usage {
                    execution: execution.clone(),
                });
                for trace_sink in graph.trace_sinks() {
                    trace_sink.record(&execution);
                }
                report.record(execution);

                sink(StreamChunk::Task {
                    step,
                    node: failure.node.clone(),
                    input,
                    result: None,
                    error: Some(failure.clone()),
                });

                match graph.error_handler(&failure.node) {
                    Some(handler) if routed_handlers.insert(handler.to_string()) => {
                        next_tasks.push(Task {
                            id: crate::kernel::node::new_task_id(handler),
                            node: handler.to_string(),
                            input: None,
                            resume: None,
                            resume_history: Vec::new(),
                            error: Some(failure),
                            timeout_millis: None,
                            interrupted: false,
                        });
                    }
                    Some(_) => {}
                    None => {
                        return Err(Error::Node {
                            node: failure.node.clone(),
                            message: format!(
                                "{} (failed after {} attempt(s))",
                                failure.message, failure.attempts
                            ),
                        });
                    }
                }
            }
            TaskOutcome::Ok(result) => {
                let TaskResult {
                    node,
                    output,
                    custom,
                    messages,
                    usage,
                    duration_millis,
                    attempts,
                    cached,
                    input,
                } = *result;
                executed.push(node.clone());
                for value in custom {
                    sink(StreamChunk::Custom {
                        node: node.clone(),
                        value,
                    });
                }
                for message in messages {
                    sink(StreamChunk::Message {
                        node: node.clone(),
                        message,
                    });
                }

                let execution = NodeExecution {
                    node: node.clone(),
                    step,
                    duration_millis,
                    attempts,
                    cached,
                    error: None,
                    usage: merge_usage(usage),
                };
                sink(StreamChunk::Usage {
                    execution: execution.clone(),
                });
                for trace_sink in graph.trace_sinks() {
                    trace_sink.record(&execution);
                }
                report.record(execution);

                let task_result = {
                    let updates = output.updates_map();
                    if updates.is_empty() {
                        None
                    } else {
                        Some(Value::Object(
                            updates
                                .iter()
                                .map(|(channel, value)| (channel.clone(), value.clone()))
                                .collect(),
                        ))
                    }
                };
                sink(StreamChunk::Task {
                    step,
                    node: node.clone(),
                    input: input.clone(),
                    result: task_result,
                    error: None,
                });
                if output.is_parent() {
                    let (updates, replaces, goto, _sends) = output.into_parts();
                    graph.record_parent_command(
                        updates.into_iter().collect::<Vec<_>>(),
                        replaces.into_iter().collect::<Vec<_>>(),
                        goto,
                    );
                    continue;
                }
                let (updates, replaces, goto, sends) = output.into_parts();
                if !updates.is_empty() {
                    let update = Value::Object(
                        updates
                            .iter()
                            .map(|(channel, value)| (channel.clone(), value.clone()))
                            .collect(),
                    );
                    sink(StreamChunk::Updates {
                        step,
                        node: node.clone(),
                        update,
                    });
                    for (channel, value) in updates {
                        channel_updates.push(ChannelUpdate {
                            channel,
                            value,
                            node: node.clone(),
                        });
                    }
                }
                for (channel, value) in replaces {
                    replacements.push((channel, value));
                }
                for target in goto {
                    goto_nodes.insert(node.clone());
                    goto_targets.push((target, node.clone()));
                }
                for send in sends {
                    let input = match send.arg {
                        None | Some(Value::Null) => None,
                        Some(Value::Object(map)) => Some(map),
                        Some(other) => {
                            return Err(Error::routing(
                                &node,
                                format!(
                                    "Send toward `{}` needs a JSON object arg, found `{other}`",
                                    send.node
                                ),
                            ));
                        }
                    };
                    if !graph.has_node(&send.node) {
                        return Err(Error::routing(
                            &node,
                            format!("Send targets unknown node `{}`", send.node),
                        ));
                    }
                    if let Some(allowed) = graph.destinations_of(&node) {
                        if !allowed.iter().any(|candidate| candidate == &send.node) {
                            return Err(Error::routing(
                                &node,
                                format!("`{}` is not a declared destination", send.node),
                            ));
                        }
                    }
                    next_tasks.push(Task {
                        id: crate::kernel::node::new_task_id(send.node.as_str()),
                        node: send.node,
                        input,
                        resume: None,
                        resume_history: Vec::new(),
                        error: None,
                        timeout_millis: send.timeout_millis,
                        interrupted: false,
                    });
                }
            }
        }
    }

    Ok(ProcessedResults {
        channel_updates,
        replacements,
        next_tasks,
        goto_targets,
        executed,
        goto_nodes,
    })
}

/// Pending channel writes of the successful tasks in a superstep.
pub(crate) fn collect_pending_writes(
    tasks: &[Task],
    outcomes: &[TaskOutcome],
) -> Vec<crate::kernel::checkpoint::PendingWrite> {
    let mut writes = Vec::new();
    for (task, outcome) in tasks.iter().zip(outcomes) {
        let TaskOutcome::Ok(result) = outcome else {
            continue;
        };
        for (channel, value) in result.output.updates_map() {
            writes.push(crate::kernel::checkpoint::PendingWrite {
                task_id: task.id.clone(),
                node: task.node.clone(),
                channel: channel.clone(),
                value: value.clone(),
                replace: false,
            });
        }
        for (channel, value) in result.output.replaces_map() {
            writes.push(crate::kernel::checkpoint::PendingWrite {
                task_id: task.id.clone(),
                node: task.node.clone(),
                channel: channel.clone(),
                value: value.clone(),
                replace: true,
            });
        }
    }
    writes
}

/// Convert the working state into the run outcome, applying the output filter and
/// stamping the total duration.
pub(crate) fn finish<S: crate::kernel::state::State>(
    state: S,
    schema: &crate::kernel::state::StateSchema,
    mut report: RunReport,
    started: std::time::Instant,
) -> Result<RunOutcome<S>> {
    report.total_duration_millis = started.elapsed().as_millis() as u64;
    let state = if schema.has_output_filter() {
        let values = crate::kernel::state::state_to_values(&state)?;
        crate::kernel::state::values_to_state::<S>(&schema.output_values(&values))?
    } else {
        state
    };
    Ok(RunOutcome { state, report })
}

/// Interrupts raised by a superstep, in task order.
pub(crate) fn interrupts_from(outcomes: &[TaskOutcome]) -> Vec<Interrupt> {
    outcomes
        .iter()
        .filter_map(|outcome| match outcome {
            TaskOutcome::Interrupted(interrupt) => Some(interrupt.clone()),
            _ => None,
        })
        .collect()
}

/// Combine replayed interrupted tasks with newly scheduled tasks by task id.
pub(crate) fn merge_interrupted_tasks(
    tasks: &[Task],
    keep: impl Fn(&Task) -> bool,
    scheduled: Vec<Task>,
) -> Vec<Task> {
    let mut merged: Vec<Task> = tasks.iter().filter(|task| keep(task)).cloned().collect();
    for task in scheduled {
        if !merged.iter().any(|existing| existing.id == task.id) {
            merged.push(task);
        }
    }
    merged
}

/// Apply processed channel updates and direct replacements to a working value map.
pub(crate) fn apply_processed_updates(
    schema: &crate::kernel::state::StateSchema,
    values: &mut serde_json::Map<String, Value>,
    channel_updates: Vec<ChannelUpdate>,
    replacements: Vec<(String, Value)>,
    finish_updates: &mut std::collections::HashMap<String, Value>,
) -> Result<()> {
    let mut immediate = Vec::with_capacity(channel_updates.len());
    for update in channel_updates {
        if schema.is_finish_value(&update.channel) {
            finish_updates.insert(update.channel, update.value);
        } else {
            immediate.push(update);
        }
    }
    crate::kernel::state::apply_updates(schema, values, immediate)?;
    for (channel, value) in replacements {
        schema.resolve(&channel)?;
        values.insert(channel, value);
    }
    Ok(())
}

/// First executed node that requests an `interrupt_after` pause.
pub(crate) fn interrupt_after_node(
    executed: &[String],
    is_after: impl Fn(&str) -> bool,
) -> Option<String> {
    executed.iter().find(|node| is_after(node)).cloned()
}

/// Move finish-value writes into the working values before the terminal checkpoint.
pub(crate) fn apply_finish_updates(
    values: &mut serde_json::Map<String, Value>,
    finish_updates: &mut std::collections::HashMap<String, Value>,
) {
    for (channel, value) in finish_updates.drain() {
        values.insert(channel, value);
    }
}

/// Stream chunk carrying the current channel values.
pub(crate) fn values_chunk(step: usize, values: &serde_json::Map<String, Value>) -> StreamChunk {
    StreamChunk::Values {
        step,
        values: Value::Object(values.clone()),
    }
}

/// Stream chunk reporting which nodes ran and which are scheduled next.
pub(crate) fn debug_chunk(step: usize, executed: &[String], scheduled: &[String]) -> StreamChunk {
    StreamChunk::Debug {
        step,
        payload: serde_json::json!({
            "step": step,
            "executed": executed,
            "next": scheduled,
        }),
    }
}

/// Whether a regular superstep checkpoint should be persisted.
pub(crate) fn should_save_after_step(
    durability: crate::kernel::graph::Durability,
    has_interrupt: bool,
) -> bool {
    durability != crate::kernel::graph::Durability::Exit || has_interrupt
}

/// Whether the terminal checkpoint should be persisted.
pub(crate) fn should_save_terminal(
    durability: crate::kernel::graph::Durability,
    has_thread: bool,
) -> bool {
    durability == crate::kernel::graph::Durability::Exit && has_thread
}

/// Interrupt raised by an `interrupt_before` hook and its stream event.
pub(crate) fn interrupt_before_pause(task: &Task) -> (Interrupt, StreamChunk) {
    let interrupt = Interrupt::new(task.node.clone(), Value::Null);
    let chunk = StreamChunk::Interrupt {
        node: task.node.clone(),
        value: Value::Null,
    };
    (interrupt, chunk)
}
