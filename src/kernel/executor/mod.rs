//! The superstep execution engine.
//!
//! [`execute`] is the only loop in the crate. It delegates:
//!
//! - node execution to [`superstep`],
//! - edge evaluation / scheduling to [`scheduling`],
//! - persistence to [`checkpointing`],
//!
//! and owns the per-step merge + control flow, which keeps each file focused on one
//! concern.

mod checkpointing;
mod scheduling;
mod superstep;

use std::collections::HashSet;
use std::time::Instant;

use serde_json::{Map, Value};

use self::checkpointing::save_checkpoint;
use self::scheduling::{collect_next_nodes, dedupe};
use self::superstep::{TaskResult, run_superstep};
use crate::kernel::error::{Error, Interrupt, Result};
use crate::kernel::graph::{CompiledGraph, RunConfig};
use crate::kernel::node::Task;
use crate::kernel::state::{ChannelUpdate, State, apply_updates, values_to_state};
use crate::kernel::stream::StreamChunk;
use crate::kernel::usage::{NodeExecution, RunOutcome, RunReport, Usage};

/// Where a run starts from.
pub(crate) enum StartInput {
    /// Full channel values (from a typed input state).
    Values(Map<String, Value>),
    /// Partial writes applied over the default state.
    Updates(Map<String, Value>),
    /// Continue a checkpointed thread.
    Resume {
        thread_id: String,
        checkpoint_id: Option<String>,
        resume: Option<Value>,
        updates: Option<Map<String, Value>>,
    },
}

/// Execute a graph and return its final state together with the consumption report.
pub(crate) fn execute<S: State>(
    graph: &CompiledGraph<S>,
    start: StartInput,
    config: &RunConfig,
    sink: &mut dyn FnMut(StreamChunk),
) -> Result<RunOutcome<S>> {
    let started = Instant::now();
    let mut report = RunReport::new();

    let recursion_limit = config
        .recursion_limit
        .unwrap_or_else(|| graph.recursion_limit());
    let thread_id = config.thread_id.clone();

    if thread_id.is_some() && graph.checkpointer().is_none() {
        return Err(Error::MissingCheckpointer("a run bound to a thread_id"));
    }

    let fresh = matches!(&start, StartInput::Values(_) | StartInput::Updates(_));
    let mut parent: Option<String> = None;
    let mut skip_interrupt_before = false;

    let (mut values, mut state, mut tasks, mut step) = match start {
        StartInput::Values(values) => {
            let state = values_to_state::<S>(&values)?;
            let tasks = graph.entry_tasks(&state)?;
            (values, state, tasks, 0usize)
        }
        StartInput::Updates(updates) => {
            let mut values = Map::new();
            apply_updates(
                graph.schema(),
                &mut values,
                updates
                    .into_iter()
                    .map(|(channel, value)| ChannelUpdate {
                        channel,
                        value,
                        node: "<input>".to_string(),
                    })
                    .collect(),
            )?;
            let state = values_to_state::<S>(&values)?;
            let tasks = graph.entry_tasks(&state)?;
            (values, state, tasks, 0usize)
        }
        StartInput::Resume {
            thread_id: resume_thread,
            checkpoint_id,
            resume,
            updates,
        } => {
            let checkpointer = graph
                .checkpointer()
                .ok_or(Error::MissingCheckpointer("resume"))?;
            let checkpoint = match checkpoint_id {
                Some(id) => checkpointer.get(&resume_thread, &id)?,
                None => checkpointer.latest(&resume_thread)?,
            }
            .ok_or_else(|| {
                Error::Checkpoint(format!("no checkpoint found for thread `{resume_thread}`"))
            })?;

            parent = Some(checkpoint.id.clone());
            skip_interrupt_before = checkpoint.paused_before;
            let mut values = checkpoint.values.clone();
            if let Some(updates) = updates {
                apply_updates(
                    graph.schema(),
                    &mut values,
                    updates
                        .into_iter()
                        .map(|(channel, value)| ChannelUpdate {
                            channel,
                            value,
                            node: "<resume>".to_string(),
                        })
                        .collect(),
                )?;
            }
            let state = values_to_state::<S>(&values)?;
            let mut tasks = checkpoint.next.clone();
            for task in &mut tasks {
                task.resume = resume.clone();
            }
            (values, state, tasks, checkpoint.step)
        }
    };

    if tasks.is_empty() {
        return Ok(finish(state, report, started));
    }

    if fresh {
        if let (Some(checkpointer), Some(tid)) = (graph.checkpointer(), thread_id.as_deref()) {
            parent = checkpointer.latest(tid)?.map(|checkpoint| checkpoint.id);
        }
        save_checkpoint(
            graph,
            thread_id.as_deref(),
            &mut parent,
            &values,
            &tasks,
            step,
            false,
            false,
            sink,
        )?;
    }

    let mut iteration = 0usize;
    loop {
        if iteration >= recursion_limit {
            return Err(Error::RecursionLimit(recursion_limit));
        }
        iteration += 1;

        // `interrupt_before` hooks stop the graph before the node runs. Resuming a
        // checkpoint that was written by such a hook must let the node run instead of
        // pausing on the same hook forever.
        let paused = if skip_interrupt_before {
            None
        } else {
            tasks
                .iter()
                .find(|task| graph.interrupts_before(&task.node))
                .cloned()
        };
        skip_interrupt_before = false;
        if let Some(interrupted) = paused {
            save_checkpoint(
                graph,
                thread_id.as_deref(),
                &mut parent,
                &values,
                &tasks,
                step,
                true,
                true,
                sink,
            )?;
            sink(StreamChunk::Interrupt {
                node: interrupted.node.clone(),
                value: Value::Null,
            });
            return Err(Error::Interrupt(Interrupt {
                node: interrupted.node,
                value: Value::Null,
            }));
        }

        let results = match run_superstep(graph, &state, &tasks) {
            Ok(results) => results,
            Err(error) => {
                if let Error::Interrupt(interrupt) = &error {
                    save_checkpoint(
                        graph,
                        thread_id.as_deref(),
                        &mut parent,
                        &values,
                        &tasks,
                        step,
                        true,
                        false,
                        sink,
                    )?;
                    sink(StreamChunk::Interrupt {
                        node: interrupt.node.clone(),
                        value: interrupt.value.clone(),
                    });
                }
                return Err(error);
            }
        };

        let processed = process_results(graph, results, step, &mut report, sink)?;
        let channel_updates = processed.channel_updates;
        let next_tasks = processed.next_tasks;
        let goto_targets = processed.goto_targets;
        let executed = processed.executed;
        let goto_nodes = processed.goto_nodes;

        apply_updates(graph.schema(), &mut values, channel_updates)?;
        state = values_to_state::<S>(&values)?;
        sink(StreamChunk::Values {
            step,
            values: Value::Object(values.clone()),
        });

        let next_nodes = collect_next_nodes(graph, &state, &executed, &goto_nodes, &goto_targets)?;
        let mut next_tasks = next_tasks;
        for node in dedupe(next_nodes) {
            next_tasks.push(Task::new(node));
        }

        step += 1;
        let interrupt_after = executed
            .iter()
            .find(|node| graph.interrupts_after(node))
            .cloned();
        save_checkpoint(
            graph,
            thread_id.as_deref(),
            &mut parent,
            &values,
            &next_tasks,
            step,
            interrupt_after.is_some(),
            false,
            sink,
        )?;
        if let Some(node) = interrupt_after {
            sink(StreamChunk::Interrupt {
                node: node.clone(),
                value: Value::Null,
            });
            return Err(Error::Interrupt(Interrupt {
                node,
                value: Value::Null,
            }));
        }

        if next_tasks.is_empty() {
            return Ok(finish(state, report, started));
        }
        tasks = next_tasks;
    }
}

fn finish<S>(state: S, mut report: RunReport, started: Instant) -> RunOutcome<S> {
    report.total_duration_millis = started.elapsed().as_millis() as u64;
    RunOutcome { state, report }
}

/// Everything one superstep's node results contribute to the run.
struct ProcessedResults {
    channel_updates: Vec<ChannelUpdate>,
    next_tasks: Vec<Task>,
    goto_targets: Vec<(String, String)>,
    executed: Vec<String>,
    goto_nodes: HashSet<String>,
}

fn process_results<S: State>(
    graph: &CompiledGraph<S>,
    results: Vec<TaskResult>,
    step: usize,
    report: &mut RunReport,
    sink: &mut dyn FnMut(StreamChunk),
) -> Result<ProcessedResults> {
    let mut channel_updates: Vec<ChannelUpdate> = Vec::new();
    let mut next_tasks: Vec<Task> = Vec::new();
    let mut goto_targets: Vec<(String, String)> = Vec::new();
    let mut executed: Vec<String> = Vec::new();
    let mut goto_nodes: HashSet<String> = HashSet::new();

    for result in results {
        let TaskResult {
            node,
            output,
            custom,
            usage,
            duration_millis,
        } = result;
        executed.push(node.clone());
        for value in custom {
            sink(StreamChunk::Custom {
                node: node.clone(),
                value,
            });
        }

        let mut combined: Option<Usage> = None;
        for item in usage {
            combined.get_or_insert_with(Usage::new).merge(&item);
        }
        let execution = NodeExecution {
            node: node.clone(),
            step,
            duration_millis,
            usage: combined,
        };
        sink(StreamChunk::Usage {
            execution: execution.clone(),
        });
        for trace_sink in graph.trace_sinks() {
            trace_sink.record(&execution);
        }
        report.record(execution);

        let (updates, goto, sends) = output.into_parts();
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
            if graph.node(&send.node).is_none() {
                return Err(Error::routing(
                    &node,
                    format!("Send targets unknown node `{}`", send.node),
                ));
            }
            next_tasks.push(Task {
                node: send.node,
                input,
                resume: None,
            });
        }
    }

    Ok(ProcessedResults {
        channel_updates,
        next_tasks,
        goto_targets,
        executed,
        goto_nodes,
    })
}
