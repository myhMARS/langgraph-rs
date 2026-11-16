//! The superstep execution engine.

use std::collections::HashSet;
use std::panic;

use serde_json::{Map, Value};

use crate::checkpoint::{Checkpoint, new_checkpoint_id, now_millis};
use crate::context::TaskContext;
use crate::error::{Error, Interrupt, Result};
use crate::graph::{CompiledGraph, RunConfig};
use crate::node::{NodeOutput, Task};
use crate::state::{ChannelUpdate, State, apply_updates, values_to_state};
use crate::stream::StreamChunk;
use crate::{END, START};

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

struct TaskResult {
    node: String,
    output: NodeOutput,
    custom: Vec<Value>,
}

/// Execute a graph.
pub(crate) fn execute<S: State>(
    graph: &CompiledGraph<S>,
    start: StartInput,
    config: &RunConfig,
    sink: &mut dyn FnMut(StreamChunk),
) -> Result<S> {
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
        return Ok(state);
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
            } = result;
            executed.push(node.clone());
            for value in custom {
                sink(StreamChunk::Custom {
                    node: node.clone(),
                    value,
                });
            }

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
                if !goto_nodes.contains(&node) {
                    goto_nodes.insert(node.clone());
                }
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

        apply_updates(graph.schema(), &mut values, channel_updates)?;
        state = values_to_state::<S>(&values)?;
        sink(StreamChunk::Values {
            step,
            values: Value::Object(values.clone()),
        });

        let mut next_nodes: Vec<String> = Vec::new();
        for node in &executed {
            // An explicit `goto` replaces the static/conditional edges of that node.
            if goto_nodes.contains(node) {
                continue;
            }
            if let Some(targets) = graph.edges_of(node) {
                for target in targets {
                    check_target(graph, node, target)?;
                    next_nodes.push(target.clone());
                }
            }
            if let Some(edges) = graph.conditional_of(node) {
                for edge in edges {
                    for target in edge.resolve(&state, node)? {
                        check_target(graph, node, &target)?;
                        next_nodes.push(target);
                    }
                }
            }
        }
        for (target, source) in goto_targets {
            if target == END {
                continue;
            }
            check_target(graph, &source, &target)?;
            next_nodes.push(target);
        }

        let mut seen = HashSet::new();
        for node in next_nodes {
            if node == END || node == START {
                continue;
            }
            if seen.insert(node.clone()) {
                next_tasks.push(Task::new(node));
            }
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
            return Ok(state);
        }
        tasks = next_tasks;
    }
}

fn run_superstep<S: State>(
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
    let (result, custom) = crate::context::with_task(context, || node(&node_state));

    match result {
        Ok(output) => Ok(TaskResult {
            node: task.node.clone(),
            output,
            custom,
        }),
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

fn check_target<S: State>(graph: &CompiledGraph<S>, source: &str, target: &str) -> Result<()> {
    if target == END {
        return Ok(());
    }
    if target == START {
        return Err(Error::routing(
            source,
            format!("`{START}` is not a valid destination"),
        ));
    }
    if graph.node(target).is_none() {
        return Err(Error::routing(
            source,
            format!("unknown destination node `{target}`"),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn save_checkpoint<S: State>(
    graph: &CompiledGraph<S>,
    thread_id: Option<&str>,
    parent: &mut Option<String>,
    values: &Map<String, Value>,
    next: &[Task],
    step: usize,
    interrupted: bool,
    paused_before: bool,
    sink: &mut dyn FnMut(StreamChunk),
) -> Result<()> {
    let Some(thread_id) = thread_id else {
        return Ok(());
    };
    let Some(checkpointer) = graph.checkpointer() else {
        return Ok(());
    };

    let id = new_checkpoint_id(thread_id, step);
    let checkpoint = Checkpoint {
        id: id.clone(),
        thread_id: thread_id.to_string(),
        step,
        values: values.clone(),
        next: next.to_vec(),
        interrupted,
        paused_before,
        parent_id: parent.clone(),
        created_at_millis: now_millis(),
    };
    checkpointer.put(checkpoint)?;
    *parent = Some(id.clone());
    sink(StreamChunk::Checkpoint { checkpoint_id: id });
    Ok(())
}
