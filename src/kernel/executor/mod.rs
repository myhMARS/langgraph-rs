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
pub(crate) mod scheduling;
mod superstep;

use std::collections::HashSet;
use std::time::Instant;

use serde_json::{Map, Value, json};

use self::checkpointing::{CheckpointWriter, save_checkpoint};
use self::scheduling::{collect_next_nodes, dedupe, send_task};
use self::superstep::{TaskFailure, TaskOutcome, TaskResult, run_superstep};
use crate::kernel::error::{Error, Interrupt, Result};
use crate::kernel::graph::{CompiledGraph, Durability, RunConfig};
use crate::kernel::node::Task;
use crate::kernel::state::{ChannelUpdate, State, apply_updates, values_to_state};
use crate::kernel::stream::StreamChunk;
use crate::kernel::usage::{NodeExecution, RunOutcome, RunReport, Usage};

/// Where a run starts from.
pub(crate) enum StartInput {
    /// Full channel values (from a typed input state).
    Values(Map<String, Value>),
    /// Partial writes applied over the default state, starting at explicit nodes
    /// (LangGraph `invoke(Command(goto=...))`).
    Gateways {
        values: Map<String, Value>,
        nodes: Vec<String>,
    },
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
    let namespace = config.checkpoint_ns.clone().unwrap_or_default();
    let durability = config.durability;
    let checkpoint_writer = if durability == Durability::Async {
        graph
            .checkpointer()
            .map(|checkpointer| CheckpointWriter::start(checkpointer.clone()))
    } else {
        None
    };
    let store = config.store.clone().or_else(|| graph.store().cloned());
    if let (Some(schema), Some(context)) = (graph.context_schema(), config.context.as_ref()) {
        crate::kernel::context::validate_response(context, Some(schema)).map_err(|error| {
            Error::validation(format!("context does not match context_schema: {error}"))
        })?;
    }
    let _run_guard = crate::kernel::context::install_run(
        thread_id.clone(),
        namespace.clone(),
        store,
        config.context.clone(),
    );
    let subgraph_collector: Option<std::sync::Arc<std::sync::Mutex<Vec<StreamChunk>>>> = config
        .stream_subgraphs
        .then(|| std::sync::Arc::new(std::sync::Mutex::new(Vec::new())));
    let _collector_guard = subgraph_collector
        .as_ref()
        .map(|collector| crate::kernel::context::install_subgraph_collector(collector.clone()));

    if thread_id.is_some() && graph.checkpointer().is_none() {
        return Err(Error::MissingCheckpointer("a run bound to a thread_id"));
    }

    let fresh = matches!(
        &start,
        StartInput::Values(_) | StartInput::Updates(_) | StartInput::Gateways { .. }
    );
    let mut parent: Option<String> = None;
    let mut skip_interrupt_before = false;

    let (mut values, mut state, mut tasks, mut step) = match start {
        StartInput::Values(values) => {
            let values = graph.schema().filter_input(values);
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
        StartInput::Gateways { values, nodes } => {
            // `Command(goto=...)` values are state updates, not run input.
            for node in &nodes {
                if graph.node(node).is_none() {
                    return Err(Error::validation(format!(
                        "Command(goto=...): `{node}` is not a registered node"
                    )));
                }
            }
            let state = values_to_state::<S>(&values)?;
            let tasks = dedupe(nodes).into_iter().map(Task::new).collect();
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
                None => checkpointer.latest_in(&resume_thread, &namespace)?,
            }
            .ok_or_else(|| {
                Error::Checkpoint(format!("no checkpoint found for thread `{resume_thread}`"))
            })?;
            crate::kernel::context::load_task_results(&checkpoint.task_results);

            parent = Some(checkpoint.id.clone());
            skip_interrupt_before = checkpoint.paused_before;
            let mut values = checkpoint.values.clone();
            apply_pending_writes(graph.schema(), &mut values, &checkpoint.pending_writes)?;
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
            let interrupt_map = match (&resume, checkpoint.interrupts.len() > 1) {
                (Some(Value::Object(map)), true) => Some(map.clone()),
                _ => None,
            };
            let mut tasks = checkpoint.next.clone();
            for task in &mut tasks {
                task.resume = match &interrupt_map {
                    Some(map) => checkpoint
                        .interrupts
                        .iter()
                        .find(|interrupt| interrupt.node == task.node)
                        .and_then(|interrupt| interrupt.resume_from(map)),
                    None => resume.clone(),
                };
            }
            // Fold the new resume value into the task's consumed history so a node that
            // calls interrupt() several times sees earlier answers again on re-runs.
            for task in &mut tasks {
                if let Some(value) = task.resume.take() {
                    task.resume_history.push(value);
                }
            }
            (values, state, tasks, checkpoint.step)
        }
    };

    if tasks.is_empty() {
        return finish(state, graph.schema(), report, started);
    }

    if fresh {
        crate::kernel::context::clear_task_results();
        if let (Some(checkpointer), Some(tid)) = (graph.checkpointer(), thread_id.as_deref()) {
            parent = checkpointer
                .latest_in(tid, &namespace)?
                .map(|checkpoint| checkpoint.id);
        }
        // LangGraph records an input checkpoint before the first superstep.
        save_checkpoint(
            graph,
            thread_id.as_deref(),
            &namespace,
            &mut parent,
            &values,
            &tasks,
            step,
            false,
            false,
            Vec::new(),
            Vec::new(),
            &[],
            sink,
            checkpoint_writer.as_ref(),
        )?;
        save_checkpoint(
            graph,
            thread_id.as_deref(),
            &namespace,
            &mut parent,
            &values,
            &tasks,
            step,
            false,
            false,
            Vec::new(),
            Vec::new(),
            &[],
            sink,
            checkpoint_writer.as_ref(),
        )?;
    }

    let mut pending_deferred: Vec<Task> = Vec::new();
    let mut finish_updates: std::collections::HashMap<String, Value> =
        std::collections::HashMap::new();
    let mut iteration = 0usize;
    loop {
        if config
            .control
            .as_ref()
            .is_some_and(crate::RunControl::is_aborted)
        {
            return Err(Error::other("run aborted by RunControl"));
        }
        apply_managed_values(
            graph.schema(),
            &mut values,
            &mut state,
            step,
            recursion_limit,
        )?;
        if iteration >= recursion_limit {
            return Err(Error::RecursionLimit(recursion_limit));
        }
        iteration += 1;
        let _step_guard = crate::kernel::context::install_step(step, recursion_limit);

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
            let interrupted_interrupt = Interrupt::new(interrupted.node.clone(), Value::Null);
            save_checkpoint(
                graph,
                thread_id.as_deref(),
                &namespace,
                &mut parent,
                &values,
                &tasks,
                step,
                true,
                true,
                vec![interrupted_interrupt.clone()],
                Vec::new(),
                &[],
                sink,
                checkpoint_writer.as_ref(),
            )?;
            sink(StreamChunk::Interrupt {
                node: interrupted.node.clone(),
                value: Value::Null,
            });
            return Err(Error::Interrupt(interrupted_interrupt));
        }

        let outcomes = run_superstep(graph, &state, &tasks);
        if let Some(collector) = &subgraph_collector {
            let chunks: Vec<StreamChunk> = collector
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .drain(..)
                .collect();
            for chunk in chunks {
                sink(chunk);
            }
        }

        // An interrupt aborts the superstep. A single interrupt keeps the pre-superstep
        // values and the whole task list (a resume replays the superstep); several
        // pending interrupts commit the successful writes and keep only the interrupted
        // tasks, resumable through an `{interrupt id: value}` map.
        let interrupts: Vec<Interrupt> = outcomes
            .iter()
            .filter_map(|outcome| match outcome {
                TaskOutcome::Interrupted(interrupt) => Some(interrupt.clone()),
                _ => None,
            })
            .collect();
        if !interrupts.is_empty() {
            if interrupts.len() == 1 {
                let interrupted = interrupts.first().expect("one interrupt").clone();
                let interrupted_ids: HashSet<String> = tasks
                    .iter()
                    .zip(&outcomes)
                    .filter(|(_, outcome)| matches!(outcome, TaskOutcome::Interrupted(_)))
                    .map(|(task, _)| task.id.clone())
                    .collect();
                let pending_writes = collect_pending_writes(&tasks, &outcomes);
                let successful: Vec<TaskOutcome> = outcomes
                    .into_iter()
                    .filter(|outcome| !matches!(outcome, TaskOutcome::Interrupted(_)))
                    .collect();
                let processed = process_outcomes(graph, successful, step, &mut report, sink)?;
                let mut next_tasks: Vec<Task> = tasks
                    .iter()
                    .filter(|task| interrupted_ids.contains(&task.id))
                    .cloned()
                    .collect();
                for task in processed.next_tasks {
                    if !next_tasks.iter().any(|existing| existing.id == task.id) {
                        next_tasks.push(task);
                    }
                }
                save_checkpoint(
                    graph,
                    thread_id.as_deref(),
                    &namespace,
                    &mut parent,
                    &values,
                    &next_tasks,
                    step,
                    true,
                    false,
                    vec![interrupted.clone()],
                    pending_writes,
                    &[],
                    sink,
                    checkpoint_writer.as_ref(),
                )?;
                sink(StreamChunk::Interrupt {
                    node: interrupted.node.clone(),
                    value: interrupted.value.clone(),
                });
                return Err(Error::Interrupt(interrupted));
            }
            let interrupted_nodes: HashSet<String> = interrupts
                .iter()
                .map(|interrupt| interrupt.node.clone())
                .collect();
            let successful: Vec<TaskOutcome> = outcomes
                .into_iter()
                .filter(|outcome| !matches!(outcome, TaskOutcome::Interrupted(_)))
                .collect();
            let processed = process_outcomes(graph, successful, step, &mut report, sink)?;
            let mut immediate = Vec::with_capacity(processed.channel_updates.len());
            for update in processed.channel_updates {
                if graph.schema().is_finish_value(&update.channel) {
                    finish_updates.insert(update.channel, update.value);
                } else {
                    immediate.push(update);
                }
            }
            apply_updates(graph.schema(), &mut values, immediate)?;
            for (channel, value) in processed.replacements {
                graph.schema().resolve(&channel)?;
                values.insert(channel, value);
            }
            sink(StreamChunk::Values {
                step,
                values: Value::Object(values.clone()),
            });
            let mut next_tasks: Vec<Task> = tasks
                .iter()
                .filter(|task| interrupted_nodes.contains(&task.node))
                .cloned()
                .collect();
            for task in processed.next_tasks {
                if !next_tasks.iter().any(|existing| existing.node == task.node) {
                    next_tasks.push(task);
                }
            }
            save_checkpoint(
                graph,
                thread_id.as_deref(),
                &namespace,
                &mut parent,
                &values,
                &next_tasks,
                step,
                true,
                false,
                interrupts.clone(),
                Vec::new(),
                &[],
                sink,
                checkpoint_writer.as_ref(),
            )?;
            for interrupt in &interrupts {
                sink(StreamChunk::Interrupt {
                    node: interrupt.node.clone(),
                    value: interrupt.value.clone(),
                });
            }
            return Err(Error::Interrupt(
                interrupts.into_iter().next().expect("interrupt"),
            ));
        }

        let processed = process_outcomes(graph, outcomes, step, &mut report, sink)?;
        let channel_updates = processed.channel_updates;
        let replacements = processed.replacements;
        let next_tasks = processed.next_tasks;
        let goto_targets = processed.goto_targets;
        let executed = processed.executed;
        let goto_nodes = processed.goto_nodes;

        let mut immediate_updates = Vec::with_capacity(channel_updates.len());
        for update in channel_updates {
            if graph.schema().is_finish_value(&update.channel) {
                finish_updates.insert(update.channel, update.value);
            } else {
                immediate_updates.push(update);
            }
        }
        apply_updates(graph.schema(), &mut values, immediate_updates)?;
        // Direct replacements (subgraph outputs) bypass reducers but must still target
        // a declared channel when the schema is strict.
        for (channel, value) in replacements {
            graph.schema().resolve(&channel)?;
            values.insert(channel, value);
        }
        state = values_to_state::<S>(&values)?;
        sink(StreamChunk::Values {
            step,
            values: Value::Object(values.clone()),
        });
        // `Command(graph=PARENT)` aborts this (sub)graph; the embedding subgraph node
        // forwards the writes and targets into the parent graph.
        if crate::kernel::context::has_parent_command() {
            return finish(state, graph.schema(), report, started);
        }

        let (next_nodes, router_sends) = collect_next_nodes(
            &state,
            &executed,
            &goto_nodes,
            &goto_targets,
            |node| graph.edges_of(node),
            |node| graph.conditional_of(node),
            |node| graph.node(node).is_some(),
            |node| graph.destinations_of(node),
        )?;
        let mut next_tasks = next_tasks;
        // A node that already has an error-handler task in this superstep must not run
        // a second time through a regular edge.
        let handler_nodes: HashSet<String> = next_tasks
            .iter()
            .filter(|task| task.error.is_some())
            .map(|task| task.node.clone())
            .collect();
        for node in dedupe(next_nodes) {
            if handler_nodes.contains(&node) {
                continue;
            }
            next_tasks.push(Task::new(node));
        }
        for (source, send) in router_sends {
            let task = send_task(&source, send)?;
            if graph.node(&task.node).is_none() {
                return Err(Error::routing(
                    &source,
                    format!("unknown destination node `{}`", task.node),
                ));
            }
            if let Some(allowed) = graph.destinations_of(&source) {
                if !allowed.iter().any(|node| node == &task.node) {
                    return Err(Error::routing(
                        &source,
                        format!("`{}` is not a declared destination", task.node),
                    ));
                }
            }
            next_tasks.push(task);
        }

        let (deferred_tasks, regular_tasks): (Vec<Task>, Vec<Task>) = next_tasks
            .into_iter()
            .partition(|task| graph.is_deferred(&task.node));
        pending_deferred.extend(deferred_tasks);
        let next_tasks = if regular_tasks.is_empty() && !pending_deferred.is_empty() {
            std::mem::take(&mut pending_deferred)
        } else {
            regular_tasks
        };

        let scheduled: Vec<String> = next_tasks.iter().map(|task| task.node.clone()).collect();
        sink(StreamChunk::Debug {
            step,
            payload: json!({
                "step": step,
                "executed": executed,
                "next": scheduled,
            }),
        });

        step += 1;
        let interrupt_after = executed
            .iter()
            .find(|node| graph.interrupts_after(node))
            .cloned();
        if durability != Durability::Exit || interrupt_after.is_some() {
            save_checkpoint(
                graph,
                thread_id.as_deref(),
                &namespace,
                &mut parent,
                &values,
                &next_tasks,
                step,
                interrupt_after.is_some(),
                false,
                interrupt_after
                    .iter()
                    .map(|node| Interrupt::new(node.clone(), Value::Null))
                    .collect(),
                Vec::new(),
                &executed,
                sink,
                checkpoint_writer.as_ref(),
            )?;
        }
        if let Some(node) = interrupt_after {
            sink(StreamChunk::Interrupt {
                node: node.clone(),
                value: Value::Null,
            });
            return Err(Error::Interrupt(Interrupt::new(node, Value::Null)));
        }

        if next_tasks.is_empty() {
            if !finish_updates.is_empty() {
                for (channel, value) in finish_updates.drain() {
                    values.insert(channel, value);
                }
                state = values_to_state::<S>(&values)?;
            }
            if durability == Durability::Exit && thread_id.is_some() {
                save_checkpoint(
                    graph,
                    thread_id.as_deref(),
                    &namespace,
                    &mut parent,
                    &values,
                    &[],
                    step,
                    false,
                    false,
                    Vec::new(),
                    Vec::new(),
                    &executed,
                    sink,
                    checkpoint_writer.as_ref(),
                )?;
            }
            return finish(state, graph.schema(), report, started);
        }
        // Ephemeral channels are visible for exactly one superstep.
        graph.schema().clear_ephemeral(&mut values);
        state = values_to_state::<S>(&values)?;
        tasks = next_tasks;
    }
}

/// Inject managed values (`remaining_steps`, `is_last_step`) before a superstep.
pub(crate) fn apply_managed_values<S: State>(
    schema: &crate::kernel::state::StateSchema,
    values: &mut Map<String, Value>,
    state: &mut S,
    step: usize,
    recursion_limit: usize,
) -> Result<()> {
    if !schema.managed_values() {
        return Ok(());
    }
    let remaining = recursion_limit.saturating_sub(step);
    values.insert("remaining_steps".to_string(), json!(remaining));
    values.insert("is_last_step".to_string(), json!(remaining <= 1));
    *state = values_to_state::<S>(values)?;
    Ok(())
}

fn finish<S: State>(
    state: S,
    schema: &crate::kernel::state::StateSchema,
    mut report: RunReport,
    started: Instant,
) -> Result<RunOutcome<S>> {
    report.total_duration_millis = started.elapsed().as_millis() as u64;
    let state = if schema.has_output_filter() {
        let values = crate::kernel::state::state_to_values(&state)?;
        values_to_state::<S>(&schema.output_values(&values))?
    } else {
        state
    };
    Ok(RunOutcome { state, report })
}

/// Everything one superstep's node outcomes contribute to the run.
struct ProcessedResults {
    channel_updates: Vec<ChannelUpdate>,
    replacements: Vec<(String, Value)>,
    next_tasks: Vec<Task>,
    goto_targets: Vec<(String, String)>,
    executed: Vec<String>,
    goto_nodes: HashSet<String>,
}

fn merge_usage(usage: Vec<Usage>) -> Option<Usage> {
    let mut combined: Option<Usage> = None;
    for item in usage {
        combined.get_or_insert_with(Usage::new).merge(&item);
    }
    combined
}

fn collect_pending_writes(
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

pub(crate) fn apply_pending_writes(
    schema: &crate::kernel::state::StateSchema,
    values: &mut Map<String, Value>,
    writes: &[crate::kernel::checkpoint::PendingWrite],
) -> Result<()> {
    if writes.is_empty() {
        return Ok(());
    }
    let updates: Vec<ChannelUpdate> = writes
        .iter()
        .filter(|write| !write.replace && !schema.is_finish_value(&write.channel))
        .map(|write| ChannelUpdate {
            channel: write.channel.clone(),
            value: write.value.clone(),
            node: write.node.clone(),
        })
        .collect();
    apply_updates(schema, values, updates)?;
    for write in writes {
        if write.replace || schema.is_finish_value(&write.channel) {
            schema.resolve(&write.channel)?;
            values.insert(write.channel.clone(), write.value.clone());
        }
    }
    Ok(())
}

fn process_outcomes<S: State>(
    graph: &CompiledGraph<S>,
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
                    Some(handler) if routed_handlers.insert(handler.clone()) => {
                        next_tasks.push(Task {
                            id: crate::kernel::node::new_task_id(handler.as_str()),
                            node: handler.clone(),
                            input: None,
                            resume: None,
                            resume_history: Vec::new(),
                            error: Some(failure),
                            timeout_millis: None,
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
                    let mut parent_updates: Vec<(String, Value)> = updates.into_iter().collect();
                    parent_updates.extend(replaces);
                    crate::kernel::context::record_parent_command(parent_updates, goto);
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
                    if graph.node(&send.node).is_none() {
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
