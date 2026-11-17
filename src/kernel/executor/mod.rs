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

pub(crate) mod checkpointing;
pub(crate) mod outcome;
pub(crate) mod scheduling;
pub(crate) mod start;
pub(crate) mod step;
mod superstep;

use std::collections::HashSet;
use std::time::Instant;

use serde_json::{Map, Value, json};

use self::checkpointing::{CheckpointWriter, save_checkpoint};
use self::outcome::{
    TaskOutcome, apply_finish_updates, apply_processed_updates, collect_pending_writes,
    debug_chunk, finish, interrupt_after_node, interrupts_from, merge_interrupted_tasks,
    process_outcomes, should_save_after_step, should_save_terminal, values_chunk,
};
use self::scheduling::{collect_next_nodes, schedule_next_tasks};
use self::superstep::run_superstep;
use crate::kernel::error::{Error, Interrupt, Result};
use crate::kernel::graph::{CompiledGraph, Durability, RunConfig};
use crate::kernel::node::Task;
use crate::kernel::state::{ChannelUpdate, State, apply_updates, values_to_state};
use crate::kernel::stream::StreamChunk;
use crate::kernel::usage::{RunOutcome, RunReport};

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

    let (mut values, mut state, mut tasks, mut step) = if let Some((values, state, tasks)) =
        start::resolve_fresh_start(
            &start,
            graph.schema(),
            |node| graph.node(node).is_some(),
            |state| graph.entry_tasks(state),
        )? {
        (values, state, tasks, 0usize)
    } else {
        let StartInput::Resume {
            thread_id: resume_thread,
            checkpoint_id,
            resume,
            updates,
        } = start
        else {
            unreachable!("fresh start inputs were resolved above");
        };
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
        parent = Some(checkpoint.id.clone());
        skip_interrupt_before = checkpoint.paused_before;
        let (values, state, tasks) =
            start::resolve_resume_start(&checkpoint, &resume, updates, graph.schema())?;
        (values, state, tasks, checkpoint.step)
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
        let prepared = step::begin_step(
            config.control.as_ref(),
            graph.schema(),
            &mut values,
            &mut state,
            step,
            recursion_limit,
            iteration,
            &tasks,
            skip_interrupt_before,
            |node| graph.interrupts_before(node),
        )?;
        iteration = prepared.iteration;
        let _step_guard = prepared.guard;
        let paused = prepared.paused;
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
        let interrupts = interrupts_from(&outcomes);
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
                let next_tasks = merge_interrupted_tasks(
                    &tasks,
                    |task| interrupted_ids.contains(&task.id),
                    processed.next_tasks,
                    true,
                );
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
            apply_processed_updates(
                graph.schema(),
                &mut values,
                processed.channel_updates,
                processed.replacements,
                &mut finish_updates,
            )?;
            sink(StreamChunk::Values {
                step,
                values: Value::Object(values.clone()),
            });
            let next_tasks = merge_interrupted_tasks(
                &tasks,
                |task| interrupted_nodes.contains(&task.node),
                processed.next_tasks,
                false,
            );
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

        // Direct replacements (subgraph outputs) bypass reducers but must still target
        // a declared channel when the schema is strict.
        apply_processed_updates(
            graph.schema(),
            &mut values,
            channel_updates,
            replacements,
            &mut finish_updates,
        )?;
        state = values_to_state::<S>(&values)?;
        sink(values_chunk(step, &values));
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
        let next_tasks = schedule_next_tasks(
            next_nodes,
            router_sends,
            next_tasks,
            |node| graph.node(node).is_some(),
            |node| graph.destinations_of(node),
            |node| graph.is_deferred(node),
            &mut pending_deferred,
        )?;

        let scheduled: Vec<String> = next_tasks.iter().map(|task| task.node.clone()).collect();
        sink(debug_chunk(step, &executed, &scheduled));

        step += 1;
        let interrupt_after = interrupt_after_node(&executed, |node| graph.interrupts_after(node));
        if should_save_after_step(durability, interrupt_after.is_some()) {
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
                apply_finish_updates(&mut values, &mut finish_updates);
                state = values_to_state::<S>(&values)?;
            }
            if should_save_terminal(durability, thread_id.is_some()) {
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
    let remaining = recursion_limit.saturating_sub(step + 1);
    values.insert("remaining_steps".to_string(), json!(remaining));
    values.insert("is_last_step".to_string(), json!(remaining <= 1));
    *state = values_to_state::<S>(values)?;
    Ok(())
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
