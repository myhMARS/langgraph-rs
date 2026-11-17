//! One Pregel loop shared by the sync and async runtimes.
//!
//! All superstep, checkpoint and interrupt decisions live here. A [`PregelBackend`]
//! supplies graph metadata, node-batch execution and checkpoint persistence; the sync
//! runtime drives this loop with a nested-safe `block_on`, the async runtime
//! awaits it.

use std::collections::{HashMap, HashSet};

use std::time::Instant;

use serde_json::Value;

use crate::kernel::context;
use crate::kernel::error::{Error, Interrupt, Result};
use crate::kernel::executor::backend::PregelBackend;
use crate::kernel::executor::checkpointing::save_checkpoint_via;
use crate::kernel::executor::outcome::{
    TaskOutcome, apply_finish_updates, apply_processed_updates, collect_pending_writes,
    debug_chunk, finish, interrupt_after_node, interrupt_before_pause, interrupts_from,
    merge_interrupted_tasks, process_outcomes, should_save_after_step, should_save_terminal,
    values_chunk,
};
use crate::kernel::executor::scheduling::{collect_next_nodes, schedule_next_tasks};
use crate::kernel::executor::{StartInput, apply_pending_writes, start, step};
use crate::kernel::graph::RunConfig;
use crate::kernel::node::Task;
use crate::kernel::state::{State, values_to_state};
use crate::kernel::stream::StreamChunk;
use crate::kernel::usage::RunOutcome;

/// Execute a graph through the shared Pregel loop.
pub(crate) async fn run_loop<S: State, B: PregelBackend<S>, F: FnMut(StreamChunk)>(
    backend: &B,
    start: StartInput,
    config: &RunConfig,
    sink: &mut F,
    emit: bool,
) -> Result<RunOutcome<S>> {
    let started = Instant::now();
    let mut report = crate::kernel::usage::RunReport::new();
    let recursion_limit = config
        .recursion_limit
        .unwrap_or_else(|| backend.recursion_limit());
    let thread_id = config.thread_id.clone();
    let namespace = config.checkpoint_ns.clone().unwrap_or_default();
    let durability = config.durability;
    let store = config.store.clone().or_else(|| backend.store());
    let setup = step::begin_run(
        thread_id.clone(),
        namespace.clone(),
        store,
        config.context.clone(),
        backend.context_schema(),
        config.stream_subgraphs,
        backend.collector(),
    )?;

    if thread_id.is_some() && !backend.has_checkpointer() {
        return Err(Error::MissingCheckpointer("a run bound to a thread_id"));
    }

    let fresh = matches!(
        &start,
        StartInput::Values(_) | StartInput::Updates(_) | StartInput::Gateways { .. }
    );
    let mut parent: Option<String> = None;
    let mut previous_checkpoint: Option<crate::kernel::checkpoint::Checkpoint> = None;
    let mut skip_interrupt_before = false;
    let (mut values, mut state, mut tasks, mut step, mut resume_deferred, resume_finish) =
        if let Some((values, state, tasks)) = start::resolve_fresh_start(
            &start,
            backend.schema(),
            |node| backend.node_exists(node),
            |state| backend.entry_tasks(state),
        )? {
            (values, state, tasks, 0usize, Vec::new(), Vec::new())
        } else {
            let StartInput::Resume {
                thread_id: resume_thread,
                checkpoint_id,
                resume,
                updates,
                goto: resume_goto,
            } = start
            else {
                unreachable!("fresh start inputs were resolved above");
            };
            let checkpoint = backend
                .load_checkpoint(&resume_thread, checkpoint_id.as_deref(), &namespace)
                .await?
                .ok_or_else(|| {
                    Error::Checkpoint(format!("no checkpoint found for thread `{resume_thread}`"))
                })?;
            parent = Some(checkpoint.id.clone());
            previous_checkpoint = Some(checkpoint.clone());
            skip_interrupt_before = checkpoint.paused_before;
            let (values, state, tasks, finish_writes) = start::resolve_resume_start(
                &checkpoint,
                &resume,
                updates,
                &resume_goto,
                backend.schema(),
            )?;
            let deferred = checkpoint.deferred.clone();
            (
                values,
                state,
                tasks,
                checkpoint.step,
                deferred,
                finish_writes,
            )
        };

    if tasks.is_empty() && !resume_deferred.is_empty() {
        tasks = std::mem::take(&mut resume_deferred);
    }
    if tasks.is_empty() {
        let mut state = state;
        if !resume_finish.is_empty() {
            for write in &resume_finish {
                values.insert(write.channel.clone(), write.value.clone());
            }
            state = values_to_state::<S>(&values)?;
        }
        return finish(state, backend.schema(), report, started);
    }
    if config
        .control
        .as_ref()
        .is_some_and(crate::RunControl::is_aborted)
    {
        return Err(Error::other("run aborted by RunControl"));
    }

    if fresh {
        context::clear_task_results();
        if let Some(tid) = thread_id.as_deref() {
            if let Some(latest) = backend.load_checkpoint(tid, None, &namespace).await? {
                parent = Some(latest.id.clone());
                previous_checkpoint = Some(latest);
            }
        }
        // LangGraph records the input checkpoint(s) before the first superstep. The
        // two records and the `total - 2` history step math are covered by checkpoint
        // and namespace tests; keep them in sync if the initial layout changes.
        save_checkpoint_via(
            backend,
            thread_id.as_deref(),
            &namespace,
            &mut parent,
            &mut previous_checkpoint,
            &values,
            &tasks,
            step,
            false,
            false,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[],
            sink,
        )
        .await?;
        save_checkpoint_via(
            backend,
            thread_id.as_deref(),
            &namespace,
            &mut parent,
            &mut previous_checkpoint,
            &values,
            &tasks,
            step,
            false,
            false,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[],
            sink,
        )
        .await?;
    }

    let mut pending_deferred: Vec<Task> = resume_deferred;
    let mut finish_updates: HashMap<String, Value> = resume_finish
        .into_iter()
        .map(|write| (write.channel, write.value))
        .collect();
    let mut iteration = 0usize;
    loop {
        let prepared = step::begin_step(
            config.control.as_ref(),
            backend.schema(),
            &mut values,
            &mut state,
            step,
            recursion_limit,
            iteration,
            &tasks,
            skip_interrupt_before,
            |node| backend.interrupts_before(node),
        )?;
        iteration = prepared.iteration;
        let _step_guard = prepared.guard;
        let paused = prepared.paused;
        skip_interrupt_before = false;
        if let Some(interrupted) = paused {
            let (interrupted_interrupt, chunk) = interrupt_before_pause(&interrupted);
            let paused_tasks: Vec<Task> = tasks
                .iter()
                .map(|task| {
                    let mut task = task.clone();
                    task.interrupted = task.id == interrupted.id;
                    task
                })
                .collect();
            save_checkpoint_via(
                backend,
                thread_id.as_deref(),
                &namespace,
                &mut parent,
                &mut previous_checkpoint,
                &values,
                &paused_tasks,
                step,
                true,
                true,
                vec![interrupted_interrupt.clone()],
                finish_pending(&finish_updates, backend.schema()),
                pending_deferred.clone(),
                &[],
                sink,
            )
            .await?;
            sink(chunk);
            return Err(Error::Interrupt(interrupted_interrupt));
        }

        let outcomes = match backend
            .run_superstep(&state, &tasks, step, config.control.as_ref())
            .await
        {
            Ok(outcomes) => outcomes,
            Err(error) => {
                drain_collector(backend, &setup, sink);
                return Err(error);
            }
        };
        if emit && let Some(collector) = backend.collector().or_else(|| setup.collector().cloned())
        {
            let chunks: Vec<StreamChunk> = collector
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .drain(..)
                .collect();
            for chunk in chunks {
                sink(chunk);
            }
        }

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
                let mut pending_writes = finish_pending(&finish_updates, backend.schema());
                pending_writes.extend(
                    collect_pending_writes(&tasks, &outcomes)
                        .into_iter()
                        .filter(|write| backend.schema().persists(&write.channel)),
                );
                let successful: Vec<TaskOutcome> = outcomes
                    .into_iter()
                    .filter(|outcome| !matches!(outcome, TaskOutcome::Interrupted(_)))
                    .collect();
                let processed =
                    process_outcomes(backend, successful, step, &mut report, sink, emit)?;
                // Successful siblings still contribute their outgoing routes even though
                // their values stay pending until resume.
                let mut routing_values = values.clone();
                let routing_writes: Vec<_> = pending_writes
                    .iter()
                    .filter(|write| !backend.schema().is_finish_value(&write.channel))
                    .cloned()
                    .collect();
                apply_pending_writes(backend.schema(), &mut routing_values, &routing_writes)?;
                let routing_state = values_to_state::<S>(&routing_values)?;
                let (next_nodes, router_sends) = collect_next_nodes(
                    &routing_state,
                    &processed.executed,
                    &processed.goto_nodes,
                    &processed.goto_targets,
                    |node| backend.edges_of(node),
                    |node| backend.conditional_of(node),
                    |node| backend.node_exists(node),
                    |node| PregelBackend::destinations_of(backend, node),
                )?;
                let scheduled = schedule_next_tasks(
                    next_nodes,
                    router_sends,
                    processed.next_tasks,
                    |node| backend.node_exists(node),
                    |node| PregelBackend::destinations_of(backend, node),
                    |node| backend.is_deferred(node),
                    &mut pending_deferred,
                    false,
                )?;
                let mut next_tasks = merge_interrupted_tasks(
                    &tasks,
                    |task| interrupted_ids.contains(&task.id),
                    scheduled,
                );
                for task in &mut next_tasks {
                    if interrupted_ids.contains(&task.id) {
                        task.interrupted = true;
                    }
                }
                save_checkpoint_via(
                    backend,
                    thread_id.as_deref(),
                    &namespace,
                    &mut parent,
                    &mut previous_checkpoint,
                    &values,
                    &next_tasks,
                    step,
                    true,
                    false,
                    vec![interrupted.clone()],
                    pending_writes,
                    pending_deferred.clone(),
                    &[],
                    sink,
                )
                .await?;
                sink(StreamChunk::Values {
                    step,
                    values: Value::Object(values.clone()),
                });
                sink(StreamChunk::Interrupt {
                    node: interrupted.node.clone(),
                    id: interrupted.id.clone(),
                    value: interrupted.value.clone(),
                });
                return Err(Error::Interrupt(interrupted));
            }
            let interrupted_task_ids: HashSet<String> = tasks
                .iter()
                .zip(&outcomes)
                .filter(|(_, outcome)| matches!(outcome, TaskOutcome::Interrupted(_)))
                .map(|(task, _)| task.id.clone())
                .collect();
            let successful: Vec<TaskOutcome> = outcomes
                .into_iter()
                .filter(|outcome| !matches!(outcome, TaskOutcome::Interrupted(_)))
                .collect();
            let processed = process_outcomes(backend, successful, step, &mut report, sink, emit)?;
            apply_processed_updates(
                backend.schema(),
                &mut values,
                processed.channel_updates,
                processed.replacements,
                &mut finish_updates,
            )?;
            if emit {
                sink(values_chunk(step, &values));
            }
            state = values_to_state::<S>(&values)?;
            let finish_writes = finish_pending(&finish_updates, backend.schema());
            let (next_nodes, router_sends) = collect_next_nodes(
                &state,
                &processed.executed,
                &processed.goto_nodes,
                &processed.goto_targets,
                |node| backend.edges_of(node),
                |node| backend.conditional_of(node),
                |node| backend.node_exists(node),
                |node| PregelBackend::destinations_of(backend, node),
            )?;
            let scheduled = schedule_next_tasks(
                next_nodes,
                router_sends,
                processed.next_tasks,
                |node| backend.node_exists(node),
                |node| PregelBackend::destinations_of(backend, node),
                |node| backend.is_deferred(node),
                &mut pending_deferred,
                false,
            )?;
            let mut next_tasks = merge_interrupted_tasks(
                &tasks,
                |task| interrupted_task_ids.contains(&task.id),
                scheduled,
            );
            for task in &mut next_tasks {
                if interrupted_task_ids.contains(&task.id) {
                    task.interrupted = true;
                }
            }
            save_checkpoint_via(
                backend,
                thread_id.as_deref(),
                &namespace,
                &mut parent,
                &mut previous_checkpoint,
                &values,
                &next_tasks,
                step,
                true,
                false,
                interrupts.clone(),
                finish_writes,
                pending_deferred.clone(),
                &[],
                sink,
            )
            .await?;
            for interrupt in &interrupts {
                sink(StreamChunk::Values {
                    step,
                    values: Value::Object(values.clone()),
                });
                sink(StreamChunk::Interrupt {
                    node: interrupt.node.clone(),
                    id: interrupt.id.clone(),
                    value: interrupt.value.clone(),
                });
            }
            return Err(Error::Interrupt(
                interrupts.into_iter().next().expect("interrupt"),
            ));
        }

        let processed = process_outcomes(backend, outcomes, step, &mut report, sink, emit)?;
        let channel_updates = processed.channel_updates;
        let replacements = processed.replacements;
        let next_tasks = processed.next_tasks;
        let goto_targets = processed.goto_targets;
        let executed = processed.executed;
        let goto_nodes = processed.goto_nodes;

        apply_processed_updates(
            backend.schema(),
            &mut values,
            channel_updates,
            replacements,
            &mut finish_updates,
        )?;
        state = values_to_state::<S>(&values)?;
        if emit {
            sink(values_chunk(step, &values));
        }
        if backend.has_parent_command() {
            if namespace.is_empty() {
                return Err(Error::validation(
                    "Command(graph=PARENT) can only be returned from inside a subgraph",
                ));
            }
            return finish(state, backend.schema(), report, started);
        }

        let (next_nodes, router_sends) = collect_next_nodes(
            &state,
            &executed,
            &goto_nodes,
            &goto_targets,
            |node| backend.edges_of(node),
            |node| backend.conditional_of(node),
            |node| backend.node_exists(node),
            |node| PregelBackend::destinations_of(backend, node),
        )?;
        let next_tasks = schedule_next_tasks(
            next_nodes,
            router_sends,
            next_tasks,
            |node| backend.node_exists(node),
            |node| PregelBackend::destinations_of(backend, node),
            |node| backend.is_deferred(node),
            &mut pending_deferred,
            true,
        )?;

        let scheduled: Vec<String> = next_tasks.iter().map(|task| task.node.clone()).collect();
        if emit {
            sink(debug_chunk(
                step,
                &executed,
                &scheduled,
                &Value::Object(values.clone()),
                thread_id.as_deref(),
                &namespace,
                parent.as_deref(),
            ));
        }

        step += 1;
        let interrupt_after =
            interrupt_after_node(&executed, |node| backend.interrupts_after(node))
                .map(|node| Interrupt::new(node, Value::Null));
        // A finished graph publishes held finish-channel values before its final checkpoint,
        // so the latest checkpoint matches the returned state.
        if next_tasks.is_empty() && interrupt_after.is_none() && !finish_updates.is_empty() {
            apply_finish_updates(&mut values, &mut finish_updates);
            state = values_to_state::<S>(&values)?;
        }
        if should_save_after_step(durability, interrupt_after.is_some()) {
            save_checkpoint_via(
                backend,
                thread_id.as_deref(),
                &namespace,
                &mut parent,
                &mut previous_checkpoint,
                &values,
                &next_tasks,
                step,
                interrupt_after.is_some(),
                false,
                interrupt_after.clone().into_iter().collect(),
                finish_pending(&finish_updates, backend.schema()),
                pending_deferred.clone(),
                &executed,
                sink,
            )
            .await?;
        }
        if let Some(interrupt) = interrupt_after {
            sink(StreamChunk::Values {
                step,
                values: Value::Object(values.clone()),
            });
            sink(StreamChunk::Interrupt {
                node: interrupt.node.clone(),
                id: interrupt.id.clone(),
                value: Value::Null,
            });
            return Err(Error::Interrupt(interrupt));
        }

        if next_tasks.is_empty() {
            if !finish_updates.is_empty() {
                apply_finish_updates(&mut values, &mut finish_updates);
                state = values_to_state::<S>(&values)?;
            }
            if should_save_terminal(durability, thread_id.is_some()) {
                save_checkpoint_via(
                    backend,
                    thread_id.as_deref(),
                    &namespace,
                    &mut parent,
                    &mut previous_checkpoint,
                    &values,
                    &[],
                    step,
                    false,
                    false,
                    Vec::new(),
                    Vec::new(),
                    pending_deferred.clone(),
                    &executed,
                    sink,
                )
                .await?;
            }
            return finish(state, backend.schema(), report, started);
        }
        // Ephemeral channels are visible for exactly one superstep.
        if backend.clear_ephemeral(&mut values) {
            state = values_to_state::<S>(&values)?;
        }
        tasks = next_tasks;
    }
}

#[path = "pregel_support.rs"]
mod pregel_support;
pub(crate) use pregel_support::*;
