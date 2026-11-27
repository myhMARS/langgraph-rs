//! One Pregel loop shared by the sync and async runtimes.
//!
//! All superstep, checkpoint and interrupt decisions live here. A [`PregelBackend`]
//! supplies graph metadata, node-batch execution and checkpoint persistence; the sync
//! runtime drives this loop with a nested-safe `block_on`, the async runtime
//! awaits it.

use std::collections::{HashMap, HashSet};

use std::time::Instant;

use serde_json::Value;

use crate::_internal::error::{Error, Interrupt, Result};
use crate::_internal::node::Task;
use crate::_internal::state::{State, values_to_state};
use crate::_internal::stream::{StreamChunk, StreamMode};
use crate::_internal::usage::RunOutcome;
use crate::graph::RunConfig;
use crate::pregel::backend::PregelBackend;
use crate::pregel::outcome::{
    apply_finish_updates, apply_processed_updates, finish, interrupt_after_node, interrupts_from,
    process_outcomes, should_save_after_step, should_save_terminal, values_chunk,
};
use crate::pregel::scheduling::{collect_next_nodes, schedule_next_tasks};
use crate::pregel::{StartInput, start, step};

/// Execute a graph through the shared Pregel loop.
pub(crate) async fn run_loop<S: State, B: PregelBackend<S>, F: FnMut(StreamChunk)>(
    backend: &B,
    start: StartInput,
    config: &RunConfig,
    sink: &mut F,
    emit: bool,
) -> Result<RunOutcome<S>> {
    let started = Instant::now();
    let modes = config.stream_modes.as_deref();
    let values_enabled = modes.is_none_or(|modes| modes.contains(&StreamMode::Values));
    let debug_enabled = modes.is_none_or(|modes| modes.contains(&StreamMode::Debug));
    let checkpoints_enabled = modes.is_none_or(|modes| {
        modes.contains(&StreamMode::Checkpoints) || modes.contains(&StreamMode::Debug)
    });
    let checkpoint_emit = emit && checkpoints_enabled;
    let mut report = crate::_internal::usage::RunReport::new();
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
        config.live_sink.clone(),
    )?;
    // `get_config()` surfaces the caller's tags and metadata, like LangGraph.
    crate::runtime::context::set_user_config(config.tags.clone(), config.metadata.clone());
    // Message channels are returned to Python as LangChain message objects.
    crate::runtime::context::set_messages_channels(backend.schema().messages_channel_names());

    // LangGraph only complains when the run genuinely needs stored state: a plain run ignores
    // `configurable.thread_id` when the graph has no checkpointer, while `Command(resume=...)`
    // cannot work without one.
    if !backend.has_checkpointer()
        && matches!(&start, StartInput::Resume { resume: Some(_), .. })
    {
        return Err(Error::MissingCheckpointer("Cannot use Command(resume=...)"));
    }

    let fresh = matches!(
        &start,
        StartInput::Values(_) | StartInput::Updates(_) | StartInput::Gateways { .. }
    );
    let mut parent: Option<String> = None;
    let mut previous_checkpoint: Option<crate::checkpoint::SharedCheckpoint> = None;
    let mut skip_interrupt_before = false;
    let mut time_travel = false;
    let mut run_start_step = 0usize;
    let mut resume_seed_nodes: Vec<String> = Vec::new();
    let mut resume_pending: Vec<Task> = Vec::new();
    let mut resume_goto_sources: Vec<String> = Vec::new();
    let raw_input = match &start {
        StartInput::Values(map) => Some(map.clone()),
        // LangGraph writes an input checkpoint for an update-start run too.
        StartInput::Updates(map) => Some(map.clone()),
        StartInput::Gateways { values, .. } => Some(values.clone()),
        StartInput::Resume { .. } => None,
    };
    let (
        mut values,
        mut state,
        mut tasks,
        mut step,
        mut resume_deferred,
        resume_finish,
        update_writes,
        mut replayed_nodes,
    ) = if let Some((values, state, tasks)) = start::resolve_fresh_start(
        &start,
        backend.schema(),
        |node| backend.node_exists(node),
        |state| backend.entry_tasks(state),
    )? {
        (
            values,
            state,
            tasks,
            0usize,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    } else {
        let StartInput::Resume {
            thread_id: resume_thread,
            checkpoint_id,
            resume,
            updates,
            goto: resume_goto,
            sends: resume_sends,
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
        previous_checkpoint = Some(std::sync::Arc::new(checkpoint.clone()));
        skip_interrupt_before = checkpoint.paused_before;
        // LangGraph excludes `Command(resume=...)` from time travel but keeps it for other
        // addressed replays (including `Command(update=...)`).
        time_travel = checkpoint_id.is_some() && resume.is_none();
        let (values, state, tasks, finish_writes, update_writes, replayed) =
            start::resolve_resume_start(
                &checkpoint,
                &resume,
                updates,
                &resume_goto,
                &resume_sends,
                backend.schema(),
            )?;
        resume_pending = checkpoint.resume_tasks.clone();
        resume_goto_sources = checkpoint.goto_sources.clone();
        resume_seed_nodes = checkpoint
            .deferred_routes
            .iter()
            .map(|task| task.node.clone())
            .collect();
        run_start_step = checkpoint.step;
        let deferred = checkpoint.deferred.clone();
        (
            values,
            state,
            tasks,
            checkpoint.step,
            deferred,
            finish_writes,
            update_writes,
            replayed,
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
        // A `Command(update=...)` on a finished thread is persisted as writes on the
        // existing checkpoint instead of a new checkpoint.
        crate::pregel::checkpointing::attach_pending_writes(
            backend,
            &mut previous_checkpoint,
            &update_writes,
        )
        .await?;
        if emit && values_enabled {
            // A run with nothing left to do still reports its final values once.
            sink(values_chunk(0, &values));
        }
        return finish(state, backend.schema(), report, started);
    }
    // A replay addressed at the raw-input checkpoint stores `__start__` as its pending
    // task; that pseudo node is not a real node, so it expands to the entry tasks.
    let mut expanded_entry = false;
    if tasks.iter().any(|task| task.node == crate::START) {
        tasks = backend.entry_tasks(&state)?;
        expanded_entry = true;
    }
    if config
        .control
        .as_ref()
        .is_some_and(crate::RunControl::is_aborted)
    {
        return Err(Error::other("run aborted by RunControl"));
    }

    // LangGraph writes a `source="fork"` checkpoint before replaying from an addressed
    // checkpoint, so the replayed branch has its own head instead of rewriting history.
    if time_travel {
        step += 1;
        crate::pregel::checkpointing::write_fork(
            backend,
            thread_id.as_deref(),
            &namespace,
            &mut parent,
            &mut previous_checkpoint,
            &values,
            &tasks,
            step,
            checkpoint_emit,
            sink,
        )
        .await?;
    }
    if expanded_entry {
        // LangGraph stores the input-merge checkpoint before the first loop tick, so a
        // replay from the raw-input checkpoint gains it too.
        step += 1;
        crate::pregel::fresh::write_entry_merge(
            backend,
            thread_id.as_deref(),
            &namespace,
            &mut parent,
            &mut previous_checkpoint,
            &values,
            &tasks,
            step,
            checkpoint_emit,
            sink,
        )
        .await?;
    }
    let mut pause_force_new = false;
    if fresh {
        let (base_step, force_new) = crate::pregel::fresh::open_run(
            backend,
            thread_id.as_deref(),
            &namespace,
            config.checkpoint_id.as_deref(),
            &mut parent,
            &mut previous_checkpoint,
            &values,
            &mut tasks,
            raw_input.as_ref(),
            durability != crate::graph::Durability::Exit,
            checkpoint_emit,
            sink,
        )
        .await?;
        step = base_step;
        pause_force_new = force_new;
    }
    // Task ids are hashed from the checkpoint the tick starts from (LangGraph v4 scheme).
    crate::pregel::scheduling::assign_v4_ids(
        &mut tasks,
        previous_checkpoint
            .as_ref()
            .map(|checkpoint| checkpoint.id.as_str()),
        &namespace,
        step,
    );
    let mut pending_deferred: Vec<Task> = resume_deferred;
    let mut finish_updates: HashMap<String, Value> = resume_finish
        .into_iter()
        .map(|write| (write.channel, write.value))
        .collect();
    if emit && values_enabled {
        // LangGraph emits the run's starting state before the first superstep.
        sink(values_chunk(0, &values));
    }
    let mut iteration = 0usize;
    loop {
        // Barriers already open before this superstep are consumed by it (LangGraph
        // consumes a named barrier when the value is read).
        let armed_barriers = step::open_named_barriers(backend.schema(), &values);
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
            let pause_context = crate::pregel::pause::PauseContext {
                thread_id: thread_id.as_deref(),
                namespace: &namespace,
                step,
                checkpoint_emit,
                force_new: pause_force_new,
                first_tick: step == run_start_step,
            };
            let interrupt = crate::pregel::pause::pause_before(
                backend,
                &pause_context,
                &mut parent,
                &mut previous_checkpoint,
                &values,
                &tasks,
                &interrupted,
                finish_pending(&finish_updates, backend.schema()),
                pending_deferred.clone(),
                raw_input.as_ref(),
                sink,
            )
            .await?;
            return Err(crate::pregel::pause::pause_error(interrupt));
        }

        if !tasks.is_empty() {
            // A task's trigger is its scheduling channel: `branch:to:<node>` for static and
            // conditional edges, `__pregel_push` for `Send` fan-out. Tasks carried in from a
            // previous superstep already have it; entry and push tasks are filled here.
            crate::pregel::scheduling::fill_triggers(
                &mut tasks,
                &[crate::START.to_string()],
                |node| backend.edges_of(node),
            );
        }
        if emit {
            let task_values = values_to_state::<S>(&values)
                .map(|state| serde_json::json!(state))
                .unwrap_or(Value::Null);
            for chunk in crate::pregel::outcome::task_start_chunks(step, &tasks, &task_values) {
                sink(chunk);
            }
        }

        let outcomes = match backend
            .run_superstep(&state, &tasks, step, config.control.as_ref())
            .await
        {
            Ok(outcomes) => outcomes,
            Err(error) => {
                drain_collector(backend, &setup, sink);
                // LangGraph leaves the failing task on the previous checkpoint with its error, so
                // `get_state()` can report what went wrong.
                record_task_failure(backend, previous_checkpoint.as_ref(), &tasks, &error).await;
                return Err(error);
            }
        };
        drain_collector(backend, &setup, sink);

        let interrupts = interrupts_from(&outcomes);
        if !interrupts.is_empty() {
            if interrupts.len() == 1 {
                let interrupted = interrupts.first().expect("one interrupt").clone();
                let mut context = crate::pregel::pause::InterruptedStep {
                    backend,
                    thread_id: thread_id.as_deref(),
                    namespace: &namespace,
                    step,
                    checkpoint_emit,
                    force_new: pause_force_new,
                    first_tick: step == run_start_step,

                    emit,
                    values_enabled,
                    parent: &mut parent,
                    previous: &mut previous_checkpoint,
                    values: &mut values,
                    pending_deferred: &mut pending_deferred,
                    report: &mut report,
                    finish_updates: &mut finish_updates,
                    raw_input: raw_input.as_ref(),
                    sink,
                    _state: std::marker::PhantomData,
                };
                let interrupt = crate::pregel::pause::pause_single_interrupt(
                    &mut context,
                    &tasks,
                    outcomes,
                    &interrupted,
                )
                .await?;
                return Err(Error::Interrupt(interrupt));
            }
            let mut context = crate::pregel::pause::InterruptedStep {
                backend,
                thread_id: thread_id.as_deref(),
                namespace: &namespace,
                step,
                checkpoint_emit,
                force_new: pause_force_new,
                first_tick: step == run_start_step,
                emit,
                values_enabled,
                parent: &mut parent,
                previous: &mut previous_checkpoint,
                values: &mut values,
                pending_deferred: &mut pending_deferred,
                report: &mut report,
                finish_updates: &mut finish_updates,
                raw_input: raw_input.as_ref(),
                sink,
                _state: std::marker::PhantomData,
            };
            let interrupt = crate::pregel::pause::pause_multi_interrupt(
                &mut context,
                &tasks,
                outcomes,
                &interrupts,
            )
            .await?;
            return Err(Error::Interrupt(interrupt));
        }

        let processed = match process_outcomes(backend, outcomes, step, &mut report, sink, emit) {
            Ok(processed) => processed,
            Err(error) => {
                // A node failure reaches `get_state()` through the previous checkpoint's task.
                record_task_failure(backend, previous_checkpoint.as_ref(), &tasks, &error).await;
                return Err(error);
            }
        };
        let written_for_barriers: HashSet<String> = processed
            .channel_updates
            .iter()
            .map(|update| update.channel.clone())
            .chain(processed.replacements.iter().map(|(name, _)| name.clone()))
            .collect();
        // Channels each executed node wrote this tick, for per-node `versions_seen`.
        let mut writes_by_node: HashMap<String, Vec<String>> = HashMap::new();
        for update in &processed.channel_updates {
            writes_by_node
                .entry(update.node.clone())
                .or_default()
                .push(update.channel.clone());
        }
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
        // Step-scoped topics that this superstep did not rewrite are emptied now.
        backend
            .schema()
            .clear_step_topics(&mut values, &written_for_barriers);
        state = values_to_state::<S>(&values)?;
        if emit && values_enabled {
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

        // Nodes replayed from pending writes still contribute their outgoing routes, but
        // only for the tick that replays them.
        let mut routing_executed = executed.clone();
        routing_executed.append(&mut replayed_nodes);
        // A sibling that chose `goto` must not have its static/conditional edges re-derived,
        // but only for the tick that replays it: later ticks must route normally.
        if !resume_goto_sources.is_empty() {
            routing_executed.retain(|node| !resume_goto_sources.contains(node));
            resume_goto_sources.clear();
        }
        for node in resume_seed_nodes.drain(..) {
            if !routing_executed.contains(&node) {
                routing_executed.push(node);
            }
        }

        let (next_nodes, router_sends) = collect_next_nodes(
            &state,
            &routing_executed,
            &goto_nodes,
            &goto_targets,
            |node| backend.edges_of(node),
            |node| backend.conditional_of(node),
            |node| backend.node_exists(node),
            |node| PregelBackend::destinations_of(backend, node),
        )?;
        let mut next_tasks = schedule_next_tasks(
            next_nodes,
            router_sends,
            next_tasks,
            |node| backend.node_exists(node),
            |node| PregelBackend::destinations_of(backend, node),
            |node| backend.is_deferred(node),
            &mut pending_deferred,
            true,
        )?; // `goto`/`Send` targets chosen by the paused tick's successful siblings run after
        // the resumed tasks complete.
        for task in resume_pending.drain(..) {
            if task.node == crate::END {
                continue;
            }
            let duplicate = next_tasks.iter().any(|existing| {
                if existing.input.is_none() && task.input.is_none() {
                    existing.node == task.node
                } else {
                    existing.node == task.node && existing.input == task.input
                }
            });
            if !duplicate {
                next_tasks.push(task);
            }
        }

        if step::consume_named_barriers(&mut values, &armed_barriers, &written_for_barriers) {
            state = values_to_state::<S>(&values)?;
        }
        crate::pregel::scheduling::fill_triggers(&mut next_tasks, &executed, |node| {
            backend.edges_of(node)
        });

        step += 1;
        // Next tick's task ids hash the checkpoint they will be stored in.
        let next_checkpoint_id = thread_id
            .as_deref()
            .map(|thread| crate::checkpoint::new_checkpoint_id(thread, step));
        if emit && debug_enabled && backend.has_checkpointer() {
            crate::pregel::pause::emit_debug_step(
                sink,
                step,
                &next_tasks,
                &values,
                thread_id.as_deref(),
                &namespace,
                parent.as_deref(),
                next_checkpoint_id.as_deref(),
            );
        }
        crate::pregel::scheduling::assign_v4_ids(
            &mut next_tasks,
            next_checkpoint_id.as_deref(),
            &namespace,
            step,
        );
        let interrupt_after =
            interrupt_after_node(&executed, |node| backend.interrupts_after(node)).map(|node| {
                let mut interrupt = Interrupt::new(node, Value::Null);
                interrupt.static_pause = true;
                interrupt
            });
        if next_tasks.is_empty() && interrupt_after.is_none() && !finish_updates.is_empty() {
            apply_finish_updates(&mut values, &mut finish_updates);
            state = values_to_state::<S>(&values)?;
        }
        if should_save_after_step(durability, interrupt_after.is_some()) {
            let pause_context = crate::pregel::pause::PauseContext {
                thread_id: thread_id.as_deref(),
                namespace: &namespace,
                step,
                checkpoint_emit,
                force_new: pause_force_new,
                first_tick: step == run_start_step,
            };
            crate::pregel::pause::save_after_step(
                backend,
                &pause_context,
                &mut parent,
                &mut previous_checkpoint,
                &values,
                &next_tasks,
                interrupt_after.as_ref(),
                finish_pending(&finish_updates, backend.schema()),
                pending_deferred.clone(),
                &executed,
                &writes_by_node,
                next_checkpoint_id.clone(),
                sink,
            )
            .await?;
        }
        if let Some(interrupt) = interrupt_after {
            if emit {
                crate::pregel::pause::emit_pause_events(
                    sink,
                    step,
                    &tasks,
                    std::slice::from_ref(&interrupt),
                    &values,
                    values_enabled,
                );
            }
            return Err(Error::Interrupt(interrupt));
        }

        if next_tasks.is_empty() {
            if !finish_updates.is_empty() {
                apply_finish_updates(&mut values, &mut finish_updates);
                state = values_to_state::<S>(&values)?;
            }
            if should_save_terminal(durability, thread_id.is_some()) {
                let pause_context = crate::pregel::pause::PauseContext {
                    thread_id: thread_id.as_deref(),
                    namespace: &namespace,
                    step,
                    checkpoint_emit,
                    force_new: pause_force_new,
                    first_tick: step == run_start_step,
                };
                crate::pregel::pause::save_terminal(
                    backend,
                    &pause_context,
                    &mut parent,
                    &mut previous_checkpoint,
                    &values,
                    step,
                    Vec::new(),
                    pending_deferred.clone(),
                    &executed,
                    &writes_by_node,
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

/// Attach a node failure to the previous checkpoint's pending task.
///
/// LangGraph reports `snapshot.tasks[i].error` after a failed run; the checkpoint keeps the same id,
/// so history does not grow by an entry.
async fn record_task_failure<S, B>(
    backend: &B,
    previous: Option<&crate::checkpoint::SharedCheckpoint>,
    tasks: &[crate::_internal::node::Task],
    error: &Error,
) where
    S: State,
    B: crate::pregel::backend::PregelBackend<S>,
{
    let Error::Node { node, message } = error else {
        return;
    };
    let Some(previous) = previous else {
        return;
    };
    let Some(task) = tasks.iter().find(|task| &task.node == node) else {
        return;
    };
    let mut checkpoint = (**previous).clone();
    let failure = crate::_internal::error::NodeFailure::new(node.clone(), message.clone(), 1);
    let mut updated = false;
    for pending in checkpoint.next.iter_mut() {
        if pending.id == task.id || pending.node == *node {
            pending.error = Some(failure.clone());
            updated = true;
        }
    }
    if !updated {
        let mut pending = task.clone();
        pending.error = Some(failure);
        checkpoint.next.push(pending);
    }
    let _ = backend
        .persist_checkpoint(std::sync::Arc::new(checkpoint))
        .await;
}
