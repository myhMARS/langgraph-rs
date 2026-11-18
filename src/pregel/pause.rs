//! Pause handling shared by `interrupt_before`, `interrupt_after` and dynamic interrupts.
//!
//! LangGraph attaches a pause to the checkpoint the interrupted superstep started from
//! instead of writing a new one, so this module owns the "build the pause, persist it,
//! emit the events" sequence.

use serde_json::Value;

use crate::_internal::error::{Error, Interrupt, Result};
use crate::_internal::node::Task;
use crate::_internal::state::State;
use crate::pregel::backend::PregelBackend;
use crate::pregel::checkpointing::{PauseDraft, persist_pause};
use crate::pregel::outcome::{interrupt_before_pause, values_chunk};

/// Run-level context needed to persist a pause.
pub(crate) struct PauseContext<'a> {
    pub(crate) thread_id: Option<&'a str>,
    pub(crate) namespace: &'a str,
    pub(crate) step: usize,
    pub(crate) checkpoint_emit: bool,
    /// Write a fresh checkpoint instead of updating the previous one in place.
    pub(crate) force_new: bool,
}

/// Handle an `interrupt_before` hook: persist the pause and emit its events.
///
/// Returns the interrupt the caller should surface to the user.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn pause_before<S, B, F>(
    backend: &B,
    context: &PauseContext<'_>,
    parent: &mut Option<String>,
    previous: &mut Option<crate::checkpoint::Checkpoint>,
    values: &serde_json::Map<String, Value>,
    tasks: &[Task],
    interrupted: &Task,
    pending_writes: Vec<crate::checkpoint::PendingWrite>,
    deferred: Vec<Task>,
    raw_input: Option<&serde_json::Map<String, Value>>,
    sink: &mut F,
) -> Result<Interrupt>
where
    S: State,
    B: PregelBackend<S>,
    F: FnMut(crate::StreamChunk),
{
    let (interrupt, chunk) = interrupt_before_pause(interrupted);
    let mut paused_tasks: Vec<Task> = tasks
        .iter()
        .map(|task| {
            let mut task = task.clone();
            task.interrupted = task.id == interrupted.id;
            task
        })
        .collect();
    crate::pregel::scheduling::assign_v4_ids(
        &mut paused_tasks,
        previous.as_ref().map(|checkpoint| checkpoint.id.as_str()),
        context.namespace,
        context.step,
    );
    persist_pause(
        backend,
        context.thread_id,
        context.namespace,
        parent,
        previous,
        &paused_tasks,
        values,
        context.step,
        PauseDraft {
            interrupted: true,
            paused_before: true,
            interrupts: vec![interrupt.clone()],
            pending_writes,
            deferred,
            executed: &[],
            values: Some(values),
            next_tasks: &paused_tasks,
            writes_by_node: &std::collections::HashMap::new(),
            raw_input,
            scheduled_after: Vec::new(),
            resume_tasks: Vec::new(),
            completed_tasks: Vec::new(),
            goto_sources: Vec::new(),
        },
        context.force_new,
        "loop",
        context.checkpoint_emit,
        sink,
    )
    .await?;
    sink(chunk);
    Ok(interrupt)
}

/// Emit a `values` chunk and a `task_result` chunk for every interrupt of a pause.
pub(crate) fn emit_pause_events<F: FnMut(crate::StreamChunk)>(
    sink: &mut F,
    step: usize,
    tasks: &[Task],
    interrupts: &[Interrupt],
    values: &serde_json::Map<String, Value>,
    values_enabled: bool,
) {
    if values_enabled {
        sink(values_chunk(step, values));
    }
    for interrupt in interrupts {
        sink(crate::pregel::outcome::interrupted_task_chunk(
            step, tasks, interrupt,
        ));
        sink(crate::StreamChunk::Interrupt {
            node: interrupt.node.clone(),
            id: interrupt.id.clone(),
            value: interrupt.value.clone(),
            response_schema: interrupt.response_schema.clone(),
            static_pause: interrupt.static_pause,
        });
    }
}

/// Wrap a pause as the engine-level interrupt error.
pub(crate) fn pause_error(interrupt: Interrupt) -> Error {
    Error::Interrupt(interrupt)
}

/// Mutable run state a dynamic interrupt needs in order to pause in place.
pub(crate) struct InterruptedStep<'a, S: State, B: PregelBackend<S>, F: FnMut(crate::StreamChunk)> {
    pub(crate) _state: std::marker::PhantomData<S>,
    pub(crate) backend: &'a B,
    pub(crate) thread_id: Option<&'a str>,
    pub(crate) namespace: &'a str,
    pub(crate) step: usize,
    pub(crate) checkpoint_emit: bool,
    pub(crate) force_new: bool,
    pub(crate) emit: bool,
    pub(crate) values_enabled: bool,
    pub(crate) parent: &'a mut Option<String>,
    pub(crate) previous: &'a mut Option<crate::checkpoint::Checkpoint>,
    pub(crate) values: &'a mut serde_json::Map<String, Value>,
    pub(crate) pending_deferred: &'a mut Vec<Task>,
    pub(crate) report: &'a mut crate::RunReport,
    pub(crate) finish_updates: &'a mut std::collections::HashMap<String, Value>,
    pub(crate) raw_input: Option<&'a serde_json::Map<String, Value>>,
    pub(crate) sink: &'a mut F,
}

/// Pause a superstep that produced exactly one dynamic interrupt.
pub(crate) async fn pause_single_interrupt<S, B, F>(
    context: &mut InterruptedStep<'_, S, B, F>,
    tasks: &[Task],
    outcomes: Vec<crate::pregel::outcome::TaskOutcome>,
    interrupted: &Interrupt,
) -> Result<Interrupt>
where
    S: State,
    B: PregelBackend<S>,
    F: FnMut(crate::StreamChunk),
{
    use crate::pregel::outcome::{TaskOutcome, collect_pending_writes, process_outcomes};
    use crate::pregel::run_loop::finish_pending;
    let interrupted_ids: std::collections::HashSet<String> = tasks
        .iter()
        .zip(&outcomes)
        .filter(|(_, outcome)| matches!(outcome, TaskOutcome::Interrupted(_)))
        .map(|(task, _)| task.id.clone())
        .collect();
    let mut pending_writes = finish_pending(context.finish_updates, context.backend.schema());
    pending_writes.extend(
        collect_pending_writes(tasks, &outcomes)
            .into_iter()
            .filter(|write| context.backend.schema().persists(&write.channel)),
    );
    let successful: Vec<TaskOutcome> = outcomes
        .into_iter()
        .filter(|outcome| !matches!(outcome, TaskOutcome::Interrupted(_)))
        .collect();
    let processed = process_outcomes(
        context.backend,
        successful,
        context.step,
        context.report,
        context.sink,
        context.emit,
    )?;
    let written_for_barriers: std::collections::HashSet<String> = processed
        .channel_updates
        .iter()
        .map(|update| update.channel.clone())
        .chain(processed.replacements.iter().map(|(name, _)| name.clone()))
        .collect();
    // The paused checkpoint keeps LangGraph's post-step topic state for rewritten topics.
    context
        .backend
        .schema()
        .clear_step_topics(context.values, &written_for_barriers);
    let executed: Vec<String> = tasks.iter().map(|task| task.node.clone()).collect();
    // A pause keeps only the interrupted tasks; successful siblings stay as pending writes
    // and their routes are taken when the resume replays them.
    let mut next_tasks: Vec<Task> = tasks
        .iter()
        .filter(|task| interrupted_ids.contains(&task.id))
        .cloned()
        .collect();
    // Successful siblings keep their node names as routing seeds: a sibling that wrote no
    // persisted channel still has to have its outgoing edges scheduled on resume.
    // A sibling that chose `goto` must not also re-derive its static/conditional edges.
    let goto_sources: std::collections::HashSet<&str> = processed
        .goto_targets
        .iter()
        .map(|(_, source)| source.as_str())
        .collect();
    let mut resume_seeds: Vec<Task> = tasks
        .iter()
        .filter(|task| {
            !interrupted_ids.contains(&task.id) && !goto_sources.contains(task.node.as_str())
        })
        .map(|task| Task::new(task.node.clone()))
        .collect();
    // Routes from an earlier pause survive another interrupt of the same superstep.
    if let Some(previous) = context.previous.as_ref() {
        for task in &previous.deferred_routes {
            if !resume_seeds.iter().any(|seed| seed.node == task.node) {
                resume_seeds.push(task.clone());
            }
        }
    }
    // `goto`/`Send` targets chosen by successful siblings are remembered so the resume
    // replays them (they cannot be derived from static edges).
    let mut resume_tasks: Vec<Task> = processed
        .goto_targets
        .iter()
        .filter(|(target, _)| target != crate::END)
        .map(|(target, _)| Task::new(target.clone()))
        .chain(processed.next_tasks.iter().cloned())
        .collect();
    if let Some(previous) = context.previous.as_ref() {
        for task in &previous.resume_tasks {
            if !resume_tasks
                .iter()
                .any(|known| known.node == task.node && known.input == task.input)
            {
                resume_tasks.push(task.clone());
            }
        }
    }
    for task in &mut next_tasks {
        task.interrupted = true;
    }
    crate::pregel::scheduling::assign_v4_ids(
        &mut next_tasks,
        context
            .previous
            .as_ref()
            .map(|checkpoint| checkpoint.id.as_str()),
        context.namespace,
        context.step,
    );
    persist_pause(
        context.backend,
        context.thread_id,
        context.namespace,
        context.parent,
        context.previous,
        &next_tasks,
        context.values,
        context.step,
        PauseDraft {
            interrupted: true,
            paused_before: false,
            interrupts: vec![interrupted.clone()],
            pending_writes,
            deferred: context.pending_deferred.clone(),
            executed: &executed,
            values: Some(context.values),
            next_tasks: &next_tasks,
            writes_by_node: &std::collections::HashMap::new(),
            raw_input: context.raw_input,
            scheduled_after: resume_seeds,
            resume_tasks,
            goto_sources: goto_sources
                .iter()
                .map(|source| source.to_string())
                .collect(),
            completed_tasks: tasks
                .iter()
                .filter(|task| !interrupted_ids.contains(&task.id))
                .cloned()
                .collect(),
        },
        context.force_new,
        "loop",
        context.checkpoint_emit,
        context.sink,
    )
    .await?;
    if context.emit {
        emit_pause_events(
            context.sink,
            context.step,
            tasks,
            std::slice::from_ref(interrupted),
            context.values,
            context.values_enabled,
        );
    }
    Ok(interrupted.clone())
}

/// Persist the checkpoint for a completed superstep, attaching an `interrupt_after` pause.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn save_after_step<S, B, F>(
    backend: &B,
    context: &PauseContext<'_>,
    parent: &mut Option<String>,
    previous: &mut Option<crate::checkpoint::Checkpoint>,
    values: &serde_json::Map<String, Value>,
    next_tasks: &[Task],
    interrupted: Option<&Interrupt>,
    pending_writes: Vec<crate::checkpoint::PendingWrite>,
    deferred: Vec<Task>,
    executed: &[String],
    writes_by_node: &std::collections::HashMap<String, Vec<String>>,
    checkpoint_id: Option<String>,
    sink: &mut F,
) -> Result<()>
where
    S: State,
    B: PregelBackend<S>,
    F: FnMut(crate::StreamChunk),
{
    crate::pregel::checkpointing::save_checkpoint_via(
        backend,
        context.thread_id,
        context.namespace,
        parent,
        previous,
        values,
        next_tasks,
        context.step,
        interrupted.is_some(),
        false,
        interrupted.cloned().into_iter().collect(),
        pending_writes,
        deferred,
        executed,
        "loop",
        writes_by_node,
        checkpoint_id,
        context.checkpoint_emit,
        sink,
    )
    .await
}
/// Persist the terminal checkpoint of a run that has no further tasks.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn save_terminal<S, B, F>(
    backend: &B,
    context: &PauseContext<'_>,
    parent: &mut Option<String>,
    previous: &mut Option<crate::checkpoint::Checkpoint>,
    values: &serde_json::Map<String, Value>,
    step: usize,
    pending_writes: Vec<crate::checkpoint::PendingWrite>,
    deferred: Vec<Task>,
    executed: &[String],
    writes_by_node: &std::collections::HashMap<String, Vec<String>>,
    sink: &mut F,
) -> Result<()>
where
    S: State,
    B: PregelBackend<S>,
    F: FnMut(crate::StreamChunk),
{
    crate::pregel::checkpointing::save_checkpoint_via(
        backend,
        context.thread_id,
        context.namespace,
        parent,
        previous,
        values,
        &[],
        step,
        false,
        false,
        Vec::new(),
        pending_writes,
        deferred,
        executed,
        "loop",
        writes_by_node,
        None,
        context.checkpoint_emit,
        sink,
    )
    .await
}
/// Emit the debug checkpoint record for a completed superstep.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_debug_step<F: FnMut(crate::StreamChunk)>(
    sink: &mut F,
    step: usize,
    executed: &[String],
    next_tasks: &[Task],
    values: &serde_json::Map<String, Value>,
    thread_id: Option<&str>,
    namespace: &str,
    parent: Option<&str>,
    checkpoint_id: Option<&str>,
) {
    sink(crate::pregel::outcome::debug_chunk(
        step,
        executed,
        next_tasks,
        &Value::Object(values.clone()),
        thread_id,
        namespace,
        parent,
        checkpoint_id,
        "loop",
    ));
}

/// Pause a superstep that produced several interrupts: commit the siblings' writes,
/// keep every interrupted task pending, and persist the pause.
pub(crate) async fn pause_multi_interrupt<S, B, F>(
    context: &mut InterruptedStep<'_, S, B, F>,
    tasks: &[Task],
    outcomes: Vec<crate::pregel::outcome::TaskOutcome>,
    interrupts: &[Interrupt],
) -> Result<Interrupt>
where
    S: State,
    B: PregelBackend<S>,
    F: FnMut(crate::StreamChunk),
{
    use crate::pregel::outcome::{
        TaskOutcome, apply_processed_updates, process_outcomes, values_chunk,
    };
    use crate::pregel::run_loop::finish_pending;
    let interrupted_ids: std::collections::HashSet<String> = tasks
        .iter()
        .zip(&outcomes)
        .filter(|(_, outcome)| matches!(outcome, TaskOutcome::Interrupted(_)))
        .map(|(task, _)| task.id.clone())
        .collect();
    let successful: Vec<TaskOutcome> = outcomes
        .into_iter()
        .filter(|outcome| !matches!(outcome, TaskOutcome::Interrupted(_)))
        .collect();
    let processed = process_outcomes(
        context.backend,
        successful,
        context.step,
        context.report,
        context.sink,
        context.emit,
    )?;
    let written: std::collections::HashSet<String> = processed
        .channel_updates
        .iter()
        .map(|update| update.channel.clone())
        .chain(processed.replacements.iter().map(|(name, _)| name.clone()))
        .collect();
    let mut writes_by_node: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for update in &processed.channel_updates {
        writes_by_node
            .entry(update.node.clone())
            .or_default()
            .push(update.channel.clone());
    }
    apply_processed_updates(
        context.backend.schema(),
        context.values,
        processed.channel_updates,
        processed.replacements,
        context.finish_updates,
    )?;
    context
        .backend
        .schema()
        .clear_step_topics(context.values, &written);
    if context.emit {
        (context.sink)(values_chunk(context.step, context.values));
    }
    let state = crate::_internal::state::values_to_state::<S>(context.values)?;
    let finish_writes = finish_pending(context.finish_updates, context.backend.schema());
    let (next_nodes, router_sends) = crate::pregel::scheduling::collect_next_nodes(
        &state,
        &processed.executed,
        &processed.goto_nodes,
        &processed.goto_targets,
        |node| context.backend.edges_of(node),
        |node| context.backend.conditional_of(node),
        |node| context.backend.node_exists(node),
        |node| PregelBackend::destinations_of(context.backend, node),
    )?;
    let scheduled = crate::pregel::scheduling::schedule_next_tasks(
        next_nodes,
        router_sends,
        processed.next_tasks,
        |node| context.backend.node_exists(node),
        |node| PregelBackend::destinations_of(context.backend, node),
        |node| context.backend.is_deferred(node),
        context.pending_deferred,
        false,
    )?;
    let mut next_tasks = crate::pregel::outcome::merge_interrupted_tasks(
        tasks,
        |task| interrupted_ids.contains(&task.id),
        scheduled,
    );
    for task in &mut next_tasks {
        if interrupted_ids.contains(&task.id) {
            task.interrupted = true;
        }
    }
    crate::pregel::scheduling::fill_triggers(&mut next_tasks, &processed.executed, |node| {
        context.backend.edges_of(node)
    });
    let executed: Vec<String> = tasks.iter().map(|task| task.node.clone()).collect();
    crate::pregel::scheduling::assign_v4_ids(
        &mut next_tasks,
        context
            .previous
            .as_ref()
            .map(|checkpoint| checkpoint.id.as_str()),
        context.namespace,
        context.step,
    );
    persist_pause(
        context.backend,
        context.thread_id,
        context.namespace,
        context.parent,
        context.previous,
        &next_tasks,
        context.values,
        context.step,
        PauseDraft {
            interrupted: true,
            paused_before: false,
            interrupts: interrupts.to_vec(),
            pending_writes: finish_writes,
            deferred: context.pending_deferred.clone(),
            executed: &executed,
            values: Some(context.values),
            next_tasks: &next_tasks,
            writes_by_node: &writes_by_node,
            raw_input: context.raw_input,
            // Routes from an earlier pause survive another interrupt of the same superstep.
            scheduled_after: context
                .previous
                .as_ref()
                .map(|checkpoint| checkpoint.deferred_routes.clone())
                .unwrap_or_default(),
            resume_tasks: context
                .previous
                .as_ref()
                .map(|checkpoint| checkpoint.resume_tasks.clone())
                .unwrap_or_default(),
            completed_tasks: Vec::new(),
            goto_sources: context
                .previous
                .as_ref()
                .map(|checkpoint| checkpoint.goto_sources.clone())
                .unwrap_or_default(),
        },
        context.force_new,
        "loop",
        context.checkpoint_emit,
        context.sink,
    )
    .await?;
    if context.emit {
        emit_pause_events(
            context.sink,
            context.step,
            tasks,
            interrupts,
            context.values,
            context.values_enabled,
        );
    }
    Ok(interrupts
        .first()
        .cloned()
        .unwrap_or_else(|| Interrupt::new(String::new(), Value::Null)))
}
