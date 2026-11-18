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
    let resume_seeds: Vec<Task> = tasks
        .iter()
        .filter(|task| !interrupted_ids.contains(&task.id))
        .map(|task| Task::new(task.node.clone()))
        .collect();
    // `goto`/`Send` targets chosen by successful siblings are remembered so the resume
    // replays them (they cannot be derived from static edges).
    let resume_tasks: Vec<Task> = processed
        .goto_targets
        .iter()
        .map(|(target, _)| Task::new(target.clone()))
        .chain(processed.next_tasks.iter().cloned())
        .collect();
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
