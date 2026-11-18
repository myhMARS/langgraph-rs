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
