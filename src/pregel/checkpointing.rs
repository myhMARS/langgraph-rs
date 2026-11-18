//! Checkpoint persistence for the executor.

use std::sync::Arc;
use std::sync::mpsc;

use serde_json::{Map, Value};

use crate::_internal::error::Interrupt;
use crate::_internal::node::Task;
use crate::_internal::ports::Checkpointer;
use crate::_internal::state::State;
use crate::_internal::stream::StreamChunk;
use crate::Result;
use crate::checkpoint::{Checkpoint, new_checkpoint_id, now_millis};

/// Background checkpoint writer used by [`Durability::Async`](crate::Durability::Async).
///
/// Checkpoints are sent to one worker thread in superstep order; dropping the writer
/// flushes the queue and joins, so a finished/interrupted run has all checkpoints.
pub(crate) struct CheckpointWriter {
    sender: Option<mpsc::Sender<Checkpoint>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl CheckpointWriter {
    pub(crate) fn start(checkpointer: Arc<dyn Checkpointer>) -> Self {
        let (sender, receiver) = mpsc::channel::<Checkpoint>();
        let handle = std::thread::spawn(move || {
            while let Ok(checkpoint) = receiver.recv() {
                let _ = checkpointer.put(checkpoint);
            }
        });
        Self {
            sender: Some(sender),
            handle: Some(handle),
        }
    }

    fn submit(&self, checkpoint: Checkpoint) -> Result<()> {
        let Some(sender) = &self.sender else {
            return Err(crate::_internal::error::Error::Checkpoint(
                "checkpoint writer is already closed".to_string(),
            ));
        };
        sender.send(checkpoint).map_err(|_| {
            crate::_internal::error::Error::Checkpoint(
                "checkpoint writer thread stopped".to_string(),
            )
        })
    }

    pub(crate) fn flush(&mut self) {
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for CheckpointWriter {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Inputs shared by the sync and async checkpoint writers.
pub(crate) struct CheckpointDraft<'a> {
    pub(crate) thread_id: &'a str,
    pub(crate) namespace: &'a str,
    pub(crate) id: String,
    pub(crate) parent: Option<String>,
    pub(crate) values: Map<String, Value>,
    pub(crate) next: &'a [Task],
    pub(crate) step: usize,
    pub(crate) interrupted: bool,
    pub(crate) paused_before: bool,
    pub(crate) interrupts: Vec<Interrupt>,
    pub(crate) pending_writes: Vec<crate::checkpoint::PendingWrite>,
    pub(crate) deferred: Vec<Task>,
    pub(crate) executed: &'a [String],
    pub(crate) previous: Option<&'a Checkpoint>,
    pub(crate) task_results: Map<String, Value>,
}

/// Build a checkpoint with the version metadata used by both runtimes.
pub(crate) fn build_checkpoint(draft: CheckpointDraft<'_>) -> Checkpoint {
    let (channel_versions, versions_seen, updated_channels) =
        crate::checkpoint::versions_for(draft.previous, &draft.values, draft.step, draft.executed);
    Checkpoint {
        id: draft.id,
        thread_id: draft.thread_id.to_string(),
        step: draft.step,
        values: draft.values,
        next: draft.next.to_vec(),
        interrupted: draft.interrupted,
        paused_before: draft.paused_before,
        parent_id: draft.parent,
        created_at_millis: now_millis(),
        interrupts: draft.interrupts,
        metadata: Map::from_iter([("source".to_string(), Value::String("loop".to_string()))]),
        namespace: draft.namespace.to_string(),
        task_results: draft.task_results,
        channel_versions,
        versions_seen,
        updated_channels,
        pending_writes: draft.pending_writes,
        deferred: draft.deferred,
    }
}

/// Rebuild pending tasks from an interrupted checkpoint and bind resume values.
///
/// Only tasks flagged as interrupted receive the resume value; a resume object with
/// several pending interrupts is a map keyed by interrupt id. The new value is
/// folded into each task's consumed history so repeated `interrupt()` calls see
/// earlier answers on re-runs.
pub(crate) fn prepare_resume_tasks(checkpoint: &Checkpoint, resume: &Option<Value>) -> Vec<Task> {
    let interrupt_ids: std::collections::HashSet<&str> = checkpoint
        .interrupts
        .iter()
        .map(|interrupt| interrupt.id.as_str())
        .collect();
    let looks_like_id = |key: &str| {
        interrupt_ids.contains(key)
            || (key.len() >= 32 && key.chars().all(|ch| ch.is_ascii_hexdigit() || ch == '-'))
    };
    let interrupt_map = match resume {
        Some(Value::Object(map)) if !map.is_empty() && map.keys().all(|key| looks_like_id(key)) => {
            Some(map.clone())
        }
        _ => None,
    };
    let mut tasks = checkpoint.next.clone();
    let flagged: std::collections::HashSet<String> = tasks
        .iter()
        .filter(|task| task.interrupted)
        .map(|task| task.id.clone())
        .collect();
    for task in &mut tasks {
        if !flagged.is_empty() && !flagged.contains(&task.id) {
            task.resume = None;
            continue;
        }
        task.resume = match &interrupt_map {
            Some(map) => checkpoint
                .interrupts
                .iter()
                .find(|interrupt| match &interrupt.task_id {
                    Some(task_id) => task_id.as_ref() == task.id.as_str(),
                    None => interrupt.node == task.node,
                })
                .and_then(|interrupt| interrupt.resume_from(map)),
            None => resume.clone(),
        };
    }
    for task in &mut tasks {
        if let Some(value) = task.resume.take() {
            task.resume_history.push(value);
        }
    }
    tasks
}

/// Record a persisted checkpoint as the new parent and emit its stream event.
pub(crate) fn checkpoint_committed(
    parent: &mut Option<String>,
    checkpoint: &Checkpoint,
    sink: &mut dyn FnMut(StreamChunk),
    emit: bool,
) {
    *parent = Some(checkpoint.id.clone());
    if !emit {
        return;
    }
    let config = serde_json::json!({
        "configurable": {
            "thread_id": checkpoint.thread_id,
            "checkpoint_ns": checkpoint.namespace,
            "checkpoint_id": checkpoint.id,
        }
    });
    let parent_config = checkpoint.parent_id.as_ref().map(|parent_id| {
        serde_json::json!({
            "configurable": {
                "thread_id": checkpoint.thread_id,
                "checkpoint_ns": checkpoint.namespace,
                "checkpoint_id": parent_id,
            }
        })
    });
    let tasks: Vec<serde_json::Value> = checkpoint
        .next
        .iter()
        .map(|task| {
            serde_json::json!({
                "id": task.id,
                "name": task.node,
                "interrupts": [],
            })
        })
        .collect();
    let payload = serde_json::json!({
        "config": config,
        "metadata": {
            "step": checkpoint.step,
            "source": checkpoint.metadata.get("source"),
            "parents": {},
        },
        "values": checkpoint.values,
        "next": checkpoint.next.iter().map(|task| task.node.clone()).collect::<Vec<_>>(),
        "parent_config": parent_config,
        "tasks": tasks,
    });
    sink(StreamChunk::Checkpoint {
        checkpoint_id: checkpoint.id.clone(),
        payload,
    });
}

/// Persist one checkpoint through a sync checkpointer or the background writer.
pub(crate) fn persist_checkpoint(
    checkpointer: &dyn Checkpointer,
    checkpoint: Checkpoint,
    writer: Option<&CheckpointWriter>,
) -> Result<()> {
    match writer {
        Some(writer) => writer.submit(checkpoint),
        None => checkpointer.put(checkpoint),
    }
}

/// Shared checkpoint save used by the unified Pregel loop.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn save_checkpoint_via<
    S: State,
    B: crate::pregel::backend::PregelBackend<S>,
    F: FnMut(StreamChunk),
>(
    backend: &B,
    thread_id: Option<&str>,
    namespace: &str,
    parent: &mut Option<String>,
    previous: &mut Option<Checkpoint>,
    values: &Map<String, Value>,
    next: &[Task],
    step: usize,
    interrupted: bool,
    paused_before: bool,
    interrupts: Vec<Interrupt>,
    pending_writes: Vec<crate::checkpoint::PendingWrite>,
    deferred: Vec<Task>,
    executed: &[String],
    emit: bool,
    sink: &mut F,
) -> Result<()> {
    let Some(thread_id) = thread_id else {
        return Ok(());
    };
    let id = new_checkpoint_id(thread_id, step);
    let persisted_values = backend.schema().persisted_values(values);
    let checkpoint = build_checkpoint(CheckpointDraft {
        thread_id,
        namespace,
        id: id.clone(),
        parent: parent.clone(),
        values: persisted_values,
        next,
        step,
        interrupted,
        paused_before,
        interrupts,
        pending_writes,
        deferred,
        executed,
        previous: previous.as_ref(),
        task_results: crate::runtime::context::snapshot_task_results(),
    });
    backend.persist_checkpoint(checkpoint.clone()).await?;
    checkpoint_committed(parent, &checkpoint, sink, emit);
    *previous = Some(checkpoint);
    Ok(())
}
