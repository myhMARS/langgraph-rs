//! Checkpoint persistence for the executor.

use std::sync::Arc;
use std::sync::mpsc;

use serde_json::{Map, Value};

use crate::Result;
use crate::kernel::checkpoint::{Checkpoint, new_checkpoint_id, now_millis};
use crate::kernel::error::Interrupt;
use crate::kernel::graph::CompiledGraph;
use crate::kernel::node::Task;
use crate::kernel::ports::Checkpointer;
use crate::kernel::state::State;
use crate::kernel::stream::StreamChunk;

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
            return Err(crate::kernel::error::Error::Checkpoint(
                "checkpoint writer is already closed".to_string(),
            ));
        };
        sender.send(checkpoint).map_err(|_| {
            crate::kernel::error::Error::Checkpoint("checkpoint writer thread stopped".to_string())
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

/// Persist a checkpoint when the graph is bound to a thread and has a checkpointer.
#[allow(clippy::too_many_arguments)]
pub(super) fn save_checkpoint<S: State>(
    graph: &CompiledGraph<S>,
    thread_id: Option<&str>,
    namespace: &str,
    parent: &mut Option<String>,
    values: &Map<String, Value>,
    next: &[Task],
    step: usize,
    interrupted: bool,
    paused_before: bool,
    interrupts: Vec<Interrupt>,
    pending_writes: Vec<crate::kernel::checkpoint::PendingWrite>,
    executed: &[String],
    sink: &mut dyn FnMut(StreamChunk),
    writer: Option<&CheckpointWriter>,
) -> Result<()> {
    let Some(thread_id) = thread_id else {
        return Ok(());
    };
    let Some(checkpointer) = graph.checkpointer() else {
        return Ok(());
    };

    let id = new_checkpoint_id(thread_id, step);
    let persisted_values = graph.schema().persisted_values(values);
    let previous = if writer.is_some() {
        None
    } else {
        parent
            .as_deref()
            .and_then(|parent_id| checkpointer.get(thread_id, parent_id).ok().flatten())
    };
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
        executed,
        previous: previous.as_ref(),
        task_results: crate::kernel::context::snapshot_task_results(),
    });
    persist_checkpoint(checkpointer.as_ref(), checkpoint, writer)?;
    checkpoint_committed(parent, &id, sink);
    Ok(())
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
    pub(crate) pending_writes: Vec<crate::kernel::checkpoint::PendingWrite>,
    pub(crate) executed: &'a [String],
    pub(crate) previous: Option<&'a Checkpoint>,
    pub(crate) task_results: Map<String, Value>,
}

/// Build a checkpoint with the version metadata used by both runtimes.
pub(crate) fn build_checkpoint(draft: CheckpointDraft<'_>) -> Checkpoint {
    let (channel_versions, versions_seen, updated_channels) =
        crate::kernel::checkpoint::versions_for(
            draft.previous,
            &draft.values,
            draft.step,
            draft.executed,
        );
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
    }
}

/// Rebuild pending tasks from an interrupted checkpoint and bind resume values.
///
/// A resume object with several pending interrupts is a map keyed by interrupt id;
/// otherwise the same value is replayed into every pending task. The new value is
/// folded into each task's consumed history so repeated `interrupt()` calls see
/// earlier answers on re-runs.
pub(crate) fn prepare_resume_tasks(checkpoint: &Checkpoint, resume: &Option<Value>) -> Vec<Task> {
    let interrupt_map = match (resume, checkpoint.interrupts.len() > 1) {
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
    id: &str,
    sink: &mut dyn FnMut(StreamChunk),
) {
    *parent = Some(id.to_string());
    sink(StreamChunk::Checkpoint {
        checkpoint_id: id.to_string(),
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
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) async fn save_checkpoint_via<
    S: State,
    B: crate::kernel::executor::backend::PregelBackend<S>,
>(
    backend: &B,
    thread_id: Option<&str>,
    namespace: &str,
    parent: &mut Option<String>,
    values: &Map<String, Value>,
    next: &[Task],
    step: usize,
    interrupted: bool,
    paused_before: bool,
    interrupts: Vec<Interrupt>,
    pending_writes: Vec<crate::kernel::checkpoint::PendingWrite>,
    executed: &[String],
    sink: &mut dyn FnMut(StreamChunk),
) -> Result<()> {
    let Some(thread_id) = thread_id else {
        return Ok(());
    };
    let id = new_checkpoint_id(thread_id, step);
    let persisted_values = backend.schema().persisted_values(values);
    let previous = match parent.as_deref() {
        Some(parent_id) => {
            backend
                .load_checkpoint(thread_id, Some(parent_id), namespace)
                .await?
        }
        None => None,
    };
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
        executed,
        previous: previous.as_ref(),
        task_results: crate::kernel::context::snapshot_task_results(),
    });
    backend.persist_checkpoint(checkpoint).await?;
    checkpoint_committed(parent, &id, sink);
    Ok(())
}
