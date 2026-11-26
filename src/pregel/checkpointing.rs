//! Checkpoint persistence for the executor.

use std::sync::Arc;
use std::sync::mpsc;

use serde_json::{Map, Value};

use crate::_internal::error::{Error, Interrupt};
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
    sender: Option<mpsc::Sender<std::sync::Arc<Checkpoint>>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl CheckpointWriter {
    pub(crate) fn start(checkpointer: Arc<dyn Checkpointer>) -> Self {
        let (sender, receiver) = mpsc::channel::<std::sync::Arc<Checkpoint>>();
        let handle = std::thread::spawn(move || {
            while let Ok(checkpoint) = receiver.recv() {
                let _ = checkpointer.put_shared(checkpoint);
            }
        });
        Self {
            sender: Some(sender),
            handle: Some(handle),
        }
    }

    fn submit(&self, checkpoint: std::sync::Arc<Checkpoint>) -> Result<()> {
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
    /// LangGraph `metadata.source` (`input`, `loop`, `update`, `fork`).
    pub(crate) source: &'a str,
    /// Channels each executed node wrote, so `versions_seen` can approximate reads.
    pub(crate) writes_by_node: &'a std::collections::HashMap<String, Vec<String>>,
    pub(crate) previous: Option<&'a Checkpoint>,
    pub(crate) task_results: Map<String, Value>,
}

/// Build a checkpoint with the version metadata used by both runtimes.
pub(crate) fn build_checkpoint(draft: CheckpointDraft<'_>) -> Checkpoint {
    let (channel_versions, versions_seen, updated_channels) = crate::checkpoint::versions_for(
        draft.previous,
        &draft.values,
        draft.step,
        draft.executed,
        draft.writes_by_node,
    );
    Checkpoint {
        id: draft.id,
        thread_id: draft.thread_id.to_string(),
        step: draft.step,
        values: draft.values,
        next: draft.next.to_vec(),
        deferred_routes: Vec::new(),
        resume_tasks: Vec::new(),
        goto_sources: Vec::new(),
        completed_tasks: Vec::new(),
        interrupted: draft.interrupted,
        paused_before: draft.paused_before,
        parent_id: draft.parent,
        created_at_millis: now_millis(),
        interrupts: draft.interrupts,
        metadata: Map::from_iter([(
            "source".to_string(),
            Value::String(draft.source.to_string()),
        )]),
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
pub(crate) fn prepare_resume_tasks(
    checkpoint: &Checkpoint,
    resume: &Option<Value>,
) -> Result<Vec<Task>> {
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
    if let Some(map) = &interrupt_map {
        if let Some(unknown) = map
            .keys()
            .find(|key| !checkpoint.interrupts.iter().any(|known| &known.id == *key))
        {
            return Err(Error::validation(format!(
                "Unknown or already-consumed interrupt: {unknown}"
            )));
        }
    }
    // Defensive: `END` is never a runnable task, even if a checkpoint stores one.
    let mut tasks: Vec<Task> = checkpoint
        .next
        .iter()
        .filter(|task| task.node != crate::END)
        .cloned()
        .collect();
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
    Ok(tasks)
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
            // LangGraph's `CheckpointTask` carries the task input plus, when available,
            // the routed error, the writes it produced and its interrupts.
            let interrupts: Vec<serde_json::Value> = checkpoint
                .interrupts
                .iter()
                .filter(|interrupt| {
                    interrupt.task_id.as_deref() == Some(task.id.as_str())
                        || (interrupt.task_id.is_none() && interrupt.node == task.node)
                })
                .filter_map(|interrupt| serde_json::to_value(interrupt).ok())
                .collect();
            let mut entry = serde_json::Map::new();
            entry.insert("id".to_string(), serde_json::json!(task.id));
            entry.insert("name".to_string(), serde_json::json!(task.node));
            entry.insert(
                "state".to_string(),
                task.input
                    .as_ref()
                    .map(|input| serde_json::Value::Object(input.clone()))
                    .unwrap_or(serde_json::Value::Null),
            );
            if let Some(error) = &task.error {
                entry.insert("error".to_string(), serde_json::json!(error.message));
            }
            if let Some(result) = checkpoint.task_results.get(&task.id) {
                entry.insert("result".to_string(), result.clone());
            }
            entry.insert("interrupts".to_string(), serde_json::json!(interrupts));
            serde_json::Value::Object(entry)
        })
        .collect();
    let payload = serde_json::json!({
        "config": config,
        "metadata": {
            // LangGraph exposes `metadata.step` one below our internal counter.
            "step": checkpoint.step as i64 - 1,
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
    checkpoint: std::sync::Arc<Checkpoint>,
    writer: Option<&CheckpointWriter>,
) -> Result<()> {
    match writer {
        Some(writer) => writer.submit(checkpoint),
        None => checkpointer.put_shared(checkpoint),
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
    previous: &mut Option<crate::checkpoint::SharedCheckpoint>,
    values: &Map<String, Value>,
    next: &[Task],
    step: usize,
    interrupted: bool,
    paused_before: bool,
    interrupts: Vec<Interrupt>,
    pending_writes: Vec<crate::checkpoint::PendingWrite>,
    deferred: Vec<Task>,
    executed: &[String],
    source: &str,
    writes_by_node: &std::collections::HashMap<String, Vec<String>>,
    id: Option<String>,
    emit: bool,
    sink: &mut F,
) -> Result<()> {
    let Some(thread_id) = thread_id else {
        return Ok(());
    };
    let id = id.unwrap_or_else(|| new_checkpoint_id(thread_id, step));
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
        source,
        writes_by_node,
        previous: previous.as_deref(),
        task_results: crate::runtime::context::snapshot_task_results(),
    });
    let checkpoint = std::sync::Arc::new(checkpoint);
    backend.persist_checkpoint(std::sync::Arc::clone(&checkpoint)).await?;
    checkpoint_committed(parent, &checkpoint, sink, emit);
    *previous = Some(checkpoint);
    Ok(())
}

/// Write the checkpoints a fresh run starts from and return the first superstep's step.
///
/// LangGraph records the raw input in its own `source="input"` checkpoint, then a
/// post-input checkpoint with the entry tasks, and continues the thread's step counter.
/// Attach a pause to the checkpoint the interrupted superstep started from.
///
/// LangGraph does not write a new checkpoint when a task interrupts: the interrupt and the
/// step's pending writes are recorded against the previous checkpoint, so history and
/// `metadata.step` are unchanged by the pause.
pub(crate) struct PauseDraft<'a> {
    pub(crate) interrupted: bool,
    pub(crate) paused_before: bool,
    pub(crate) interrupts: Vec<crate::_internal::error::Interrupt>,
    pub(crate) pending_writes: Vec<crate::checkpoint::PendingWrite>,
    pub(crate) deferred: Vec<crate::_internal::node::Task>,
    pub(crate) executed: &'a [String],
    /// Updated channel values, when the tick's writes were already applied.
    pub(crate) values: Option<&'a Map<String, Value>>,
    /// Tasks still pending after the pause.
    pub(crate) next_tasks: &'a [crate::_internal::node::Task],
    /// Channels each executed node wrote, for `versions_seen`.
    pub(crate) writes_by_node: &'a std::collections::HashMap<String, Vec<String>>,
    /// Raw input of this turn, used to flush the deferred `Durability::Exit` checkpoints.
    pub(crate) raw_input: Option<&'a Map<String, Value>>,
    /// Routing seeds (node names) of successful siblings.
    pub(crate) scheduled_after: Vec<Task>,
    /// `goto`/`Send` targets the successful siblings scheduled.
    pub(crate) resume_tasks: Vec<Task>,
    /// Siblings that completed before the interrupt.
    pub(crate) completed_tasks: Vec<Task>,
    /// Siblings that chose `goto` (their static edges are suppressed on replay).
    pub(crate) goto_sources: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn persist_pause<S, B, F>(
    backend: &B,
    thread_id: Option<&str>,
    namespace: &str,
    parent: &mut Option<String>,
    previous: &mut Option<crate::checkpoint::SharedCheckpoint>,
    next: &[crate::_internal::node::Task],
    values: &Map<String, Value>,
    step: usize,
    draft: PauseDraft<'_>,
    force_new: bool,
    source: &str,
    emit: bool,
    sink: &mut F,
) -> Result<()>
where
    S: State,
    B: crate::pregel::backend::PregelBackend<S>,
    F: FnMut(StreamChunk),
{
    let _ = next;
    // Durability::Exit defers this turn's input/post checkpoints; when the run pauses they
    // are flushed first, and the pause then attaches to the post checkpoint in place.
    // Durability::Exit defers this turn's checkpoints; a pause flushes the *post* checkpoint
    // only, because LangGraph does not keep the raw-input one for a deferred turn (its pause
    // history has a single entry).
    if force_new && draft.raw_input.is_some() {
        save_checkpoint_via(
            backend,
            thread_id,
            namespace,
            parent,
            previous,
            draft.values.unwrap_or(values),
            draft.next_tasks,
            step,
            false,
            false,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[],
            "loop",
            &std::collections::HashMap::new(),
            None,
            emit,
            sink,
        )
        .await?;
    }
    let in_place = previous.is_some() && (!force_new || draft.raw_input.is_some());
    let Some(mut checkpoint) = previous.take().filter(|_| in_place) else {
        // Durability::Exit skips the input checkpoints, so a pause needs to be written
        // explicitly or it would be lost.
        let next_tasks: &[crate::_internal::node::Task] = draft.next_tasks;
        return save_checkpoint_via(
            backend,
            thread_id,
            namespace,
            parent,
            previous,
            draft.values.unwrap_or(values),
            next_tasks,
            step + 1,
            draft.interrupted,
            draft.paused_before,
            draft.interrupts.clone(),
            draft.pending_writes.clone(),
            draft.deferred.clone(),
            draft.executed,
            source,
            draft.writes_by_node,
            None,
            emit,
            sink,
        )
        .await;
    };
    // Pauses are rare: copy the shared checkpoint once so it can be updated in place.
    let checkpoint = std::sync::Arc::make_mut(&mut checkpoint);
    checkpoint.next = draft.next_tasks.to_vec();
    if let Some(values) = draft.values {
        checkpoint.values = backend.schema().persisted_values(values);
    }
    checkpoint.interrupted = draft.interrupted;
    checkpoint.paused_before = draft.paused_before;
    checkpoint.interrupts = draft.interrupts;
    // The draft always carries the complete pending set: writes applied to `values` are
    // dropped here, writes from the interrupted tick are kept.
    let mut written_by_node: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    let written = draft.pending_writes;
    for write in &written {
        written_by_node
            .entry(write.node.clone())
            .or_default()
            .push(write.channel.clone());
    }
    checkpoint.pending_writes = written;
    checkpoint.deferred = draft.deferred;
    checkpoint.deferred_routes = draft.scheduled_after;
    checkpoint.resume_tasks = draft.resume_tasks;
    checkpoint.completed_tasks = draft.completed_tasks;
    checkpoint.goto_sources = draft.goto_sources;
    // Functional-API tasks that finished during the interrupted superstep keep their
    // recorded results so a resume does not re-run them.
    checkpoint
        .task_results
        .extend(crate::runtime::context::snapshot_task_results());
    if !draft.executed.is_empty() {
        // Record the interrupted superstep's nodes and the channels they wrote, so
        // `versions_seen` and `updated_channels` match a normally committed step.

        // `versions_for` takes the maps out of `checkpoint`, so snapshot what it needs
        // first: this path runs once per pause, not once per superstep.
        let pause_values = checkpoint.values.clone();
        let pause_step = checkpoint.step;
        let (channel_versions, versions_seen, updated_channels) = crate::checkpoint::versions_for(
            Some(checkpoint),
            &pause_values,
            pause_step,
            draft.executed,
            &written_by_node,
        );
        checkpoint.channel_versions = channel_versions;
        checkpoint.versions_seen = versions_seen;
        checkpoint.updated_channels = updated_channels;
    }
    let checkpoint = std::sync::Arc::new(checkpoint.clone());
    backend.persist_checkpoint(std::sync::Arc::clone(&checkpoint)).await?;
    *previous = Some(checkpoint);
    Ok(())
}

/// Attach pending writes to the latest checkpoint without creating a new one.
///
/// LangGraph stores `Command(update=...)` on a finished thread as writes against the
/// existing checkpoint, so `get_state` shows them through the live view.
pub(crate) async fn attach_pending_writes<S, B>(
    backend: &B,
    previous: &mut Option<crate::checkpoint::SharedCheckpoint>,
    writes: &[crate::checkpoint::PendingWrite],
) -> Result<()>
where
    S: State,
    B: crate::pregel::backend::PregelBackend<S>,
{
    if writes.is_empty() {
        return Ok(());
    }
    let Some(existing) = previous.clone() else {
        return Ok(());
    };
    // Rare path (a `Command(update=...)` on a finished thread): copy the shared checkpoint
    // once, then store the new version under the same id.
    let mut checkpoint = (*existing).clone();
    checkpoint.pending_writes.extend(writes.iter().cloned());
    let checkpoint = std::sync::Arc::new(checkpoint);
    backend.persist_checkpoint(std::sync::Arc::clone(&checkpoint)).await?;
    *previous = Some(checkpoint);
    Ok(())
}

/// Write the `source="fork"` checkpoint LangGraph creates when a run replays from an
/// addressed checkpoint, so the replayed branch gets its own head.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn write_fork<S, B, F>(
    backend: &B,
    thread_id: Option<&str>,
    namespace: &str,
    parent: &mut Option<String>,
    previous: &mut Option<crate::checkpoint::SharedCheckpoint>,
    values: &Map<String, Value>,
    tasks: &[crate::_internal::node::Task],
    step: usize,
    emit: bool,
    sink: &mut F,
) -> Result<()>
where
    S: State,
    B: crate::pregel::backend::PregelBackend<S>,
    F: FnMut(StreamChunk),
{
    save_checkpoint_via(
        backend,
        thread_id,
        namespace,
        parent,
        previous,
        values,
        tasks,
        step,
        false,
        false,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        &[],
        "fork",
        &std::collections::HashMap::new(),
        None,
        emit,
        sink,
    )
    .await
}
