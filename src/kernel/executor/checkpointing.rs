//! Checkpoint persistence for the executor.

use serde_json::{Map, Value};

use crate::Result;
use crate::kernel::checkpoint::{Checkpoint, new_checkpoint_id, now_millis};
use crate::kernel::error::Interrupt;
use crate::kernel::graph::CompiledGraph;
use crate::kernel::node::Task;
use crate::kernel::state::State;
use crate::kernel::stream::StreamChunk;

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
    sink: &mut dyn FnMut(StreamChunk),
) -> Result<()> {
    let Some(thread_id) = thread_id else {
        return Ok(());
    };
    let Some(checkpointer) = graph.checkpointer() else {
        return Ok(());
    };

    let id = new_checkpoint_id(thread_id, step);
    let persisted_values = graph.schema().persisted_values(values);
    let previous = parent
        .as_deref()
        .and_then(|parent_id| checkpointer.get(thread_id, parent_id).ok().flatten());
    let (channel_versions, versions_seen, updated_channels) =
        crate::kernel::checkpoint::versions_for(previous.as_ref(), &persisted_values, step);
    let checkpoint = Checkpoint {
        id: id.clone(),
        thread_id: thread_id.to_string(),
        step,
        values: persisted_values,
        next: next.to_vec(),
        interrupted,
        paused_before,
        parent_id: parent.clone(),
        created_at_millis: now_millis(),
        interrupts,
        metadata: Map::from_iter([("source".to_string(), Value::String("loop".to_string()))]),
        namespace: namespace.to_string(),
        task_results: crate::kernel::context::snapshot_task_results(),
        channel_versions,
        versions_seen,
        updated_channels,
    };
    checkpointer.put(checkpoint)?;
    *parent = Some(id.clone());
    sink(StreamChunk::Checkpoint { checkpoint_id: id });
    Ok(())
}
