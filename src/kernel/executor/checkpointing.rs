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
    let checkpoint = Checkpoint {
        id: id.clone(),
        thread_id: thread_id.to_string(),
        step,
        values: values.clone(),
        next: next.to_vec(),
        interrupted,
        paused_before,
        parent_id: parent.clone(),
        created_at_millis: now_millis(),
        interrupts,
        metadata: Map::from_iter([("source".to_string(), Value::String("loop".to_string()))]),
    };
    checkpointer.put(checkpoint)?;
    *parent = Some(id.clone());
    sink(StreamChunk::Checkpoint { checkpoint_id: id });
    Ok(())
}
