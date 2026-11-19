//! Writing the two checkpoints that open a run: the raw input checkpoint and the
//! post-input checkpoint that carries the entry tasks.

use serde_json::{Map, Value};

use crate::_internal::node::Task;
use crate::_internal::state::State;
use crate::_internal::stream::StreamChunk;
use crate::Result;
use crate::checkpoint::new_checkpoint_id;
use crate::pregel::checkpointing::save_checkpoint_via;
use crate::pregel::outcome::debug_chunk;
#[allow(clippy::too_many_arguments)]
pub(crate) async fn save_fresh_start<S, B, F>(
    backend: &B,
    thread_id: Option<&str>,
    namespace: &str,
    addressed: Option<&str>,
    parent: &mut Option<String>,
    previous: &mut Option<crate::checkpoint::SharedCheckpoint>,
    values: &Map<String, Value>,
    tasks: &mut [Task],
    raw_input: Option<&Map<String, Value>>,
    persist: bool,
    emit: bool,
    sink: &mut F,
) -> Result<(usize, bool)>
where
    S: State,
    B: crate::pregel::backend::PregelBackend<S>,
    F: FnMut(StreamChunk),
{
    let mut base_step = 0usize;
    if let Some(thread_id) = thread_id
        && let Some(latest) = backend
            .load_checkpoint(thread_id, addressed, namespace)
            .await?
    {
        // Steps are thread-cumulative in LangGraph: a new turn continues one past the
        // latest checkpoint instead of restarting at zero.
        base_step = latest.step + 1;
        *parent = Some(latest.id.clone());
        *previous = Some(std::sync::Arc::new(latest));
    }
    if let Some(raw) = raw_input {
        let start_tasks = vec![Task::new(crate::START)];
        let raw_id = thread_id.map(|thread| new_checkpoint_id(thread, base_step));
        let raw_parent = parent.clone();
        if persist {
            save_checkpoint_via(
                backend,
                thread_id,
                namespace,
                parent,
                previous,
                raw,
                &start_tasks,
                base_step,
                false,
                false,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                &[],
                "input",
                &std::collections::HashMap::new(),
                raw_id.clone(),
                emit,
                sink,
            )
            .await?;
        }
        if persist && emit {
            sink(debug_chunk(
                base_step,
                &[],
                &start_tasks,
                &Value::Object(raw.clone()),
                thread_id,
                namespace,
                raw_parent.as_deref(),
                raw_id.as_deref(),
                "input",
            ));
        }
        base_step += 1;
    }
    let post_id = thread_id.map(|thread| new_checkpoint_id(thread, base_step));
    if let Some(id) = post_id.as_deref() {
        crate::pregel::scheduling::assign_v4_ids(tasks, Some(id), namespace, base_step);
    }
    let post_parent = parent.clone();
    if persist {
        save_checkpoint_via(
            backend,
            thread_id,
            namespace,
            parent,
            previous,
            values,
            tasks,
            base_step,
            false,
            false,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[],
            "loop",
            &std::collections::HashMap::new(),
            post_id.clone(),
            emit,
            sink,
        )
        .await?;
    }
    if persist && emit {
        sink(debug_chunk(
            base_step,
            &[],
            tasks,
            &Value::Object(values.clone()),
            thread_id,
            namespace,
            post_parent.as_deref(),
            post_id.as_deref(),
            "loop",
        ));
    }
    Ok((base_step, persist))
}
/// Write the post-input ("loop") checkpoint created when a replay expands its entry tasks.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn write_entry_merge<S, B, F>(
    backend: &B,
    thread_id: Option<&str>,
    namespace: &str,
    parent: &mut Option<String>,
    previous: &mut Option<crate::checkpoint::SharedCheckpoint>,
    values: &Map<String, Value>,
    tasks: &[Task],
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
        "loop",
        &std::collections::HashMap::new(),
        None,
        emit,
        sink,
    )
    .await
}

/// Open a fresh run: write the input checkpoints (unless durability defers them) and
/// return the starting step plus whether a pause must be written as a new checkpoint.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn open_run<S, B, F>(
    backend: &B,
    thread_id: Option<&str>,
    namespace: &str,
    addressed: Option<&str>,
    parent: &mut Option<String>,
    previous: &mut Option<crate::checkpoint::SharedCheckpoint>,
    values: &Map<String, Value>,
    tasks: &mut [Task],
    raw_input: Option<&Map<String, Value>>,
    persist: bool,
    emit: bool,
    sink: &mut F,
) -> Result<(usize, bool)>
where
    S: State,
    B: crate::pregel::backend::PregelBackend<S>,
    F: FnMut(StreamChunk),
{
    crate::runtime::context::clear_task_results();
    let (step, persisted) = save_fresh_start(
        backend, thread_id, namespace, addressed, parent, previous, values, tasks, raw_input,
        persist, emit, sink,
    )
    .await?;
    Ok((step, !persisted))
}
