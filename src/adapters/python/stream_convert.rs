//! Snapshot and stream-chunk conversion helpers.

use pyo3::exceptions::PyTypeError;
use pyo3::types::{PyAny, PyDict, PyList, PyString, PyTuple};

use super::*;
use crate::_internal::stream::StreamChunk;
use crate::JsonState;

pub(crate) fn snapshot_to_py(
    py: Python<'_>,
    snapshot: &crate::StateSnapshot<JsonState>,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("values", json_state_to_py(py, &snapshot.values)?)?;
    // LangGraph exposes `next`, `interrupts` and `tasks` as tuples.
    dict.set_item("next", PyTuple::new(py, snapshot.next.clone())?)?;
    dict.set_item("checkpoint_id", snapshot.checkpoint_id.clone())?;
    dict.set_item("thread_id", snapshot.thread_id.clone())?;
    dict.set_item("step", snapshot.step)?;
    dict.set_item("interrupted", snapshot.interrupted)?;
    dict.set_item("created_at_millis", snapshot.created_at_millis)?;
    dict.set_item("created_at", snapshot.created_at.clone())?;
    dict.set_item(
        "parent_checkpoint_id",
        snapshot.parent_checkpoint_id.clone(),
    )?;
    dict.set_item("checkpoint_ns", snapshot.checkpoint_ns.clone())?;
    // LangGraph exposes the checkpoint metadata (including `step`) on `StateSnapshot`.
    let metadata = PyDict::new(py);
    for (key, value) in &snapshot.metadata {
        metadata.set_item(key, value_to_py(py, value)?)?;
    }
    metadata.set_item("step", snapshot.step)?;
    // LangGraph always reports the parent checkpoint map (empty at the top level).
    metadata.set_item("parents", PyDict::new(py))?;
    dict.set_item("metadata", metadata)?;
    let interrupts = snapshot
        .interrupts
        .iter()
        .map(|interrupt| {
            let item = PyDict::new(py);
            item.set_item("id", interrupt.id.clone())?;
            item.set_item("node", interrupt.node.clone())?;
            item.set_item("value", value_to_py(py, &interrupt.value)?)?;
            if let Some(schema) = &interrupt.response_schema {
                item.set_item("response_schema", value_to_py(py, schema)?)?;
            }
            Ok(item.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("interrupts", PyTuple::new(py, interrupts)?)?;
    dict.set_item("config", value_to_py(py, &snapshot.config)?)?;
    let parent_config: Py<PyAny> = match &snapshot.parent_config {
        Some(config) => value_to_py(py, config)?,
        None => py.None(),
    };
    dict.set_item("parent_config", parent_config)?;
    let tasks = snapshot
        .tasks
        .iter()
        .map(|task| {
            let item = PyDict::new(py);
            item.set_item("id", task.id.clone())?;
            item.set_item("node", task.node.clone())?;
            item.set_item("name", task.node.clone())?;
            let task_interrupts: Vec<Py<PyAny>> = task
                .interrupts
                .iter()
                .map(|interrupt| {
                    Py::new(
                        py,
                        crate::adapters::python::types::PyInterrupt {
                            id: interrupt.id.clone(),
                            value: value_to_py(py, &interrupt.value)?,
                            node: Some(interrupt.node.clone()),
                            resumable: true,
                            response_schema: match &interrupt.response_schema {
                                Some(schema) => Some(value_to_py(py, schema)?),
                                None => None,
                            },
                        },
                    )
                    .map(|interrupt| interrupt.into_any())
                })
                .collect::<PyResult<Vec<_>>>()?;
            item.set_item("interrupts", task_interrupts)?;
            item.set_item(
                "state",
                match &task.state {
                    Some(state) => value_to_py(py, state)?,
                    None => py.None(),
                },
            )?;
            if let Some(result) = &task.result {
                item.set_item("result", value_to_py(py, result)?)?;
            }
            if let Some(error) = &task.error {
                let failure = PyDict::new(py);
                failure.set_item("node", error.node.clone())?;
                failure.set_item("message", error.message.clone())?;
                failure.set_item("attempts", error.attempts)?;
                item.set_item("error", failure)?;
            }
            if let Some(resume) = &task.resume {
                item.set_item("resume", value_to_py(py, resume)?)?;
            }
            Ok(item.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("tasks", PyTuple::new(py, tasks)?)?;
    Ok(Py::new(
        py,
        crate::adapters::python::types::PyStateSnapshot {
            data: dict.unbind(),
        },
    )?
    .into_any())
}

/// LangGraph v2 stream-part shape: `{type, ns, data}`.
pub(crate) fn empty_snapshot_to_py(py: Python<'_>, thread_id: Option<&str>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("values", PyDict::new(py))?;
    dict.set_item("next", PyTuple::new(py, Vec::<String>::new())?)?;
    dict.set_item("checkpoint_id", py.None())?;
    dict.set_item("thread_id", thread_id)?;
    dict.set_item("step", 0)?;
    dict.set_item("interrupted", false)?;
    dict.set_item("created_at", py.None())?;
    dict.set_item("created_at_millis", 0)?;
    dict.set_item("parent_checkpoint_id", py.None())?;
    dict.set_item("checkpoint_ns", "")?;
    dict.set_item("metadata", py.None())?;
    dict.set_item("interrupts", PyTuple::new(py, Vec::<Py<PyAny>>::new())?)?;
    dict.set_item("config", PyDict::new(py))?;
    dict.set_item("parent_config", py.None())?;
    dict.set_item("tasks", PyTuple::new(py, Vec::<Py<PyAny>>::new())?)?;
    Ok(Py::new(
        py,
        crate::adapters::python::types::PyStateSnapshot {
            data: dict.unbind(),
        },
    )?
    .into_any())
}

/// Convert a task chunk into LangGraph debug `task` / `task_result` records.
pub(crate) fn convert_stream_chunks(
    py: Python<'_>,
    chunks: &[StreamChunk],
    v2: bool,
    requested: &[crate::StreamMode],
) -> PyResult<Vec<Py<PyAny>>> {
    convert_stream_chunks_with(py, chunks, v2, requested, false)
}

/// Convert chunks, optionally wrapping **every** chunk as `(namespace, payload)`.
///
/// LangGraph's `subgraphs=True` reports top-level chunks with an empty namespace too; the engine
/// only tags chunks that come from a subgraph, so the top level is wrapped here.
pub(crate) fn convert_stream_chunks_with(
    py: Python<'_>,
    chunks: &[StreamChunk],
    v2: bool,
    requested: &[crate::StreamMode],
    subgraphs: bool,
) -> PyResult<Vec<Py<PyAny>>> {
    if subgraphs {
        let wrapped: Vec<StreamChunk> = chunks
            .iter()
            .map(|chunk| match chunk {
                StreamChunk::Subgraph { .. } => chunk.clone(),
                other => StreamChunk::Subgraph {
                    namespace: String::new(),
                    chunk: Box::new(other.clone()),
                },
            })
            .collect();
        return convert_stream_chunks_inner(py, &wrapped, v2, requested);
    }
    convert_stream_chunks_inner(py, chunks, v2, requested)
}

fn convert_stream_chunks_inner(
    py: Python<'_>,
    chunks: &[StreamChunk],
    v2: bool,
    requested: &[crate::StreamMode],
) -> PyResult<Vec<Py<PyAny>>> {
    let multi_mode = requested.len() > 1;
    let debug_mode = requested.contains(&crate::StreamMode::Debug);
    let tasks_enabled = requested.contains(&crate::StreamMode::Tasks);
    let mut converted: Vec<Py<PyAny>> = Vec::with_capacity(chunks.len());
    let mut values_indices = Vec::new();
    let mut interrupt_payloads: Vec<Py<PyAny>> = Vec::new();
    let mut interrupt_chunks: Vec<&StreamChunk> = Vec::new();
    // Static `interrupt_before`/`interrupt_after` breakpoints pause like LangGraph: no
    // interrupt payload, just an empty `__interrupt__` marker in updates mode.
    let mut static_pauses = 0usize;

    let convert_regular = |converted: &mut Vec<Py<PyAny>>,
                           values_indices: &mut Vec<usize>,
                           chunk: &StreamChunk|
     -> PyResult<()> {
        if matches!(chunk, StreamChunk::Values { .. }) {
            values_indices.push(converted.len());
        }
        if v2 {
            converted.extend(chunk_to_py_v2_parts(py, chunk)?);
        } else {
            converted.push(chunk_to_py_v1(py, chunk, multi_mode)?);
        }
        Ok(())
    };

    for chunk in chunks {
        if debug_mode {
            match chunk {
                StreamChunk::Debug { .. } => {
                    convert_regular(&mut converted, &mut values_indices, chunk)?
                }
                StreamChunk::Task { .. } => {
                    let parts = task_debug_parts(py, chunk, v2)?;
                    if v2 || !multi_mode {
                        converted.extend(parts);
                    } else {
                        for part in parts {
                            converted.push(
                                PyTuple::new(
                                    py,
                                    [PyString::new(py, "debug").into_any(), part.into_bound(py)],
                                )?
                                .into_any()
                                .unbind(),
                            );
                        }
                    }
                    if tasks_enabled {
                        convert_regular(&mut converted, &mut values_indices, chunk)?;
                    }
                }
                StreamChunk::Interrupt {
                    static_pause: true, ..
                } => static_pauses += 1,
                StreamChunk::Interrupt { .. } => {
                    // Recorded only: the shared tail injects it into the values payload or
                    // emits it standalone, so it is never emitted twice.
                    interrupt_payloads.push(interrupt_payload(py, chunk, v2)?);
                    interrupt_chunks.push(chunk);
                }
                _ if chunk.matches_modes(requested) => {
                    convert_regular(&mut converted, &mut values_indices, chunk)?;
                }
                _ => {}
            }
            continue;
        }
        if let StreamChunk::Interrupt { static_pause, .. } = chunk {
            if *static_pause {
                static_pauses += 1;
                continue;
            }
        }
        if matches!(chunk, StreamChunk::Interrupt { .. }) {
            interrupt_payloads.push(interrupt_payload(py, chunk, v2)?);
            interrupt_chunks.push(chunk);
            continue;
        }
        convert_regular(&mut converted, &mut values_indices, chunk)?;
    }

    if let Some(index) = values_indices.last().copied()
        && let Some(item) = converted.get_mut(index)
        && let Some(dict) = values_dict(item.bind(py))
    {
        let key = if v2 { "interrupts" } else { "__interrupt__" };
        if !interrupt_payloads.is_empty() {
            dict.set_item(key, PyList::new(py, interrupt_payloads)?)?;
        }
    } else {
        // Updates/other modes without a values payload surface interrupts as their own chunk.
        for chunk in interrupt_chunks {
            if v2 {
                converted.extend(chunk_to_py_v2_parts(py, chunk)?);
            } else {
                converted.push(chunk_to_py_v1(py, chunk, multi_mode)?);
            }
        }
        // A static breakpoint shows up only as an empty `__interrupt__` marker in
        // updates mode, matching LangGraph.
        if static_pauses > 0
            && interrupt_payloads.is_empty()
            && requested.contains(&crate::StreamMode::Updates)
        {
            let marker = PyDict::new(py);
            marker.set_item("__interrupt__", PyTuple::empty(py))?;
            if v2 {
                let part = PyDict::new(py);
                part.set_item("type", "updates")?;
                part.set_item("data", marker)?;
                part.set_item("ns", PyTuple::empty(py))?;
                converted.push(part.into_any().unbind());
            } else if multi_mode {
                converted.push(
                    PyTuple::new(
                        py,
                        [PyString::new(py, "updates").into_any(), marker.into_any()],
                    )?
                    .into_any()
                    .unbind(),
                );
            } else {
                converted.push(marker.into_any().unbind());
            }
        }
    }
    Ok(converted)
}

fn values_dict<'py>(item: &Bound<'py, PyAny>) -> Option<Bound<'py, PyDict>> {
    if let Ok(tuple) = item.cast::<PyTuple>() {
        return tuple.get_item(1).ok()?.cast::<PyDict>().ok().cloned();
    }
    item.cast::<PyDict>().ok().cloned()
}

fn interrupt_payload(py: Python<'_>, chunk: &StreamChunk, v2: bool) -> PyResult<Py<PyAny>> {
    let item = if v2 {
        chunk_to_py_v2_parts(py, chunk)?
            .into_iter()
            .next()
            .ok_or_else(|| PyTypeError::new_err("interrupt chunk produced no part"))?
    } else {
        chunk_to_py_v1(py, chunk, false)?
    };
    let dict = item
        .bind(py)
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("interrupt chunk is not a dict"))?;
    let key = "__interrupt__";
    let payload = if v2 {
        dict.get_item("data")?
            .and_then(|data| data.cast_into::<PyDict>().ok())
            .and_then(|data| data.get_item(key).ok().flatten())
    } else {
        dict.get_item(key)?
    };
    payload
        .map(|value| value.unbind())
        .ok_or_else(|| PyTypeError::new_err("interrupt chunk has no payload"))
}

#[path = "stream_parts.rs"]
mod parts;
pub(crate) use parts::*;
