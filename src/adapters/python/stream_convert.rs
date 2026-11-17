//! Snapshot and stream-chunk conversion helpers.

use pyo3::exceptions::PyTypeError;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};
use serde_json::Value;

use super::*;
use crate::JsonState;
use crate::kernel::stream::StreamChunk;

pub(crate) fn snapshot_to_py(
    py: Python<'_>,
    snapshot: &crate::StateSnapshot<JsonState>,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("values", json_state_to_py(py, &snapshot.values)?)?;
    dict.set_item("next", snapshot.next.clone())?;
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
    dict.set_item(
        "metadata",
        value_to_py(py, &Value::Object(snapshot.metadata.clone()))?,
    )?;
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
    dict.set_item("interrupts", interrupts)?;
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
            item.set_item("interrupts", Vec::<Py<PyAny>>::new())?;
            item.set_item("state", py.None())?;
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
    dict.set_item("tasks", tasks)?;
    Ok(Py::new(
        py,
        crate::adapters::python::types::PyStateSnapshot {
            data: dict.unbind(),
        },
    )?
    .into_any())
}

/// LangGraph v2 stream-part shape: `{type, ns, data}`.
pub(crate) fn chunk_to_py_v2(py: Python<'_>, chunk: &StreamChunk) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    match chunk {
        StreamChunk::Subgraph { namespace, chunk } => {
            let inner = chunk_to_py_v2(py, chunk)?;
            let inner = inner.bind(py).cast::<PyDict>()?;
            let ns: Vec<String> = if namespace.is_empty() {
                Vec::new()
            } else {
                namespace.split('|').map(str::to_owned).collect()
            };
            inner.set_item("ns", ns)?;
            return Ok(inner.clone().into_any().unbind());
        }
        StreamChunk::Values { values, .. } => {
            dict.set_item("type", "values")?;
            dict.set_item("data", value_to_py(py, values)?)?;
            dict.set_item("interrupts", PyTuple::new(py, Vec::<Py<PyAny>>::new())?)?;
        }
        StreamChunk::Updates { node, update, .. } => {
            dict.set_item("type", "updates")?;
            let data = PyDict::new(py);
            data.set_item(node, value_to_py(py, update)?)?;
            dict.set_item("data", data)?;
        }
        StreamChunk::Custom { value, .. } => {
            dict.set_item("type", "custom")?;
            dict.set_item("data", value_to_py(py, value)?)?;
        }
        StreamChunk::Message { node, message, .. } => {
            dict.set_item("type", "messages")?;
            let data = PyList::empty(py);
            data.append(value_to_py(py, message)?)?;
            let metadata = PyDict::new(py);
            metadata.set_item("langgraph_node", node)?;
            let tags = PyList::empty(py);
            if let Some(object) = message.as_object() {
                if let Some(values) = object.get("tags").and_then(serde_json::Value::as_array) {
                    for tag in values {
                        tags.append(value_to_py(py, tag)?)?;
                    }
                }
                if let Some(extra) = object
                    .get("metadata")
                    .and_then(serde_json::Value::as_object)
                {
                    for (key, value) in extra {
                        if key != "langgraph_node" && key != "tags" {
                            metadata.set_item(key, value_to_py(py, value)?)?;
                        }
                    }
                }
            }
            metadata.set_item("tags", tags)?;
            let message = data.get_item(0)?;
            let tuple = PyTuple::new(py, [message, metadata.into_any()])?;
            dict.set_item("data", tuple)?;
        }
        StreamChunk::Debug { step, payload } => {
            dict.set_item("type", "debug")?;
            let data = PyDict::new(py);
            data.set_item("type", "checkpoint")?;
            data.set_item(
                "timestamp",
                crate::kernel::checkpoint::iso_timestamp(crate::kernel::checkpoint::now_millis()),
            )?;
            data.set_item("step", step)?;
            data.set_item("payload", value_to_py(py, payload)?)?;
            dict.set_item("data", data)?;
        }
        StreamChunk::Task {
            node,
            input,
            result,
            error,
            ..
        } => {
            dict.set_item("type", "tasks")?;
            let data = PyDict::new(py);
            data.set_item("node", node)?;
            data.set_item("name", node)?;
            data.set_item("path", Vec::<String>::new())?;
            data.set_item("triggers", Vec::<String>::new())?;
            data.set_item("interrupts", Vec::<Py<PyAny>>::new())?;
            if let Some(input) = input {
                data.set_item("input", value_to_py(py, input)?)?;
            }
            if let Some(result) = result {
                data.set_item("result", value_to_py(py, result)?)?;
            }
            if let Some(error) = error {
                let error = serde_json::to_value(error)
                    .map_err(|error| PyTypeError::new_err(error.to_string()))?;
                data.set_item("error", value_to_py(py, &error)?)?;
            }
            dict.set_item("data", data)?;
        }
        StreamChunk::Checkpoint { payload, .. } => {
            dict.set_item("type", "checkpoints")?;
            dict.set_item("data", value_to_py(py, payload)?)?;
        }
        StreamChunk::Interrupt { node, id, value } => {
            dict.set_item("type", "updates")?;
            let data = PyDict::new(py);
            let interrupt = PyDict::new(py);
            interrupt.set_item("id", id)?;
            interrupt.set_item("node", node)?;
            interrupt.set_item("value", value_to_py(py, value)?)?;
            data.set_item("__interrupt__", vec![interrupt])?;
            dict.set_item("data", data)?;
        }
        StreamChunk::Usage { execution } => {
            dict.set_item("type", "usage")?;
            let execution = serde_json::to_value(execution)
                .map_err(|error| PyTypeError::new_err(error.to_string()))?;
            dict.set_item("data", value_to_py(py, &execution)?)?;
        }
    }
    dict.set_item("ns", Vec::<String>::new())?;
    Ok(dict.into_any().unbind())
}

pub(crate) fn chunk_to_py(py: Python<'_>, chunk: &StreamChunk) -> PyResult<Py<PyAny>> {
    match chunk {
        StreamChunk::Subgraph { namespace, chunk } => {
            let namespace: Vec<String> = if namespace.is_empty() {
                Vec::new()
            } else {
                namespace.split('|').map(str::to_owned).collect()
            };
            let inner = chunk_to_py(py, chunk)?;
            Ok((namespace, inner).into_pyobject(py)?.into_any().unbind())
        }
        StreamChunk::Usage { execution } => {
            let dict = PyDict::new(py);
            dict.set_item("node", execution.node.clone())?;
            dict.set_item("step", execution.step)?;
            dict.set_item("duration_millis", execution.duration_millis)?;
            if let Some(usage) = &execution.usage {
                let usage = serde_json::to_value(usage)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                dict.set_item("usage", value_to_py(py, &usage)?)?;
            }
            Ok(dict.into_any().unbind())
        }
        StreamChunk::Values { values, .. } => value_to_py(py, values),
        StreamChunk::Updates { node, update, .. } => {
            let data = PyDict::new(py);
            data.set_item(node, value_to_py(py, update)?)?;
            Ok(data.into_any().unbind())
        }
        StreamChunk::Custom { value, .. } => value_to_py(py, value),
        StreamChunk::Interrupt { node, id, value } => {
            let interrupt = PyDict::new(py);
            interrupt.set_item("id", id)?;
            interrupt.set_item("node", node)?;
            interrupt.set_item("value", value_to_py(py, value)?)?;
            let data = PyDict::new(py);
            data.set_item("__interrupt__", vec![interrupt])?;
            Ok(data.into_any().unbind())
        }
        StreamChunk::Checkpoint { payload, .. } => value_to_py(py, payload),
        StreamChunk::Message { node, message } => {
            let metadata = PyDict::new(py);
            metadata.set_item("langgraph_node", node)?;
            let tags = PyList::empty(py);
            if let Some(object) = message.as_object() {
                if let Some(values) = object.get("tags").and_then(serde_json::Value::as_array) {
                    for tag in values {
                        tags.append(value_to_py(py, tag)?)?;
                    }
                }
                if let Some(extra) = object
                    .get("metadata")
                    .and_then(serde_json::Value::as_object)
                {
                    for (key, value) in extra {
                        if key != "langgraph_node" && key != "tags" {
                            metadata.set_item(key, value_to_py(py, value)?)?;
                        }
                    }
                }
            }
            metadata.set_item("tags", tags)?;
            let result = PyTuple::new(
                py,
                [value_to_py(py, message)?, metadata.into_any().unbind()],
            )?;
            Ok(result.into_any().unbind())
        }
        StreamChunk::Task {
            step,
            node,
            input,
            result,
            error,
        } => {
            let dict = PyDict::new(py);
            dict.set_item("id", node)?;
            dict.set_item("name", node)?;
            dict.set_item("step", *step)?;
            if let Some(input) = input {
                dict.set_item("input", value_to_py(py, input)?)?;
            }
            if let Some(result) = result {
                dict.set_item("result", value_to_py(py, result)?)?;
            }
            if let Some(error) = error {
                let failure = PyDict::new(py);
                failure.set_item("node", error.node.clone())?;
                failure.set_item("message", error.message.clone())?;
                failure.set_item("attempts", error.attempts)?;
                dict.set_item("error", failure)?;
            }
            Ok(dict.into_any().unbind())
        }
        StreamChunk::Debug { step, payload } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "checkpoint")?;
            dict.set_item(
                "timestamp",
                crate::kernel::checkpoint::iso_timestamp(crate::kernel::checkpoint::now_millis()),
            )?;
            dict.set_item("step", *step)?;
            dict.set_item("payload", value_to_py(py, payload)?)?;
            Ok(dict.into_any().unbind())
        }
    }
}
