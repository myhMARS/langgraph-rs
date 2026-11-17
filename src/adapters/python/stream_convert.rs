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
    let tasks = snapshot
        .tasks
        .iter()
        .map(|task| {
            let item = PyDict::new(py);
            item.set_item("id", task.id.clone())?;
            item.set_item("node", task.node.clone())?;
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

fn iso_timestamp(millis: i64) -> String {
    let seconds = millis.div_euclid(1000);
    let millis_part = millis.rem_euclid(1000);
    let days = seconds.div_euclid(86_400);
    let day_seconds = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = day_seconds / 3_600;
    let minute = (day_seconds % 3_600) / 60;
    let second = day_seconds % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis_part:03}Z")
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    if month <= 2 {
        year += 1;
    }
    (year, month, day)
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
                iso_timestamp(crate::kernel::checkpoint::now_millis() as i64),
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
        StreamChunk::Interrupt { node, value } => {
            dict.set_item("type", "interrupt")?;
            let data = PyDict::new(py);
            data.set_item("node", node)?;
            data.set_item("value", value_to_py(py, value)?)?;
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
    let dict = PyDict::new(py);
    match chunk {
        StreamChunk::Subgraph { namespace, chunk } => {
            dict.set_item("type", "subgraph")?;
            dict.set_item("namespace", namespace.clone())?;
            dict.set_item("chunk", chunk_to_py(py, chunk)?)?;
        }
        StreamChunk::Usage { execution } => {
            dict.set_item("type", "usage")?;
            dict.set_item("node", execution.node.clone())?;
            dict.set_item("step", execution.step)?;
            dict.set_item("duration_millis", execution.duration_millis)?;
            if let Some(usage) = &execution.usage {
                let usage = serde_json::to_value(usage)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                dict.set_item("usage", value_to_py(py, &usage)?)?;
            }
        }
        StreamChunk::Values { step, values } => {
            dict.set_item("type", "values")?;
            dict.set_item("step", *step)?;
            dict.set_item("values", value_to_py(py, values)?)?;
        }
        StreamChunk::Updates { step, node, update } => {
            dict.set_item("type", "updates")?;
            dict.set_item("step", *step)?;
            dict.set_item("node", node.clone())?;
            dict.set_item("update", value_to_py(py, update)?)?;
        }
        StreamChunk::Custom { node, value } => {
            dict.set_item("type", "custom")?;
            dict.set_item("node", node.clone())?;
            dict.set_item("value", value_to_py(py, value)?)?;
        }
        StreamChunk::Interrupt { node, value } => {
            dict.set_item("type", "interrupt")?;
            dict.set_item("node", node.clone())?;
            dict.set_item("value", value_to_py(py, value)?)?;
        }
        StreamChunk::Checkpoint { checkpoint_id, .. } => {
            dict.set_item("type", "checkpoint")?;
            dict.set_item("checkpoint_id", checkpoint_id.clone())?;
        }
        StreamChunk::Message { node, message } => {
            dict.set_item("type", "messages")?;
            dict.set_item("node", node.clone())?;
            dict.set_item("message", value_to_py(py, message)?)?;
        }
        StreamChunk::Task {
            step,
            node,
            input,
            result,
            error,
        } => {
            dict.set_item("type", "tasks")?;
            dict.set_item("step", *step)?;
            dict.set_item("node", node.clone())?;
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
        }
        StreamChunk::Debug { step, payload } => {
            dict.set_item("type", "debug")?;
            dict.set_item("step", *step)?;
            dict.set_item("payload", value_to_py(py, payload)?)?;
        }
    }
    Ok(dict.into_any().unbind())
}
