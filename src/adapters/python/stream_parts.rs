//! LangGraph v1/v2 stream part conversion (chunks -> Python payloads).

use pyo3::exceptions::PyTypeError;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};
use serde_json::Value;

use super::super::*;
use crate::_internal::stream::StreamChunk;

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
            inner.set_item("ns", PyTuple::new(py, ns)?)?;
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
        StreamChunk::Tool { value, .. } => {
            dict.set_item("type", "tools")?;
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
                crate::checkpoint::iso_timestamp(crate::checkpoint::now_millis()),
            )?;
            data.set_item("step", step)?;
            data.set_item("payload", value_to_py(py, payload)?)?;
            dict.set_item("data", data)?;
        }
        StreamChunk::Task {
            id,
            node,
            input,
            result,
            error,
            started,
            interrupts,
            triggers,
            ..
        } => {
            dict.set_item("type", "tasks")?;
            let data = PyDict::new(py);
            data.set_item("id", id)?;
            data.set_item("name", node)?;
            if *started {
                data.set_item(
                    "input",
                    match input {
                        Some(input) => value_to_py(py, input)?,
                        None => py.None(),
                    },
                )?;
                data.set_item("triggers", triggers)?;
            } else {
                data.set_item(
                    "error",
                    match error {
                        Some(error) => {
                            value_to_py(py, &serde_json::Value::String(error.message.clone()))?
                        }
                        None => py.None(),
                    },
                )?;
                let interrupt_values: Vec<serde_json::Value> = interrupts
                    .iter()
                    .filter_map(|interrupt| serde_json::to_value(interrupt).ok())
                    .collect();
                data.set_item(
                    "interrupts",
                    value_to_py(py, &serde_json::Value::Array(interrupt_values))?,
                )?;
                data.set_item(
                    "result",
                    match result {
                        Some(result) => value_to_py(py, result)?,
                        None => py.None(),
                    },
                )?;
            }
            dict.set_item("data", data)?;
        }
        StreamChunk::Checkpoint { payload, .. } => {
            dict.set_item("type", "checkpoints")?;
            dict.set_item("data", value_to_py(py, payload)?)?;
        }
        StreamChunk::Interrupt {
            node,
            id,
            value,
            response_schema,
            ..
        } => {
            dict.set_item("type", "updates")?;
            let data = PyDict::new(py);
            let interrupt = Py::new(
                py,
                crate::adapters::python::types::PyInterrupt {
                    id: id.clone(),
                    value: value_to_py(py, value)?,
                    node: Some(node.clone()),
                    resumable: true,
                    response_schema: match response_schema {
                        Some(schema) => Some(value_to_py(py, schema)?),
                        None => None,
                    },
                },
            )?;
            data.set_item("__interrupt__", PyTuple::new(py, [interrupt])?)?;
            dict.set_item("data", data)?;
        }
        StreamChunk::Usage { execution } => {
            dict.set_item("type", "usage")?;
            let execution = serde_json::to_value(execution)
                .map_err(|error| PyTypeError::new_err(error.to_string()))?;
            dict.set_item("data", value_to_py(py, &execution)?)?;
        }
    }
    dict.set_item("ns", PyTuple::new(py, Vec::<String>::new())?)?;
    Ok(dict.into_any().unbind())
}

/// LangGraph v2 task events: emit a start payload and a result payload.
pub(crate) fn chunk_to_py_v2_parts(
    py: Python<'_>,
    chunk: &StreamChunk,
) -> PyResult<Vec<Py<PyAny>>> {
    if let StreamChunk::Subgraph {
        namespace,
        chunk: inner,
    } = chunk
    {
        let namespace: Vec<String> = if namespace.is_empty() {
            Vec::new()
        } else {
            namespace.split('|').map(str::to_owned).collect()
        };
        let parts = chunk_to_py_v2_parts(py, inner)?;
        let mut wrapped = Vec::with_capacity(parts.len());
        for part in parts {
            if let Ok(dict) = part.bind(py).cast::<PyDict>() {
                dict.set_item("ns", PyTuple::new(py, namespace.clone())?)?;
            }
            wrapped.push(part);
        }
        return Ok(wrapped);
    }
    if let StreamChunk::Task {
        id,
        node,
        input,
        result,
        error,
        started,
        interrupts,
        triggers,
        ..
    } = chunk
    {
        if *started {
            let start = PyDict::new(py);
            start.set_item("type", "tasks")?;
            let start_data = PyDict::new(py);
            start_data.set_item("id", id)?;
            start_data.set_item("name", node)?;
            start_data.set_item(
                "input",
                match input {
                    Some(input) => value_to_py(py, input)?,
                    None => py.None(),
                },
            )?;
            start_data.set_item("triggers", triggers)?;
            start.set_item("data", start_data)?;
            start.set_item("ns", PyTuple::new(py, Vec::<String>::new())?)?;
            return Ok(vec![start.into_any().unbind()]);
        }
        let start = PyDict::new(py);
        start.set_item("type", "tasks")?;
        let start_data = PyDict::new(py);
        start_data.set_item("id", id)?;
        start_data.set_item("name", node)?;
        start_data.set_item(
            "input",
            match input {
                Some(input) => value_to_py(py, input)?,
                None => py.None(),
            },
        )?;
        start_data.set_item("triggers", Vec::<String>::new())?;
        start.set_item("data", start_data)?;
        start.set_item("ns", PyTuple::new(py, Vec::<String>::new())?)?;

        let finish = PyDict::new(py);
        finish.set_item("type", "tasks")?;
        let finish_data = PyDict::new(py);
        finish_data.set_item("id", id)?;
        finish_data.set_item("name", node)?;
        finish_data.set_item(
            "error",
            value_to_py(
                py,
                &match error {
                    Some(error) => Value::String(error.message.clone()),
                    None => Value::Null,
                },
            )?,
        )?;
        let interrupt_values: Vec<serde_json::Value> = interrupts
            .iter()
            .filter_map(|interrupt| serde_json::to_value(interrupt).ok())
            .collect();
        finish_data.set_item(
            "interrupts",
            value_to_py(py, &serde_json::Value::Array(interrupt_values))?,
        )?;
        finish_data.set_item(
            "result",
            match result {
                Some(result) => value_to_py(py, result)?,
                None => py.None(),
            },
        )?;
        finish.set_item("data", finish_data)?;
        finish.set_item("ns", PyTuple::new(py, Vec::<String>::new())?)?;
        return Ok(vec![finish.into_any().unbind()]);
    }
    Ok(vec![chunk_to_py_v2(py, chunk)?])
}

/// LangGraph returns an empty `StateSnapshot` for a thread without checkpoints.
pub(crate) fn task_debug_parts(
    py: Python<'_>,
    chunk: &StreamChunk,
    v2: bool,
) -> PyResult<Vec<Py<PyAny>>> {
    let StreamChunk::Task {
        step,
        id,
        node,
        input,
        result,
        error,
        started,
        interrupts,
        triggers,
        ..
    } = chunk
    else {
        return Ok(Vec::new());
    };
    let timestamp = crate::checkpoint::iso_timestamp(crate::checkpoint::now_millis());
    let records = if *started {
        vec![(
            "task",
            serde_json::json!({
                "id": id,
                "name": node,
                "input": input,
                "triggers": triggers,
            }),
        )]
    } else {
        vec![(
            "task_result",
            serde_json::json!({
                "id": id,
                "name": node,
                "error": error.as_ref().map(|error| error.message.clone()),
                "interrupts": interrupts,
                "result": result,
            }),
        )]
    };
    let mut parts = Vec::new();
    for (kind, payload) in records {
        if v2 {
            let data = PyDict::new(py);
            data.set_item("type", kind)?;
            data.set_item("timestamp", &timestamp)?;
            data.set_item("step", step)?;
            data.set_item("payload", value_to_py(py, &payload)?)?;
            let part = PyDict::new(py);
            part.set_item("type", "debug")?;
            part.set_item("ns", PyTuple::new(py, Vec::<String>::new())?)?;
            part.set_item("data", data)?;
            parts.push(part.into_any().unbind());
        } else {
            let record = PyDict::new(py);
            record.set_item("type", kind)?;
            record.set_item("timestamp", &timestamp)?;
            record.set_item("step", step)?;
            record.set_item("payload", value_to_py(py, &payload)?)?;
            parts.push(record.into_any().unbind());
        }
    }
    Ok(parts)
}

/// Convert collected chunks into Python stream items (v1 or v2), including
/// debug task synthesis and interrupt injection into the matching values part.
pub(crate) fn chunk_to_py_v1(
    py: Python<'_>,
    chunk: &StreamChunk,
    multi_mode: bool,
) -> PyResult<Py<PyAny>> {
    if let StreamChunk::Subgraph {
        namespace,
        chunk: inner,
    } = chunk
    {
        let namespace: Vec<String> = if namespace.is_empty() {
            Vec::new()
        } else {
            namespace.split('|').map(str::to_owned).collect()
        };
        let namespace = PyTuple::new(py, namespace)?;
        let result = if multi_mode {
            let payload = chunk_to_py(py, inner)?;
            let mode = inner.mode().map(|mode| mode.as_str()).unwrap_or("updates");
            (namespace, mode, payload)
                .into_pyobject(py)?
                .into_any()
                .unbind()
        } else {
            let payload = chunk_to_py_v1(py, inner, false)?;
            (namespace, payload).into_pyobject(py)?.into_any().unbind()
        };
        return Ok(result);
    }
    let payload = chunk_to_py(py, chunk)?;
    if multi_mode {
        let mode = chunk.mode().map(|mode| mode.as_str()).unwrap_or("updates");
        return Ok((mode, payload).into_pyobject(py)?.into_any().unbind());
    }
    Ok(payload)
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
            let namespace = PyTuple::new(py, namespace)?;
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
        StreamChunk::Tool { value, .. } => value_to_py(py, value),
        StreamChunk::Interrupt {
            node,
            id,
            value,
            response_schema,
            ..
        } => {
            let interrupt = Py::new(
                py,
                crate::adapters::python::types::PyInterrupt {
                    id: id.clone(),
                    value: value_to_py(py, value)?,
                    node: Some(node.clone()),
                    resumable: true,
                    response_schema: match response_schema {
                        Some(schema) => Some(value_to_py(py, schema)?),
                        None => None,
                    },
                },
            )?;
            let data = PyDict::new(py);
            data.set_item("__interrupt__", PyTuple::new(py, [interrupt])?)?;
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
            id,
            node,
            input,
            result,
            error,
            started,
            interrupts,
            triggers,
        } => {
            let dict = PyDict::new(py);
            dict.set_item("id", id)?;
            dict.set_item("name", node)?;
            dict.set_item("step", *step)?;
            if *started {
                dict.set_item(
                    "input",
                    match input {
                        Some(input) => value_to_py(py, input)?,
                        None => py.None(),
                    },
                )?;
                dict.set_item("triggers", triggers)?;
            } else {
                dict.set_item(
                    "error",
                    match error {
                        Some(error) => {
                            value_to_py(py, &serde_json::Value::String(error.message.clone()))?
                        }
                        None => py.None(),
                    },
                )?;
                let interrupt_values: Vec<serde_json::Value> = interrupts
                    .iter()
                    .filter_map(|interrupt| serde_json::to_value(interrupt).ok())
                    .collect();
                dict.set_item(
                    "interrupts",
                    value_to_py(py, &serde_json::Value::Array(interrupt_values))?,
                )?;
                dict.set_item(
                    "result",
                    match result {
                        Some(result) => value_to_py(py, result)?,
                        None => py.None(),
                    },
                )?;
            }
            Ok(dict.into_any().unbind())
        }
        StreamChunk::Debug { step, payload } => {
            let dict = PyDict::new(py);
            dict.set_item("type", "checkpoint")?;
            dict.set_item(
                "timestamp",
                crate::checkpoint::iso_timestamp(crate::checkpoint::now_millis()),
            )?;
            dict.set_item("step", *step)?;
            dict.set_item("payload", value_to_py(py, payload)?)?;
            Ok(dict.into_any().unbind())
        }
    }
}
