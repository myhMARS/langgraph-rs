//! Conversion helpers between Python objects and kernel values.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use super::types::{PyCommand, PySend};
use serde_json::Map;

use super::GraphInterrupt;
use crate::*;

// ---------------------------------------------------------------------------
// JSON <-> Python conversion
// ---------------------------------------------------------------------------

pub(super) fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string())
}

pub(super) fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(value) = obj.extract::<bool>() {
        return Ok(Value::Bool(value));
    }
    if let Ok(value) = obj.extract::<i64>() {
        return Ok(Value::from(value));
    }
    if let Ok(value) = obj.extract::<f64>() {
        return Ok(Value::from(value));
    }
    if let Ok(value) = obj.extract::<String>() {
        return Ok(Value::String(value));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = Map::new();
        for (key, value) in dict.iter() {
            let key = key.extract::<String>().map_err(|_| {
                PyTypeError::new_err("JSON objects emitted from Python need string keys")
            })?;
            map.insert(key, py_to_value(&value)?);
        }
        return Ok(Value::Object(map));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_to_value(&item)?);
        }
        return Ok(Value::Array(items));
    }
    if let Ok(tuple) = obj.cast::<pyo3::types::PyTuple>() {
        let mut items = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            items.push(py_to_value(&item)?);
        }
        return Ok(Value::Array(items));
    }
    Err(PyTypeError::new_err(format!(
        "value of type `{}` cannot be converted to JSON; use None, bool, int, float, str, list, tuple or dict",
        type_name(obj)
    )))
}

pub(super) fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    use pyo3::conversion::IntoPyObjectExt;

    Ok(match value {
        Value::Null => py.None(),
        Value::Bool(value) => value.into_py_any(py)?,
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                value.into_py_any(py)?
            } else if let Some(value) = number.as_u64() {
                value.into_py_any(py)?
            } else {
                number.as_f64().unwrap_or_default().into_py_any(py)?
            }
        }
        Value::String(value) => value.into_py_any(py)?,
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, value_to_py(py, item)?)?;
            }
            dict.into_any().unbind()
        }
    })
}

pub(super) fn json_state_to_py(py: Python<'_>, state: &JsonState) -> PyResult<Py<PyAny>> {
    value_to_py(py, &Value::Object(state.0.clone()))
}

pub(super) fn state_from_py(obj: Option<&Bound<'_, PyAny>>) -> PyResult<JsonState> {
    let Some(obj) = obj else {
        return Ok(JsonState::default());
    };
    if obj.is_none() {
        return Ok(JsonState::default());
    }
    match py_to_value(obj)? {
        Value::Object(map) => Ok(JsonState(map)),
        other => Err(PyTypeError::new_err(format!(
            "graph state must be a dict, got {}",
            type_name(obj).to_string() + " " + &other.to_string()
        ))),
    }
}

pub(super) fn usage_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Usage> {
    let dict = obj
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("usage must be a dict"))?;
    let mut usage = Usage::new();
    if let Some(value) = dict.get_item("model")? {
        usage.model = Some(value.extract::<String>()?);
    }
    if let Some(value) = dict.get_item("input_tokens")? {
        usage.input_tokens = Some(value.extract::<u64>()?);
    }
    if let Some(value) = dict.get_item("output_tokens")? {
        usage.output_tokens = Some(value.extract::<u64>()?);
    }
    if let Some(value) = dict.get_item("cost_usd")? {
        usage.cost_usd = Some(value.extract::<f64>()?);
    }
    if let Some(value) = dict.get_item("metadata")? {
        if let Value::Object(metadata) = py_to_value(&value)? {
            usage.metadata = Some(metadata);
        }
    }
    Ok(usage)
}

pub(super) fn to_pyerr(py: Python<'_>, error: Error) -> PyErr {
    match error {
        Error::Interrupt(interrupt) => {
            let payload = value_to_py(py, &interrupt.value).unwrap_or_else(|_| py.None());
            PyErr::new::<GraphInterrupt, _>((payload,))
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

pub(super) fn pyerr_to_error(py: Python<'_>, error: PyErr) -> Error {
    if error.is_instance_of::<GraphInterrupt>(py) {
        let value = error
            .value(py)
            .getattr("args")
            .ok()
            .and_then(|args| args.get_item(0).ok())
            .and_then(|payload| py_to_value(&payload).ok())
            .unwrap_or(Value::Null);
        Error::Interrupt(Interrupt::new(String::new(), value))
    } else {
        Error::other(format!(
            "python node raised {}: {error}",
            type_name(error.value(py))
        ))
    }
}

// ---------------------------------------------------------------------------
// Graph construction helpers
// ---------------------------------------------------------------------------

/// Translate the LangGraph-style `"START"` / `"END"` aliases to the engine markers.
pub(super) fn normalize_node(name: &str) -> String {
    match name {
        "START" => crate::START.to_string(),
        "END" => crate::END.to_string(),
        other => other.to_string(),
    }
}

pub(super) fn route_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Route> {
    if obj.is_none() {
        return Ok(Route::end());
    }
    if let Ok(name) = obj.extract::<String>() {
        let name = normalize_node(&name);
        return Ok(if name == crate::END {
            Route::end()
        } else {
            Route::node(name)
        });
    }
    if let Ok(send) = obj.extract::<PyRef<'_, super::types::PySend>>() {
        return Ok(Route::send(send.to_send(obj.py())?));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut names = Vec::with_capacity(list.len());
        let mut sends = Vec::new();
        for item in list.iter() {
            if let Ok(send) = item.extract::<PyRef<'_, super::types::PySend>>() {
                sends.push(send.to_send(obj.py())?);
                continue;
            }
            let name = normalize_node(&item.extract::<String>()?);
            if name != crate::END {
                names.push(name);
            }
        }
        return Ok(Route::nodes(names).with_sends(sends));
    }
    Err(PyTypeError::new_err(
        "router must return None, a node name, a Send, or a list of node names / Send objects",
    ))
}

pub(super) fn node_output_from_py(obj: &Bound<'_, PyAny>) -> PyResult<NodeOutput> {
    if obj.is_none() {
        return Ok(NodeOutput::new());
    }
    if let Ok(command) = obj.extract::<PyRef<PyCommand>>() {
        return command.to_output(obj.py());
    }
    if let Ok(send) = obj.extract::<PyRef<PySend>>() {
        return Ok(NodeOutput::new().send(send.to_send(obj.py())?));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut output = NodeOutput::new();
        for item in list.iter() {
            if let Ok(send) = item.extract::<PyRef<PySend>>() {
                output = output.send(send.to_send(obj.py())?);
            } else if let Ok(command) = item.extract::<PyRef<PyCommand>>() {
                let converted = command.to_output(obj.py())?;
                let (updates, replaces, goto, sends) = converted.into_parts();
                for (channel, value) in updates {
                    output = output.update(channel, value);
                }
                for (channel, value) in replaces {
                    output = output.replace(channel, value);
                }
                for target in goto {
                    output = output.goto(target);
                }
                for send in sends {
                    output = output.send(send);
                }
            } else {
                return Err(PyTypeError::new_err(
                    "a node list must contain Send or Command objects",
                ));
            }
        }
        return Ok(output);
    }
    let dict = obj.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err("a node must return a dict of channel updates, Send/Command, or None")
    })?;

    let mut output = NodeOutput::new();
    for (key, value) in dict.iter() {
        let key = key.extract::<String>()?;
        match key.as_str() {
            "__goto__" => {
                if let Ok(names) = value.extract::<Vec<String>>() {
                    for name in names {
                        output = output.goto(name);
                    }
                } else {
                    output = output.goto(value.extract::<String>()?);
                }
            }
            "__usage__" => output = output.with_usage(usage_from_py(&value)?),
            "__send__" | "send" => {
                let sends = value
                    .extract::<Vec<Py<PyAny>>>()
                    .map_err(|_| PyTypeError::new_err("`send` must be a list of Send objects"))?;
                for send in sends {
                    let send = send.bind(obj.py()).extract::<PyRef<PySend>>()?;
                    output = output.send(send.to_send(obj.py())?);
                }
            }
            _ => output = output.update(key, py_to_value(&value)?),
        }
    }
    Ok(output)
}

pub(super) fn schema_from_py(schema: Option<&Bound<'_, PyAny>>) -> PyResult<StateSchema> {
    let Some(schema) = schema else {
        return Ok(StateSchema::dynamic());
    };
    if schema.is_none() {
        return Ok(StateSchema::dynamic());
    }
    let dict = schema
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("schema must be a dict of channel -> spec"))?;

    let mut entries = Vec::with_capacity(dict.len());
    for (key, spec) in dict.iter() {
        let name: String = key.extract()?;
        let spec = spec.extract::<String>().map_err(|_| {
            PyTypeError::new_err(format!(
                "channel spec for `{name}` must be a string such as \"last_value\", \"topic\", \"append\""
            ))
        })?;
        let spec = match spec.as_str() {
            "last" | "value" => "last_value".to_string(),
            "add_messages" | "messages" => "messages".to_string(),
            "any" | "any_value" => "any_value".to_string(),
            "ephemeral" | "ephemeral_value" => "ephemeral_value".to_string(),
            "untracked" | "untracked_value" => "untracked_value".to_string(),
            other => other.to_string(),
        };
        entries.push((name, spec));
    }
    Kernel::builtin()
        .schema(entries)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(super) fn targets_from_py(to: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(name) = to.extract::<String>() {
        return Ok(vec![normalize_node(&name)]);
    }
    to.extract::<Vec<String>>()
        .map(|names| names.iter().map(|name| normalize_node(name)).collect())
        .map_err(|_| {
            PyTypeError::new_err("edge target must be a node name or a list of node names")
        })
}

pub(super) fn path_map_from_py(path_map: &Bound<'_, PyAny>) -> PyResult<Option<PathMap>> {
    if path_map.is_none() {
        return Ok(None);
    }
    let dict = path_map
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("path_map must be a dict"))?;
    let mut map = PathMap::new();
    for (key, targets) in dict.iter() {
        let key: String = key.extract()?;
        let targets = if let Ok(name) = targets.extract::<String>() {
            vec![normalize_node(&name)]
        } else {
            targets
                .extract::<Vec<String>>()?
                .iter()
                .map(|name| normalize_node(name))
                .collect()
        };
        map = map.insert(key, targets);
    }
    Ok(Some(map))
}

pub(super) fn node_from_python(
    name: String,
    callable: Py<PyAny>,
) -> impl Fn(&JsonState) -> crate::Result<NodeOutput> + std::marker::Send + std::marker::Sync + 'static
{
    move |state: &JsonState| {
        Python::attach(|py| {
            let state_obj = json_state_to_py(py, state).map_err(|error| {
                Error::other(format!(
                    "failed to convert state for node `{name}`: {error}"
                ))
            })?;
            let result = callable
                .call1(py, (state_obj,))
                .map_err(|error| pyerr_to_error(py, error))?;
            node_output_from_py(result.bind(py)).map_err(|error| {
                Error::other(format!("node `{name}` returned an invalid output: {error}"))
            })
        })
    }
}

pub(super) fn router_from_python(
    callable: Py<PyAny>,
) -> impl Fn(&JsonState) -> crate::Result<Route> + std::marker::Send + std::marker::Sync + 'static {
    move |state: &JsonState| {
        Python::attach(|py| {
            let state_obj = json_state_to_py(py, state).map_err(|error| {
                Error::other(format!("failed to convert router state: {error}"))
            })?;
            let result = callable
                .call1(py, (state_obj,))
                .map_err(|error| pyerr_to_error(py, error))?;
            route_from_py(result.bind(py))
                .map_err(|error| Error::other(format!("router returned an invalid route: {error}")))
        })
    }
}

pub(super) fn run_config(thread_id: Option<String>) -> RunConfig {
    match thread_id {
        Some(thread_id) => RunConfig::new().with_thread(thread_id),
        None => RunConfig::new(),
    }
}

pub(super) fn snapshot_to_py(
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
            Ok(item.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("interrupts", interrupts)?;
    let tasks = snapshot
        .tasks
        .iter()
        .map(|task| {
            let item = PyDict::new(py);
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
    Ok(dict.into_any().unbind())
}

pub(super) fn chunk_to_py(py: Python<'_>, chunk: &StreamChunk) -> PyResult<Py<PyAny>> {
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
        StreamChunk::Checkpoint { checkpoint_id } => {
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
