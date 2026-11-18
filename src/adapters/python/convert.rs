//! Conversion helpers between Python objects and kernel values.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyInt, PyList};

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
    if let Ok(remove) = obj.extract::<PyRef<'_, super::types::PyRemoveMessage>>() {
        return Ok(crate::RemoveMessage::new(remove.id.clone()).to_value());
    }
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(value) = obj.extract::<bool>() {
        return Ok(Value::Bool(value));
    }
    if obj.is_instance_of::<PyInt>() {
        return obj.extract::<i64>().map(Value::from).map_err(|_| {
            pyo3::exceptions::PyOverflowError::new_err(
                "Python integer is outside the i64 range supported by JSON state",
            )
        });
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
            let schema = match &interrupt.response_schema {
                Some(schema) => value_to_py(py, schema).unwrap_or_else(|_| py.None()),
                None => py.None(),
            };
            PyErr::new::<GraphInterrupt, _>((payload, schema, interrupt.id.clone()))
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

pub(super) fn pyerr_to_error(py: Python<'_>, error: PyErr) -> Error {
    if error.is_instance_of::<GraphInterrupt>(py) {
        let args = error.value(py).getattr("args").ok();
        let value = args
            .as_ref()
            .and_then(|args| args.get_item(0).ok())
            .and_then(|payload| py_to_value(&payload).ok())
            .unwrap_or(Value::Null);
        let schema = args
            .as_ref()
            .and_then(|args| args.get_item(1).ok())
            .filter(|schema| !schema.is_none())
            .and_then(|schema| py_to_value(&schema).ok());
        let node = crate::runtime::context::current_node().unwrap_or_default();
        let id = args
            .as_ref()
            .and_then(|args| args.get_item(2).ok())
            .and_then(|id| id.extract::<String>().ok());
        match id {
            Some(id) => Error::Interrupt(Interrupt::with_schema_and_id(node, value, schema, id)),
            None => Error::Interrupt(Interrupt::with_schema(node, value, schema)),
        }
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
    let dict = match schema.cast::<PyDict>() {
        Ok(dict) => dict.clone(),
        Err(_) => {
            let module = schema.py().import("langgraph_rs")?;
            module
                .getattr("_schema_from_class")?
                .call1((schema,))?
                .cast_into::<PyDict>()
                .map_err(|_| PyTypeError::new_err("schema must be a dict of channel -> spec"))?
        }
    };

    let kernel = Kernel::builtin();
    let mut out = StateSchema::dynamic();
    for (key, spec) in dict.iter() {
        let name: String = key.extract()?;
        if let Ok(spec) = spec.extract::<String>() {
            let spec = match spec.as_str() {
                "last" | "value" => "last_value".to_string(),
                "add_messages" | "messages" => "messages".to_string(),
                "any" | "any_value" => "any_value".to_string(),
                "ephemeral" | "ephemeral_value" => "ephemeral_value".to_string(),
                "untracked" | "untracked_value" => "untracked_value".to_string(),
                "topic_current" | "topic_no_accumulate" => "topic_current".to_string(),
                other => other.to_string(),
            };
            let channel = kernel.resolve_channel(&spec).ok_or_else(|| {
                PyValueError::new_err(format!("unknown channel spec `{spec}` for `{name}`"))
            })?;
            out = out.with_channel(name, channel);
            continue;
        }
        if spec.is_callable() {
            let callable = spec.clone().unbind();
            out = out.with_channel(
                name,
                Channel::reducer(move |current: &Value, incoming: &Value| {
                    Python::attach(|py| {
                        let current =
                            value_to_py(py, current).map_err(|error| pyerr_to_error(py, error))?;
                        let incoming =
                            value_to_py(py, incoming).map_err(|error| pyerr_to_error(py, error))?;
                        let result = callable
                            .call1(py, (current, incoming))
                            .map_err(|error| pyerr_to_error(py, error))?;
                        py_to_value(result.bind(py)).map_err(|error| pyerr_to_error(py, error))
                    })
                }),
            );
            continue;
        }
        return Err(PyTypeError::new_err(format!(
            "channel spec for `{name}` must be a string policy or a callable reducer"
        )));
    }
    Ok(out)
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

pub(super) fn install_langchain_callbacks<'py>(
    py: Python<'py>,
) -> Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let handler = super::langchain_callback_handler(py, None, None).ok()?;
    let callbacks = PyList::empty(py);
    callbacks.append(handler.bind(py)).ok()?;
    let config = PyDict::new(py);
    config.set_item("callbacks", callbacks).ok()?;
    let module = py.import("langchain_core.runnables.config").ok()?;
    let variable = module.getattr("var_child_runnable_config").ok()?;
    let token = variable.call_method1("set", (config,)).ok()?;
    Some((variable, token))
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
            let callback_guard = install_langchain_callbacks(py);
            let result = callable.call1(py, (state_obj,));
            if let Some((variable, token)) = callback_guard {
                let _ = variable.call_method1("reset", (token,));
            }
            let result = result.map_err(|error| pyerr_to_error(py, error))?;
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

#[path = "stream_convert.rs"]
mod stream_convert;
pub(crate) use stream_convert::*;

/// Parse a plain thread id or a LangGraph RunnableConfig-shaped dict into run config.
pub(crate) fn config_from_py(
    _py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Option<String>, crate::graph::RunConfig)> {
    let mut config = crate::graph::RunConfig::new();
    let Some(value) = value else {
        return Ok((None, config));
    };
    if value.is_none() {
        return Ok((None, config));
    }
    if let Ok(thread) = value.extract::<String>() {
        config.thread_id = Some(thread.clone());
        return Ok((Some(thread), config));
    }
    let Some(object) = value.cast::<PyDict>().ok() else {
        return Err(PyTypeError::new_err(
            "expected a thread id string or a config dict",
        ));
    };
    let mut thread = None;
    if let Some(configurable) = object.get_item("configurable")? {
        if let Ok(configurable) = configurable.cast::<PyDict>() {
            if let Some(item) = configurable.get_item("thread_id")? {
                thread = item.extract::<String>().ok();
            }
            if let Some(item) = configurable.get_item("checkpoint_id")? {
                if let Ok(id) = item.extract::<String>() {
                    config.checkpoint_id = Some(id);
                }
            }
            if let Some(item) = configurable.get_item("checkpoint_ns")? {
                if let Ok(namespace) = item.extract::<String>() {
                    config = config.with_checkpoint_ns(namespace);
                }
            }
        }
    }
    if let Some(item) = object.get_item("checkpoint_ns")? {
        if let Ok(namespace) = item.extract::<String>() {
            config = config.with_checkpoint_ns(namespace);
        }
    }
    if let Some(item) = object.get_item("recursion_limit")? {
        if let Ok(limit) = item.extract::<usize>() {
            config = config.with_recursion_limit(limit);
        }
    }
    if let Some(item) = object.get_item("max_concurrency")? {
        if let Ok(limit) = item.extract::<usize>() {
            config = config.with_max_concurrency(limit);
        }
    }
    if let Some(item) = object.get_item("durability")? {
        if let Ok(name) = item.extract::<String>()
            && let Some(durability) = crate::Durability::parse(&name)
        {
            config = config.with_durability(durability);
        }
    }
    config.thread_id = thread.clone();
    Ok((thread, config))
}
