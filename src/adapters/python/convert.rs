//! Conversion helpers between Python objects and kernel values.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyInt, PyList};

use super::types::{PyCommand, PySend};
use serde_json::Map;

use super::{GraphInterrupt, GraphRecursionError};
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
    // LangGraph stores LangChain objects (messages, documents, ...) as JSON-plus constructor
    // dicts; the shim owns that encoding so checkpoints stay interchangeable.
    if let Some(encoded) = encode_object(obj)? {
        return Ok(encoded);
    }
    Err(PyTypeError::new_err(format!(
        "value of type `{}` cannot be converted to JSON; use None, bool, int, float, str, list, tuple or dict",
        type_name(obj)
    )))
}


/// Encode a LangChain object as LangGraph's JSON-plus constructor dict.
///
/// The encoding is `{"lc": 1, "type": "constructor", "id": [module, name], "kwargs": ...}` and is
/// produced from the object's own `model_dump` / `to_json`, so the extension carries the data
/// without calling into the shim.
fn encode_object(obj: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    let kind = obj.get_type();
    let module: String = kind.getattr("__module__")?.extract()?;
    let name: String = kind.getattr("__qualname__")?.extract().or_else(|_| {
        kind.getattr("__name__")
            .and_then(|value| value.extract::<String>())
    })?;
    let kwargs = if let Ok(model_dump) = obj.getattr("model_dump") {
        if model_dump.is_callable() {
            Some(model_dump.call0()?)
        } else {
            None
        }
    } else {
        None
    };
    let kwargs = match kwargs {
        Some(kwargs) => kwargs,
        None => match obj.getattr("to_json") {
            Ok(to_json) if to_json.is_callable() => {
                let payload = to_json.call0()?;
                if payload.cast::<PyDict>().is_ok() {
                    payload
                } else {
                    let wrapper = PyDict::new(obj.py());
                    wrapper.set_item("value", payload)?;
                    wrapper.into_any()
                }
            }
            _ => return Ok(None),
        },
    };
    let dict = PyDict::new(obj.py());
    dict.set_item("lc", 1)?;
    dict.set_item("type", "constructor")?;
    dict.set_item("id", (module, name))?;
    dict.set_item("kwargs", kwargs)?;
    Ok(Some(py_to_value(&dict.into_any())?))
}

/// Revive a JSON-plus constructor dict: import the recorded class and rebuild it from `kwargs`.
fn decode_object(py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<Option<Py<PyAny>>> {
    let Some(serde_json::Value::Number(lc)) = dict
        .get_item("lc")?
        .map(|value| py_to_value(&value))
        .transpose()?
    else {
        return Ok(None);
    };
    if lc.as_i64() != Some(1) {
        return Ok(None);
    }
    let Some(serde_json::Value::String(kind)) = dict
        .get_item("type")?
        .map(|value| py_to_value(&value))
        .transpose()?
    else {
        return Ok(None);
    };
    if kind != "constructor" {
        return Ok(None);
    }
    let Some(serde_json::Value::Array(path)) = dict
        .get_item("id")?
        .map(|value| py_to_value(&value))
        .transpose()?
    else {
        return Ok(None);
    };
    let names: Vec<String> = path
        .iter()
        .filter_map(|item| item.as_str().map(str::to_owned))
        .collect();
    if names.is_empty() {
        return Ok(None);
    }
    let Some(module_name) = names.first() else {
        return Ok(None);
    };
    let Some(class_name) = names.last() else {
        return Ok(None);
    };
    let Ok(module) = py.import(module_name.as_str()) else {
        return Ok(None);
    };
    let Ok(factory) = module.getattr(class_name.as_str()) else {
        return Ok(None);
    };
    let kwargs = match dict.get_item("kwargs")? {
        Some(value) if value.cast::<PyDict>().is_ok() => value,
        Some(value) => {
            let wrapper = PyDict::new(py);
            wrapper.set_item("value", value)?;
            wrapper.into_any()
        }
        None => PyDict::new(py).into_any(),
    };
    let kwargs = kwargs.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err("a constructor payload must carry a `kwargs` mapping")
    })?;
    Ok(match factory.call((), Some(kwargs)) {
        Ok(value) => Some(value.unbind()),
        Err(_) => None,
    })
}

pub(crate) fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
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
            if map.get("lc").and_then(Value::as_i64) == Some(1) {
                if let Some(decoded) = decode_object(py, &dict)? {
                    return Ok(decoded);
                }
            }
            dict.into_any().unbind()
        }
    })
}

/// Convert one channel value, turning message lists into LangChain message objects.
fn channel_value_to_py(py: Python<'_>, name: &str, value: &Value) -> PyResult<Py<PyAny>> {
    if crate::runtime::context::messages_as_objects()
        && crate::runtime::context::is_messages_channel(name)
        && let Value::Array(items) = value
        && !items.is_empty()
        && items.iter().all(Value::is_object)
    {
        if let Some(converted) = messages_to_py_objects(py, value) {
            return Ok(converted);
        }
    }
    value_to_py(py, value)
}

/// Build LangChain message objects from a list of message dictionaries.
fn messages_to_py_objects(py: Python<'_>, value: &Value) -> Option<Py<PyAny>> {
    let module = py.import("langchain_core.messages").ok()?;
    let convert = module.getattr("convert_to_messages").ok()?;
    let payload = value_to_py(py, value).ok()?;
    match convert.call1((payload,)) {
        Ok(converted) => Some(converted.unbind()),
        Err(_) => None,
    }
}

pub(super) fn json_state_to_py(py: Python<'_>, state: &JsonState) -> PyResult<Py<PyAny>> {
    // Build the dict straight from the borrowed map: wrapping it in `Value::Object` used to
    // deep-clone every channel value on each node call and snapshot conversion.
    let dict = PyDict::new(py);
    for (key, value) in &state.0 {
        dict.set_item(key, channel_value_to_py(py, key, value)?)?;
    }
    Ok(dict.into_any().unbind())
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

/// Milliseconds of a run timeout recorded in a node failure message, when it is one.
///
/// The engine formats timeouts as ``node `<name>` timed out after <millis> ms``; the retry layer then
/// appends the attempt count, so the prefix is what identifies the shape.
/// The `Expected dict, got ...` detail of an invalid node output, when the message carries one.
fn invalid_node_output(message: &str) -> Option<String> {
    let marker = "Expected dict, got ";
    let start = message.find(marker)?;
    let mut detail = message[start..].trim_end().to_string();
    // The retry bookkeeping appends `(failed after N attempt(s))`; LangGraph does not show it.
    if let Some(position) = detail.find(" (failed after ") {
        detail.truncate(position);
    }
    Some(detail)
}

fn idle_millis(node: &str, message: &str) -> Option<u64> {
    let prefix = format!("node `{node}` went idle for ");
    let rest = message.strip_prefix(&prefix)?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

fn timeout_millis(node: &str, message: &str) -> Option<u64> {
    let prefix = format!("node `{node}` timed out after ");
    let rest = message.strip_prefix(&prefix)?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
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
        // LangGraph reports the recursion budget as `GraphRecursionError(RecursionError)` with a
        // fixed message shape that includes the troubleshooting link.
        Error::RecursionLimit(limit) => PyErr::new::<GraphRecursionError, _>(format!(
            "Recursion limit of {limit} reached without hitting a stop condition. \
             You can increase the limit by setting the `recursion_limit` config key.\n\
             For troubleshooting, visit: https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT"
        )),
        // A node that returns something other than an update mapping is an invalid update.
        Error::Node { message, .. } if invalid_node_output(&message).is_some() => {
            let detail = invalid_node_output(&message).unwrap_or_default();
            let text = format!(
                "{detail}\nFor troubleshooting, visit: \
                 https://docs.langchain.com/oss/python/langgraph/errors/INVALID_CONCURRENT_GRAPH_UPDATE"
            );
            match py
                .import("langgraph_rs.errors")
                .and_then(|module| module.getattr("InvalidUpdateError"))
                .and_then(|class| class.call1((text.clone(),)))
            {
                Ok(error) => PyErr::from_value(error),
                Err(_) => PyRuntimeError::new_err(text),
            }
        }
        Error::Other(message) if invalid_node_output(&message).is_some() => {
            let detail = invalid_node_output(&message).unwrap_or_default();
            match py
                .import("langgraph_rs.errors")
                .and_then(|module| module.getattr("InvalidUpdateError"))
                .and_then(|class| class.call1((detail,)))
            {
                Ok(error) => PyErr::from_value(error),
                Err(_) => PyRuntimeError::new_err(message),
            }
        }
        // LangGraph reports conflicting writes to a plain channel as `InvalidUpdateError`.
        Error::ConcurrentUpdate { channel, .. } => {
            let message = format!(
                "At key '{channel}': Can receive only one value per step. Use an Annotated key to \
                 handle multiple values.\nFor troubleshooting, visit: \
                 https://docs.langchain.com/oss/python/langgraph/errors/INVALID_CONCURRENT_GRAPH_UPDATE"
            );
            match py
                .import("langgraph_rs.errors")
                .and_then(|module| module.getattr("InvalidUpdateError"))
                .and_then(|class| class.call1((message.clone(),)))
            {
                Ok(error) => PyErr::from_value(error),
                Err(_) => PyRuntimeError::new_err(message),
            }
        }
        // An idle timeout (the node stopped reporting progress) is a distinct LangGraph error kind.
        Error::IdleTimeout { node, millis } => {
            let seconds = millis as f64 / 1000.0;
            match py.import("langgraph_rs.errors").and_then(|module| {
                let factory = module.getattr("NodeTimeoutError")?;
                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item("kind", "idle")?;
                kwargs.set_item("idle_timeout", seconds)?;
                factory.call((node.clone(), seconds), Some(&kwargs))
            }) {
                Ok(error) => PyErr::from_value(error),
                Err(_) => PyRuntimeError::new_err(format!("node `{node}` went idle")),
            }
        }
        // A run timeout that went through the retry bookkeeping arrives wrapped as a node
        // failure; LangGraph reports both shapes as `NodeTimeoutError`.
        Error::Node { node, message }
            if timeout_millis(&node, &message).is_some()
                || idle_millis(&node, &message).is_some() =>
        {
            let idle = idle_millis(&node, &message);
            let millis = idle.or_else(|| timeout_millis(&node, &message)).unwrap_or_default();
            let seconds = millis as f64 / 1000.0;
            match py.import("langgraph_rs.errors").and_then(|module| {
                let factory = module.getattr("NodeTimeoutError")?;
                let kwargs = pyo3::types::PyDict::new(py);
                if idle.is_some() {
                    kwargs.set_item("kind", "idle")?;
                    kwargs.set_item("idle_timeout", seconds)?;
                } else {
                    kwargs.set_item("kind", "run")?;
                    kwargs.set_item("run_timeout", seconds)?;
                }
                factory.call((node.clone(), seconds), Some(&kwargs))
            }) {
                Ok(error) => PyErr::from_value(error),
                Err(_) => PyRuntimeError::new_err(format!("node `{node}` timed out")),
            }
        }
        // LangGraph raises `NodeTimeoutError` (an `os.error`-style exception carrying the node,
        // the elapsed time and the configured timeout) instead of a bare `RuntimeError`.
        Error::Timeout { node, millis } => {
            let seconds = millis as f64 / 1000.0;
            match py.import("langgraph_rs.errors").and_then(|module| {
                let factory = module.getattr("NodeTimeoutError")?;
                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item("kind", "run")?;
                kwargs.set_item("run_timeout", seconds)?;
                factory.call((node.clone(), seconds), Some(&kwargs))
            }) {
                Ok(error) => PyErr::from_value(error),
                Err(_) => PyRuntimeError::new_err(Error::Timeout { node, millis }.to_string()),
            }
        }
        // A Python exception raised inside a node is re-raised with its original type, as
        // LangGraph does (`ValueError: kaboom`, pydantic `ValidationError`, ...).
        Error::Node { message, .. } | Error::Other(message)
            if python_error_id(&message).is_some() =>
        {
            match python_error_id(&message).and_then(take_python_error) {
                Some(exception) => PyErr::from_value(exception.bind(py).clone()),
                None => PyRuntimeError::new_err(strip_python_error_marker(&message)),
            }
        }
        // LangGraph's message here is exactly "No checkpointer set"; `Command(resume=...)` has its
        // own wording and stays a `RuntimeError`.
        Error::MissingCheckpointer(message) => {
            if message.contains("Command(resume") {
                PyRuntimeError::new_err(format!("{message} without checkpointer"))
            } else {
                PyValueError::new_err("No checkpointer set")
            }
        }
        // LangGraph reports every graph-construction problem as a `ValueError`.
        Error::Validation(message) => PyValueError::new_err(message),
        other => PyRuntimeError::new_err(strip_python_error_marker(&other.to_string())),
    }
}

/// Python exceptions raised inside a node, kept until the run surfaces them again.
///
/// LangGraph re-raises the original exception (`ValueError: kaboom`, pydantic `ValidationError`,
/// ...), so the object has to survive the engine's error handling. The message carries a marker
/// with the id; `to_pyerr` swaps it back for the exception.
static PY_ERRORS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<u64, Py<PyAny>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

static PY_ERROR_IDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub(super) fn stash_python_error(py: Python<'_>, error: &PyErr) -> u64 {
    let id = PY_ERROR_IDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Ok(mut stash) = PY_ERRORS.lock() {
        // Keep the map small when many errors are produced and never consumed.
        if stash.len() > 64 {
            stash.clear();
        }
        stash.insert(id, error.value(py).clone().into_any().unbind());
    }
    id
}

/// Look up a stashed exception without consuming it (used when building snapshots).
pub(super) fn peek_python_error(py: Python<'_>, id: u64) -> Option<Py<PyAny>> {
    PY_ERRORS
        .lock()
        .ok()
        .and_then(|stash| stash.get(&id).map(|value| value.clone_ref(py)))
}

/// Fetch a stashed exception. The entry is kept (the map is capped in `stash_python_error`), so the
/// snapshot builder can still describe the failure after the exception was re-raised.
pub(super) fn take_python_error(id: u64) -> Option<Py<PyAny>> {
    PY_ERRORS.lock().ok().and_then(|stash| {
        stash
            .get(&id)
            .map(|value| pyo3::Python::attach(|py| value.clone_ref(py)))
    })
}

/// Id of the Python exception carried by an engine error message, when it has one.
pub(crate) fn python_error_id(message: &str) -> Option<u64> {
    let marker = "\u{1}pyerr:";
    let start = message.rfind(marker)? + marker.len();
    let rest = &message[start..];
    let end = rest.find('\u{1}')?;
    rest[..end].parse().ok()
}

/// Strip the marker from a message that is shown to users.
pub(crate) fn strip_python_error_marker(message: &str) -> String {
    match message.rfind("\u{1}pyerr:") {
        Some(start) => {
            let rest = &message[start..];
            match rest.find('\u{1}') {
                Some(_) => {
                    let end = rest[1..].find('\u{1}').map(|offset| start + 1 + offset + 1);
                    match end {
                        Some(end) => format!("{}{}", &message[..start], &message[end..]),
                        None => message.to_string(),
                    }
                }
                None => message.to_string(),
            }
        }
        None => message.to_string(),
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
        // Keep the exception so `to_pyerr` can re-raise it, as LangGraph does.
        let id = stash_python_error(py, &error);
        Error::other(format!(
            "python node raised {}: {error}\u{1}pyerr:{id}\u{1}",
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
    // Type checks first: raise-free lookups avoid building a cleared exception per call, which
    // matters because routers run once per superstep.
    if obj.cast::<pyo3::types::PyString>().is_ok() {
        let name: String = obj.extract()?;
        let name = normalize_node(&name);
        return Ok(if name == crate::END {
            Route::end()
        } else {
            Route::node(name)
        });
    }
    if let Ok(send) = obj.cast::<super::types::PySend>().map(|cell| cell.borrow()) {
        return Ok(Route::send(send.to_send(obj.py())?));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut names = Vec::with_capacity(list.len());
        let mut sends = Vec::new();
        for item in list.iter() {
            if let Ok(send) = item.cast::<super::types::PySend>().map(|cell| cell.borrow()) {
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
    // Type checks first: `extract::<PyRef<..>>()` raises and clears a Python exception when
    // the object is not a `Command`/`Send`, which used to cost ~5 us per ordinary dict node
    // result (two failed extracts).
    if let Ok(command) = obj.cast::<PyCommand>().map(|cell| cell.borrow()) {
        return command.to_output(obj.py());
    }
    if let Ok(send) = obj.cast::<PySend>().map(|cell| cell.borrow()) {
        return Ok(NodeOutput::new().send(send.to_send(obj.py())?));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut output = NodeOutput::new();
        for item in list.iter() {
            if let Ok(send) = item.cast::<PySend>().map(|cell| cell.borrow()) {
                output = output.send(send.to_send(obj.py())?);
            } else if let Ok(command) = item.cast::<PyCommand>().map(|cell| cell.borrow()) {
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
                // LangGraph reports the whole value: `Expected dict, got [...]`.
                let repr = obj
                    .repr()
                    .map(|text| text.to_string())
                    .unwrap_or_else(|_| "?".to_string());
                return Err(PyTypeError::new_err(format!("Expected dict, got {repr}")));
            }
        }
        return Ok(output);
    }
    let dict = obj.cast::<PyDict>().map_err(|_| {
        // LangGraph reports this as `InvalidUpdateError: Expected dict, got <repr>`.
        let repr = obj
            .repr()
            .map(|text| text.to_string())
            .unwrap_or_else(|_| "?".to_string());
        PyTypeError::new_err(format!("Expected dict, got {repr}"))
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
    // The Python builder normalises class schemas and marks custom channels before calling here.
    let dict = schema.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err("schema must be a dict of channel -> spec (use langgraph_rs.StateGraph)")
    })?.clone();

    let kernel = Kernel::builtin();
    let mut out = StateSchema::dynamic();
    for (key, spec) in dict.iter() {
        let name: String = key.extract()?;
        // A channel that implements its own merge arrives wrapped by the Python builder as
        // `{"__custom_channel__": channel}`; LangGraph's protocol is `update([write]) -> bool`
        // then `get() -> value`.
        let custom_channel = spec
            .cast::<PyDict>()
            .ok()
            .and_then(|dict| dict.get_item("__custom_channel__").ok().flatten());
        if let Some(channel) = custom_channel {
            let channel = channel.unbind();
            out = out.with_channel(
                name,
                Channel::reducer(move |_current: &Value, incoming: &Value| {
                    Python::attach(|py| -> Result<Value> {
                        let bound = channel.bind(py);
                        let write = value_to_py(py, incoming)
                            .map_err(|error| crate::Error::other(error.to_string()))?;
                        let writes = PyList::new(py, [write])
                            .map_err(|error| crate::Error::other(error.to_string()))?;
                        bound
                            .call_method1("update", (writes,))
                            .map_err(|error| crate::Error::other(error.to_string()))?;
                        let current = bound
                            .call_method0("get")
                            .map_err(|error| crate::Error::other(error.to_string()))?;
                        py_to_value(&current)
                            .map_err(|error| crate::Error::other(error.to_string()))
                    })
                }),
            );
            continue;
        }
        // LangGraph's channel objects (`LastValue`, `Topic`, ...) carry their reducer spec.
        if let Ok(channel_spec) = spec.getattr("__channel_spec__") {
            if channel_spec.is_none() {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                    "channel `{name}` uses {}, which the langgraph_rs extension does not implement",
                    spec.get_type().name().map(|n| n.to_string()).unwrap_or_default()
                )));
            }
            if let Ok(text) = channel_spec.extract::<String>() {
                let channel = kernel.resolve_channel(&text).ok_or_else(|| {
                    PyValueError::new_err(format!("unknown channel spec `{text}` for `{name}`"))
                })?;
                out = out.with_channel(name, channel);
                continue;
            }
            if channel_spec.is_callable() {
                let callable = channel_spec.clone().unbind();
                out = out.with_channel(
                    name,
                    Channel::reducer(move |current: &Value, incoming: &Value| {
                        Python::attach(|py| -> Result<Value> {
                            let left = value_to_py(py, current)
                                .map_err(|error| crate::Error::other(error.to_string()))?;
                            let right = value_to_py(py, incoming)
                                .map_err(|error| crate::Error::other(error.to_string()))?;
                            let result = callable
                                .call1(py, (left, right))
                                .map_err(|error| crate::Error::other(error.to_string()))?;
                            py_to_value(result.bind(py))
                                .map_err(|error| crate::Error::other(error.to_string()))
                        })
                    }),
                );
                continue;
            }
        }
        if let Ok(spec) = spec.extract::<String>() {
            // `IsLastStep` / `RemainingSteps` are managed values in LangGraph: injected for
            // every superstep, never persisted, and only present when declared.
            if matches!(spec.as_str(), "is_last_step" | "remaining_steps") {
                out = out.with_managed_values();
                continue;
            }
            if matches!(spec.as_str(), "add_messages" | "messages") {
                out = out.with_messages_channel(name.clone());
            }
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
            // `Annotated[list, operator.add]` is the most common reducer; concatenating two
            // JSON arrays natively avoids a Python round trip (and two conversions) per
            // superstep. Anything else falls back to calling the Python callable.
            let is_operator_add = Python::attach(|py| {
                py.import("operator")
                    .and_then(|module| module.getattr("add"))
                    .is_ok_and(|add| add.is(&callable))
            });
            out = out.with_channel(
                name,
                Channel::reducer(move |current: &Value, incoming: &Value| {
                    if is_operator_add
                        && let (Value::Array(left), Value::Array(right)) = (current, incoming)
                    {
                        let mut merged = Vec::with_capacity(left.len() + right.len());
                        merged.extend(left.iter().cloned());
                        merged.extend(right.iter().cloned());
                        return Ok(Value::Array(merged));
                    }
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
    // LangGraph also accepts a plain list of destination names.
    if let Ok(names) = path_map.extract::<Vec<String>>() {
        let mut map = PathMap::new();
        for name in &names {
            map = map.insert(name.as_str(), vec![normalize_node(name)]);
        }
        return Ok(Some(map));
    }
    let dict = path_map
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("path_map must be a dict or a list of node names"))?;
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

/// `(var_child_runnable_config, {"callbacks": [handler]})`: the LangChain callback bridge.
type CallbackConfig = (Py<PyAny>, Py<PyAny>);

std::thread_local! {
    /// The callback bridge per OS thread (plain and token-streaming variants).
    ///
    /// Building the handler and importing `langchain_core.runnables.config` for every node
    /// call cost ~13 us per node; the contextvar itself is process-wide, so the pair can be
    /// reused. Nodes still set/reset it per call, which is what LangChain expects.
    static CALLBACK_CONFIG: std::cell::RefCell<[Option<CallbackConfig>; 2]> =
        const { std::cell::RefCell::new([None, None]) };
}

fn cached_callback_config(py: Python<'_>) -> Option<(Py<PyAny>, Py<PyAny>)> {
    let streaming = STREAM_TOKENS.with(std::cell::Cell::get);
    let slot = usize::from(streaming);
    if let Some((variable, config)) = CALLBACK_CONFIG.with(|cell| {
        cell.borrow()[slot]
            .as_ref()
            .map(|(variable, config)| (variable.clone_ref(py), config.clone_ref(py)))
    }) {
        return Some((variable, config));
    }
    let handler = super::langchain_callback_handler(py, None, None, streaming).ok()?;
    let callbacks = PyList::empty(py);
    callbacks.append(handler.bind(py)).ok()?;
    let config = PyDict::new(py);
    config.set_item("callbacks", callbacks).ok()?;
    let module = py.import("langchain_core.runnables.config").ok()?;
    let variable = module.getattr("var_child_runnable_config").ok()?;
    let pair = (variable.unbind(), config.into_any().unbind());
    CALLBACK_CONFIG.with(|cell| {
        cell.borrow_mut()[slot] = Some((pair.0.clone_ref(py), pair.1.clone_ref(py)));
    });
    Some(pair)
}

thread_local! {
    /// `true` while the binding has installed the LangChain callback context for a whole run,
    /// so per-node installation can be skipped on this thread.
    static CALLBACK_SCOPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Whether the current run wants model tokens on the stream. LangGraph only makes `invoke`
    /// stream (its callback handler is a `_StreamingCallbackHandler`) when a streaming mode is
    /// requested; otherwise the model is called normally.
    static STREAM_TOKENS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Installs the LangChain callback context once for a whole graph run.
///
/// Nodes executed on this thread then reuse it instead of setting and resetting the context
/// variable per node (two Python calls each). Parallel nodes run on worker threads, where the
/// per-node installation still happens.
pub(super) struct CallbackScope {
    pair: Option<(Py<PyAny>, Py<PyAny>)>,
}

impl Drop for CallbackScope {
    fn drop(&mut self) {
        if let Some((variable, token)) = self.pair.take() {
            Python::attach(|py| {
                let _ = variable
                    .bind(py)
                    .call_method1("reset", (token.bind(py),));
            });
            CALLBACK_SCOPE.with(|cell| cell.set(false));
        }
    }
}

/// Ask the callback handler for model tokens (`stream_mode="messages"` / `stream_events`).
pub(super) fn set_stream_tokens(enabled: bool) {
    STREAM_TOKENS.with(|cell| cell.set(enabled));
}

pub(super) fn begin_callback_scope(py: Python<'_>) -> CallbackScope {
    if CALLBACK_SCOPE.with(std::cell::Cell::get) {
        return CallbackScope { pair: None };
    }
    match install_langchain_callbacks(py) {
        Some((variable, token)) => {
            CALLBACK_SCOPE.with(|cell| cell.set(true));
            CallbackScope {
                pair: Some((variable.unbind(), token.unbind())),
            }
        }
        None => CallbackScope { pair: None },
    }
}

pub(super) fn install_langchain_callbacks<'py>(
    py: Python<'py>,
) -> Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    if CALLBACK_SCOPE.with(std::cell::Cell::get) {
        return None;
    }
    let (variable, config) = cached_callback_config(py)?;
    let token = variable
        .bind(py)
        .call_method1("set", (config.bind(py),))
        .ok()?;
    Some((variable.into_bound(py), token))
}

/// Convert a `context_schema` argument: a JSON-schema dict or a pydantic/TypedDict class.
pub(super) fn context_schema_value(
    py: Python<'_>,
    schema: &Bound<'_, PyAny>,
) -> PyResult<serde_json::Value> {
    if let Ok(value) = py_to_value(schema) {
        return Ok(value);
    }
    if let Ok(adapter) = py.import("pydantic")?.getattr("TypeAdapter")
        && let Ok(adapter) = adapter.call1((schema,))
        && let Ok(json_schema) = adapter.call_method0("json_schema")
        && let Ok(value) = py_to_value(&json_schema)
    {
        return Ok(value);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "context_schema must be a JSON schema dict or a pydantic/TypedDict class",
    ))
}

/// Apply the writes a node performed itself to its output, preserving multiple writes per channel.
///
/// Several `send()` calls to the same channel arrive as one list, which reducers treat exactly like
/// the node returning each value in turn.
pub(super) fn apply_node_writes(
    mut output: crate::_internal::node::NodeOutput,
    writes: Vec<(String, serde_json::Value)>,
) -> crate::_internal::node::NodeOutput {
    if writes.is_empty() {
        return output;
    }
    let mut grouped: Vec<(String, Vec<serde_json::Value>)> = Vec::new();
    for (channel, value) in writes {
        match grouped.iter_mut().find(|(name, _)| *name == channel) {
            Some((_, values)) => values.push(value),
            None => grouped.push((channel, vec![value])),
        }
    }
    let existing = output.updates_map().clone();
    for (channel, values) in grouped {
        let mut all: Vec<serde_json::Value> = Vec::new();
        if let Some(previous) = existing.get(&channel) {
            all.push(previous.clone());
        }
        all.extend(values);
        let value = if all.len() == 1 {
            all.pop().unwrap_or(serde_json::Value::Null)
        } else {
            serde_json::Value::Array(all)
        };
        output = output.update(&channel, value);
    }
    output
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
            // The engine already installed this node's task context; only the channel writes it
            // performed itself are collected here.
            let mut result = callable.call1(py, (state_obj,));
            // LangGraph accepts async nodes in a sync builder: the call returns a coroutine and
            // `ainvoke` drives it. Resolve it here so both run styles work.
            if let Ok(value) = &result {
                let bound = value.bind(py);
                if bound.hasattr("__await__").unwrap_or(false) {
                    result =
                        crate::adapters::python::async_graph::async_support::run_timed_awaitable(
                            py, bound,
                        )
                        .map(|value| value.unbind());
                }
            }
            let writes = crate::runtime::context::take_writes();
            if let Some((variable, token)) = callback_guard {
                let _ = variable.call_method1("reset", (token,));
            }
            let result = result.map_err(|error| pyerr_to_error(py, error))?;
            let output = node_output_from_py(result.bind(py)).map_err(|error| {
                Error::other(format!("node `{name}` returned an invalid output: {error}"))
            })?;
            Ok(apply_node_writes(output, writes))
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
pub(crate) mod stream_convert;
pub(crate) use stream_convert::*;


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
    // Check the type before extracting: `extract::<String>()` on a dict builds and clears a
    // Python exception, which is a measurable cost on every config-dict call.
    if value.cast::<pyo3::types::PyString>().is_ok() {
        let thread: String = value.extract()?;
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
    if let Some(item) = object.get_item("tags")? {
        if let Ok(tags) = item.extract::<Vec<String>>() {
            config.tags = tags;
        }
    }
    if let Some(item) = object.get_item("metadata")? {
        if let Ok(metadata) = py_to_value(&item)
            && let serde_json::Value::Object(map) = metadata
        {
            config.metadata = map;
        }
    }
    config.thread_id = thread.clone();
    Ok((thread, config))
}
