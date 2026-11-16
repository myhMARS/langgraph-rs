//! Python bindings (enabled with the `python` feature).
//!
//! The Python surface mirrors the Rust API using [`JsonState`]: a graph is built from
//! Python callables, every node receives the state as a `dict` and returns a `dict` of
//! channel writes (optionally with the reserved `__goto__` and `__usage__` keys), and
//! `interrupt()` inside a node pauses the run so the caller can `resume()` it later.
//!
//! ```python
//! import langgraph_rs as lg
//!
//! graph = lg.StateGraph({"count": "last_value", "log": "append"})
//! graph.add_node("inc", lambda state: {"count": state["count"] + 1, "log": [state["count"] + 1]})
//! graph.add_edge("START", "inc")
//! graph.add_conditional_edges("inc", lambda state: "inc" if state["count"] < 3 else None)
//! compiled = graph.compile()
//! compiled.invoke({"count": 0, "log": []})
//! ```

use std::sync::Arc;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use serde_json::{Map, Value};

use crate::{
    CompileOptions, CompiledGraph, Error, InMemorySaver, Interrupt, JsonState, NodeOutput, PathMap,
    Route, RunConfig, StateSchema, StreamChunk, Usage, reducers,
};

create_exception!(
    langgraph_rs,
    GraphInterrupt,
    PyException,
    "Raised when a node pauses the graph through langgraph_rs.interrupt()."
);

// ---------------------------------------------------------------------------
// JSON <-> Python conversion
// ---------------------------------------------------------------------------

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string())
}

fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
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

fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
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

fn json_state_to_py(py: Python<'_>, state: &JsonState) -> PyResult<Py<PyAny>> {
    value_to_py(py, &Value::Object(state.0.clone()))
}

fn state_from_py(obj: Option<&Bound<'_, PyAny>>) -> PyResult<JsonState> {
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

fn usage_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Usage> {
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

fn to_pyerr(py: Python<'_>, error: Error) -> PyErr {
    match error {
        Error::Interrupt(interrupt) => {
            let payload = value_to_py(py, &interrupt.value).unwrap_or_else(|_| py.None());
            PyErr::new::<GraphInterrupt, _>((payload,))
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn pyerr_to_error(py: Python<'_>, error: PyErr) -> Error {
    if error.is_instance_of::<GraphInterrupt>(py) {
        let value = error
            .value(py)
            .getattr("args")
            .ok()
            .and_then(|args| args.get_item(0).ok())
            .and_then(|payload| py_to_value(&payload).ok())
            .unwrap_or(Value::Null);
        Error::Interrupt(Interrupt {
            node: String::new(),
            value,
        })
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
fn normalize_node(name: &str) -> String {
    match name {
        "START" => crate::START.to_string(),
        "END" => crate::END.to_string(),
        other => other.to_string(),
    }
}

fn route_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Route> {
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
    if let Ok(list) = obj.cast::<PyList>() {
        let mut names = Vec::with_capacity(list.len());
        for item in list.iter() {
            let name = normalize_node(&item.extract::<String>()?);
            if name != crate::END {
                names.push(name);
            }
        }
        return Ok(Route::nodes(names));
    }
    Err(PyTypeError::new_err(
        "router must return None, a node name, or a list of node names",
    ))
}

fn node_output_from_py(obj: &Bound<'_, PyAny>) -> PyResult<NodeOutput> {
    if obj.is_none() {
        return Ok(NodeOutput::new());
    }
    let dict = obj.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err("a node must return a dict of channel updates or None")
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
            _ => output = output.update(key, py_to_value(&value)?),
        }
    }
    Ok(output)
}

fn schema_from_py(schema: Option<&Bound<'_, PyAny>>) -> PyResult<StateSchema> {
    let Some(schema) = schema else {
        return Ok(StateSchema::dynamic());
    };
    if schema.is_none() {
        return Ok(StateSchema::dynamic());
    }
    let dict = schema
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("schema must be a dict of channel -> spec"))?;

    let mut result = StateSchema::dynamic();
    for (key, spec) in dict.iter() {
        let name: String = key.extract()?;
        let spec = spec.extract::<String>().map_err(|_| {
            PyTypeError::new_err(format!(
                "channel spec for `{name}` must be a string such as \"last_value\", \"topic\", \"append\""
            ))
        })?;
        result = match spec.as_str() {
            "last_value" | "last" | "value" => result.last_value(name),
            "topic" => result.topic(name),
            "append" => result.with_reducer(name, reducers::append()),
            "add" => result.with_reducer(name, reducers::add()),
            "messages" => result.with_reducer(name, reducers::messages()),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown channel spec `{other}` for `{name}`; use last_value, topic, append, add or messages"
                )));
            }
        };
    }
    Ok(result)
}

fn targets_from_py(to: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(name) = to.extract::<String>() {
        return Ok(vec![normalize_node(&name)]);
    }
    to.extract::<Vec<String>>()
        .map(|names| names.iter().map(|name| normalize_node(name)).collect())
        .map_err(|_| {
            PyTypeError::new_err("edge target must be a node name or a list of node names")
        })
}

fn path_map_from_py(path_map: &Bound<'_, PyAny>) -> PyResult<Option<PathMap>> {
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

fn node_from_python(
    name: String,
    callable: Py<PyAny>,
) -> impl Fn(&JsonState) -> crate::Result<NodeOutput> + Send + Sync + 'static {
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

fn router_from_python(
    callable: Py<PyAny>,
) -> impl Fn(&JsonState) -> crate::Result<Route> + Send + Sync + 'static {
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

fn run_config(thread_id: Option<String>) -> RunConfig {
    match thread_id {
        Some(thread_id) => RunConfig::new().with_thread(thread_id),
        None => RunConfig::new(),
    }
}

// ---------------------------------------------------------------------------
// Python classes
// ---------------------------------------------------------------------------

/// A mutable graph builder.
#[pyclass(name = "StateGraph")]
pub struct PyStateGraph {
    inner: crate::StateGraph<JsonState>,
}

#[pymethods]
impl PyStateGraph {
    /// Create a graph. `schema` maps channel names to a merge policy
    /// (`"last_value"`, `"topic"`, `"append"`, `"add"` or `"messages"`).
    #[new]
    #[pyo3(signature = (schema=None))]
    fn new(schema: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            inner: crate::StateGraph::with_schema(schema_from_py(schema)?),
        })
    }

    /// Register a Python callable as a node.
    fn add_node(&mut self, name: String, function: Py<PyAny>) -> PyResult<()> {
        let graph = std::mem::take(&mut self.inner);
        self.inner = graph.add_node(name.clone(), node_from_python(name, function));
        Ok(())
    }

    /// Add a static edge (`"START"` / `"END"` are accepted as source/target).
    fn add_edge(&mut self, from: String, to: &Bound<'_, PyAny>) -> PyResult<()> {
        let targets = targets_from_py(to)?;
        let graph = std::mem::take(&mut self.inner);
        self.inner = graph.add_edge(normalize_node(&from), targets);
        Ok(())
    }

    /// Add a conditional edge. Either return node names directly or pass a `path_map`.
    #[pyo3(signature = (source, router, path_map=None))]
    fn add_conditional_edges(
        &mut self,
        source: String,
        router: Py<PyAny>,
        path_map: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let graph = std::mem::take(&mut self.inner);
        let route = router_from_python(router);
        self.inner = match path_map.map(path_map_from_py).transpose()?.flatten() {
            Some(path_map) => graph.add_conditional_edges_mapped(source, route, path_map),
            None => graph.add_conditional_edges(source, route),
        };
        Ok(())
    }

    /// Set the static entry point.
    fn set_entry_point(&mut self, node: String) {
        let graph = std::mem::take(&mut self.inner);
        self.inner = graph.set_entry_point(node);
    }

    /// Use a router to choose the entry point.
    fn set_conditional_entry_point(&mut self, router: Py<PyAny>) {
        let graph = std::mem::take(&mut self.inner);
        self.inner = graph.set_conditional_entry_point(router_from_python(router));
    }

    /// Add an edge from `node` to `"END"`.
    fn set_finish_point(&mut self, node: String) {
        let graph = std::mem::take(&mut self.inner);
        self.inner = graph.set_finish_point(node);
    }

    /// Validate and compile the graph.
    #[pyo3(signature = (checkpointer=false, recursion_limit=None, interrupt_before=None, interrupt_after=None))]
    fn compile(
        &self,
        py: Python<'_>,
        checkpointer: bool,
        recursion_limit: Option<usize>,
        interrupt_before: Option<Vec<String>>,
        interrupt_after: Option<Vec<String>>,
    ) -> PyResult<PyCompiledGraph> {
        let mut options = CompileOptions::new();
        let mut saver = None;
        if checkpointer {
            let memory = Arc::new(InMemorySaver::new());
            options = options.with_checkpointer(memory.clone());
            saver = Some(memory);
        }
        if let Some(limit) = recursion_limit {
            options = options.with_recursion_limit(limit);
        }
        if let Some(nodes) = interrupt_before {
            options = options.interrupt_before(nodes);
        }
        if let Some(nodes) = interrupt_after {
            options = options.interrupt_after(nodes);
        }
        let compiled = self
            .inner
            .clone()
            .compile_with(options)
            .map_err(|error| to_pyerr(py, error))?;
        Ok(PyCompiledGraph {
            graph: compiled,
            saver,
        })
    }
}

/// A compiled, runnable graph.
#[pyclass(name = "CompiledGraph")]
pub struct PyCompiledGraph {
    graph: CompiledGraph<JsonState>,
    saver: Option<Arc<InMemorySaver>>,
}

#[pymethods]
impl PyCompiledGraph {
    /// Run the graph and return the final state dict.
    #[pyo3(signature = (input=None, thread_id=None))]
    fn invoke(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let state = state_from_py(input)?;
        let config = run_config(thread_id);
        let result = py.detach(|| self.graph.invoke(&state, &config));
        match result {
            Ok(state) => json_state_to_py(py, &state),
            Err(error) => Err(to_pyerr(py, error)),
        }
    }

    /// Run the graph and return `{"state": ..., "report": ...}`.
    #[pyo3(signature = (input=None, thread_id=None))]
    fn run(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let state = state_from_py(input)?;
        let config = run_config(thread_id);
        let outcome = py
            .detach(|| self.graph.run(&state, &config))
            .map_err(|error| to_pyerr(py, error))?;

        let result = PyDict::new(py);
        result.set_item("state", json_state_to_py(py, &outcome.state)?)?;
        result.set_item(
            "report",
            value_to_py(
                py,
                &serde_json::to_value(&outcome.report)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
            )?,
        )?;
        Ok(result.into_any().unbind())
    }

    /// Run the graph and return the list of stream chunks.
    #[pyo3(signature = (input=None, thread_id=None))]
    fn stream(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<String>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let state = state_from_py(input)?;
        let config = run_config(thread_id);
        let chunks = py
            .detach(|| self.graph.stream(&state, &config))
            .map_err(|error| to_pyerr(py, error))?;
        chunks.iter().map(|chunk| chunk_to_py(py, chunk)).collect()
    }

    /// Resume an interrupted thread.
    fn resume(
        &self,
        py: Python<'_>,
        thread_id: String,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let value = py_to_value(value)?;
        let result = py.detach(|| self.graph.resume(&thread_id, value));
        match result {
            Ok(state) => json_state_to_py(py, &state),
            Err(error) => Err(to_pyerr(py, error)),
        }
    }

    /// Resume an interrupted thread after applying state updates.
    #[pyo3(signature = (thread_id, value, updates=None))]
    fn resume_with_updates(
        &self,
        py: Python<'_>,
        thread_id: String,
        value: &Bound<'_, PyAny>,
        updates: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let value = py_to_value(value)?;
        let updates = match updates {
            Some(updates) => match py_to_value(updates)? {
                Value::Object(map) => map.into_iter().collect::<Vec<_>>(),
                _ => return Err(PyTypeError::new_err("updates must be a dict")),
            },
            None => Vec::new(),
        };
        let result = py.detach(|| self.graph.resume_with_updates(&thread_id, value, updates));
        match result {
            Ok(state) => json_state_to_py(py, &state),
            Err(error) => Err(to_pyerr(py, error)),
        }
    }

    /// Continue a thread paused by an `interrupt_before` / `interrupt_after` hook.
    fn continue_thread(&self, py: Python<'_>, thread_id: String) -> PyResult<Py<PyAny>> {
        let result = py.detach(|| self.graph.continue_thread(&thread_id));
        match result {
            Ok(state) => json_state_to_py(py, &state),
            Err(error) => Err(to_pyerr(py, error)),
        }
    }

    /// Return the latest checkpoint of a thread, or `None`.
    fn get_state(&self, py: Python<'_>, thread_id: String) -> PyResult<Option<Py<PyAny>>> {
        let snapshot = self
            .graph
            .get_state(&thread_id)
            .map_err(|error| to_pyerr(py, error))?;
        snapshot
            .map(|snapshot| snapshot_to_py(py, &snapshot))
            .transpose()
    }

    /// Return the full checkpoint history of a thread, newest first.
    fn get_state_history(&self, py: Python<'_>, thread_id: String) -> PyResult<Vec<Py<PyAny>>> {
        let history = self
            .graph
            .get_state_history(&thread_id)
            .map_err(|error| to_pyerr(py, error))?;
        history
            .iter()
            .map(|snapshot| snapshot_to_py(py, snapshot))
            .collect()
    }

    /// Registered node names.
    fn node_names(&self) -> Vec<String> {
        self.graph.node_names().map(str::to_string).collect()
    }

    /// Whether the graph was compiled with a checkpointer.
    fn has_checkpointer(&self) -> bool {
        self.saver.is_some()
    }
}

fn snapshot_to_py(
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
    Ok(dict.into_any().unbind())
}

fn chunk_to_py(py: Python<'_>, chunk: &StreamChunk) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    match chunk {
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
    }
    Ok(dict.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Module level functions
// ---------------------------------------------------------------------------

/// Pause the graph and wait for human input; returns the resume value on retry.
#[pyfunction]
fn interrupt(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let value = py_to_value(value)?;
    match crate::context::interrupt(value) {
        Ok(resume) => value_to_py(py, &resume),
        Err(error) => Err(to_pyerr(py, error)),
    }
}

/// Emit a custom value into the stream from inside a node.
#[pyfunction]
fn custom(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let value = py_to_value(value)?;
    crate::custom(value).map_err(|error| to_pyerr(py, error))
}

/// Record consumption for the current node.
#[pyfunction]
fn record_usage(py: Python<'_>, usage: &Bound<'_, PyAny>) -> PyResult<()> {
    let usage = usage_from_py(usage)?;
    crate::record_usage(usage).map_err(|error| to_pyerr(py, error))
}

/// Convenience shorthand for `record_usage` with token counts.
#[pyfunction]
fn record_tokens(py: Python<'_>, input_tokens: u64, output_tokens: u64) -> PyResult<()> {
    crate::record_tokens(input_tokens, output_tokens).map_err(|error| to_pyerr(py, error))
}

/// Crate version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python module definition.
#[pymodule]
fn langgraph_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStateGraph>()?;
    module.add_class::<PyCompiledGraph>()?;
    module.add("GraphInterrupt", module.py().get_type::<GraphInterrupt>())?;
    module.add_function(wrap_pyfunction!(interrupt, module)?)?;
    module.add_function(wrap_pyfunction!(custom, module)?)?;
    module.add_function(wrap_pyfunction!(record_usage, module)?)?;
    module.add_function(wrap_pyfunction!(record_tokens, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add("START", crate::START)?;
    module.add("END", crate::END)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
