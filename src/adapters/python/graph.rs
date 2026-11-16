//! Python graph builder and compiled graph classes.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use super::convert::*;
use super::types::{PyCachePolicy, PyCommand, PyRetryPolicy, checkpointer_from_py, store_from_py};
use crate::*;

// ---------------------------------------------------------------------------
// Python classes
// ---------------------------------------------------------------------------

/// A mutable graph builder.
#[pyclass(name = "StateGraph")]
pub struct PyStateGraph {
    inner: crate::StateGraph<JsonState>,
}

impl PyStateGraph {
    pub(super) fn from_inner(inner: crate::StateGraph<JsonState>) -> Self {
        Self { inner }
    }
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
    ///
    /// Optional LangGraph-style keywords: ``retry_policy``, ``cache_policy`` and
    /// ``error_handler`` (node name).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, function, retry_policy=None, cache_policy=None, error_handler=None, deferred=false, destinations=None, input_schema=None))]
    fn add_node(
        &mut self,
        name: String,
        function: Py<PyAny>,
        retry_policy: Option<PyRef<'_, PyRetryPolicy>>,
        cache_policy: Option<PyRef<'_, PyCachePolicy>>,
        error_handler: Option<String>,
        deferred: bool,
        destinations: Option<Vec<String>>,
        input_schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let mut graph = std::mem::take(&mut self.inner);
        if deferred {
            graph = graph.set_deferred(name.clone());
        }
        if let Some(destinations) = destinations {
            graph = graph.set_destinations(name.clone(), destinations);
        }
        if let Some(schema) = input_schema {
            graph = graph.set_node_input_schema(name.clone(), py_to_value(schema)?);
        }
        graph = graph.add_node(name.clone(), node_from_python(name.clone(), function));
        if let Some(policy) = retry_policy {
            graph = graph.set_retry(name.clone(), policy.to_policy());
        }
        if let Some(policy) = cache_policy {
            graph = graph.set_cache(name.clone(), policy.to_policy());
        }
        if let Some(handler) = error_handler {
            graph = graph.add_error_edge(name, normalize_node(&handler));
        }
        self.inner = graph;
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

    /// Embed a compiled graph as a node, sharing the parent thread and state type.
    ///
    /// The child runs in a checkpoint namespace derived from the node name, so an
    /// `interrupt()` inside it pauses the parent and a parent `resume` continues it.
    fn add_subgraph(&mut self, name: String, subgraph: PyRef<'_, PyCompiledGraph>) -> PyResult<()> {
        let child = subgraph.graph.clone();
        let builder = std::mem::take(&mut self.inner);
        self.inner = builder.add_subgraph(name, child);
        Ok(())
    }

    /// Validate and compile the graph.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (checkpointer=None, recursion_limit=None, interrupt_before=None, interrupt_after=None, cache=None, store=None, name=None))]
    fn compile(
        &self,
        py: Python<'_>,
        checkpointer: Option<&Bound<'_, PyAny>>,
        recursion_limit: Option<usize>,
        interrupt_before: Option<Vec<String>>,
        interrupt_after: Option<Vec<String>>,
        cache: Option<bool>,
        store: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
    ) -> PyResult<PyCompiledGraph> {
        let mut options = CompileOptions::new();
        if let Some(name) = name {
            options = options.with_name(name);
        }
        let mut saver = None;
        let mut builder = self.inner.clone();
        if let Some(checkpointer) = checkpointer_from_py(checkpointer)? {
            options = options.with_checkpointer(checkpointer.clone());
            saver = Some(checkpointer);
        }
        if cache == Some(true) {
            builder = builder.with_cache(Arc::new(crate::InMemoryCache::new()));
        }
        if let Some(store) = store_from_py(store)? {
            options = options.with_store(store);
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
        let compiled = builder
            .compile_with(options)
            .map_err(|error| to_pyerr(py, error))?;
        Ok(PyCompiledGraph {
            graph: Arc::new(compiled),
            saver,
        })
    }
}

/// A compiled, runnable graph.
#[pyclass(name = "CompiledGraph")]
pub struct PyCompiledGraph {
    graph: Arc<CompiledGraph<JsonState>>,
    saver: Option<Arc<dyn Checkpointer>>,
}

/// Graph description object returned by `CompiledGraph.get_graph()`.
#[pyclass(name = "Graph")]
pub struct PyGraph {
    mermaid: String,
    spec: Value,
}

#[pymethods]
impl PyGraph {
    /// Mermaid flowchart source.
    fn draw_mermaid(&self) -> String {
        self.mermaid.clone()
    }

    /// Serializable graph description (nodes, edges, conditional edges).
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_py(py, &self.spec)
    }

    fn __repr__(&self) -> String {
        "Graph()".to_string()
    }
}

#[pymethods]
impl PyCompiledGraph {
    /// Run the graph and return the final state dict.
    #[pyo3(signature = (input=None, thread_id=None, context=None, control=None))]
    fn invoke(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<String>,
        context: Option<&Bound<'_, PyAny>>,
        control: Option<PyRef<'_, super::types::PyRunControl>>,
    ) -> PyResult<Py<PyAny>> {
        let control = control.map(|control| control.inner.clone());
        if let Some(obj) = input {
            if let Ok(command) = obj.extract::<PyRef<PyCommand>>() {
                let thread = thread_id.as_deref().ok_or_else(|| {
                    PyTypeError::new_err("invoke(Command(...)) requires thread_id")
                })?;
                let (resume, updates) = command.to_resume_command(py)?;
                let result = py.detach(|| match resume {
                    Some(value) => self.graph.resume_with_updates(thread, value, updates),
                    None => self.graph.resume_command(
                        thread,
                        crate::Command {
                            resume: None,
                            update: Some(updates),
                            goto: Vec::new(),
                            parent: false,
                        },
                    ),
                });
                return match result {
                    Ok(state) => json_state_to_py(py, &state),
                    Err(error) => Err(to_pyerr(py, error)),
                };
            }
        }
        let state = state_from_py(input)?;
        let mut config = run_config(thread_id);
        if let Some(control) = control {
            config = config.with_control(control);
        }
        if let Some(context) = context {
            config = config.with_context(py_to_value(context)?);
        }
        let result = py.detach(|| self.graph.invoke(&state, &config));
        match result {
            Ok(state) => json_state_to_py(py, &state),
            Err(error) => Err(to_pyerr(py, error)),
        }
    }

    /// Run the graph and return `{"state": ..., "report": ...}`.
    #[pyo3(signature = (input=None, thread_id=None, context=None))]
    fn run(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<String>,
        context: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let state = state_from_py(input)?;
        let mut config = run_config(thread_id);
        if let Some(context) = context {
            config = config.with_context(py_to_value(context)?);
        }
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
    #[pyo3(signature = (input=None, thread_id=None, stream_mode=None, context=None, subgraphs=false, version=None, control=None))]
    #[allow(clippy::too_many_arguments)]
    fn stream(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<String>,
        stream_mode: Option<&Bound<'_, PyAny>>,
        context: Option<&Bound<'_, PyAny>>,
        subgraphs: bool,
        version: Option<String>,
        control: Option<PyRef<'_, super::types::PyRunControl>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let state = state_from_py(input)?;
        let mut config = run_config(thread_id);
        if let Some(control) = control {
            config = config.with_control(control.inner.clone());
        }
        if let Some(context) = context {
            config = config.with_context(py_to_value(context)?);
        }
        if subgraphs {
            config = config.with_stream_subgraphs(true);
        }
        let modes = stream_modes_from_py(stream_mode)?;
        let chunks = py
            .detach(|| match &modes {
                Some(modes) => self.graph.stream_modes(&state, &config, modes),
                None => self.graph.stream(&state, &config),
            })
            .map_err(|error| to_pyerr(py, error))?;
        let v2 = match version.as_deref() {
            None | Some("v1") => false,
            Some("v2") => true,
            Some(other) => {
                return Err(PyTypeError::new_err(format!(
                    "stream version must be 'v1' or 'v2', got {other:?}"
                )));
            }
        };
        chunks
            .iter()
            .map(|chunk| {
                if v2 {
                    chunk_to_py_v2(py, chunk)
                } else {
                    chunk_to_py(py, chunk)
                }
            })
            .collect()
    }

    /// Invoke the graph over a list of inputs.
    fn batch(&self, py: Python<'_>, inputs: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        let list = inputs
            .cast::<PyList>()
            .map_err(|_| PyTypeError::new_err("batch expects a list of state dicts"))?;
        let states: Vec<JsonState> = list
            .iter()
            .map(|item| state_from_py(Some(&item)))
            .collect::<PyResult<Vec<_>>>()?;
        let config = RunConfig::new();
        let results = py.detach(|| self.graph.batch(states, &config));
        results
            .into_iter()
            .map(|result| match result {
                Ok(state) => json_state_to_py(py, &state),
                Err(error) => Err(to_pyerr(py, error)),
            })
            .collect()
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
    #[pyo3(signature = (thread_id, checkpoint_ns=None))]
    fn get_state(
        &self,
        py: Python<'_>,
        thread_id: String,
        checkpoint_ns: Option<String>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let snapshot = match checkpoint_ns.as_deref() {
            Some(namespace) => self.graph.get_state_in(&thread_id, namespace),
            None => self.graph.get_state(&thread_id),
        }
        .map_err(|error| to_pyerr(py, error))?;
        snapshot
            .map(|snapshot| snapshot_to_py(py, &snapshot))
            .transpose()
    }

    /// Apply state updates to the latest checkpoint of a thread.
    ///
    /// With ``as_node`` the next tasks are derived from that node's outgoing edges.
    #[pyo3(signature = (thread_id, values, as_node=None))]
    fn update_state(
        &self,
        py: Python<'_>,
        thread_id: String,
        values: &Bound<'_, PyAny>,
        as_node: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let value = py_to_value(values)?;
        let map = value
            .as_object()
            .ok_or_else(|| PyTypeError::new_err("update_state expects a dict of channel writes"))?;
        let updates: Vec<(String, crate::Value)> = map
            .iter()
            .map(|(channel, value)| (channel.clone(), value.clone()))
            .collect();
        let snapshot = py
            .detach(|| {
                self.graph
                    .update_state(&thread_id, updates, as_node.as_deref())
            })
            .map_err(|error| to_pyerr(py, error))?;
        snapshot_to_py(py, &snapshot)
    }

    /// Return the checkpoint history of a thread, newest first.
    ///
    /// ``before`` is an exclusive checkpoint id and ``limit`` caps the result size.
    #[pyo3(signature = (thread_id, before=None, limit=None, checkpoint_ns=None))]
    fn get_state_history(
        &self,
        py: Python<'_>,
        thread_id: String,
        before: Option<String>,
        limit: Option<usize>,
        checkpoint_ns: Option<String>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let history = match checkpoint_ns.as_deref() {
            Some(namespace) => self.graph.get_state_history_filtered_in(
                &thread_id,
                namespace,
                before.as_deref(),
                limit,
            ),
            None => self
                .graph
                .get_state_history_filtered(&thread_id, before.as_deref(), limit),
        }
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

    /// Render the graph as a Mermaid `graph TD` diagram.
    fn to_mermaid(&self) -> String {
        self.graph.to_mermaid()
    }

    /// Export the latest checkpoint of a thread as LangGraph-shaped JSON.
    #[pyo3(signature = (thread_id, checkpoint_ns=None))]
    fn export_checkpoint(
        &self,
        py: Python<'_>,
        thread_id: String,
        checkpoint_ns: Option<String>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let document = match checkpoint_ns {
            Some(namespace) => self
                .graph
                .checkpointer()
                .and_then(|checkpointer| checkpointer.latest_in(&thread_id, &namespace).ok())
                .flatten()
                .map(|checkpoint| crate::interop::checkpoint_to_langgraph(&checkpoint)),
            None => self
                .graph
                .export_checkpoint(&thread_id)
                .map_err(|error| to_pyerr(py, error))?,
        };
        document
            .map(|document| value_to_py(py, &document))
            .transpose()
    }

    /// Import a LangGraph-shaped checkpoint document as a checkpoint of `thread_id`.
    fn import_checkpoint(
        &self,
        py: Python<'_>,
        thread_id: String,
        document: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let document = py_to_value(document)?;
        self.graph
            .import_checkpoint(&thread_id, &document)
            .map_err(|error| to_pyerr(py, error))
    }

    /// Optional name given at compile time.
    #[getter]
    fn name(&self) -> Option<String> {
        self.graph.name().map(str::to_owned)
    }

    /// Graph description object (`draw_mermaid()` / `to_dict()`).
    fn get_graph(&self) -> PyGraph {
        PyGraph {
            mermaid: self.graph.to_mermaid(),
            spec: self.graph.graph_spec(),
        }
    }

    /// List threads that have at least one checkpoint.
    fn threads(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.graph.threads().map_err(|error| to_pyerr(py, error))
    }

    /// Keep only the newest `keep_last` checkpoints of a thread.
    #[pyo3(signature = (thread_id, keep_last=1))]
    fn prune_thread(&self, py: Python<'_>, thread_id: String, keep_last: usize) -> PyResult<()> {
        self.graph
            .prune_thread(&thread_id, keep_last)
            .map_err(|error| to_pyerr(py, error))
    }

    /// Delete every checkpoint of a thread.
    fn delete_thread(&self, py: Python<'_>, thread_id: String) -> PyResult<()> {
        self.graph
            .delete_thread(&thread_id)
            .map_err(|error| to_pyerr(py, error))
    }

    /// Copy every checkpoint of one thread to another thread id.
    fn copy_thread(
        &self,
        py: Python<'_>,
        source_thread: String,
        target_thread: String,
    ) -> PyResult<()> {
        self.graph
            .copy_thread(&source_thread, &target_thread)
            .map_err(|error| to_pyerr(py, error))
    }
}

/// Parse a `stream_mode` argument: one name or a list of names.
pub(super) fn stream_modes_from_py(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<crate::StreamMode>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let names: Vec<String> = if let Ok(name) = value.extract::<String>() {
        vec![name]
    } else {
        value.extract::<Vec<String>>().map_err(|_| {
            PyTypeError::new_err("stream_mode must be a string or a list of strings")
        })?
    };
    let mut modes = Vec::with_capacity(names.len());
    for name in names {
        let mode = crate::StreamMode::parse(&name)
            .ok_or_else(|| PyTypeError::new_err(format!("unknown stream_mode `{name}`")))?;
        modes.push(mode);
    }
    Ok(Some(modes))
}
