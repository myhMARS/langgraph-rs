//! Python graph builder and compiled graph classes.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::convert::*;
use crate::*;

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

    /// Render the graph as a Mermaid `graph TD` diagram.
    fn to_mermaid(&self) -> String {
        self.graph.to_mermaid()
    }
}
