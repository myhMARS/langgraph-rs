//! Python graph builder and compiled graph classes.

use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use super::convert::*;
use super::types::{
    PyCachePolicy, PyCommand, PyRetryPolicy, PyTimeoutPolicy, checkpointer_from_py, store_from_py,
};
use crate::*;

// ---------------------------------------------------------------------------
// Python classes
// ---------------------------------------------------------------------------

/// A mutable graph builder.
#[pyclass(name = "StateGraph", subclass)]
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
    #[pyo3(signature = (schema=None, input_schema=None, output_schema=None, private_schema=None, managed_values=false))]
    fn new(
        schema: Option<&Bound<'_, PyAny>>,
        input_schema: Option<Vec<String>>,
        output_schema: Option<Vec<String>>,
        private_schema: Option<Vec<String>>,
        managed_values: bool,
    ) -> PyResult<Self> {
        let mut schema = schema_from_py(schema)?;
        if let Some(channels) = input_schema {
            schema = schema.with_input_channels(channels);
        }
        if let Some(channels) = output_schema {
            schema = schema.with_output_channels(channels);
        }
        if let Some(channels) = private_schema {
            schema = schema.with_private_channels(channels);
        }
        if managed_values {
            schema = schema.with_managed_values();
        }
        Ok(Self {
            inner: crate::StateGraph::with_schema(schema),
        })
    }

    /// Register a Python callable as a node.
    ///
    /// Optional LangGraph-style keywords: ``retry_policy``, ``cache_policy`` and
    /// ``error_handler`` (node name).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, function, retry_policy=None, cache_policy=None, error_handler=None, deferred=false, destinations=None, input_schema=None, timeout=None))]
    fn add_node(
        mut slf: PyRefMut<'_, Self>,
        name: String,
        function: Py<PyAny>,
        retry_policy: Option<PyRef<'_, PyRetryPolicy>>,
        cache_policy: Option<PyRef<'_, PyCachePolicy>>,
        error_handler: Option<String>,
        deferred: bool,
        destinations: Option<Vec<String>>,
        input_schema: Option<&Bound<'_, PyAny>>,
        timeout: Option<PyRef<'_, PyTimeoutPolicy>>,
    ) -> PyResult<Py<Self>> {
        let mut graph = std::mem::take(&mut slf.inner);
        if deferred {
            graph = graph.set_deferred(name.clone());
        }
        if let Some(destinations) = destinations {
            graph = graph.set_destinations(name.clone(), destinations);
        }
        if let Some(schema) = input_schema {
            graph = graph.set_node_input_schema(name.clone(), py_to_value(schema)?);
        }
        // LangGraph lets a compiled subgraph be registered with `add_node(name, subgraph)`; route
        // it to the subgraph path so both spellings behave the same.
        let subgraph = function
            .bind(slf.py())
            .extract::<PyRef<'_, PyCompiledGraph>>()
            .ok()
            .map(|child| child.graph.clone());
        graph = match subgraph {
            Some(child) => graph.add_subgraph(name.clone(), child),
            None => graph.add_node(name.clone(), node_from_python(name.clone(), function)),
        };
        if let Some(policy) = retry_policy {
            graph = graph.set_retry(name.clone(), policy.to_policy());
        }
        if let Some(policy) = cache_policy {
            graph = graph.set_cache(name.clone(), policy.to_policy());
        }
        if let Some(handler) = error_handler {
            graph = graph.add_error_edge(name.clone(), normalize_node(&handler));
        }
        if let Some(policy) = timeout {
            graph = graph.set_timeout(name, policy.to_policy());
        }
        slf.inner = graph;
        Ok(slf.into())
    }

    /// Add a static edge (`"START"` / `"END"` are accepted as source/target).
    fn add_edge(
        mut slf: PyRefMut<'_, Self>,
        from: String,
        to: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let targets = targets_from_py(to)?;
        let graph = std::mem::take(&mut slf.inner);
        slf.inner = graph.add_edge(normalize_node(&from), targets);
        Ok(slf.into())
    }

    /// Add a conditional edge. Either return node names directly or pass a `path_map`.
    #[pyo3(signature = (source, router, path_map=None))]
    fn add_conditional_edges(
        mut slf: PyRefMut<'_, Self>,
        source: String,
        router: Py<PyAny>,
        path_map: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<Self>> {
        let graph = std::mem::take(&mut slf.inner);
        let route = router_from_python(router);
        slf.inner = match path_map.map(path_map_from_py).transpose()?.flatten() {
            Some(path_map) => graph.add_conditional_edges_mapped(source, route, path_map),
            None => graph.add_conditional_edges(source, route),
        };
        Ok(slf.into())
    }

    /// Set the static entry point.
    fn set_entry_point(mut slf: PyRefMut<'_, Self>, node: String) -> Py<Self> {
        let graph = std::mem::take(&mut slf.inner);
        slf.inner = graph.set_entry_point(node);
        slf.into()
    }

    /// Use a router to choose the entry point.
    fn set_conditional_entry_point(mut slf: PyRefMut<'_, Self>, router: Py<PyAny>) -> Py<Self> {
        let graph = std::mem::take(&mut slf.inner);
        slf.inner = graph.set_conditional_entry_point(router_from_python(router));
        slf.into()
    }

    /// Add an edge from `node` to `"END"`.
    fn set_finish_point(mut slf: PyRefMut<'_, Self>, node: String) -> Py<Self> {
        let graph = std::mem::take(&mut slf.inner);
        slf.inner = graph.set_finish_point(node);
        slf.into()
    }

    /// Embed a compiled graph as a node, sharing the parent thread and state type.
    ///
    /// The child runs in a checkpoint namespace derived from the node name, so an
    /// `interrupt()` inside it pauses the parent and a parent `resume` continues it.
    fn add_subgraph(
        mut slf: PyRefMut<'_, Self>,
        name: String,
        subgraph: PyRef<'_, PyCompiledGraph>,
    ) -> PyResult<Py<Self>> {
        let child = subgraph.graph.clone();
        let builder = std::mem::take(&mut slf.inner);
        slf.inner = builder.add_subgraph(name, child);
        Ok(slf.into())
    }

    /// Validate and compile the graph.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (checkpointer=None, recursion_limit=None, interrupt_before=None, interrupt_after=None, cache=None, store=None, name=None, context_schema=None, transformers=None))]
    fn compile(
        &self,
        py: Python<'_>,
        checkpointer: Option<&Bound<'_, PyAny>>,
        recursion_limit: Option<usize>,
        interrupt_before: Option<Vec<String>>,
        interrupt_after: Option<Vec<String>>,
        cache: Option<&Bound<'_, PyAny>>,
        store: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
        context_schema: Option<&Bound<'_, PyAny>>,
        transformers: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyCompiledGraph> {
        let mut options = CompileOptions::new();
        if let Some(name) = name {
            options = options.with_name(name);
        }
        if let Some(schema) = context_schema {
            options = options.with_context_schema(super::convert::context_schema_value(py, schema)?);
        }
        let mut saver = None;
        let mut builder = self.inner.clone();
        if let Some(checkpointer) = checkpointer_from_py(checkpointer)? {
            options = options.with_checkpointer(checkpointer.clone());
            saver = Some(checkpointer);
        }
        if let Some(cache) = cache {
            if cache.is_none() {
                // no cache object
            } else if let Ok(flag) = cache.extract::<bool>() {
                if flag {
                    builder = builder.with_cache(Arc::new(crate::InMemoryCache::new()));
                }
            } else if let Some(cache_object) = super::types::cache_from_py(cache) {
                builder = builder.with_cache(cache_object);
            } else {
                return Err(PyTypeError::new_err("cache must be True or a cache object"));
            }
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
        let stream_transformers = match transformers.filter(|value| !value.is_none()) {
            Some(list) => list
                .try_iter()?
                .map(|item| item.map(|item| item.unbind()))
                .collect::<PyResult<Vec<_>>>()?,
            None => Vec::new(),
        };
        Ok(PyCompiledGraph {
            graph: Arc::new(compiled),
            saver,
            stream_transformers,
        })
    }
}

#[path = "graph_compiled.rs"]
mod graph_compiled;
#[path = "graph_invoke.rs"]
mod graph_invoke;
pub(crate) use graph_compiled::*;
