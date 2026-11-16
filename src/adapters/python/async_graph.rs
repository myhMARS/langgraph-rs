//! Async Python bindings (`python-async` feature): `AsyncStateGraph` / `AsyncCompiledGraph`.

use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

use super::convert::*;
use super::types::{checkpointer_from_py, store_from_py};
use crate::BlockingSaver;
use crate::JsonState;
use crate::kernel::context::TaskContext;
use crate::kernel::node::NodeOutput;
use crate::kernel::stream::StreamChunk;
use crate::tokio_runtime::{AsyncCompiledGraph, AsyncContext, AsyncStateGraph};
use crate::*;

fn async_node_from_python(
    name: String,
    function: Py<PyAny>,
) -> impl Fn(
    JsonState,
    AsyncContext,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = crate::Result<NodeOutput>> + std::marker::Send>,
> + std::marker::Send
+ Sync
+ 'static {
    let name = Arc::new(name);
    let function = Arc::new(function);
    move |state: JsonState, context: AsyncContext| {
        let name = name.clone();
        let function = function.clone();
        Box::pin(async move {
            Python::attach(|py| -> crate::Result<NodeOutput> {
                let py_state =
                    json_state_to_py(py, &state).map_err(|error| pyerr_to_error(py, error))?;
                let resume = context.take_resume();
                let node_name = name.as_ref().clone();
                let (result, custom, messages, usage) = crate::kernel::context::with_task(
                    TaskContext::new(node_name, resume),
                    || -> crate::Result<NodeOutput> {
                        let value = function
                            .as_ref()
                            .bind(py)
                            .call1((py_state,))
                            .map_err(|error| pyerr_to_error(py, error))?;
                        // Python async nodes: drive the returned awaitable to completion on a
                        // private event loop while the task context stays installed, so
                        // interrupt()/custom()/usage helpers keep working inside coroutines.
                        let value = if value
                            .hasattr("__await__")
                            .map_err(|error| pyerr_to_error(py, error))?
                        {
                            py.import("asyncio")
                                .and_then(|asyncio| asyncio.getattr("run"))
                                .and_then(|run| run.call1((&value,)))
                                .map_err(|error| pyerr_to_error(py, error))?
                        } else {
                            value
                        };
                        node_output_from_py(&value).map_err(|error| pyerr_to_error(py, error))
                    },
                );
                for item in usage {
                    context.record_usage(item);
                }
                for item in custom {
                    let _ = context.custom(item);
                }
                for item in messages {
                    let _ = context.message(item);
                }
                result
            })
        })
            as std::pin::Pin<
                Box<
                    dyn std::future::Future<Output = crate::Result<NodeOutput>> + std::marker::Send,
                >,
            >
    }
}

/// Async graph builder (Python).
#[pyclass(name = "AsyncStateGraph")]
pub struct PyAsyncStateGraph {
    inner: AsyncStateGraph<JsonState>,
}

#[pymethods]
impl PyAsyncStateGraph {
    #[new]
    #[pyo3(signature = (schema=None))]
    fn new(schema: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            inner: AsyncStateGraph::with_schema(schema_from_py(schema)?),
        })
    }

    /// Register a Python callable as an async node.
    fn add_node(&mut self, name: String, function: Py<PyAny>) -> PyResult<()> {
        let builder = std::mem::replace(
            &mut self.inner,
            AsyncStateGraph::with_schema(StateSchema::dynamic()),
        );
        self.inner = builder.add_node(name.clone(), async_node_from_python(name, function));
        Ok(())
    }

    fn set_entry_point(&mut self, node: String) {
        let builder = std::mem::replace(
            &mut self.inner,
            AsyncStateGraph::with_schema(StateSchema::dynamic()),
        );
        self.inner = builder.set_entry_point(normalize_node(&node));
    }

    fn set_finish_point(&mut self, node: String) {
        let builder = std::mem::replace(
            &mut self.inner,
            AsyncStateGraph::with_schema(StateSchema::dynamic()),
        );
        self.inner = builder.set_finish_point(normalize_node(&node));
    }

    fn add_edge(&mut self, from: String, to: &Bound<'_, PyAny>) -> PyResult<()> {
        let targets = targets_from_py(to)?;
        let mut builder = std::mem::replace(
            &mut self.inner,
            AsyncStateGraph::with_schema(StateSchema::dynamic()),
        );
        for target in targets {
            builder = builder.add_edge(normalize_node(&from), normalize_node(&target));
        }
        self.inner = builder;
        Ok(())
    }

    /// Embed a compiled async graph as a node (child checkpoint namespace).
    fn add_subgraph(
        &mut self,
        name: String,
        subgraph: PyRef<'_, PyAsyncCompiledGraph>,
    ) -> PyResult<()> {
        let child = subgraph.graph.clone();
        let builder = std::mem::replace(
            &mut self.inner,
            AsyncStateGraph::with_schema(StateSchema::dynamic()),
        );
        self.inner = builder.add_subgraph(name, child);
        Ok(())
    }

    fn add_conditional_edges(&mut self, from: String, router: Py<PyAny>) -> PyResult<()> {
        let builder = std::mem::replace(
            &mut self.inner,
            AsyncStateGraph::with_schema(StateSchema::dynamic()),
        );
        let router = Arc::new(router);
        self.inner =
            builder.add_conditional_edges(normalize_node(&from), move |state: &JsonState| {
                Python::attach(|py| -> crate::Result<Route> {
                    let py_state =
                        json_state_to_py(py, state).map_err(|error| pyerr_to_error(py, error))?;
                    let value = router
                        .as_ref()
                        .bind(py)
                        .call1((py_state,))
                        .map_err(|error| pyerr_to_error(py, error))?;
                    route_from_py(&value).map_err(|error| pyerr_to_error(py, error))
                })
            });
        Ok(())
    }

    #[pyo3(signature = (checkpointer=None, recursion_limit=None, cache=None, store=None))]
    fn compile(
        &self,
        py: Python<'_>,
        checkpointer: Option<&Bound<'_, PyAny>>,
        recursion_limit: Option<usize>,
        cache: Option<bool>,
        store: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyAsyncCompiledGraph> {
        let mut builder = self.inner.clone();
        let mut saver = None;
        if let Some(checkpointer) = checkpointer_from_py(checkpointer)? {
            let adapter =
                BlockingSaver::new(checkpointer.clone()).map_err(|error| to_pyerr(py, error))?;
            builder = builder.with_async_checkpointer(Arc::new(adapter));
            saver = Some(checkpointer);
        }
        if let Some(limit) = recursion_limit {
            builder = builder.with_recursion_limit(limit);
        }
        if cache == Some(true) {
            builder = builder.with_cache(Arc::new(crate::InMemoryCache::new()));
        }
        if let Some(store) = store_from_py(store)? {
            builder = builder.with_store(store);
        }
        let compiled = builder.compile().map_err(|error| to_pyerr(py, error))?;
        Ok(PyAsyncCompiledGraph {
            graph: Arc::new(compiled),
            saver,
        })
    }
}

/// Compiled async graph (Python).
#[pyclass(name = "AsyncCompiledGraph")]
pub struct PyAsyncCompiledGraph {
    pub(crate) graph: Arc<AsyncCompiledGraph<JsonState>>,
    saver: Option<Arc<dyn Checkpointer>>,
}

#[pymethods]
impl PyAsyncCompiledGraph {
    /// Awaitable graph invocation returning the final state dict.
    #[pyo3(signature = (input=None, thread_id=None, context=None))]
    fn ainvoke<'py>(
        &self,
        py: Python<'py>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<String>,
        context: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = state_from_py(input)?;
        let mut config = run_config(thread_id);
        if let Some(context) = context {
            config = config.with_context(py_to_value(context)?);
        }
        let graph = self.graph.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match graph
                .run(&state, &config)
                .await
                .map(|outcome| outcome.state)
            {
                Ok(state) => Python::attach(|py| json_state_to_py(py, &state)),
                Err(error) => Python::attach(|py| Err(to_pyerr(py, error))),
            }
        })
    }

    /// Awaitable resume of an interrupted thread.
    fn aresume<'py>(
        &self,
        py: Python<'py>,
        thread_id: String,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let value = py_to_value(value)?;
        let graph = self.graph.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match graph.resume(&thread_id, value).await {
                Ok(state) => Python::attach(|py| json_state_to_py(py, &state)),
                Err(error) => Python::attach(|py| Err(to_pyerr(py, error))),
            }
        })
    }

    /// Awaitable batch invocation returning the final state of every input.
    #[pyo3(signature = (inputs, thread_id=None, context=None))]
    fn abatch<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'_, PyAny>,
        thread_id: Option<String>,
        context: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let list = inputs
            .cast::<PyList>()
            .map_err(|_| PyTypeError::new_err("abatch expects a list of state dicts"))?;
        let states: Vec<JsonState> = list
            .iter()
            .map(|item| state_from_py(Some(&item)))
            .collect::<PyResult<Vec<_>>>()?;
        let mut config = run_config(thread_id);
        if let Some(context) = context {
            config = config.with_context(py_to_value(context)?);
        }
        let graph = self.graph.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let results = graph.batch(states, &config).await;
            Python::attach(|py| {
                let output = PyList::empty(py);
                for result in results {
                    match result {
                        Ok(state) => output.append(json_state_to_py(py, &state)?)?,
                        Err(error) => return Err(to_pyerr(py, error)),
                    }
                }
                Ok(output.into_any().unbind())
            })
        })
    }

    /// Awaitable stream collection returning a list of chunk dicts.
    #[pyo3(signature = (input=None, thread_id=None, context=None, stream_mode=None, subgraphs=false, version=None))]
    #[allow(clippy::too_many_arguments)]
    fn astream<'py>(
        &self,
        py: Python<'py>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<String>,
        context: Option<&Bound<'_, PyAny>>,
        stream_mode: Option<&Bound<'_, PyAny>>,
        subgraphs: bool,
        version: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v2 = match version.as_deref() {
            None | Some("v1") => false,
            Some("v2") => true,
            Some(other) => {
                return Err(PyTypeError::new_err(format!(
                    "stream version must be 'v1' or 'v2', got {other:?}"
                )));
            }
        };
        let state = state_from_py(input)?;
        let mut config = run_config(thread_id);
        if let Some(context) = context {
            config = config.with_context(py_to_value(context)?);
        }
        if subgraphs {
            config = config.with_stream_subgraphs(true);
        }
        let modes = super::graph::stream_modes_from_py(stream_mode)?;
        let graph = self.graph.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut collected: Vec<StreamChunk> = Vec::new();
            let result = match &modes {
                Some(modes) => {
                    graph
                        .stream_modes_with(&state, &config, modes, |chunk| collected.push(chunk))
                        .await
                }
                None => {
                    graph
                        .stream_with(&state, &config, |chunk| collected.push(chunk))
                        .await
                }
            };
            if let Err(error) = result {
                return Python::attach(|py| Err(to_pyerr(py, error)));
            }
            Python::attach(|py| {
                let list = PyList::empty(py);
                for chunk in &collected {
                    let item = if v2 {
                        chunk_to_py_v2(py, chunk)?
                    } else {
                        chunk_to_py(py, chunk)?
                    };
                    list.append(item)?;
                }
                Ok(list.into_any().unbind())
            })
        })
    }

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

    fn has_checkpointer(&self) -> bool {
        self.saver.is_some()
    }

    /// Read the full checkpoint history of a thread, newest first.
    #[pyo3(signature = (thread_id, checkpoint_ns=None))]
    fn get_state_history(
        &self,
        py: Python<'_>,
        thread_id: String,
        checkpoint_ns: Option<String>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let snapshots = match checkpoint_ns.as_deref() {
            Some(namespace) => self
                .graph
                .get_state_history_in(&thread_id, namespace)
                .map_err(|error| to_pyerr(py, error))?,
            None => self
                .graph
                .get_state_history(&thread_id)
                .map_err(|error| to_pyerr(py, error))?,
        };
        snapshots
            .iter()
            .map(|snapshot| snapshot_to_py(py, snapshot))
            .collect()
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

    fn to_mermaid(&self) -> String {
        self.graph.to_mermaid()
    }
}
