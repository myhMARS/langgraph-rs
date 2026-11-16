//! Async Python bindings (`python-async` feature): `AsyncStateGraph` / `AsyncCompiledGraph`.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use super::convert::*;
use super::types::{checkpointer_from_py, store_from_py};
use crate::JsonState;
use crate::kernel::context::TaskContext;
use crate::kernel::node::NodeOutput;
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
            builder = builder.with_checkpointer(checkpointer.clone());
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

    fn to_mermaid(&self) -> String {
        self.graph.to_mermaid()
    }
}
