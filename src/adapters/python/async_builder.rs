//! Async Python graph builder and node adapter.

use std::sync::Arc;

use pyo3::types::PyAny;

use super::*;
use crate::Route;
use crate::kernel::context::TaskContext;
use crate::kernel::state::StateSchema;
use crate::tokio_runtime::AsyncStateGraph;

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
                let run_timeout = context.run_timeout();
                let node_name = name.as_ref().clone();
                let callback_guard = install_langchain_callbacks(py);
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
                            let value = match run_timeout {
                                Some(timeout) => py
                                    .import("asyncio")
                                    .and_then(|asyncio| asyncio.getattr("wait_for"))
                                    .and_then(|wait_for| {
                                        wait_for.call1((&value, timeout.as_secs_f64()))
                                    })
                                    .map_err(|error| pyerr_to_error(py, error))?,
                                None => value,
                            };
                            if run_timeout.is_some() {
                                run_timed_awaitable(py, &value)
                                    .map_err(|error| pyerr_to_error(py, error))?
                            } else {
                                py.import("asyncio")
                                    .and_then(|asyncio| asyncio.getattr("run"))
                                    .and_then(|run| run.call1((&value,)))
                                    .map_err(|error| pyerr_to_error(py, error))?
                            }
                        } else {
                            value
                        };
                        node_output_from_py(&value).map_err(|error| pyerr_to_error(py, error))
                    },
                );
                if let Some((variable, token)) = callback_guard {
                    let _ = variable.call_method1("reset", (token,));
                }
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
    #[pyo3(signature = (name, function, timeout=None))]
    fn add_node(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        name: String,
        function: Py<PyAny>,
        timeout: Option<PyRef<'_, PyTimeoutPolicy>>,
    ) -> PyResult<Py<Self>> {
        let builder = std::mem::replace(
            &mut slf.inner,
            AsyncStateGraph::with_schema(StateSchema::dynamic()),
        );
        // Python coroutines are driven by a private asyncio loop, so a Tokio
        // timeout cannot interrupt them; enforce the run timeout with
        // asyncio.wait_for inside the node itself instead.
        let function = match timeout.as_ref().and_then(|policy| policy.run_timeout) {
            Some(seconds) => wrap_async_timeout(py, function, seconds)?,
            None => function,
        };
        let mut builder =
            builder.add_node(name.clone(), async_node_from_python(name.clone(), function));
        if let Some(policy) = timeout {
            builder = builder.set_timeout(name, policy.to_policy());
        }
        slf.inner = builder;
        Ok(slf.into())
    }

    fn set_entry_point(mut slf: PyRefMut<'_, Self>, node: String) -> Py<Self> {
        let builder = std::mem::replace(
            &mut slf.inner,
            AsyncStateGraph::with_schema(StateSchema::dynamic()),
        );
        slf.inner = builder.set_entry_point(normalize_node(&node));
        slf.into()
    }

    fn set_finish_point(mut slf: PyRefMut<'_, Self>, node: String) -> Py<Self> {
        let builder = std::mem::replace(
            &mut slf.inner,
            AsyncStateGraph::with_schema(StateSchema::dynamic()),
        );
        slf.inner = builder.set_finish_point(normalize_node(&node));
        slf.into()
    }

    fn add_edge(
        mut slf: PyRefMut<'_, Self>,
        from: String,
        to: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let targets = targets_from_py(to)?;
        let mut builder = std::mem::replace(
            &mut slf.inner,
            AsyncStateGraph::with_schema(StateSchema::dynamic()),
        );
        for target in targets {
            builder = builder.add_edge(normalize_node(&from), normalize_node(&target));
        }
        slf.inner = builder;
        Ok(slf.into())
    }

    /// Embed a compiled async graph as a node (child checkpoint namespace).
    fn add_subgraph(
        mut slf: PyRefMut<'_, Self>,
        name: String,
        subgraph: PyRef<'_, PyAsyncCompiledGraph>,
    ) -> PyResult<Py<Self>> {
        let child = subgraph.graph.clone();
        let builder = std::mem::replace(
            &mut slf.inner,
            AsyncStateGraph::with_schema(StateSchema::dynamic()),
        );
        slf.inner = builder.add_subgraph(name, child);
        Ok(slf.into())
    }

    fn add_conditional_edges(
        mut slf: PyRefMut<'_, Self>,
        from: String,
        router: Py<PyAny>,
    ) -> PyResult<Py<Self>> {
        let builder = std::mem::replace(
            &mut slf.inner,
            AsyncStateGraph::with_schema(StateSchema::dynamic()),
        );
        let router = Arc::new(router);
        slf.inner =
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
        Ok(slf.into())
    }

    #[pyo3(signature = (checkpointer=None, recursion_limit=None, cache=None, store=None, interrupt_before=None, interrupt_after=None, name=None))]
    #[allow(clippy::too_many_arguments)]
    fn compile(
        &self,
        py: Python<'_>,
        checkpointer: Option<&Bound<'_, PyAny>>,
        recursion_limit: Option<usize>,
        cache: Option<bool>,
        store: Option<&Bound<'_, PyAny>>,
        interrupt_before: Option<Vec<String>>,
        interrupt_after: Option<Vec<String>>,
        name: Option<String>,
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
        if let Some(nodes) = interrupt_before {
            builder = builder.interrupt_before(nodes);
        }
        if let Some(nodes) = interrupt_after {
            builder = builder.interrupt_after(nodes);
        }
        if let Some(name) = name {
            builder = builder.with_name(name);
        }
        let compiled = builder.compile().map_err(|error| to_pyerr(py, error))?;
        Ok(PyAsyncCompiledGraph {
            graph: Arc::new(compiled),
            saver,
        })
    }
}
