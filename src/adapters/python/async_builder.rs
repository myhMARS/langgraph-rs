//! Async Python graph builder and node adapter.

use std::sync::Arc;

use pyo3::types::PyAny;

use super::*;
use crate::_internal::state::StateSchema;
use crate::adapters::python::types::{PyCachePolicy, PyRetryPolicy};
use crate::Route;
use crate::runtime::context::TaskContext;
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
                let task_id = crate::runtime::context::current_task_id();
                let callback_guard = install_langchain_callbacks(py);
                let task_context = match task_id {
                    Some(task_id) => TaskContext::new(node_name, resume).with_task_id(task_id),
                    None => TaskContext::new(node_name, resume),
                };
                let (result, custom, tools, messages, writes, usage) = crate::runtime::context::with_task(
                    task_context,
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
                            // Reuse the thread's asyncio loop: `asyncio.run` would create and
                            // tear down a loop per node (~1 ms each), which dominates an
                            // otherwise sub-millisecond orchestration step.
                            run_timed_awaitable(py, &value)
                                .map_err(|error| pyerr_to_error(py, error))?
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
                for item in tools {
                    let _ = context.tool_event(item);
                }
                for item in messages {
                    let _ = context.message(item);
                }
                // Writes the node performed itself (`send()` / `push_ui_message`) are merged into
                // its output so the reducer applies them exactly like a returned value.
                match result {
                    Ok(output) => Ok(crate::adapters::python::convert::apply_node_writes(output, writes)),
                    other => other,
                }
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
#[pyclass(name = "AsyncStateGraph", subclass)]
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
    #[pyo3(signature = (name, function, retry_policy=None, cache_policy=None, error_handler=None, deferred=false, destinations=None, timeout=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_node(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        name: String,
        function: Py<PyAny>,
        retry_policy: Option<PyRef<'_, PyRetryPolicy>>,
        cache_policy: Option<PyRef<'_, PyCachePolicy>>,
        error_handler: Option<String>,
        deferred: bool,
        destinations: Option<Vec<String>>,
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
        let mut builder = builder;
        if deferred {
            builder = builder.set_deferred(name.clone());
        }
        if let Some(destinations) = destinations {
            builder = builder.set_destinations(name.clone(), destinations);
        }
        builder = builder.add_node(name.clone(), async_node_from_python(name.clone(), function));
        if let Some(policy) = retry_policy {
            builder = builder.set_retry(name.clone(), policy.to_policy());
        }
        if let Some(policy) = cache_policy {
            builder = builder.set_cache(name.clone(), policy.to_policy());
        }
        if let Some(handler) = error_handler {
            builder = builder.add_error_edge(name.clone(), normalize_node(&handler));
        }
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
        subgraph: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let child = {
            let direct = subgraph
                .extract::<PyRef<'_, PyAsyncCompiledGraph>>()
                .ok()
                .map(|child| child.graph.clone());
            match direct {
                Some(child) => child,
                None => {
                    let inner = subgraph
                        .getattr("_inner")
                        .map_err(|_| PyTypeError::new_err("add_subgraph expects a compiled graph"))?;
                    inner
                        .extract::<PyRef<'_, PyAsyncCompiledGraph>>()
                        .map_err(|_| PyTypeError::new_err("add_subgraph expects a compiled graph"))?
                        .graph
                        .clone()
                }
            }
        };
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

    #[pyo3(signature = (checkpointer=None, recursion_limit=None, cache=None, store=None, interrupt_before=None, interrupt_after=None, name=None, transformers=None))]
    #[allow(clippy::too_many_arguments)]
    fn compile(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        checkpointer: Option<&Bound<'_, PyAny>>,
        recursion_limit: Option<usize>,
        cache: Option<bool>,
        store: Option<&Bound<'_, PyAny>>,
        interrupt_before: Option<Vec<String>>,
        interrupt_after: Option<Vec<String>>,
        name: Option<String>,
        transformers: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let mut builder = slf.inner.clone();
        let mut saver = None;
        if let Some(checkpointer) = checkpointer_from_py(checkpointer)? {
            // An in-memory saver cannot block, so the `BlockingSaver` wrapper would only add
            // a `spawn_blocking` round trip (~40 us) to every read and write. Keep those
            // savers on the sync path, which the async engine already supports inline.
            if checkpointer.is_in_memory() {
                builder = builder.with_checkpointer(checkpointer.clone());
            } else {
                let adapter =
                    BlockingSaver::new(checkpointer.clone()).map_err(|error| to_pyerr(py, error))?;
                builder = builder.with_async_checkpointer(Arc::new(adapter));
            }
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
        let stream_transformers = match transformers.filter(|value| !value.is_none()) {
            Some(list) => list
                .try_iter()?
                .map(|item| item.map(|item| item.unbind()))
                .collect::<PyResult<Vec<_>>>()?,
            None => Vec::new(),
        };
        let native = Py::new(
            py,
            PyAsyncCompiledGraph {
                graph: Arc::new(compiled),
                saver,
                stream_transformers,
                checkpointer_py: checkpointer
                    .filter(|value| !value.is_none())
                    .map(|value| value.clone().unbind()),
                builder_py: Some(slf.into_pyobject(py)?.into_any().unbind()),
                last_lifecycle: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            },
        )?;
        // Return the native object: the Python builder wraps it in the `Runnable` facade.
        Ok(native.into_any())
    }
}
