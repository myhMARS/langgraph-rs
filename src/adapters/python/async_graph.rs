//! Async Python bindings (`python-async` feature): `AsyncStateGraph` / `AsyncCompiledGraph`.

use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use super::convert::install_langchain_callbacks;
use super::convert::*;
use super::types::{PyCommand, PyTimeoutPolicy, checkpointer_from_py, store_from_py};
use crate::_internal::node::NodeOutput;
use crate::_internal::stream::StreamChunk;
use crate::BlockingSaver;
use crate::JsonState;
use crate::tokio_runtime::{AsyncCompiledGraph, AsyncContext};
use crate::*;

/// Compiled async graph (Python).
#[pyclass(name = "AsyncCompiledGraph")]
pub struct PyAsyncCompiledGraph {
    pub(crate) graph: Arc<AsyncCompiledGraph<JsonState>>,
    saver: Option<Arc<dyn Checkpointer>>,
    /// Transformers registered at compile time for `astream_events(version="v3")`.
    pub(crate) stream_transformers: Vec<Py<PyAny>>,
    /// The checkpointer object passed to `compile()` (LangGraph's `checkpointer` attribute).
    pub(crate) checkpointer_py: Option<Py<PyAny>>,
    /// The builder this graph was compiled from (LangGraph's `builder` attribute).
    pub(crate) builder_py: Option<Py<PyAny>>,
    /// Pause/resume transitions recorded by the last run (drained by the facade).
    pub(crate) last_lifecycle:
        std::sync::Arc<std::sync::Mutex<Vec<crate::_internal::lifecycle::LifecycleEvent>>>,
}

#[path = "async_support.rs"]
mod async_support;
use async_support::*;
#[pymethods]
impl PyAsyncCompiledGraph {
    /// Awaitable graph invocation returning the final state dict.
    #[pyo3(signature = (input=None, thread_id=None, context=None, config=None))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn ainvoke<'py>(
        &self,
        py: Python<'py>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<&Bound<'_, PyAny>>,
        context: Option<&Bound<'_, PyAny>>,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (thread, mut config) =
            super::convert::config_from_py(py, config.or(thread_id))?;
        if let Some(context) = context {
            config = config.with_context(py_to_value(context)?);
        }
        let mut resume_start: Option<(
            Option<crate::Value>,
            Option<serde_json::Map<String, crate::Value>>,
            Vec<String>,
            Vec<crate::Send>,
        )> = None;
        let mut command: Option<crate::Command> = None;
        if let Some(obj) = input {
            if let Ok(py_command) = obj.cast::<PyCommand>().map(|cell| cell.borrow()) {
                let (resume, updates) = py_command.to_resume_command(py)?;
                let (goto, sends) = py_command.goto_parts(py)?;
                let command_update = Some(updates.clone());
                resume_start = Some((
                    resume.clone(),
                    Some(updates.into_iter().collect()),
                    goto.clone(),
                    sends.clone(),
                ));
                command = Some(crate::Command {
                    resume,
                    update: command_update,
                    goto,
                    sends: Vec::new(),
                    parent: false,
                });
            }
        }
        let lifecycle: crate::_internal::lifecycle::LifecycleSink =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        config = config.with_lifecycle_sink(lifecycle.clone());
        let continue_thread =
            resume_start.is_none() && input.is_none_or(|value| value.is_none()) && thread.is_some();
        // Report a resume to `GraphCallbackHandler` before the run continues.
        if resume_start.as_ref().is_some_and(|(resume, ..)| resume.is_some()) || continue_thread {
            if let Some(thread_id) = thread.as_deref() {
                let snapshot = match config.checkpoint_ns.as_deref() {
                    Some(namespace) => self.graph.get_state_in(thread_id, namespace).ok().flatten(),
                    None => self.graph.get_state(thread_id).ok().flatten(),
                };
                if let Some(snapshot) = snapshot {
                    crate::adapters::python::graph_pause::record_resume(
                        config.lifecycle.as_ref(),
                        &snapshot.checkpoint_id,
                        config.checkpoint_ns.as_deref().unwrap_or(""),
                    );
                }
            }
        }
        let state = if resume_start.is_none() && !continue_thread {
            Some(state_from_py(input)?)
        } else {
            None
        };
        let graph = self.graph.clone();
        let stream_thread = thread.clone();
        let static_ns = config.checkpoint_ns.clone();
        let recorded = self.last_lifecycle.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = if let Some((resume, updates, goto, sends)) = resume_start {
                let thread = stream_thread.as_deref();
                match thread {
                    Some(thread) => {
                        let exists = futures::executor::block_on(graph.has_thread(thread)).unwrap_or(false);
                        match (resume, exists) {
                            (Some(value), _) if goto.is_empty() => {
                                graph
                                    .resume_with_updates_config(
                                        thread,
                                        value,
                                        updates.unwrap_or_default(),
                                        sends.clone(),
                                        &config,
                                    )
                                    .await
                            }
                            (Some(value), _) => {
                                graph
                                    .resume_with_updates_and_goto_config(
                                        thread,
                                        value,
                                        updates.unwrap_or_default(),
                                        goto,
                                        sends.clone(),
                                        &config,
                                    )
                                    .await
                            }
                            (None, true) => {
                                graph
                                    .resume_continue_with_goto_config(
                                        thread,
                                        updates.unwrap_or_default(),
                                        goto,
                                        sends.clone(),
                                        &config,
                                    )
                                    .await
                            }
                            (None, false) => {
                                let command = command.unwrap_or_default();
                                graph
                                    .run_command(command, &config)
                                    .await
                                    .map(|outcome| outcome.state)
                            }
                        }
                    }
                    None => {
                        let command = command.unwrap_or_default();
                        graph
                            .run_command(command, &config)
                            .await
                            .map(|outcome| outcome.state)
                    }
                }
            } else if continue_thread {
                let thread = stream_thread.as_deref().unwrap_or_default();
                graph.continue_thread_with_config(thread, &config).await
            } else {
                let state = state.expect("input state checked");
                graph
                    .run(&state, &config)
                    .await
                    .map(|outcome| outcome.state)
            };
            let outcome = match result {
                Ok(state) => Python::attach(|py| json_state_to_py(py, &state)),
                Err(error) => Python::attach(|py| {
                    match async_support::static_pause_values(
                        py,
                        &graph,
                        stream_thread.as_deref(),
                        static_ns.as_deref(),
                        &error,
                        config.lifecycle.as_ref(),
                    )? {
                        Some(value) => Ok(value),
                        None => Err(to_pyerr(py, error)),
                    }
                }),
            };
            if let Ok(mut events) = lifecycle.lock()
                && let Ok(mut target) = recorded.lock()
            {
                target.clear();
                target.append(&mut events);
            }
            outcome
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
                Err(error) => Python::attach(|py| {
                    match async_support::static_pause_values(
                        py,
                        &graph,
                        Some(thread_id.as_str()),
                        None,
                        &error,
                        None,
                    )? {
                        Some(value) => Ok(value),
                        None => Err(to_pyerr(py, error)),
                    }
                }),
            }
        })
    }

    /// Blocking invocation, matching LangGraph's single-class API.
    ///
    /// The async engine runs on the extension's Tokio runtime while the calling thread waits, so
    /// this works from ordinary Python code as well as from a worker thread.
    #[pyo3(signature = (input=None, config=None, *, thread_id=None, context=None, checkpoint_ns=None))]
    #[allow(clippy::too_many_arguments)]
    fn invoke(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        config: Option<&Bound<'_, PyAny>>,
        thread_id: Option<&Bound<'_, PyAny>>,
        context: Option<&Bound<'_, PyAny>>,
        checkpoint_ns: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let (thread, mut config) =
            super::convert::config_from_py(py, config.or(thread_id))?;
        if let Some(context) = context.filter(|value| !value.is_none()) {
            config = config.with_context(py_to_value(context)?);
        }
        if let Some(namespace) = checkpoint_ns {
            config.checkpoint_ns = Some(namespace);
        }
        let mut command: Option<crate::Command> = None;
        if let Some(obj) = input {
            if let Ok(py_command) = obj.cast::<PyCommand>().map(|cell| cell.borrow()) {
                let (resume, updates) = py_command.to_resume_command(py)?;
                let (goto, _sends) = py_command.goto_parts(py)?;
                // `Command(sends=...)` fan-out is not supported on the blocking path.
                command = Some(crate::Command {
                    resume,
                    update: Some(updates.into_iter().collect()),
                    goto,
                    sends: Vec::new(),
                    parent: false,
                });
            }
        }
        let continue_thread =
            command.is_none() && input.is_none_or(|value| value.is_none()) && thread.is_some();
        let state = if command.is_none() && !continue_thread {
            Some(state_from_py(input)?)
        } else {
            None
        };
        let graph = self.graph.clone();
        let error_thread = thread.clone();
        let error_ns = config.checkpoint_ns.clone();
        let result = py.detach(move || {
            pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
                match (command, state, continue_thread) {
                    (Some(command), _, _) => graph.run_command(command, &config).await.map(|outcome| outcome.state),
                    (None, Some(state), _) => graph.run(&state, &config).await.map(|outcome| outcome.state),
                    (None, None, true) => {
                        let thread = thread.clone().unwrap_or_default();
                        graph.continue_thread_with_config(&thread, &config).await
                    }
                    (None, None, false) => Err(crate::Error::validation("invoke requires an input")),
                }
            })
        });
        match result {
            Ok(state) => json_state_to_py(py, &state),
            Err(error) => match async_support::static_pause_values(
                py,
                &self.graph,
                error_thread.as_deref(),
                error_ns.as_deref(),
                &error,
                None,
            )? {
                Some(value) => Ok(value),
                None => Err(to_pyerr(py, error)),
            },
        }
    }

    /// The graph name recorded at compile time (used by the Python `Runnable` facade).
    #[pyo3(signature = ())]
    fn _graph_name(&self) -> Option<String> {
        self.graph.name().map(|value| value.to_string())
    }

    /// Renderable graph (LangGraph's `get_graph`).
    #[pyo3(signature = (config=None, *, xray=false))]
    fn get_graph(
        &self,
        config: Option<&Bound<'_, PyAny>>,
        xray: bool,
    ) -> crate::adapters::python::graph::graph_compiled::PyGraph {
        let _ = (config, xray);
        crate::adapters::python::graph::graph_compiled::PyGraph {
            mermaid: self.graph.to_mermaid(),
            ascii: self.graph.to_ascii(),
            spec: self.graph.graph_spec(),
        }
    }

    /// The builder this graph was compiled from (LangGraph's `builder`).
    #[getter]
    fn builder(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.builder_py.as_ref() {
            Some(value) => Ok(value.clone_ref(py)),
            None => Ok(py.None()),
        }
    }

    /// Default stream mode of this graph (`values`, as in LangGraph).
    #[getter]
    fn stream_mode(&self) -> &'static str {
        "values"
    }

    /// Chunks are streamed per node, never eagerly (LangGraph's default).
    #[getter]
    fn stream_eager(&self) -> bool {
        false
    }

    /// Graph-level config defaults: the extension has none.
    #[getter]
    fn config(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(pyo3::types::PyDict::new(py).into_any().unbind())
    }

    /// Transformers registered at compile time (drained by the facade's run stream).
    #[getter]
    fn stream_transformers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list = pyo3::types::PyList::empty(py);
        for transformer in &self.stream_transformers {
            list.append(transformer.bind(py))?;
        }
        Ok(list.into_any().unbind())
    }

    /// Marker: this native graph implements the async API itself (no shim delegation).
    #[getter]
    fn _native_async(&self) -> bool {
        true
    }

    /// The checkpointer passed to `compile()`, exactly as the caller supplied it.
    #[getter]
    fn checkpointer(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.checkpointer_py.as_ref() {
            Some(value) => Ok(value.clone_ref(py)),
            None => Ok(py.None()),
        }
    }

    /// Nodes reachable from each node through static edges.
    #[getter]
    fn trigger_to_nodes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = pyo3::types::PyDict::new(py);
        for (node, targets) in self.graph.static_triggers() {
            dict.set_item(node, targets)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Nodes paused before they run.
    #[getter]
    fn interrupt_before_nodes(&self) -> Vec<String> {
        let mut nodes: Vec<String> = self.graph.interrupt_before().iter().cloned().collect();
        nodes.sort();
        nodes
    }

    /// Nodes paused after they run.
    #[getter]
    fn interrupt_after_nodes(&self) -> Vec<String> {
        let mut nodes: Vec<String> = self.graph.interrupt_after().iter().cloned().collect();
        nodes.sort();
        nodes
    }

    /// Per-node retry policies.
    #[getter]
    fn retry_policy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = pyo3::types::PyDict::new(py);
        for (node, policy) in self.graph.retries() {
            let entry = pyo3::types::PyDict::new(py);
            entry.set_item("max_attempts", policy.max_attempts)?;
            entry.set_item("initial_interval", policy.initial_backoff.as_secs_f64())?;
            entry.set_item("backoff_factor", policy.multiplier)?;
            entry.set_item("max_interval", policy.max_backoff.as_secs_f64())?;
            entry.set_item("jitter", policy.jitter)?;
            entry.set_item("retry_on", policy.retry_on.clone())?;
            dict.set_item(node, entry)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Per-node run timeouts, in seconds.
    #[getter]
    fn step_timeout(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = pyo3::types::PyDict::new(py);
        for (node, policy) in self.graph.timeouts() {
            if let Some(timeout) = policy.run_timeout {
                dict.set_item(node, timeout.as_secs_f64())?;
            }
        }
        Ok(dict.into_any().unbind())
    }

    /// LangGraph's `debug` flag: the extension has no per-run debug switch.
    #[getter]
    fn debug(&self) -> bool {
        false
    }

    /// Drain the pause/resume transitions recorded by the last run.
    #[pyo3(signature = ())]
    fn take_lifecycle_events(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let events = match self.last_lifecycle.lock() {
            Ok(mut events) => std::mem::take(&mut *events),
            Err(_) => Vec::new(),
        };
        let list = pyo3::types::PyList::empty(py);
        for event in events {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("kind", event.kind.as_str())?;
            dict.set_item("status", event.status)?;
            dict.set_item("checkpoint_id", event.checkpoint_id)?;
            dict.set_item("namespace", event.namespace)?;
            let interrupts = pyo3::types::PyList::empty(py);
            for interrupt in &event.interrupts {
                let item = Py::new(
                    py,
                    crate::adapters::python::types::PyInterrupt {
                        id: interrupt.id.clone(),
                        value: value_to_py(py, &interrupt.value)?,
                        node: Some(interrupt.node.clone()),
                        resumable: true,
                        response_schema: match &interrupt.response_schema {
                            Some(schema) => Some(value_to_py(py, schema)?),
                            None => None,
                        },
                    },
                )?;
                interrupts.append(item)?;
            }
            dict.set_item("interrupts", interrupts)?;
            list.append(dict)?;
        }
        Ok(list.into_any().unbind())
    }

    /// Blocking streaming over the async engine.
    #[pyo3(signature = (input=None, config=None, *, thread_id=None, stream_mode=None, context=None, subgraphs=false, version=None))]
    #[allow(clippy::too_many_arguments)]
    fn stream(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        config: Option<&Bound<'_, PyAny>>,
        thread_id: Option<&Bound<'_, PyAny>>,
        stream_mode: Option<&Bound<'_, PyAny>>,
        context: Option<&Bound<'_, PyAny>>,
        subgraphs: bool,
        version: Option<String>,
    ) -> PyResult<Py<crate::adapters::python::types::PyStreamChunks>> {
        let v2 = match version.as_deref() {
            None | Some("v1") => false,
            Some("v2") => true,
            Some(other) => {
                return Err(PyTypeError::new_err(format!(
                    "stream version must be 'v1' or 'v2', got {other:?}"
                )));
            }
        };
        let (thread, mut config) = super::convert::config_from_py(py, config.or(thread_id))?;
        let _ = thread;
        if let Some(context) = context.filter(|value| !value.is_none()) {
            config = config.with_context(py_to_value(context)?);
        }
        if subgraphs {
            config = config.with_stream_subgraphs(true);
        }
        let requested = super::graph::stream_modes_from_py(stream_mode)?
            .unwrap_or_else(|| vec![crate::StreamMode::Values]);
        let config = config.with_stream_modes(requested.clone());
        let state = state_from_py(input)?;
        let graph = self.graph.clone();
        let modes = requested.clone();
        let result = py.detach(move || {
            pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
                graph.stream_modes(&state, &config, &modes).await
            })
        });
        let chunks = result.map_err(|error| to_pyerr(py, error))?;
        let converted = convert_stream_chunks(py, &chunks, v2, &requested)?;
        Py::new(py, crate::adapters::python::types::PyStreamChunks::new(converted))
    }

    /// Blocking batch over the async engine.
    #[pyo3(signature = (inputs, config=None, *, return_exceptions=false, context=None))]
    fn batch(
        &self,
        py: Python<'_>,
        inputs: &Bound<'_, PyAny>,
        config: Option<&Bound<'_, PyAny>>,
        return_exceptions: bool,
        context: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let list = inputs
            .cast::<PyList>()
            .map_err(|_| PyTypeError::new_err("batch expects a list of state dicts"))?;
        let states: Vec<JsonState> = list
            .iter()
            .map(|item| state_from_py(Some(&item)))
            .collect::<PyResult<Vec<_>>>()?;
        let (_, mut config) = super::convert::config_from_py(py, config)?;
        if let Some(context) = context.filter(|value| !value.is_none()) {
            config = config.with_context(py_to_value(context)?);
        }
        let graph = self.graph.clone();
        let results = py.detach(move || {
            pyo3_async_runtimes::tokio::get_runtime().block_on(async move { graph.batch(states, &config).await })
        });
        results
            .into_iter()
            .map(|result| match result {
                Ok(state) => json_state_to_py(py, &state),
                Err(error) if return_exceptions => Ok(to_pyerr(py, error).into_value(py).into_any()),
                Err(error) => Err(to_pyerr(py, error)),
            })
            .collect()
    }

    /// Awaitable batch invocation returning the final state of every input.
    #[pyo3(signature = (inputs, config=None, *, return_exceptions=false, context=None, thread_id=None))]
    #[allow(clippy::too_many_arguments)]
    fn abatch<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'_, PyAny>,
        config: Option<&Bound<'_, PyAny>>,
        return_exceptions: bool,
        context: Option<&Bound<'_, PyAny>>,
        thread_id: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let list = inputs
            .cast::<PyList>()
            .map_err(|_| PyTypeError::new_err("abatch expects a list of state dicts"))?;
        let states: Vec<JsonState> = list
            .iter()
            .map(|item| state_from_py(Some(&item)))
            .collect::<PyResult<Vec<_>>>()?;
        // LangGraph's `abatch` accepts one config or one per input.
        let configs = super::graph::graph_compiled::per_input_configs(py, config, states.len())?;
        let mut base = run_config(thread_id);
        if let Some(context) = context.filter(|value| !value.is_none()) {
            base = base.with_context(py_to_value(context)?);
        }
        let graph = self.graph.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let results: Vec<crate::Result<JsonState>> = match configs {
                Some(configs) => {
                    let mut out = Vec::with_capacity(configs.len());
                    for (state, config) in states.into_iter().zip(configs) {
                        out.push(graph.run(&state, &config).await.map(|outcome| outcome.state));
                    }
                    out
                }
                None => graph.batch(states, &base).await,
            };
            Python::attach(|py| {
                let output = PyList::empty(py);
                for result in results {
                    match result {
                        Ok(state) => output.append(json_state_to_py(py, &state)?)?,
                        Err(error) if return_exceptions => {
                            output.append(to_pyerr(py, error).into_value(py).into_any())?
                        }
                        Err(error) => return Err(to_pyerr(py, error)),
                    }
                }
                Ok(output.into_any().unbind())
            })
        })
    }


    #[pyo3(signature = (input=None, thread_id=None, context=None, stream_mode=None, subgraphs=false, version=None, live_sink=None, config=None))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn astream<'py>(
        &self,
        py: Python<'py>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<&Bound<'_, PyAny>>,
        context: Option<&Bound<'_, PyAny>>,
        stream_mode: Option<&Bound<'_, PyAny>>,
        subgraphs: bool,
        version: Option<String>,
        live_sink: Option<&Bound<'_, PyAny>>,
        config: Option<&Bound<'_, PyAny>>,
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
        let (thread, mut config) =
            super::convert::config_from_py(py, config.or(thread_id))?;

        if let Some(context) = context {
            config = config.with_context(py_to_value(context)?);
        }
        if subgraphs {
            config = config.with_stream_subgraphs(true);
        }
        let requested = super::graph::stream_modes_from_py(stream_mode)?
            .unwrap_or_else(|| vec![crate::StreamMode::Values]);
        let mut config = config.with_stream_modes(requested.clone());
        if let Some(callback) = live_sink.filter(|value| !value.is_none()) {
            // Node-produced chunks go straight to the caller instead of being buffered per node.
            let callback = callback.clone().unbind();
            let modes = requested.clone();
            let sink: crate::runtime::context::LiveSink =
                std::sync::Arc::new(std::sync::Mutex::new(
                    move |chunk: crate::_internal::stream::StreamChunk| {
                        Python::attach(|py| {
                            if let Ok(parts) = convert_stream_chunks(py, &[chunk], true, &modes)
                                && let Some(part) = parts.into_iter().next()
                            {
                                let _ = callback.bind(py).call1((part,));
                            }
                        });
                    },
                ));
            config = config.with_live_sink(sink);
        }
        let mut resume_start: Option<(
            Option<crate::Value>,
            Option<serde_json::Map<String, crate::Value>>,
            Vec<String>,
            Vec<crate::Send>,
        )> = None;
        if let Some(obj) = input {
            if let Ok(command) = obj.cast::<PyCommand>().map(|cell| cell.borrow()) {
                let (resume, updates) = command.to_resume_command(py)?;
                let (goto, sends) = command.goto_parts(py)?;
                resume_start = Some((resume, Some(updates.into_iter().collect()), goto, sends));
            }
        }
        let continue_thread =
            resume_start.is_none() && input.is_none_or(|value| value.is_none()) && thread.is_some();
        let state = if resume_start.is_none() && !continue_thread {
            Some(state_from_py(input)?)
        } else {
            None
        };
        if resume_start.is_some() && thread.is_none() {
            return Err(PyTypeError::new_err(
                "astream(Command(...)) requires thread_id",
            ));
        }
        let stream_thread = thread.clone();
        let graph = self.graph.clone();
        let future = pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut collected: Vec<StreamChunk> = Vec::new();
            let result = if let Some((resume, updates, goto, sends)) = resume_start {
                let thread = stream_thread.as_deref().expect("command thread checked");
                graph
                    .stream_thread_with(
                        thread,
                        resume,
                        updates,
                        goto,
                        sends,
                        &config,
                        &requested,
                        |chunk| collected.push(chunk),
                    )
                    .await
            } else if continue_thread {
                let thread = stream_thread.as_deref().expect("continue thread checked");
                graph
                    .stream_thread_with(
                        thread,
                        None,
                        None,
                        Vec::new(),
                        Vec::new(),
                        &config,
                        &requested,
                        |chunk| collected.push(chunk),
                    )
                    .await
            } else {
                let state = state.expect("input state checked");
                graph
                    .stream_modes_with(&state, &config, &requested, |chunk| collected.push(chunk))
                    .await
            };
            if let Err(error) = result {
                if !matches!(error, crate::Error::Interrupt(_)) {
                    return Python::attach(|py| Err(to_pyerr(py, error)));
                }
            }
            collected
                .retain(|chunk| chunk.is_interrupt() || chunk.matches_modes_strict(&requested));
            Python::attach(|py| {
                let converted = convert_stream_chunks(py, &collected, v2, &requested)?;
                let list = PyList::empty(py);
                for item in converted {
                    list.append(item)?;
                }
                Ok(list.into_any().unbind())
            })
        })?;
        // Return the awaitable directly: the Python facade wraps it in `AsyncStream`.
        Ok(future)
    }

    #[pyo3(signature = (thread_id, checkpoint_ns=None, subgraphs=false))]
    fn get_state(
        &self,
        py: Python<'_>,
        thread_id: &Bound<'_, PyAny>,
        checkpoint_ns: Option<String>,
        subgraphs: bool,
    ) -> PyResult<Option<Py<PyAny>>> {
        let (thread_id, parsed_config) = super::convert::config_from_py(py, Some(thread_id))?;
        let thread_id =
            thread_id.ok_or_else(|| PyTypeError::new_err("get_state requires a thread id"))?;
        let checkpoint_ns = checkpoint_ns.or_else(|| parsed_config.checkpoint_ns.clone());
        if subgraphs {
            let states = self
                .graph
                .get_state_subgraphs(&thread_id)
                .map_err(|error| to_pyerr(py, error))?;
            return match states {
                Some((root, children)) => embed_subgraph_states(py, &root, &children).map(Some),
                None => Ok(None),
            };
        }
        let snapshot = match (
            checkpoint_ns.as_deref(),
            parsed_config.checkpoint_id.as_deref(),
        ) {
            (Some(namespace), Some(id)) => self.graph.get_state_in_at(&thread_id, namespace, id),
            (None, Some(id)) => self.graph.get_state_at(&thread_id, id),
            (Some(namespace), None) => self.graph.get_state_in(&thread_id, namespace),
            (None, None) => self.graph.get_state(&thread_id),
        }
        .map_err(|error| to_pyerr(py, error))?;
        match snapshot {
            Some(snapshot) => snapshot_to_py(py, &snapshot).map(Some),
            None => empty_snapshot_to_py(py, Some(&thread_id)).map(Some),
        }
    }

    fn has_checkpointer(&self) -> bool {
        self.saver.is_some()
    }

    /// Awaitable `get_state` for async callers.
    #[pyo3(signature = (thread_id, checkpoint_ns=None, subgraphs=false))]
    fn aget_state<'py>(
        &self,
        py: Python<'py>,
        thread_id: &Bound<'_, PyAny>,
        checkpoint_ns: Option<String>,
        subgraphs: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (thread_id, parsed_config) = super::convert::config_from_py(py, Some(thread_id))?;
        let thread_id =
            thread_id.ok_or_else(|| PyTypeError::new_err("aget_state requires a thread id"))?;
        let checkpoint_ns = checkpoint_ns.or_else(|| parsed_config.checkpoint_ns.clone());
        let checkpoint_id = parsed_config.checkpoint_id.clone();
        let graph = self.graph.clone();
        immediate_awaitable(py, move || {
            Python::attach(|py| {
                if subgraphs {
                    let states = graph
                        .get_state_subgraphs(&thread_id)
                        .map_err(|error| to_pyerr(py, error))?;
                    return match states {
                        Some((root, children)) => {
                            embed_subgraph_states(py, &root, &children).map(Some)
                        }
                        None => Ok(None),
                    };
                }
                let snapshot = match (checkpoint_ns.as_deref(), checkpoint_id.as_deref()) {
                    (Some(namespace), Some(id)) => graph.get_state_in_at(&thread_id, namespace, id),
                    (None, Some(id)) => graph.get_state_at(&thread_id, id),
                    (Some(namespace), None) => graph.get_state_in(&thread_id, namespace),
                    (None, None) => graph.get_state(&thread_id),
                }
                .map_err(|error| to_pyerr(py, error))?;
                let Some(snapshot) = snapshot else {
                    return empty_snapshot_to_py(py, Some(&thread_id)).map(Some);
                };
                snapshot_to_py(py, &snapshot).map(Some)
            })
        })
    }

    /// Awaitable `get_state_history` for async callers.
    #[pyo3(signature = (thread_id, checkpoint_ns=None, before=None, limit=None, filter=None))]
    fn aget_state_history<'py>(
        &self,
        py: Python<'py>,
        thread_id: &Bound<'_, PyAny>,
        checkpoint_ns: Option<String>,
        before: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
        filter: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (thread_id, parsed_config) = super::convert::config_from_py(py, Some(thread_id))?;
        let before: Option<String> = match before.filter(|value| !value.is_none()) {
            Some(value) => match value.extract::<String>() {
                Ok(id) => Some(id),
                Err(_) => {
                    let (_, config) = super::convert::config_from_py(py, Some(value))?;
                    config.checkpoint_id
                }
            },
            None => None,
        };
        let filter = match filter {
            Some(filter) => {
                let value = super::convert::py_to_value(&filter)?;
                match value {
                    serde_json::Value::Object(map) => Some(map),
                    _ => {
                        return Err(PyTypeError::new_err(
                            "aget_state_history filter must be a dict",
                        ));
                    }
                }
            }
            None => None,
        };
        let thread_id = thread_id
            .ok_or_else(|| PyTypeError::new_err("aget_state_history requires a thread id"))?;
        let checkpoint_ns = checkpoint_ns.or_else(|| parsed_config.checkpoint_ns.clone());
        let graph = self.graph.clone();
        immediate_awaitable(py, move || {
            Python::attach(|py| {
                let namespace = checkpoint_ns.clone().unwrap_or_default();
                graph
                    .history_handles(&thread_id, &namespace, None, Some(0))
                    .map_err(|error| to_pyerr(py, error))?;
                // A metadata filter cannot be answered from checkpoint handles, so those
                // queries stay eager (the same split as the synchronous reader).
                if let Some(filter) = filter {
                    let snapshots = graph
                        .get_state_history_where_in(
                            &thread_id,
                            &namespace,
                            Some(&filter),
                            before.as_deref(),
                            limit,
                        )
                        .map_err(|error| to_pyerr(py, error))?;
                    let items = snapshots
                        .iter()
                        .map(|snapshot| snapshot_to_py(py, snapshot))
                        .collect::<PyResult<Vec<_>>>()?;
                    let history = crate::adapters::python::types::PyStateHistory::from_items(items);
                    return Py::new(py, history)
                        .map(|history| history.into_any())
                        .map(Some);
                }
                let source: std::sync::Arc<
                    dyn crate::adapters::python::types::history::HistorySource,
                > = graph.clone();
                let history = crate::adapters::python::types::PyStateHistory::new(
                    source, thread_id, namespace, before, limit,
                );
                Py::new(py, history)
                    .map(|history| history.into_any())
                    .map(Some)
            })
        })
    }

    /// Awaitably update state without running nodes.
    #[pyo3(signature = (thread_id, values, as_node=None, task_id=None, checkpoint_ns=None))]
    fn aupdate_state<'py>(
        &self,
        py: Python<'py>,
        thread_id: String,
        values: &Bound<'_, PyDict>,
        as_node: Option<String>,
        task_id: Option<String>,
        checkpoint_ns: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut updates: Vec<(String, serde_json::Value)> = Vec::with_capacity(values.len());
        for (key, value) in values.iter() {
            updates.push((key.extract::<String>()?, py_to_value(&value)?));
        }
        let graph = self.graph.clone();
        let in_memory = self
            .saver
            .as_ref()
            .is_some_and(|saver| saver.is_in_memory());
        let work = async move {
            let result = match checkpoint_ns.as_deref() {
                Some(namespace) => {
                    graph
                        .update_state_in(&thread_id, namespace, updates, as_node.as_deref())
                        .await
                }
                None => {
                    graph
                        .update_state_with_task(
                            &thread_id,
                            updates,
                            as_node.as_deref(),
                            task_id.as_deref(),
                        )
                        .await
                }
            };
            match result {
                Ok(snapshot) => Python::attach(|py| snapshot_to_py(py, &snapshot).map(Some)),
                Err(error) => Python::attach(|py| Err(to_pyerr(py, error))),
            }
        };
        // An in-memory saver cannot block, so the update completes synchronously; running it
        // here avoids a Tokio task and an asyncio wakeup per call (~0.3 ms).
        if in_memory {
            let result = futures::executor::block_on(work);
            return immediate_awaitable(py, || result);
        }
        pyo3_async_runtimes::tokio::future_into_py(py, work)
    }

    /// Awaitably resume from a specific checkpoint id.
    #[pyo3(signature = (thread_id, checkpoint_id, value, checkpoint_ns=None))]
    fn aresume_from<'py>(
        &self,
        py: Python<'py>,
        thread_id: String,
        checkpoint_id: String,
        value: &Bound<'_, PyAny>,
        checkpoint_ns: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let value = py_to_value(value)?;
        let graph = self.graph.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = match checkpoint_ns.as_deref() {
                Some(namespace) => {
                    graph
                        .resume_from_in(&thread_id, namespace, &checkpoint_id, value)
                        .await
                }
                None => graph.resume_from(&thread_id, &checkpoint_id, value).await,
            };
            match result {
                Ok(state) => Python::attach(|py| json_state_to_py(py, &state)),
                Err(error) => Python::attach(|py| {
                    match async_support::static_pause_values(
                        py,
                        &graph,
                        Some(thread_id.as_str()),
                        checkpoint_ns.as_deref(),
                        &error,
                        None,
                    )? {
                        Some(value) => Ok(value),
                        None => Err(to_pyerr(py, error)),
                    }
                }),
            }
        })
    }

    /// Export the latest checkpoint as a LangGraph-shaped JSON document.
    #[pyo3(signature = (thread_id))]
    fn export_checkpoint(&self, py: Python<'_>, thread_id: String) -> PyResult<Option<Py<PyAny>>> {
        let document = self
            .graph
            .export_checkpoint(&thread_id)
            .map_err(|error| to_pyerr(py, error))?;
        document
            .map(|document| value_to_py(py, &document))
            .transpose()
    }

    /// Import a LangGraph-shaped checkpoint document into a thread.
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

    /// Read the full checkpoint history of a thread, newest first.
    #[pyo3(signature = (thread_id, checkpoint_ns=None, before=None, limit=None))]
    fn get_state_history(
        &self,
        py: Python<'_>,
        thread_id: &Bound<'_, PyAny>,
        checkpoint_ns: Option<String>,
        before: Option<String>,
        limit: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let (thread_id, parsed_config) = super::convert::config_from_py(py, Some(thread_id))?;
        let thread_id = thread_id
            .ok_or_else(|| PyTypeError::new_err("get_state_history requires a thread id"))?;
        let checkpoint_ns = checkpoint_ns.or_else(|| parsed_config.checkpoint_ns.clone());
        let namespace = checkpoint_ns.clone().unwrap_or_default();
        self.graph
            .history_handles(&thread_id, &namespace, None, Some(0))
            .map_err(|error| to_pyerr(py, error))?;
        let source: std::sync::Arc<dyn crate::adapters::python::types::history::HistorySource> =
            self.graph.clone();
        let history = crate::adapters::python::types::PyStateHistory::new(
            source, thread_id, namespace, before, limit,
        );
        Py::new(py, history).map(|history| history.into_any())
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

#[path = "async_builder.rs"]
pub(crate) mod async_builder;
pub(crate) use async_builder::*;
