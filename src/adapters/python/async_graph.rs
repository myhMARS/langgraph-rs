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
}

#[path = "async_support.rs"]
mod async_support;
use async_support::*;
#[pymethods]
impl PyAsyncCompiledGraph {
    /// Awaitable graph invocation returning the final state dict.
    #[pyo3(signature = (input=None, thread_id=None, context=None))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn ainvoke<'py>(
        &self,
        py: Python<'py>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<&Bound<'_, PyAny>>,
        context: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (thread, mut config) = super::convert::config_from_py(py, thread_id)?;
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
        let continue_thread =
            resume_start.is_none() && input.is_none_or(|value| value.is_none()) && thread.is_some();
        let state = if resume_start.is_none() && !continue_thread {
            Some(state_from_py(input)?)
        } else {
            None
        };
        let graph = self.graph.clone();
        let stream_thread = thread.clone();
        let static_ns = config.checkpoint_ns.clone();
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
            match result {
                Ok(state) => Python::attach(|py| json_state_to_py(py, &state)),
                Err(error) => Python::attach(|py| {
                    match async_support::static_pause_values(
                        py,
                        &graph,
                        stream_thread.as_deref(),
                        static_ns.as_deref(),
                        &error,
                    )? {
                        Some(value) => Ok(value),
                        None => Err(to_pyerr(py, error)),
                    }
                }),
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
                Err(error) => Python::attach(|py| {
                    match async_support::static_pause_values(
                        py,
                        &graph,
                        Some(thread_id.as_str()),
                        None,
                        &error,
                    )? {
                        Some(value) => Ok(value),
                        None => Err(to_pyerr(py, error)),
                    }
                }),
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
    /// `astream_events(version="v3")`: an awaitable resolving to an `AsyncGraphRunStream`.
    #[pyo3(signature = (input=None, thread_id=None, version="v3", transformers=None, context=None))]
    fn astream_events<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        input: Option<&Bound<'py, PyAny>>,
        thread_id: Option<&Bound<'py, PyAny>>,
        version: &str,
        transformers: Option<&Bound<'py, PyAny>>,
        context: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let module = py.import("langgraph_rs.stream._events")?;
        let function = module.getattr("astream_events")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("version", version)?;
        if !slf.stream_transformers.is_empty() {
            let compiled = pyo3::types::PyList::new(
                py,
                slf.stream_transformers.iter().map(|item| item.bind(py)),
            )?;
            kwargs.set_item("compile_transformers", compiled)?;
        }
        if let Some(thread_id) = thread_id {
            kwargs.set_item("thread_id", thread_id)?;
        }
        if let Some(transformers) = transformers {
            kwargs.set_item("transformers", transformers)?;
        }
        if let Some(context) = context.filter(|value| !value.is_none()) {
            kwargs.set_item("context", context)?;
        }
        let graph = slf.into_pyobject(py)?.into_any();
        function.call((graph, input), Some(&kwargs))
    }
    #[pyo3(signature = (input=None, thread_id=None, context=None, stream_mode=None, subgraphs=false, version=None))]
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
        let (thread, mut config) = super::convert::config_from_py(py, thread_id)?;
        if let Some(context) = context {
            config = config.with_context(py_to_value(context)?);
        }
        if subgraphs {
            config = config.with_stream_subgraphs(true);
        }
        let requested = super::graph::stream_modes_from_py(stream_mode)?
            .unwrap_or_else(|| vec![crate::StreamMode::Values]);
        let config = config.with_stream_modes(requested.clone());
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
        let module = py.import("langgraph_rs")?;
        let stream = module.getattr("AsyncStream")?.call1((future,))?;
        Ok(stream)
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
    #[pyo3(signature = (thread_id, checkpoint_ns=None, before=None, limit=None))]
    fn aget_state_history<'py>(
        &self,
        py: Python<'py>,
        thread_id: &Bound<'_, PyAny>,
        checkpoint_ns: Option<String>,
        before: Option<String>,
        limit: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (thread_id, parsed_config) = super::convert::config_from_py(py, Some(thread_id))?;
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
mod async_builder;
pub(crate) use async_builder::*;
