//! Async Python bindings (`python-async` feature): `AsyncStateGraph` / `AsyncCompiledGraph`.

use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use super::convert::install_langchain_callbacks;
use super::convert::*;
use super::types::{PyCommand, PyTimeoutPolicy, checkpointer_from_py, store_from_py};
use crate::BlockingSaver;
use crate::JsonState;
use crate::kernel::node::NodeOutput;
use crate::kernel::stream::StreamChunk;
use crate::tokio_runtime::{AsyncCompiledGraph, AsyncContext};
use crate::*;

fn wrap_async_timeout(py: Python<'_>, function: Py<PyAny>, seconds: f64) -> PyResult<Py<PyAny>> {
    let locals = pyo3::types::PyDict::new(py);
    locals.set_item("fn", function.bind(py))?;
    locals.set_item("seconds", seconds.max(0.0))?;
    py.run(
        c"async def _timeout_wrapper(state, fn=fn, seconds=seconds):
    import asyncio
    return await asyncio.wait_for(fn(state), seconds)
wrapper = _timeout_wrapper",
        None,
        Some(&locals),
    )?;
    locals
        .get_item("wrapper")?
        .map(|value| value.unbind())
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("timeout wrapper missing"))
}

fn global_step_for(
    history: &[crate::StateSnapshot<crate::JsonState>],
    checkpoint_id: &str,
) -> Option<i64> {
    let total = history.len() as i64;
    let index = history
        .iter()
        .position(|snapshot| snapshot.checkpoint_id == checkpoint_id)? as i64;
    Some(total - 2 - index)
}

fn snapshot_to_py_with_step(
    py: Python<'_>,
    snapshot: &crate::StateSnapshot<crate::JsonState>,
    step: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let value = snapshot_to_py(py, snapshot)?;
    let Some(step) = step else {
        return Ok(value);
    };
    if let Ok(snapshot) = value
        .bind(py)
        .extract::<pyo3::PyRef<'_, crate::adapters::python::types::PyStateSnapshot>>()
    {
        let dict = snapshot.dict(py);
        dict.set_item("step", step)?;
        if let Some(metadata) = dict.get_item("metadata")? {
            if let Ok(metadata) = metadata.cast::<pyo3::types::PyDict>() {
                metadata.set_item("step", step)?;
            }
        }
        return Ok(value);
    }
    let dict = value.bind(py).cast::<pyo3::types::PyDict>()?;
    dict.set_item("step", step)?;
    if let Some(metadata) = dict.get_item("metadata")? {
        if let Ok(metadata) = metadata.cast::<pyo3::types::PyDict>() {
            metadata.set_item("step", step)?;
        }
    }
    Ok(value)
}

fn run_timed_awaitable<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // Timed nodes reuse a per-thread asyncio loop so the wait_for overhead is not
    // paid on every attempt; ordinary nodes keep the cheaper asyncio.run path.
    let asyncio = py.import("asyncio")?;
    let loop_object = match asyncio
        .getattr("get_event_loop")
        .and_then(|get| get.call0())
    {
        Ok(loop_object) => loop_object,
        Err(_) => {
            let loop_object = asyncio.getattr("new_event_loop")?.call0()?;
            asyncio.getattr("set_event_loop")?.call1((&loop_object,))?;
            loop_object
        }
    };
    loop_object.call_method1("run_until_complete", (value,))
}

/// Compiled async graph (Python).
#[pyclass(name = "AsyncCompiledGraph")]
pub struct PyAsyncCompiledGraph {
    pub(crate) graph: Arc<AsyncCompiledGraph<JsonState>>,
    saver: Option<Arc<dyn Checkpointer>>,
}

fn embed_subgraph_states(
    py: Python<'_>,
    root: &crate::StateSnapshot<JsonState>,
    subgraphs: &[crate::StateSnapshot<JsonState>],
) -> PyResult<Py<PyAny>> {
    let root_py = snapshot_to_py(py, root)?;
    if let Ok(root_snapshot) = root_py
        .bind(py)
        .extract::<PyRef<crate::adapters::python::types::PyStateSnapshot>>()
        && let Ok(Some(tasks)) = root_snapshot.dict(py).get_item("tasks")
        && let Ok(tasks) = tasks.cast::<PyList>()
    {
        for task in tasks {
            let Ok(task_dict) = task.cast::<PyDict>() else {
                continue;
            };
            let Ok(Some(node)) = task_dict.get_item("node") else {
                continue;
            };
            let Ok(node) = node.extract::<String>() else {
                continue;
            };
            let child = subgraphs.iter().find(|snapshot| {
                snapshot
                    .checkpoint_ns
                    .split('|')
                    .next_back()
                    .is_some_and(|segment| {
                        segment == node
                            || segment
                                .strip_prefix(&node)
                                .is_some_and(|rest| rest.starts_with(':'))
                    })
            });
            if let Some(child) = child {
                task_dict.set_item("state", snapshot_to_py(py, child)?)?;
            }
        }
    }
    Ok(root_py)
}

#[pymethods]
impl PyAsyncCompiledGraph {
    /// Awaitable graph invocation returning the final state dict.
    #[pyo3(signature = (input=None, thread_id=None, context=None))]
    #[allow(clippy::too_many_arguments)]
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
        )> = None;
        let mut command: Option<crate::Command> = None;
        if let Some(obj) = input {
            if let Ok(py_command) = obj.extract::<PyRef<PyCommand>>() {
                let (resume, updates) = py_command.to_resume_command(py)?;
                let (goto, sends) = py_command.goto_parts(py)?;
                if !sends.is_empty() {
                    return Err(PyTypeError::new_err(
                        "ainvoke Command(goto=Send) is not supported yet",
                    ));
                }
                let command_update = Some(updates.clone());
                resume_start = Some((
                    resume.clone(),
                    Some(updates.into_iter().collect()),
                    goto.clone(),
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = if let Some((resume, updates, goto)) = resume_start {
                let thread = stream_thread.as_deref();
                match thread {
                    Some(thread) => {
                        let exists = graph.get_state(thread).ok().flatten().is_some();
                        match (resume, exists) {
                            (Some(value), _) if goto.is_empty() => {
                                graph
                                    .resume_with_updates_config(
                                        thread,
                                        value,
                                        updates.unwrap_or_default(),
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
        let mut resume_start: Option<(
            Option<crate::Value>,
            Option<serde_json::Map<String, crate::Value>>,
            Vec<String>,
        )> = None;
        if let Some(obj) = input {
            if let Ok(command) = obj.extract::<PyRef<PyCommand>>() {
                let (resume, updates) = command.to_resume_command(py)?;
                let goto = command.goto_targets(py)?;
                resume_start = Some((resume, Some(updates.into_iter().collect()), goto));
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
            let result = if let Some((resume, updates, goto)) = resume_start {
                let thread = stream_thread.as_deref().expect("command thread checked");
                graph
                    .stream_thread_with(
                        thread,
                        resume,
                        updates,
                        goto,
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
            collected.retain(|chunk| {
                matches!(chunk, StreamChunk::Interrupt { .. })
                    || chunk.matches_modes_strict(&requested)
            });
            Python::attach(|py| {
                let list = PyList::empty(py);
                let multi_mode = requested.len() > 1;
                for chunk in &collected {
                    if v2 {
                        for item in chunk_to_py_v2_parts(py, chunk)? {
                            list.append(item)?;
                        }
                    } else {
                        list.append(chunk_to_py_v1(py, chunk, multi_mode)?)?;
                    }
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
            Some(snapshot) => {
                let history = self
                    .graph
                    .get_state_history(&thread_id)
                    .map_err(|error| to_pyerr(py, error))?;
                let global_step = global_step_for(&history, &snapshot.checkpoint_id);
                snapshot_to_py_with_step(py, &snapshot, global_step).map(Some)
            }
            None => Ok(None),
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
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
                    return Ok(None);
                };
                let history = graph
                    .get_state_history(&thread_id)
                    .map_err(|error| to_pyerr(py, error))?;
                let global_step = global_step_for(&history, &snapshot.checkpoint_id);
                snapshot_to_py_with_step(py, &snapshot, global_step).map(Some)
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Python::attach(|py| {
                let snapshots = match checkpoint_ns.as_deref() {
                    Some(namespace) => graph.get_state_history_filtered_in(
                        &thread_id,
                        namespace,
                        before.as_deref(),
                        limit,
                    ),
                    None => graph.get_state_history_filtered(&thread_id, before.as_deref(), limit),
                }
                .map_err(|error| to_pyerr(py, error))?;
                let full = graph
                    .get_state_history(&thread_id)
                    .map_err(|error| to_pyerr(py, error))?;
                let list = PyList::empty(py);
                for snapshot in snapshots {
                    let global_step = global_step_for(&full, &snapshot.checkpoint_id);
                    list.append(snapshot_to_py_with_step(py, &snapshot, global_step)?)?;
                }
                Ok(list.into_any().unbind())
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
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
                Ok(snapshot) => Python::attach(|py| snapshot_to_py(py, &snapshot)),
                Err(error) => Python::attach(|py| Err(to_pyerr(py, error))),
            }
        })
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
                Err(error) => Python::attach(|py| Err(to_pyerr(py, error))),
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
    ) -> PyResult<Vec<Py<PyAny>>> {
        let (thread_id, parsed_config) = super::convert::config_from_py(py, Some(thread_id))?;
        let thread_id = thread_id
            .ok_or_else(|| PyTypeError::new_err("get_state_history requires a thread id"))?;
        let checkpoint_ns = checkpoint_ns.or_else(|| parsed_config.checkpoint_ns.clone());
        let snapshots = match checkpoint_ns.as_deref() {
            Some(namespace) => self
                .graph
                .get_state_history_filtered_in(&thread_id, namespace, before.as_deref(), limit)
                .map_err(|error| to_pyerr(py, error))?,
            None => self
                .graph
                .get_state_history_filtered(&thread_id, before.as_deref(), limit)
                .map_err(|error| to_pyerr(py, error))?,
        };
        let full_history = self
            .graph
            .get_state_history(&thread_id)
            .map_err(|error| to_pyerr(py, error))?;
        snapshots
            .iter()
            .map(|snapshot| {
                let global_step = global_step_for(&full_history, &snapshot.checkpoint_id);
                snapshot_to_py_with_step(py, snapshot, global_step)
            })
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

#[path = "async_builder.rs"]
mod async_builder;
pub(crate) use async_builder::*;
