//! Compiled Python graph and graph-description classes.

use std::sync::Arc;

#[cfg(feature = "msgpack")]
use pyo3::exceptions::PyLookupError;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};

use super::*;

#[pyclass(name = "CompiledGraph")]
pub struct PyCompiledGraph {
    pub(crate) graph: Arc<CompiledGraph<JsonState>>,
    pub(crate) saver: Option<Arc<dyn Checkpointer>>,
}

impl PyCompiledGraph {
    pub(crate) fn from_inner_with_saver(
        graph: CompiledGraph<JsonState>,
        saver: Option<Arc<dyn Checkpointer>>,
    ) -> Self {
        Self {
            graph: Arc::new(graph),
            saver,
        }
    }
}

#[pymethods]
impl PyCompiledGraph {
    /// Run the graph and return the final state dict.
    #[pyo3(signature = (input=None, thread_id=None, context=None, control=None, checkpoint_ns=None, config=None))]
    #[allow(clippy::too_many_arguments)]
    fn invoke(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<&Bound<'_, PyAny>>,
        context: Option<&Bound<'_, PyAny>>,
        control: Option<PyRef<'_, crate::adapters::python::types::PyRunControl>>,
        checkpoint_ns: Option<String>,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let (thread_id, base_config) =
            crate::adapters::python::convert::config_from_py(py, config.or(thread_id))?;
        let mut invoke_config = base_config.clone();
        if let Some(namespace) = checkpoint_ns.clone() {
            invoke_config = invoke_config.with_checkpoint_ns(namespace);
        }
        if let Some(control) = control.as_ref() {
            invoke_config = invoke_config.with_control(control.inner.clone());
        }
        if let Some(context) = context {
            invoke_config = invoke_config.with_context(py_to_value(context)?);
        }
        if let Some(obj) = input {
            if let Ok(command) = obj.extract::<PyRef<PyCommand>>() {
                let (targets, sends) = command.goto_parts(py)?;
                if (!targets.is_empty() || !sends.is_empty()) && command.resume.is_none() {
                    let updates = command.to_update_map(py)?;
                    let config = invoke_config.clone();
                    let thread = thread_id.clone();
                    let command = crate::Command {
                        resume: None,
                        update: Some(updates.clone()),
                        goto: targets.clone(),
                        sends: sends.clone(),
                        parent: false,
                    };
                    let exists = thread
                        .as_deref()
                        .map(|thread| {
                            match config.checkpoint_ns.as_deref() {
                                Some(namespace) => self.graph.get_state_in(thread, namespace),
                                None => self.graph.get_state(thread),
                            }
                            .ok()
                            .flatten()
                            .is_some()
                        })
                        .unwrap_or(false);
                    let result = py.detach(|| {
                        if let (true, Some(thread)) =
                            (exists && sends.is_empty(), thread.as_deref())
                        {
                            self.graph.resume_continue_with_goto_config(
                                thread,
                                updates.clone(),
                                targets.clone(),
                                Vec::new(),
                                &config,
                            )
                        } else {
                            self.graph
                                .run_command(command, &config)
                                .map(|outcome| outcome.state)
                        }
                    });
                    return match result {
                        Ok(state) => json_state_to_py(py, &state),
                        Err(error) => Err(to_pyerr(py, error)),
                    };
                }
                if thread_id.is_none() && command.resume.is_none() {
                    let updates = command.to_update_map(py)?;
                    if updates.is_empty() {
                        return Err(PyTypeError::new_err(
                            "Command requires either `goto` or `update`",
                        ));
                    }
                    let updates = updates.into_iter().collect();
                    let command = crate::Command {
                        resume: None,
                        update: Some(updates),
                        goto: Vec::new(),
                        sends: Vec::new(),
                        parent: false,
                    };
                    let config = invoke_config.clone();
                    let result = py.detach(|| self.graph.run_command(command, &config));
                    return match result {
                        Ok(outcome) => json_state_to_py(py, &outcome.state),
                        Err(error) => Err(to_pyerr(py, error)),
                    };
                }
                let thread = thread_id.as_deref().ok_or_else(|| {
                    PyTypeError::new_err("invoke(Command(...)) requires thread_id")
                })?;
                let (resume, updates) = command.to_resume_command(py)?;
                let (goto, sends) = command.goto_parts(py)?;
                let resume_config = invoke_config.clone();
                let result = py.detach(|| match resume {
                    Some(value) if goto.is_empty() && sends.is_empty() => {
                        self.graph.resume_with_updates_config(
                            thread,
                            value,
                            updates,
                            Vec::new(),
                            &resume_config,
                        )
                    }
                    Some(value) => self.graph.resume_with_updates_and_goto_config(
                        thread,
                        value,
                        updates,
                        goto,
                        sends,
                        &resume_config,
                    ),
                    None => self.graph.resume_continue_with_goto_config(
                        thread,
                        updates,
                        goto,
                        sends,
                        &resume_config,
                    ),
                });
                return match result {
                    Ok(state) => json_state_to_py(py, &state),
                    Err(error) => Err(to_pyerr(py, error)),
                };
            }
        }
        let config = invoke_config;
        let input_is_none = input.map(|value| value.is_none()).unwrap_or(true);
        if input_is_none {
            if let Some(thread) = thread_id.as_deref() {
                let result = py.detach(|| self.graph.continue_thread_with_config(thread, &config));
                return match result {
                    Ok(state) => json_state_to_py(py, &state),
                    Err(error) => Err(to_pyerr(py, error)),
                };
            }
        }
        if let (Some(obj), Some(thread)) = (input, thread_id.as_deref()) {
            let exists = match config.checkpoint_ns.as_deref() {
                Some(namespace) => self.graph.get_state_in(thread, namespace),
                None => self.graph.get_state(thread),
            }
            .ok()
            .flatten()
            .is_some();
            if exists {
                let updates = py_to_value(obj)?;
                if let Some(map) = updates.as_object() {
                    let updates: Vec<(String, crate::Value)> = map
                        .iter()
                        .map(|(channel, value)| (channel.clone(), value.clone()))
                        .collect();
                    let result = py.detach(|| self.graph.run_on_thread(thread, updates, &config));
                    return match result {
                        Ok(outcome) => json_state_to_py(py, &outcome.state),
                        Err(error) => Err(to_pyerr(py, error)),
                    };
                }
            }
        }
        let state = state_from_py(input)?;
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
    #[pyo3(signature = (input=None, thread_id=None, stream_mode=None, context=None, subgraphs=false, version=None, control=None, config=None))]
    #[allow(clippy::too_many_arguments)]
    fn stream(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<&Bound<'_, PyAny>>,
        stream_mode: Option<&Bound<'_, PyAny>>,
        context: Option<&Bound<'_, PyAny>>,
        subgraphs: bool,
        version: Option<String>,
        control: Option<PyRef<'_, crate::adapters::python::types::PyRunControl>>,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let v2 = match version.as_deref() {
            None | Some("v1") => false,
            Some("v2") => true,
            Some(other) => {
                return Err(PyTypeError::new_err(format!(
                    "stream version must be 'v1' or 'v2', got {other:?}"
                )));
            }
        };
        let (thread, mut config) = config_from_py(py, config.or(thread_id))?;
        if let Some(control) = control {
            config = config.with_control(control.inner.clone());
        }
        if let Some(context) = context {
            config = config.with_context(py_to_value(context)?);
        }
        if subgraphs {
            config = config.with_stream_subgraphs(true);
        }
        let requested =
            stream_modes_from_py(stream_mode)?.unwrap_or_else(|| vec![crate::StreamMode::Values]);
        let config = config.clone().with_stream_modes(requested.clone());
        let mut streamed = Vec::new();
        let mut handled = false;
        if let Some(obj) = input {
            if let Ok(command) = obj.extract::<PyRef<PyCommand>>() {
                let thread = thread.as_deref().ok_or_else(|| {
                    PyTypeError::new_err("stream(Command(...)) requires thread_id")
                })?;
                let (resume, updates) = command.to_resume_command(py)?;
                let (goto, sends) = command.goto_parts(py)?;
                let updates = Some(updates.into_iter().collect());
                match py.detach(|| {
                    self.graph.stream_thread_with(
                        thread,
                        resume,
                        updates,
                        goto,
                        sends,
                        &config,
                        &requested,
                        |chunk| streamed.push(chunk),
                    )
                }) {
                    Ok(_) | Err(crate::Error::Interrupt(_)) => {}
                    Err(error) => return Err(to_pyerr(py, error)),
                }
                handled = true;
            }
        }
        if !handled && input.is_none_or(|value| value.is_none()) {
            if let Some(thread) = thread.as_deref() {
                match py.detach(|| {
                    self.graph.stream_thread_with(
                        thread,
                        None,
                        None,
                        Vec::new(),
                        Vec::new(),
                        &config,
                        &requested,
                        |chunk| streamed.push(chunk),
                    )
                }) {
                    Ok(_) | Err(crate::Error::Interrupt(_)) => {}
                    Err(error) => return Err(to_pyerr(py, error)),
                }
                handled = true;
            }
        }
        let chunks = if handled {
            streamed
        } else {
            let state = state_from_py(input)?;
            match py.detach(|| {
                self.graph
                    .stream_modes_with(&state, &config, &requested, |chunk| streamed.push(chunk))
            }) {
                Ok(_) | Err(crate::Error::Interrupt(_)) => {}
                Err(error) => return Err(to_pyerr(py, error)),
            }
            streamed
                .into_iter()
                .filter(|chunk| {
                    matches!(chunk, crate::StreamChunk::Interrupt { .. })
                        || chunk.matches_modes_strict(&requested)
                })
                .collect::<Vec<_>>()
        };
        convert_stream_chunks(py, &chunks, v2, &requested)
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

    /// Replay a thread from a specific checkpoint id.
    #[pyo3(signature = (thread_id, checkpoint_id, value=None))]
    fn resume_from(
        &self,
        py: Python<'_>,
        thread_id: String,
        checkpoint_id: String,
        value: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let value = match value {
            Some(value) => py_to_value(value)?,
            None => Value::Null,
        };
        let result = py.detach(|| self.graph.resume_from(&thread_id, &checkpoint_id, value));
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
    #[pyo3(signature = (thread_id=None, checkpoint_ns=None, subgraphs=false, config=None))]
    fn get_state(
        &self,
        py: Python<'_>,
        thread_id: Option<&Bound<'_, PyAny>>,
        checkpoint_ns: Option<String>,
        subgraphs: bool,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let (parsed_thread, parsed_config) = config_from_py(py, config.or(thread_id))?;
        let thread_id =
            parsed_thread.ok_or_else(|| PyTypeError::new_err("get_state requires a thread id"))?;
        let checkpoint_ns = checkpoint_ns.or_else(|| parsed_config.checkpoint_ns.clone());
        if subgraphs {
            let states = self
                .graph
                .get_state_subgraphs(&thread_id)
                .map_err(|error| to_pyerr(py, error))?;
            let Some((root, subgraphs)) = states else {
                return Ok(py.None());
            };
            let root_py = snapshot_to_py(py, &root)?;
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
            return Ok(root_py);
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
                if checkpoint_ns.is_none() {
                    // Root snapshots already carry the LangGraph global step.
                    snapshot_to_py(py, &snapshot)
                } else {
                    let history = self
                        .graph
                        .get_state_history(&thread_id)
                        .map_err(|error| to_pyerr(py, error))?;
                    let global_step = global_step_for(&history, &snapshot.checkpoint_id);
                    snapshot_to_py_with_step(py, &snapshot, global_step)
                }
            }
            None => empty_snapshot_to_py(py, Some(&thread_id)),
        }
    }

    /// Apply state updates to the latest checkpoint of a thread.
    ///
    /// With ``as_node`` the next tasks are derived from that node's outgoing edges.
    #[pyo3(signature = (thread_id, values, as_node=None, task_id=None))]
    fn update_state(
        &self,
        py: Python<'_>,
        thread_id: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        as_node: Option<String>,
        task_id: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let thread_id = config_from_py(py, Some(thread_id))?
            .0
            .ok_or_else(|| PyTypeError::new_err("update_state requires a thread id"))?;
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
                self.graph.update_state_with_task(
                    &thread_id,
                    updates,
                    as_node.as_deref(),
                    task_id.as_deref(),
                )
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
        thread_id: &Bound<'_, PyAny>,
        before: Option<String>,
        limit: Option<usize>,
        checkpoint_ns: Option<String>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let (thread_id, parsed_config) = config_from_py(py, Some(thread_id))?;
        let thread_id = thread_id
            .ok_or_else(|| PyTypeError::new_err("get_state_history requires a thread id"))?;
        let checkpoint_ns = checkpoint_ns.or_else(|| parsed_config.checkpoint_ns.clone());
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
        if checkpoint_ns.is_none() {
            return history
                .iter()
                .map(|snapshot| snapshot_to_py(py, snapshot))
                .collect();
        }
        let full_history = self
            .graph
            .get_state_history(&thread_id)
            .map_err(|error| to_pyerr(py, error))?;
        history
            .iter()
            .map(|snapshot| {
                let global_step = global_step_for(&full_history, &snapshot.checkpoint_id);
                snapshot_to_py_with_step(py, snapshot, global_step)
            })
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

    /// Render the graph as a PNG image using Graphviz `dot`.
    fn draw_png(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let bytes = self.graph.draw_png().map_err(|error| to_pyerr(py, error))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    /// Render the graph as a Graphviz DOT document.
    fn draw_dot(&self) -> String {
        self.graph.to_dot()
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
            ascii: self.graph.to_ascii(),
            spec: self.graph.graph_spec(),
        }
    }

    /// List threads that have at least one checkpoint.
    fn threads(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.graph.threads().map_err(|error| to_pyerr(py, error))
    }

    /// Export the latest checkpoint as LangGraph-compatible MessagePack bytes.
    #[cfg(feature = "msgpack")]
    #[pyo3(signature = (thread_id, checkpoint_ns=None))]
    fn export_checkpoint_msgpack(
        &self,
        py: Python<'_>,
        thread_id: String,
        checkpoint_ns: Option<String>,
    ) -> PyResult<Py<PyBytes>> {
        let checkpointer = self
            .graph
            .checkpointer()
            .ok_or_else(|| PyRuntimeError::new_err("graph has no checkpointer"))?;
        let checkpoint = match checkpoint_ns.as_deref() {
            Some(namespace) => checkpointer.latest_in(&thread_id, namespace),
            None => checkpointer.latest(&thread_id),
        }
        .map_err(|error| to_pyerr(py, error))?;
        let Some(checkpoint) = checkpoint else {
            return Err(PyLookupError::new_err(format!(
                "no checkpoint found for thread `{thread_id}`"
            )));
        };
        let bytes = crate::interop::checkpoint_to_langgraph_msgpack(&checkpoint)
            .map_err(|error| to_pyerr(py, error))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    /// Import LangGraph-compatible MessagePack checkpoint bytes into a thread.
    #[cfg(feature = "msgpack")]
    fn import_checkpoint_msgpack(
        &self,
        py: Python<'_>,
        thread_id: String,
        data: &[u8],
    ) -> PyResult<()> {
        let checkpoint = crate::interop::checkpoint_from_langgraph_msgpack(data)
            .map_err(|error| to_pyerr(py, error))?;
        let document = crate::interop::checkpoint_to_langgraph(&checkpoint);
        self.graph
            .import_checkpoint(&thread_id, &document)
            .map_err(|error| to_pyerr(py, error))
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

#[path = "graph_support.rs"]
mod graph_support;
pub(crate) use graph_support::*;
