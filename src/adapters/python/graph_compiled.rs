//! Compiled Python graph and graph-description classes.

use std::sync::Arc;

#[cfg(feature = "msgpack")]
use pyo3::exceptions::PyLookupError;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};

use super::*;

/// An item travelling from the graph's worker thread to the Python consumer.
enum LiveItem {
    Chunk(Box<crate::StreamChunk>),
    Error(String),
}




/// Run a sync state call on a worker thread and expose it as an awaitable.
fn async_state_call<'py>(
    slf: PyRef<'py, PyCompiledGraph>,
    py: Python<'py>,
    method: &str,
    thread_id: Option<&Bound<'py, PyAny>>,
    checkpoint_ns: Option<String>,
    subgraphs: bool,
    config: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let module = py.import("langgraph_rs.stream._events")?;
    let helper = module.getattr("call_via_sync")?;
    let graph = slf.into_pyobject(py)?.into_any();
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(thread_id) = thread_id {
        kwargs.set_item("thread_id", thread_id)?;
    }
    if let Some(checkpoint_ns) = checkpoint_ns {
        kwargs.set_item("checkpoint_ns", checkpoint_ns)?;
    }
    if subgraphs {
        kwargs.set_item("subgraphs", true)?;
    }
    if let Some(config) = config {
        kwargs.set_item("config", config)?;
    }
    helper.call1((graph, method, kwargs))
}

fn async_update_call<'py>(
    slf: PyRef<'py, PyCompiledGraph>,
    py: Python<'py>,
    thread_id: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    as_node: Option<String>,
    task_id: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let module = py.import("langgraph_rs.stream._events")?;
    let helper = module.getattr("call_via_sync")?;
    let graph = slf.into_pyobject(py)?.into_any();
    let kwargs = pyo3::types::PyDict::new(py);
    let thread_id = thread_id.clone().unbind();
    kwargs.set_item("thread_id", thread_id.bind(py))?;
    kwargs.set_item("values", values)?;
    if let Some(as_node) = as_node {
        kwargs.set_item("as_node", as_node)?;
    }
    if let Some(task_id) = task_id {
        kwargs.set_item("task_id", task_id)?;
    }
    helper.call1((graph, "update_state", kwargs))
}

fn async_history_call<'py>(
    slf: PyRef<'py, PyCompiledGraph>,
    py: Python<'py>,
    thread_id: &Bound<'py, PyAny>,
    before: Option<&Bound<'py, PyAny>>,
    limit: Option<usize>,
    checkpoint_ns: Option<String>,
    filter: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let module = py.import("langgraph_rs.stream._events")?;
    let helper = module.getattr("call_via_sync")?;
    let graph = slf.into_pyobject(py)?.into_any();
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("thread_id", thread_id)?;
    if let Some(before) = before {
        kwargs.set_item("before", before)?;
    }
    if let Some(limit) = limit {
        kwargs.set_item("limit", limit)?;
    }
    if let Some(checkpoint_ns) = checkpoint_ns {
        kwargs.set_item("checkpoint_ns", checkpoint_ns)?;
    }
    if let Some(filter) = filter {
        kwargs.set_item("filter", filter)?;
    }
    helper.call1((graph, "get_state_history", kwargs))
}

/// Parse LangGraph's `batch` config argument: `None`, one config, or one config per input.
pub(crate) fn per_input_configs(
    py: Python<'_>,
    config: Option<&Bound<'_, PyAny>>,
    count: usize,
) -> PyResult<Option<Vec<RunConfig>>> {
    let Some(config) = config.filter(|value| !value.is_none()) else {
        return Ok(None);
    };
    let items: Vec<Bound<'_, PyAny>> = if let Ok(list) = config.cast::<PyList>() {
        list.iter().collect()
    } else if let Ok(tuple) = config.cast::<pyo3::types::PyTuple>() {
        tuple.iter().collect()
    } else {
        return Ok(None);
    };
    if items.len() != count {
        return Err(PyTypeError::new_err(format!(
            "batch config list length {} does not match the number of inputs {count}",
            items.len()
        )));
    }
    let mut configs = Vec::with_capacity(count);
    for item in items {
        let (_, parsed) = crate::adapters::python::convert::config_from_py(py, Some(&item))?;
        configs.push(parsed);
    }
    Ok(Some(configs))
}

/// Rebuild the interrupt payload carried by a paused run's chunks.
fn interrupt_from_chunks(chunks: &[crate::StreamChunk]) -> Option<crate::Interrupt> {
    chunks.iter().find_map(|chunk| match chunk {
        crate::StreamChunk::Interrupt {
            node,
            id,
            value,
            response_schema,
            static_pause,
        } => Some(crate::Interrupt {
            id: id.clone(),
            node: node.clone(),
            value: value.clone(),
            response_schema: response_schema.clone(),
            values: None,
            static_pause: *static_pause,
            task_id: None,
        }),
        _ => None,
    })
}

/// Pull-based iterator over a run's chunks (`stream_live`).
///
/// The graph runs on a worker thread that pushes every chunk as it is produced — including chunks a
/// node emits through `message()` / custom writers, which therefore reach the consumer while the
/// node is still running.
#[pyclass(name = "_LiveStream")]
pub struct PyLiveStream {
    receiver: std::sync::Mutex<std::sync::mpsc::Receiver<LiveItem>>,
    handle: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
    v2: bool,
    requested: Vec<crate::StreamMode>,
}

impl PyLiveStream {
    fn new(
        receiver: std::sync::mpsc::Receiver<LiveItem>,
        handle: std::thread::JoinHandle<()>,
        v2: bool,
        requested: Vec<crate::StreamMode>,
    ) -> Self {
        Self {
            receiver: std::sync::Mutex::new(receiver),
            handle: std::sync::Mutex::new(Some(handle)),
            v2,
            requested,
        }
    }
}

#[pymethods]
impl PyLiveStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Whether the run finished and every chunk was already handed out.
    fn finished(&self) -> bool {
        let receiver = self
            .receiver
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        matches!(receiver.try_recv(), Err(std::sync::mpsc::TryRecvError::Disconnected))
    }

    /// Take the next chunk, if one is ready (`None` = nothing yet).
    ///
    /// Waiting happens on the Python side (`time.sleep`), which releases the GIL; blocking here
    /// would stall the worker thread that needs it to run Python nodes.
    fn poll(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        loop {
            let item = {
                let receiver = self
                    .receiver
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                receiver.try_recv()
            };
            match item {
                Err(std::sync::mpsc::TryRecvError::Empty) => return Ok(None),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(pyo3::exceptions::PyStopIteration::new_err(()));
                }
                Ok(LiveItem::Error(message)) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(message));
                }
                Ok(LiveItem::Chunk(chunk)) => {
                    if !chunk.is_interrupt() && !chunk.matches_modes_strict(&self.requested) {
                        continue;
                    }
                    let converted =
                        convert_stream_chunks(py, &[*chunk], self.v2, &self.requested)?;
                    if let Some(item) = converted.into_iter().next() {
                        return Ok(Some(item));
                    }
                    continue;
                }
            }
        }
    }
}


impl Drop for PyLiveStream {
    fn drop(&mut self) {
        // The worker stops on its own when the receiver disappears; joining here would block the
        // interpreter thread on a run the caller already abandoned.
        self.handle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
    }
}
#[pyclass(name = "CompiledGraph")]
pub struct PyCompiledGraph {
    pub(crate) graph: Arc<CompiledGraph<JsonState>>,
    pub(crate) saver: Option<Arc<dyn Checkpointer>>,
    /// Transformers registered at compile time for `stream_events(version="v3")`.
    pub(crate) stream_transformers: Vec<Py<PyAny>>,
}

impl PyCompiledGraph {
    pub(crate) fn from_inner_with_saver(
        graph: CompiledGraph<JsonState>,
        saver: Option<Arc<dyn Checkpointer>>,
    ) -> Self {
        Self {
            graph: Arc::new(graph),
            saver,
            stream_transformers: Vec::new(),
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
        graph_invoke::invoke(
            py,
            &self.graph,
            input,
            thread_id,
            context,
            control,
            checkpoint_ns,
            config,
        )
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

    /// Stream chunks as the graph produces them (pull-based, live).
    #[pyo3(signature = (input=None, thread_id=None, stream_mode=None, context=None, subgraphs=false, version=None))]
    #[allow(clippy::too_many_arguments)]
    fn stream_live(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        thread_id: Option<&Bound<'_, PyAny>>,
        stream_mode: Option<&Bound<'_, PyAny>>,
        context: Option<&Bound<'_, PyAny>>,
        subgraphs: bool,
        version: Option<String>,
    ) -> PyResult<Py<PyLiveStream>> {
        let v2 = match version.as_deref() {
            None | Some("v1") => false,
            Some("v2") => true,
            Some(other) => {
                return Err(PyTypeError::new_err(format!(
                    "version must be 'v1' or 'v2', got {other:?}"
                )));
            }
        };
        let thread = thread_id.filter(|value| !value.is_none()).map(|value| {
            value
                .extract::<String>()
                .or_else(|_| {
                    value
                        .get_item("thread_id")
                        .and_then(|inner| inner.extract::<String>())
                })
                .unwrap_or_default()
        });
        let requested = stream_modes_from_py(stream_mode)?
            .unwrap_or_else(|| vec![crate::StreamMode::Values]);
        let mut config = RunConfig::new().with_stream_modes(requested.clone());
        if let Some(thread) = thread.as_deref() {
            config = config.with_thread(thread.to_string());
        }
        if subgraphs {
            config = config.with_stream_subgraphs(true);
        }
        if let Some(context) = context.filter(|value| !value.is_none()) {
            config = config.with_context(py_to_value(context)?);
        }
        let state = match input.filter(|value| !value.is_none()) {
            Some(value) => state_from_py(Some(value))?,
            None => return Err(PyTypeError::new_err("stream_live requires an input state")),
        };
        let graph = self.graph.clone();
        let modes_for_worker = requested.clone();
        let config_for_worker = config.clone();
        let (sender, receiver) = std::sync::mpsc::channel::<LiveItem>();
        let sink_sender = sender.clone();
        let handle = std::thread::Builder::new()
            .name("langgraph_rs-live-stream".to_string())
            .spawn(move || {
                // Nodes run on this thread, so the LangChain callback context has to be installed
                // here for model tokens to reach `messages` (the token is intentionally leaked:
                // the thread ends with the run).
                Python::attach(|py| {
                    let _ = crate::adapters::python::convert::install_langchain_callbacks(py);
                });
                crate::runtime::context::set_pending_live_sink(std::sync::Arc::new(
                    std::sync::Mutex::new(move |chunk: crate::StreamChunk| {
                        let _ = sink_sender.send(LiveItem::Chunk(Box::new(chunk)));
                    }),
                ));
                let outcome = graph.stream_modes_with(&state, &config_for_worker, &modes_for_worker, |chunk| {
                    let _ = sender.send(LiveItem::Chunk(Box::new(chunk)));
                });
                match outcome {
                    Ok(_) | Err(crate::Error::Interrupt(_)) => {}
                    Err(error) => {
                        let _ = sender.send(LiveItem::Error(error.to_string()));
                    }
                }
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Py::new(py, PyLiveStream::new(receiver, handle, v2, requested))
    }
    /// Run the graph and return the list of stream chunks.
    /// `stream_events(version="v3")`: a `GraphRunStream` driven by the caller.
    #[pyo3(signature = (input=None, thread_id=None, version="v3", transformers=None, context=None))]
    fn stream_events<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        input: Option<&Bound<'py, PyAny>>,
        thread_id: Option<&Bound<'py, PyAny>>,
        version: &str,
        transformers: Option<&Bound<'py, PyAny>>,
        context: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let module = py.import("langgraph_rs.stream._events")?;
        let function = module.getattr("stream_events")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("version", version)?;
        if !slf.stream_transformers.is_empty() {
            let compiled = pyo3::types::PyList::new(py, slf.stream_transformers.iter().map(|item| item.bind(py)))?;
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
        let py_config: Option<Py<PyAny>> = config.map(|value| value.clone().unbind());
        let (thread, mut config) = config_from_py(py, config.or(thread_id))?;
        let _callback_scope = crate::adapters::python::convert::begin_callback_scope(py);
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
            if let Ok(command) = obj.cast::<PyCommand>().map(|cell| cell.borrow()) {
                let thread = thread.as_deref().ok_or_else(|| {
                    PyTypeError::new_err("stream(Command(...)) requires thread_id")
                })?;
                let (resume, updates) = command.to_resume_command(py)?;
                let (goto, sends) = command.goto_parts(py)?;
                if resume.is_some()
                    && let Ok(Some(snapshot)) = self.graph.get_state(thread)
                {
                    crate::adapters::python::graph_pause::dispatch_resume(
                        py,
                        py_config.as_ref().map(|value| value.bind(py)),
                        &snapshot.checkpoint_id,
                        None,
                    )?;
                }
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
                .filter(|chunk| chunk.is_interrupt() || chunk.matches_modes_strict(&requested))
                .collect::<Vec<_>>()
        };
        // Report a paused run to `GraphCallbackHandler` (LangGraph dispatches during the run).
        if let Some(thread_id) = thread.as_deref() {
            if let Ok(Some(snapshot)) = self.graph.get_state(thread_id) {
                if snapshot.interrupted {
                    if let Some(interrupt) = interrupt_from_chunks(&chunks) {
                        crate::adapters::python::graph_pause::dispatch_interrupt(
                            py,
                            py_config.as_ref().map(|value| value.bind(py)),
                            &snapshot.checkpoint_id,
                            &interrupt,
                        )?;
                    }
                }
            }
        }
        let converted = convert_stream_chunks(py, &chunks, v2, &requested)?;
        Py::new(py, crate::adapters::python::types::PyStreamChunks::new(converted))
    }

    /// Invoke the graph over a list of inputs.
    #[pyo3(signature = (inputs, config=None, *, return_exceptions=false, context=None))]
    #[allow(clippy::too_many_arguments)]
    fn batch(
        &self,
        py: Python<'_>,
        inputs: &Bound<'_, PyAny>,
        config: Option<&Bound<'_, PyAny>>,
        return_exceptions: bool,
        context: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let _callback_scope = crate::adapters::python::convert::begin_callback_scope(py);
        let list = inputs
            .cast::<PyList>()
            .map_err(|_| PyTypeError::new_err("batch expects a list of state dicts"))?;
        let states: Vec<JsonState> = list
            .iter()
            .map(|item| state_from_py(Some(&item)))
            .collect::<PyResult<Vec<_>>>()?;
        let configs = per_input_configs(py, config, states.len())?;
        let shared = configs.as_ref().is_none_or(|configs| {
            configs
                .windows(2)
                .all(|pair| pair[0].thread_id == pair[1].thread_id)
        });
        let mut base = RunConfig::new();
        if let Some(context) = context.filter(|value| !value.is_none()) {
            base = base.with_context(py_to_value(context)?);
        }
        let results: Vec<crate::Result<JsonState>> = match (shared, configs) {
            (true, None) => py.detach(|| self.graph.batch(states, &base)),
            (true, Some(configs)) => {
                let config = configs.into_iter().next().unwrap_or(base);
                py.detach(|| self.graph.batch(states, &config))
            }
            (false, Some(configs)) => py.detach(|| {
                states
                    .into_iter()
                    .zip(configs)
                    .map(|(state, config)| self.graph.invoke(&state, &config))
                    .collect()
            }),
            (false, None) => py.detach(|| self.graph.batch(states, &base)),
        };
        results
            .into_iter()
            .map(|result| match result {
                Ok(state) => json_state_to_py(py, &state),
                Err(error) => {
                    if return_exceptions {
                        Ok(to_pyerr(py, error).into_value(py).into_any())
                    } else {
                        Err(to_pyerr(py, error))
                    }
                }
            })
            .collect()
    }

    #[pyo3(signature = (inputs, config=None, *, return_exceptions=false, context=None))]
    #[allow(clippy::too_many_arguments)]
    fn abatch<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        inputs: &Bound<'_, PyAny>,
        config: Option<&Bound<'_, PyAny>>,
        return_exceptions: bool,
        context: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // LangGraph exposes `abatch` next to `batch`; the synchronous engine runs on a worker
        // thread so the event loop stays free.
        let kwargs = pyo3::types::PyDict::new(py);
        if return_exceptions {
            kwargs.set_item("return_exceptions", true)?;
        }
        if let Some(context) = context {
            kwargs.set_item("context", context)?;
        }
        let module = py.import("langgraph_rs.stream._events")?;
        let helper = module.getattr("abatch_via_sync")?;
        let graph = slf.into_pyobject(py)?.into_any();
        let config = config.map(|value| value.clone().unbind());
        let inputs = inputs.clone().unbind();
        helper.call1((
            graph,
            inputs.bind(py),
            config.as_ref().map(|value| value.bind(py)),
            kwargs,
        ))
    }

    /// Apply several supersteps of updates at once (LangGraph's `bulk_update_state`).
    #[pyo3(signature = (config=None, supersteps=None, *, thread_id=None))]
    fn bulk_update_state(
        &self,
        py: Python<'_>,
        config: Option<&Bound<'_, PyAny>>,
        supersteps: Option<&Bound<'_, PyAny>>,
        thread_id: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let (parsed_thread, _) = config_from_py(py, config.or(thread_id))?;
        let thread_id = parsed_thread
            .ok_or_else(|| PyTypeError::new_err("bulk_update_state requires a thread id"))?;
        let supersteps = supersteps.ok_or_else(|| {
            PyTypeError::new_err("bulk_update_state requires a list of supersteps")
        })?;
        let steps = supersteps
            .cast::<PyList>()
            .map_err(|_| PyTypeError::new_err("supersteps must be a list of lists"))?;
        let mut latest: Option<Py<PyAny>> = None;
        for step in steps.iter() {
            let updates = step
                .cast::<PyList>()
                .map_err(|_| PyTypeError::new_err("every superstep must be a list of updates"))?;
            for update in updates.iter() {
                let values = update.get_item("values").ok();
                let as_node: Option<String> = update
                    .get_item("as_node")
                    .ok()
                    .and_then(|node| node.extract::<String>().ok());
                let task_id: Option<String> = update
                    .get_item("task_id")
                    .ok()
                    .and_then(|task| task.extract::<String>().ok());
                let values = match values {
                    Some(values) => values,
                    None => continue,
                };
                let value = py_to_value(&values)?;
                let map = value.as_object().cloned().ok_or_else(|| {
                    PyTypeError::new_err("bulk_update_state expects dict updates")
                })?;
                let writes: Vec<(String, crate::Value)> =
                    map.into_iter().collect();
                let result = py.detach(|| {
                    self.graph
                        .update_state_with_task(&thread_id, writes, as_node.as_deref(), task_id.as_deref())
                });
                latest = Some(match result {
                    Ok(snapshot) => snapshot_to_py(py, &snapshot)?,
                    Err(error) => return Err(to_pyerr(py, error)),
                });
            }
        }
        Ok(latest.unwrap_or_else(|| py.None()))
    }

    /// Clear cached node outputs (LangGraph's `clear_cache`).
    #[pyo3(signature = (nodes=None))]
    fn clear_cache(&self, py: Python<'_>, nodes: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        // The cache backends only support a full clear; `nodes` is accepted for compatibility.
        let _ = nodes;
        match self.graph.cache() {
            Some(cache) => cache.clear().map_err(|error| to_pyerr(py, error)),
            None => Ok(()),
        }
    }

    #[pyo3(signature = (thread_id=None, checkpoint_ns=None, subgraphs=false, config=None))]
    fn aget_state<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        thread_id: Option<&Bound<'py, PyAny>>,
        checkpoint_ns: Option<String>,
        subgraphs: bool,
        config: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        async_state_call(slf, py, "get_state", thread_id, checkpoint_ns, subgraphs, config)
    }

    #[pyo3(signature = (thread_id, values, as_node=None, task_id=None))]
    fn aupdate_state<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        thread_id: &Bound<'py, PyAny>,
        values: &Bound<'py, PyAny>,
        as_node: Option<String>,
        task_id: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        async_update_call(slf, py, thread_id, values, as_node, task_id)
    }

    #[pyo3(signature = (thread_id, before=None, limit=None, checkpoint_ns=None, filter=None))]
    fn aget_state_history<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        thread_id: &Bound<'py, PyAny>,
        before: Option<&Bound<'py, PyAny>>,
        limit: Option<usize>,
        checkpoint_ns: Option<String>,
        filter: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        async_history_call(slf, py, thread_id, before, limit, checkpoint_ns, filter)
    }

    #[pyo3(signature = (config=None, supersteps=None, *, thread_id=None))]
    fn abulk_update_state<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        config: Option<&Bound<'py, PyAny>>,
        supersteps: Option<&Bound<'py, PyAny>>,
        thread_id: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let module = py.import("langgraph_rs.stream._events")?;
        let helper = module.getattr("call_via_sync")?;
        let graph = slf.into_pyobject(py)?.into_any();
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(config) = config {
            kwargs.set_item("config", config)?;
        }
        if let Some(supersteps) = supersteps {
            kwargs.set_item("supersteps", supersteps)?;
        }
        if let Some(thread_id) = thread_id {
            kwargs.set_item("thread_id", thread_id)?;
        }
        helper.call1((graph, "bulk_update_state", kwargs))
    }

    #[pyo3(signature = (nodes=None))]
    fn aclear_cache<'py>(
        &self,
        py: Python<'py>,
        nodes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        PyCompiledGraph::clear_cache(self, py, nodes)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok::<_, pyo3::PyErr>(()) })
    }

    /// The long-term store this graph was compiled with.
    #[getter]
    fn store(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.graph.store() {
            Some(store) => Ok(crate::adapters::python::types::PyStore::new(store.clone())
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            None => Ok(py.None()),
        }
    }

    /// The declared channel names of the compiled graph.
    #[getter]
    fn channels(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = pyo3::types::PyDict::new(py);
        for name in self.graph.schema().names() {
            dict.set_item(name, py.None())?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Iterate over the subgraphs embedded in this graph (LangGraph's `get_subgraphs`).
    #[pyo3(signature = (*, namespace=None, recurse=false))]
    fn get_subgraphs(
        &self,
        py: Python<'_>,
        namespace: Option<String>,
        recurse: bool,
    ) -> PyResult<Py<PyAny>> {
        let _ = namespace;
        let items = pyo3::types::PyList::empty(py);
        for (name, child) in self.graph.subgraphs() {
            let child_py = Py::new(
                py,
                PyCompiledGraph {
                    graph: child.clone(),
                    saver: self.saver.clone(),
                    stream_transformers: Vec::new(),
                },
            )?;
            items.append((name.clone(), child_py))?;
            if recurse {
                for (inner_name, inner) in child.subgraphs() {
                    let nested = Py::new(
                        py,
                        PyCompiledGraph {
                            graph: inner.clone(),
                            saver: self.saver.clone(),
                            stream_transformers: Vec::new(),
                        },
                    )?;
                    items.append((format!("{name}|{inner_name}"), nested))?;
                }
            }
        }
        Ok(items.call_method0("__iter__")?.unbind().into_any())
    }

    /// Copy the compiled graph (LangGraph's `copy`).
    #[pyo3(signature = (update=None))]
    fn copy(&self, py: Python<'_>, update: Option<&Bound<'_, PyAny>>) -> PyResult<Py<Self>> {
        let mut graph = (*self.graph).clone();
        if let Some(update) = update.filter(|value| !value.is_none()) {
            if let Some(name) = update.get_item("name").ok().and_then(|value| value.extract::<String>().ok()) {
                graph.name = Some(name);
            }
        }
        Py::new(
            py,
            PyCompiledGraph {
                graph: std::sync::Arc::new(graph),
                saver: self.saver.clone(),
                stream_transformers: self
                    .stream_transformers
                    .iter()
                    .map(|item| item.clone_ref(py))
                    .collect(),
            },
        )
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
            Err(error) => {
                match crate::adapters::python::graph_pause::pause_values(py, &self.graph, Some(thread_id.as_str()), None, &error, None)?
                {
                    Some(value) => Ok(value),
                    None => Err(to_pyerr(py, error)),
                }
            }
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
        // `resume_from(..., None)` replays without a resume value, like `invoke(None)`.
        let result = match value {
            Some(value) => {
                let value = py_to_value(value)?;
                py.detach(|| self.graph.resume_from(&thread_id, &checkpoint_id, value))
            }
            None => py.detach(|| self.graph.replay_from(&thread_id, &checkpoint_id)),
        };
        match result {
            Ok(state) => json_state_to_py(py, &state),
            Err(error) => {
                match crate::adapters::python::graph_pause::pause_values(py, &self.graph, Some(thread_id.as_str()), None, &error, None)?
                {
                    Some(value) => Ok(value),
                    None => Err(to_pyerr(py, error)),
                }
            }
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
            Err(error) => {
                match crate::adapters::python::graph_pause::pause_values(py, &self.graph, Some(thread_id.as_str()), None, &error, None)?
                {
                    Some(value) => Ok(value),
                    None => Err(to_pyerr(py, error)),
                }
            }
        }
    }

    /// Continue a thread paused by an `interrupt_before` / `interrupt_after` hook.
    fn continue_thread(&self, py: Python<'_>, thread_id: String) -> PyResult<Py<PyAny>> {
        let result = py.detach(|| self.graph.continue_thread(&thread_id));
        match result {
            Ok(state) => json_state_to_py(py, &state),
            Err(error) => {
                match crate::adapters::python::graph_pause::pause_values(py, &self.graph, Some(thread_id.as_str()), None, &error, None)?
                {
                    Some(value) => Ok(value),
                    None => Err(to_pyerr(py, error)),
                }
            }
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
            return graph_support::embed_subgraph_states(py, &root, &subgraphs);
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
            Some(snapshot) => snapshot_to_py(py, &snapshot),
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
    #[pyo3(signature = (thread_id, before=None, limit=None, checkpoint_ns=None, filter=None))]
    fn get_state_history(
        &self,
        py: Python<'_>,
        thread_id: &Bound<'_, PyAny>,
        before: Option<String>,
        limit: Option<usize>,
        checkpoint_ns: Option<String>,
        filter: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let (thread_id, parsed_config) = config_from_py(py, Some(thread_id))?;
        let filter = match filter {
            Some(filter) => {
                let value = py_to_value(&filter)?;
                match value {
                    Value::Object(map) => Some(map),
                    _ => {
                        return Err(PyTypeError::new_err(
                            "get_state_history filter must be a dict",
                        ));
                    }
                }
            }
            None => None,
        };
        let thread_id = thread_id
            .ok_or_else(|| PyTypeError::new_err("get_state_history requires a thread id"))?;
        let checkpoint_ns = checkpoint_ns.or_else(|| parsed_config.checkpoint_ns.clone());
        let namespace = checkpoint_ns.clone().unwrap_or_default();
        // Validate the query (missing checkpointer, unknown namespace) the way the eager read
        // did, without materialising anything: an empty page still performs the lookup.
        self.graph
            .history_handles(&thread_id, &namespace, None, Some(0))
            .map_err(|error| to_pyerr(py, error))?;
        // A metadata filter cannot be answered from checkpoint handles, so those queries stay
        // eager; everything else converts snapshots on demand.
        let history = match filter {
            Some(filter) => {
                let snapshots = match checkpoint_ns.as_deref() {
                    Some(namespace) => self.graph.get_state_history_where_in(
                        &thread_id,
                        namespace,
                        Some(&filter),
                        before.as_deref(),
                        limit,
                    ),
                    None => self.graph.get_state_history_where(
                        &thread_id,
                        Some(&filter),
                        before.as_deref(),
                        limit,
                    ),
                }
                .map_err(|error| to_pyerr(py, error))?;
                let items = snapshots
                    .iter()
                    .map(|snapshot| snapshot_to_py(py, snapshot))
                    .collect::<PyResult<Vec<_>>>()?;
                crate::adapters::python::types::PyStateHistory::from_items(items)
            }
            None => {
                let source: Arc<dyn crate::adapters::python::types::history::HistorySource> =
                    self.graph.clone();
                crate::adapters::python::types::PyStateHistory::new(
                    source, thread_id, namespace, before, limit,
                )
            }
        };
        Py::new(py, history).map(|history| history.into_any())
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
    #[pyo3(signature = (config=None, *, xray=false))]
    fn get_graph(&self, config: Option<&Bound<'_, PyAny>>, xray: bool) -> PyGraph {
        // `xray` (and the config) are accepted for LangGraph compatibility; the extension renders
        // the same graph either way because subgraphs are embedded as nodes already.
        let _ = (config, xray);
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
    #[pyo3(signature = (input=None, config=None, *, thread_id=None, context=None, checkpoint_ns=None))]
    #[allow(clippy::too_many_arguments)]
    fn ainvoke<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        input: Option<&Bound<'_, PyAny>>,
        config: Option<&Bound<'_, PyAny>>,
        thread_id: Option<&Bound<'_, PyAny>>,
        context: Option<&Bound<'_, PyAny>>,
        checkpoint_ns: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // LangGraph exposes `ainvoke` on the same object; the extension runs the synchronous
        // engine on a worker thread so the event loop stays free.
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(thread_id) = thread_id {
            kwargs.set_item("thread_id", thread_id)?;
        }
        if let Some(context) = context {
            kwargs.set_item("context", context)?;
        }
        if let Some(checkpoint_ns) = checkpoint_ns {
            kwargs.set_item("checkpoint_ns", checkpoint_ns)?;
        }
        let module = py.import("langgraph_rs.stream._events")?;
        let helper = module.getattr("ainvoke_via_sync")?;
        let graph = slf.into_pyobject(py)?.into_any();
        let input = input.map(|value| value.clone().unbind());
        let config = config.map(|value| value.clone().unbind());
        helper.call1((
            graph,
            input.as_ref().map(|value| value.bind(py)),
            config.as_ref().map(|value| value.bind(py)),
            kwargs,
        ))
    }

    #[pyo3(signature = (input=None, config=None, *, thread_id=None, stream_mode=None, context=None, subgraphs=false, version=None))]
    #[allow(clippy::too_many_arguments)]
    fn astream<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        input: Option<&Bound<'_, PyAny>>,
        config: Option<&Bound<'_, PyAny>>,
        thread_id: Option<&Bound<'_, PyAny>>,
        stream_mode: Option<&Bound<'_, PyAny>>,
        context: Option<&Bound<'_, PyAny>>,
        subgraphs: bool,
        version: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Async generator over the collected run, matching LangGraph's `astream` shape.
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(thread_id) = thread_id {
            kwargs.set_item("thread_id", thread_id)?;
        }
        if let Some(stream_mode) = stream_mode {
            kwargs.set_item("stream_mode", stream_mode)?;
        }
        if let Some(context) = context {
            kwargs.set_item("context", context)?;
        }
        if subgraphs {
            kwargs.set_item("subgraphs", true)?;
        }
        if let Some(version) = version {
            kwargs.set_item("version", version)?;
        }
        let module = py.import("langgraph_rs.stream._events")?;
        let helper = module.getattr("astream_via_sync")?;
        let graph = slf.into_pyobject(py)?.into_any();
        let input = input.map(|value| value.clone().unbind());
        let config = config.map(|value| value.clone().unbind());
        helper.call1((
            graph,
            input.as_ref().map(|value| value.bind(py)),
            config.as_ref().map(|value| value.bind(py)),
            kwargs,
        ))
    }

    #[pyo3(signature = (input=None, config=None, *, thread_id=None, version="v3", transformers=None, context=None))]
    #[allow(clippy::too_many_arguments)]
    fn astream_events<'py>(
        &self,
        py: Python<'py>,
        input: Option<&Bound<'_, PyAny>>,
        config: Option<&Bound<'_, PyAny>>,
        thread_id: Option<&Bound<'_, PyAny>>,
        version: &str,
        transformers: Option<&Bound<'_, PyAny>>,
        context: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = config;
        let module = py.import("langgraph_rs.stream._events")?;
        let function = module.getattr("astream_events")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("version", version)?;
        if !self.stream_transformers.is_empty() {
            let compiled = pyo3::types::PyList::new(
                py,
                self.stream_transformers.iter().map(|item| item.bind(py)),
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
        let graph = self.graph.clone();
        let saver = self.saver.clone();
        let transform = self
            .stream_transformers
            .iter()
            .map(|item| item.clone_ref(py))
            .collect::<Vec<_>>();
        let bound_graph =
            Py::new(py, PyCompiledGraph { graph, saver, stream_transformers: transform })?;
        let input = input.map(|value| value.clone().unbind());
        let input = input.as_ref().map(|value| value.bind(py));
        function.call((bound_graph, input), Some(&kwargs))
    }

}

#[path = "graph_support.rs"]
mod graph_support;
pub(crate) use graph_support::*;
