//! Python classes mirroring LangGraph's `Send`, `Command`, `RetryPolicy`,
//! `CachePolicy` and the checkpoint saver objects.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde_json::Value;

pub(crate) mod history;
pub(crate) mod store_embedder;

pub(crate) use history::PyStateHistory;

use super::convert::{normalize_node, py_to_value, to_pyerr};
use crate::_internal::node::{NodeOutput, Send};
use crate::_internal::retry::RetryPolicy;

/// A dynamic fan-out task, mirroring `langgraph.types.Send`.
#[pyclass(name = "Send")]
pub struct PySend {
    #[pyo3(get)]
    pub node: String,
    pub arg: Option<Py<PyAny>>,
    /// Run timeout in seconds, from `float`, `timedelta` or a `TimeoutPolicy`.
    pub timeout: Option<f64>,
}

#[pymethods]
impl PySend {
    #[new]
    #[pyo3(signature = (node, arg=None, timeout=None))]
    fn new(
        py: Python<'_>,
        node: String,
        arg: Option<Py<PyAny>>,
        timeout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            node,
            arg,
            timeout: timeout.and_then(|value| send_timeout_seconds(py, value)),
        })
    }

    fn __repr__(&self) -> String {
        format!("Send(node={:?})", self.node)
    }
}

/// Exception names accepted by `RetryPolicy(retry_on=...)`.
///
/// LangGraph takes exception types (or a tuple of them) or a callable; the engine matches by name.
fn retry_on_names(value: &Bound<'_, PyAny>) -> Vec<String> {
    let mut names = Vec::new();
    let mut push = |item: &Bound<'_, PyAny>| {
        let name = item
            .getattr("__name__")
            .ok()
            .and_then(|name| name.extract::<String>().ok())
            .unwrap_or_else(|| item.to_string());
        if !names.contains(&name) {
            names.push(name);
        }
    };
    if let Ok(items) = value.try_iter() {
        for item in items.flatten() {
            push(&item);
        }
    } else {
        push(value);
    }
    names
}

/// Seconds described by a `Send(timeout=...)` value: a number, a `timedelta` or a
/// `TimeoutPolicy(run_timeout=...)`.
fn send_timeout_seconds(_py: Python<'_>, value: &Bound<'_, PyAny>) -> Option<f64> {
    if let Ok(seconds) = value.extract::<f64>() {
        return Some(seconds);
    }
    if let Ok(policy) = value.cast::<PyTimeoutPolicy>() {
        return policy.borrow().run_timeout;
    }
    // `datetime.timedelta.total_seconds()`
    value
        .call_method0("total_seconds")
        .ok()
        .and_then(|seconds| seconds.extract::<f64>().ok())
}

impl PySend {
    pub(super) fn to_send(&self, py: Python<'_>) -> PyResult<Send> {
        let node = normalize_node(&self.node);
        let send = match &self.arg {
            None => Send::to(node),
            Some(arg) => Send::with_arg(node, py_to_value(arg.bind(py))?),
        };
        Ok(match self.timeout {
            Some(timeout) => send.with_timeout(std::time::Duration::from_secs_f64(timeout)),
            None => send,
        })
    }
}

/// A `Command` returned from a node or passed to `invoke`/`stream`.
/// Python `Interrupt` value surfaced by `__interrupt__` stream entries.
/// The runtime handle LangGraph injects into tools.
///
/// Constructed by the engine when a tool declares a `ToolRuntime` parameter, so the extension does
/// not have to import the shim's dataclass.
#[pyclass(name = "ToolRuntime", module = "langgraph_rs")]
pub struct PyToolRuntime {
    #[pyo3(get, set)]
    pub state: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    pub context: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    pub config: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    pub stream_writer: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    pub tool_call_id: Option<String>,
    #[pyo3(get, set)]
    pub store: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    pub tools: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    pub execution_info: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    pub server_info: Option<Py<PyAny>>,
}

#[pymethods]
impl PyToolRuntime {
    #[new]
    #[pyo3(signature = (state=None, context=None, config=None, stream_writer=None, tool_call_id=None, store=None, tools=None, execution_info=None, server_info=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        state: Option<Py<PyAny>>,
        context: Option<Py<PyAny>>,
        config: Option<Py<PyAny>>,
        stream_writer: Option<Py<PyAny>>,
        tool_call_id: Option<String>,
        store: Option<Py<PyAny>>,
        tools: Option<Py<PyAny>>,
        execution_info: Option<Py<PyAny>>,
        server_info: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            state,
            context,
            config,
            stream_writer,
            tool_call_id,
            store,
            tools,
            execution_info,
            server_info,
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let state = self
            .state
            .as_ref()
            .map(|value| value.bind(py).repr().map(|text| text.to_string()).unwrap_or_default())
            .unwrap_or_else(|| "None".to_string());
        format!("ToolRuntime(state={state})")
    }
}

#[pyclass(name = "Interrupt", module = "langgraph_rs")]
pub struct PyInterrupt {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub value: Py<PyAny>,
    #[pyo3(get)]
    pub node: Option<String>,
    #[pyo3(get)]
    pub resumable: bool,
    #[pyo3(get)]
    pub response_schema: Option<Py<PyAny>>,
}

#[pymethods]
impl PyInterrupt {
    #[new]
    #[pyo3(signature = (value, id=None, node=None, response_schema=None))]
    fn new(
        value: Py<PyAny>,
        id: Option<String>,
        node: Option<String>,
        response_schema: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            id: id.unwrap_or_default(),
            value,
            node,
            resumable: true,
            response_schema,
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Interrupt(id={:?}, value={})",
            self.id,
            self.value
                .bind(py)
                .repr()
                .map(|repr| repr.to_string())
                .unwrap_or_else(|_| "<value>".to_string())
        )
    }
}

/// `Command.PARENT`: target the enclosing graph from inside a subgraph.
pub(crate) const PARENT_GRAPH: &str = "__parent__";

#[pyclass(name = "Command")]
#[derive(Default)]
pub struct PyCommand {
    pub graph: Option<String>,
    pub update: Option<Py<PyAny>>,
    pub resume: Option<Py<PyAny>>,
    pub goto: Option<Py<PyAny>>,
}

#[pymethods]
impl PyCommand {
    #[new]
    #[pyo3(signature = (update=None, resume=None, goto=None, graph=None))]
    fn new(
        update: Option<Py<PyAny>>,
        resume: Option<Py<PyAny>>,
        goto: Option<Py<PyAny>>,
        graph: Option<String>,
    ) -> Self {
        Self {
            graph,
            update,
            resume,
            goto,
        }
    }

    #[classattr]
    const PARENT: &'static str = "__parent__";

    fn __repr__(&self) -> String {
        format!(
            "Command(update={}, resume={}, goto={}, graph={:?})",
            self.update.is_some(),
            self.resume.is_some(),
            self.goto.is_some(),
            self.graph
        )
    }
}

impl PyCommand {
    /// Convert a Command returned by a node into channel writes + goto targets.
    pub(super) fn to_output(&self, py: Python<'_>) -> PyResult<NodeOutput> {
        let mut output = NodeOutput::new();
        // LangGraph only treats `Command.PARENT` (`"__parent__"`) as parent routing; a plain
        // graph *name* is a different feature and has no effect at the parent level.
        let parent = matches!(self.graph.as_deref(), Some(PARENT_GRAPH));
        if let Some(update) = &self.update {
            if !update.bind(py).is_none() {
                let dict = update.bind(py).cast::<PyDict>().map_err(|_| {
                    PyTypeError::new_err("Command(update=...) expects a dict of channel writes")
                })?;
                for (key, value) in dict.iter() {
                    output = output.update(key.extract::<String>()?, py_to_value(&value)?);
                }
            }
        }
        if let Some(goto) = &self.goto {
            let goto = goto.bind(py);
            if !goto.is_none() {
                let mut handled = false;
                if goto.extract::<String>().is_err()
                    && let Ok(items) = goto.try_iter()
                {
                    for item in items {
                        let item = item?;
                        if let Ok(send) = item.extract::<PyRef<PySend>>() {
                            output = output.send(send.to_send(py)?);
                            handled = true;
                            continue;
                        }
                        if let Ok(name) = item.extract::<String>() {
                            let target = normalize_node(&name);
                            output = if parent {
                                output.goto_parent(target)
                            } else {
                                output.goto(target)
                            };
                            handled = true;
                        }
                    }
                }
                if !handled {
                    if let Ok(send) = goto.extract::<PyRef<PySend>>() {
                        output = output.send(send.to_send(py)?);
                    } else {
                        let target = normalize_node(&goto.extract::<String>()?);
                        output = if parent {
                            output.goto_parent(target)
                        } else {
                            output.goto(target)
                        };
                    }
                }
            }
        }
        Ok(output)
    }

    /// Convert a Command passed to `invoke` into resume value + updates.
    pub(super) fn to_resume_command(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Option<Value>, indexmap::IndexMap<String, Value>)> {
        if self.graph.is_some() {
            return Err(PyTypeError::new_err(
                "Command(graph=...) is only supported for subgraphs",
            ));
        }
        let resume = match &self.resume {
            Some(value) if !value.bind(py).is_none() => Some(py_to_value(value.bind(py))?),
            _ => None,
        };
        Ok((resume, self.to_update_map(py)?))
    }

    pub(super) fn to_update_map(
        &self,
        py: Python<'_>,
    ) -> PyResult<indexmap::IndexMap<String, Value>> {
        let mut updates = indexmap::IndexMap::new();
        if let Some(update) = &self.update {
            if !update.bind(py).is_none() {
                let dict = update.bind(py).cast::<PyDict>().map_err(|_| {
                    PyTypeError::new_err("Command(update=...) expects a dict of channel writes")
                })?;
                for (key, value) in dict.iter() {
                    updates.insert(key.extract::<String>()?, py_to_value(&value)?);
                }
            }
        }
        Ok(updates)
    }

    pub(super) fn goto_parts(&self, py: Python<'_>) -> PyResult<(Vec<String>, Vec<crate::Send>)> {
        if self.graph.is_some() {
            return Err(PyTypeError::new_err(
                "Command(graph=...) is only supported for subgraphs",
            ));
        }
        let Some(goto) = &self.goto else {
            return Ok((Vec::new(), Vec::new()));
        };
        let goto = goto.bind(py);
        if goto.is_none() {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut names = Vec::new();
        let mut sends = Vec::new();
        let mut push_item = |item: &Bound<'_, PyAny>| -> PyResult<()> {
            if let Ok(send) = item.extract::<PyRef<PySend>>() {
                sends.push(send.to_send(py)?);
            } else if let Ok(name) = item.extract::<String>() {
                names.push(normalize_node(&name));
            } else {
                return Err(PyTypeError::new_err(
                    "Command(goto=...) expects node names or Send objects",
                ));
            }
            Ok(())
        };
        if goto.extract::<String>().is_err() {
            if let Ok(send) = goto.extract::<PyRef<PySend>>() {
                sends.push(send.to_send(py)?);
                return Ok((names, sends));
            }
            if let Ok(items) = goto.try_iter() {
                for item in items {
                    push_item(&item?)?;
                }
                return Ok((names, sends));
            }
        }
        push_item(goto)?;
        Ok((names, sends))
    }
}

/// Cooperative run-control handle (LangGraph `RunControl` basics).
#[pyclass(name = "RunControl")]
pub struct PyRunControl {
    pub(super) inner: crate::RunControl,
}

#[pymethods]
impl PyRunControl {
    #[new]
    fn new() -> Self {
        Self {
            inner: crate::RunControl::new(),
        }
    }

    /// Abort the run at the next superstep boundary.
    fn abort(&self) {
        self.inner.abort();
    }

    fn is_aborted(&self) -> bool {
        self.inner.is_aborted()
    }

    fn __repr__(&self) -> String {
        format!("RunControl(aborted={})", self.inner.is_aborted())
    }
}

/// Writer returned by `langgraph_rs.get_stream_writer()`.
#[pyclass(name = "StreamWriter")]
pub struct PyStreamWriter;

#[pymethods]
impl PyStreamWriter {
    /// Emit a custom stream value.
    fn __call__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = py_to_value(value)?;
        // Progress for the idle watchdog: a writer call is the node reporting activity.
        crate::runtime::context::bump_progress();
        crate::push_custom(value).map_err(|error| to_pyerr(py, error))
    }

    fn __repr__(&self) -> String {
        "StreamWriter()".to_string()
    }
}

/// Retry policy usable as `StateGraph.add_node(..., retry_policy=RetryPolicy(...))`.
#[pyclass(name = "RetryPolicy", from_py_object)]
#[derive(Clone)]
pub struct PyRetryPolicy {
    #[pyo3(get)]
    pub max_attempts: usize,
    #[pyo3(get)]
    pub initial_interval: f64,
    #[pyo3(get)]
    pub backoff_factor: f64,
    #[pyo3(get)]
    pub max_interval: f64,
    #[pyo3(get)]
    pub jitter: f64,
    #[pyo3(get)]
    pub retry_on: Vec<String>,
}

#[pymethods]
impl PyRetryPolicy {
    #[new]
    #[pyo3(signature = (max_attempts=3, initial_interval=0.5, backoff_factor=2.0, max_interval=128.0, jitter=None, retry_on=None))]
    fn new(
        max_attempts: usize,
        initial_interval: f64,
        backoff_factor: f64,
        max_interval: f64,
        jitter: Option<&Bound<'_, PyAny>>,
        retry_on: Option<&Bound<'_, PyAny>>,
    ) -> Self {
        Self {
            max_attempts,
            initial_interval,
            backoff_factor,
            max_interval,
            // LangGraph's `jitter` is a bool (default True); the engine wants a factor.
            jitter: match jitter {
                None => 1.0,
                Some(value) => match value.extract::<bool>() {
                    Ok(true) => 1.0,
                    Ok(false) => 0.0,
                    Err(_) => value.extract::<f64>().unwrap_or(0.0),
                },
            },
            retry_on: retry_on.map(retry_on_names).unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        let jitter = if self.jitter > 0.0 { "True" } else { "False" };
        let retry_on = if self.retry_on.is_empty() {
            "()".to_string()
        } else {
            format!(
                "({})",
                self.retry_on
                    .iter()
                    .map(|name| format!("'{name}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        format!(
            "RetryPolicy(initial_interval={}, backoff_factor={:?}, max_interval={:?}, \
             max_attempts={}, jitter={jitter}, retry_on={retry_on})",
            self.initial_interval, self.backoff_factor, self.max_interval, self.max_attempts
        )
    }
}

impl PyRetryPolicy {
    pub(super) fn to_policy(&self) -> RetryPolicy {
        RetryPolicy::new(self.max_attempts.max(1))
            .with_backoff(Duration::from_secs_f64(self.initial_interval.max(0.0)))
            .with_multiplier(self.backoff_factor.max(1.0))
            .with_max_backoff(Duration::from_secs_f64(self.max_interval.max(0.0)))
            .with_jitter(self.jitter)
            .with_retry_on(self.retry_on.clone())
    }
}

/// Cache policy usable as `StateGraph.add_node(..., cache_policy=CachePolicy(...))`.
#[pyclass(name = "CachePolicy", from_py_object)]
#[derive(Default)]
pub struct PyCachePolicy {
    #[pyo3(get)]
    pub ttl: Option<u64>,
    /// Custom cache-key function (LangGraph's `CachePolicy.key_func`).
    #[pyo3(get)]
    pub key_func: Option<Py<PyAny>>,
}

#[pymethods]
impl PyCachePolicy {
    #[new]
    #[pyo3(signature = (ttl=None, key_func=None))]
    fn new(ttl: Option<u64>, key_func: Option<Py<PyAny>>) -> Self {
        Self { ttl, key_func }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let key_func = match &self.key_func {
            Some(value) => value
                .bind(py)
                .repr()
                .map(|text| text.to_string())
                .unwrap_or_else(|_| "None".to_string()),
            None => "None".to_string(),
        };
        let ttl = match self.ttl {
            Some(ttl) => ttl.to_string(),
            None => "None".to_string(),
        };
        format!("CachePolicy(key_func={key_func}, ttl={ttl})")
    }
}

impl Clone for PyCachePolicy {
    fn clone(&self) -> Self {
        // `Py<PyAny>` needs the GIL to clone a reference, so the callable is rebound lazily.
        Self {
            ttl: self.ttl,
            key_func: Python::attach(|py| self.key_func.as_ref().map(|value| value.clone_ref(py))),
        }
    }
}

impl PyCachePolicy {
    pub(super) fn to_policy(&self) -> crate::_internal::cache::CachePolicy {
        let policy = crate::_internal::cache::CachePolicy::new();
        let policy = match self.ttl {
            Some(ttl) => policy.with_ttl(Duration::from_secs(ttl)),
            None => policy,
        };
        match &self.key_func {
            // LangGraph calls `key_func(val)` with the node's input state.
            Some(key_func) => {
                let key_func = Python::attach(|py| key_func.clone_ref(py));
                policy.with_key(move |input: &serde_json::Value| {
                    Python::attach(|py| {
                        let argument =
                            crate::adapters::python::convert::value_to_py(py, input)?;
                        let key = key_func.bind(py).call1((argument,))?;
                        key.extract::<String>()
                            .or_else(|_| key.str().and_then(|text| text.extract::<String>()))
                    })
                    .unwrap_or_else(|_| input.to_string())
                })
            }
            None => policy,
        }
    }
}

/// Node timeout policy usable as `StateGraph.add_node(..., timeout=TimeoutPolicy(...))`.
#[pyclass(name = "TimeoutPolicy", from_py_object)]
#[derive(Clone, Default)]
pub struct PyTimeoutPolicy {
    #[pyo3(get)]
    pub run_timeout: Option<f64>,
    #[pyo3(get)]
    pub idle_timeout: Option<f64>,
    /// What refreshes the idle timer (`"auto"` or `"heartbeat"`), as in LangGraph.
    #[pyo3(get)]
    pub refresh_on: String,
}

#[pymethods]
impl PyTimeoutPolicy {
    #[new]
    #[pyo3(signature = (run_timeout=None, idle_timeout=None, refresh_on="auto"))]
    fn new(
        py: Python<'_>,
        run_timeout: Option<&Bound<'_, PyAny>>,
        idle_timeout: Option<&Bound<'_, PyAny>>,
        refresh_on: &str,
    ) -> Self {
        let seconds = |value: Option<&Bound<'_, PyAny>>| {
            value.and_then(|value| send_timeout_seconds(py, value))
        };
        Self {
            run_timeout: seconds(run_timeout),
            idle_timeout: seconds(idle_timeout),
            refresh_on: refresh_on.to_string(),
        }
    }

    fn __repr__(&self) -> String {
        let show = |value: Option<f64>| match value {
            Some(seconds) => seconds.to_string(),
            None => "None".to_string(),
        };
        format!(
            "TimeoutPolicy(run_timeout={}, idle_timeout={}, refresh_on={:?})",
            show(self.run_timeout),
            show(self.idle_timeout),
            self.refresh_on
        )
    }
}

impl PyTimeoutPolicy {
    pub(super) fn to_policy(&self) -> crate::_internal::timeout::TimeoutPolicy {
        let mut policy = crate::_internal::timeout::TimeoutPolicy::new();
        if let Some(seconds) = self.run_timeout {
            policy = policy.with_run_timeout(Duration::from_secs_f64(seconds.max(0.0)));
        }
        if let Some(seconds) = self.idle_timeout {
            policy = policy.with_idle_timeout(Duration::from_secs_f64(seconds.max(0.0)));
        }
        policy
    }
}

/// In-memory checkpointer object for `StateGraph.compile(checkpointer=...)`.
mod objects;
mod snapshot;
mod stream_chunks;
pub(crate) use objects::*;
pub(crate) use snapshot::PyStateSnapshot;
pub(crate) use stream_chunks::PyStreamChunks;
