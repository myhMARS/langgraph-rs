//! Python classes mirroring LangGraph's `Send`, `Command`, `RetryPolicy`,
//! `CachePolicy` and the checkpoint saver objects.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde_json::Value;

use super::convert::{normalize_node, py_to_value, to_pyerr};
use crate::_internal::node::{NodeOutput, Send};
use crate::_internal::retry::RetryPolicy;

/// A dynamic fan-out task, mirroring `langgraph.types.Send`.
#[pyclass(name = "Send")]
pub struct PySend {
    #[pyo3(get)]
    pub node: String,
    pub arg: Option<Py<PyAny>>,
    pub timeout: Option<f64>,
}

#[pymethods]
impl PySend {
    #[new]
    #[pyo3(signature = (node, arg=None, timeout=None))]
    fn new(node: String, arg: Option<Py<PyAny>>, timeout: Option<f64>) -> Self {
        Self { node, arg, timeout }
    }

    fn __repr__(&self) -> String {
        format!("Send(node={:?})", self.node)
    }
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
        let parent = self.graph.is_some();
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
    #[pyo3(signature = (max_attempts=3, initial_interval=0.5, backoff_factor=2.0, max_interval=128.0, jitter=0.0, retry_on=None))]
    fn new(
        max_attempts: usize,
        initial_interval: f64,
        backoff_factor: f64,
        max_interval: f64,
        jitter: f64,
        retry_on: Option<Vec<String>>,
    ) -> Self {
        Self {
            max_attempts,
            initial_interval,
            backoff_factor,
            max_interval,
            jitter,
            retry_on: retry_on.unwrap_or_default(),
        }
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
#[derive(Clone, Default)]
pub struct PyCachePolicy {
    #[pyo3(get)]
    pub ttl: Option<u64>,
}

#[pymethods]
impl PyCachePolicy {
    #[new]
    #[pyo3(signature = (ttl=None))]
    fn new(ttl: Option<u64>) -> Self {
        Self { ttl }
    }
}

impl PyCachePolicy {
    pub(super) fn to_policy(&self) -> crate::_internal::cache::CachePolicy {
        let policy = crate::_internal::cache::CachePolicy::new();
        match self.ttl {
            Some(ttl) => policy.with_ttl(Duration::from_secs(ttl)),
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
}

#[pymethods]
impl PyTimeoutPolicy {
    #[new]
    #[pyo3(signature = (run_timeout=None, idle_timeout=None))]
    fn new(run_timeout: Option<f64>, idle_timeout: Option<f64>) -> Self {
        Self {
            run_timeout,
            idle_timeout,
        }
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
pub(crate) use objects::*;
pub(crate) use snapshot::PyStateSnapshot;
