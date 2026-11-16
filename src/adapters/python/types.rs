//! Python classes mirroring LangGraph's `Send`, `Command`, `RetryPolicy`,
//! `CachePolicy` and the checkpoint saver objects.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde_json::Value;

use super::convert::{normalize_node, py_to_value, value_to_py};
use crate::Checkpointer;
use crate::InMemoryStore;
#[cfg(feature = "postgres")]
use crate::PostgresSaver;
#[cfg(feature = "redis")]
use crate::RedisSaver;
use crate::kernel::node::{NodeOutput, Send};
use crate::kernel::retry::RetryPolicy;
use crate::kernel::store::{BaseStore, StoreItem};

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
            if !goto.bind(py).is_none() {
                if let Ok(names) = goto.bind(py).extract::<Vec<String>>() {
                    for name in names {
                        let target = normalize_node(&name);
                        output = if parent {
                            output.goto_parent(target)
                        } else {
                            output.goto(target)
                        };
                    }
                } else {
                    let target = normalize_node(&goto.bind(py).extract::<String>()?);
                    output = if parent {
                        output.goto_parent(target)
                    } else {
                        output.goto(target)
                    };
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
        if self.goto.is_some() {
            return Err(PyTypeError::new_err(
                "Command(goto=...) is only supported when returned by a node",
            ));
        }
        let resume = match &self.resume {
            Some(value) if !value.bind(py).is_none() => Some(py_to_value(value.bind(py))?),
            _ => None,
        };
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
        Ok((resume, updates))
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
}

#[pymethods]
impl PyRetryPolicy {
    #[new]
    #[pyo3(signature = (max_attempts=3, initial_interval=0.5, backoff_factor=2.0, max_interval=128.0))]
    fn new(
        max_attempts: usize,
        initial_interval: f64,
        backoff_factor: f64,
        max_interval: f64,
    ) -> Self {
        Self {
            max_attempts,
            initial_interval,
            backoff_factor,
            max_interval,
        }
    }
}

impl PyRetryPolicy {
    pub(super) fn to_policy(&self) -> RetryPolicy {
        RetryPolicy::new(self.max_attempts.max(1))
            .with_backoff(Duration::from_secs_f64(self.initial_interval.max(0.0)))
            .with_multiplier(self.backoff_factor.max(1.0))
            .with_max_backoff(Duration::from_secs_f64(self.max_interval.max(0.0)))
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
    pub(super) fn to_policy(&self) -> crate::kernel::cache::CachePolicy {
        let policy = crate::kernel::cache::CachePolicy::new();
        match self.ttl {
            Some(ttl) => policy.with_ttl(Duration::from_secs(ttl)),
            None => policy,
        }
    }
}

/// In-memory checkpointer object for `StateGraph.compile(checkpointer=...)`.
#[pyclass(name = "InMemorySaver")]
pub struct PyInMemorySaver {
    pub(super) inner: Arc<crate::InMemorySaver>,
}

#[pymethods]
impl PyInMemorySaver {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(crate::InMemorySaver::new()),
        }
    }

    fn __repr__(&self) -> &'static str {
        "InMemorySaver()"
    }
}

/// SQLite checkpointer object (available with the `sqlite` feature).
#[cfg(feature = "sqlite")]
#[pyclass(name = "SqliteSaver")]
pub struct PySqliteSaver {
    pub(super) inner: Arc<crate::SqliteSaver>,
}

#[cfg(feature = "sqlite")]
#[pymethods]
impl PySqliteSaver {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(
                crate::SqliteSaver::open(path)
                    .map_err(|error| PyTypeError::new_err(error.to_string()))?,
            ),
        })
    }

    #[staticmethod]
    fn in_memory() -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(
                crate::SqliteSaver::in_memory()
                    .map_err(|error| PyTypeError::new_err(error.to_string()))?,
            ),
        })
    }

    fn __repr__(&self) -> &'static str {
        "SqliteSaver()"
    }
}

/// Extract a checkpointer from a Python `bool` or saver object.
pub(super) fn checkpointer_from_py(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Arc<dyn Checkpointer>>> {
    let Some(obj) = obj else {
        return Ok(None);
    };
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(enabled) = obj.extract::<bool>() {
        return Ok(enabled.then(|| Arc::new(crate::InMemorySaver::new()) as Arc<dyn Checkpointer>));
    }
    if let Ok(saver) = obj.extract::<PyRef<PyInMemorySaver>>() {
        return Ok(Some(saver.inner.clone() as Arc<dyn Checkpointer>));
    }
    #[cfg(feature = "sqlite")]
    if let Ok(saver) = obj.extract::<PyRef<PySqliteSaver>>() {
        return Ok(Some(saver.inner.clone() as Arc<dyn Checkpointer>));
    }
    #[cfg(feature = "redis")]
    if let Ok(saver) = obj.extract::<PyRef<PyRedisSaver>>() {
        return Ok(Some(saver.inner.clone() as Arc<dyn Checkpointer>));
    }
    #[cfg(feature = "postgres")]
    if let Ok(saver) = obj.extract::<PyRef<PyPostgresSaver>>() {
        return Ok(Some(saver.inner.clone() as Arc<dyn Checkpointer>));
    }
    Err(PyTypeError::new_err(
        "checkpointer must be a bool or a saver object (InMemorySaver, SqliteSaver, RedisSaver, PostgresSaver)",
    ))
}

/// Removal marker for `add_messages`-style reducers (LangGraph `RemoveMessage`).
#[pyclass(name = "RemoveMessage")]
pub struct PyRemoveMessage {
    #[pyo3(get)]
    pub id: String,
}

#[pymethods]
impl PyRemoveMessage {
    #[new]
    fn new(id: String) -> Self {
        Self { id }
    }

    fn __repr__(&self) -> String {
        format!("RemoveMessage(id={:?})", self.id)
    }
}

/// A Python callable whose result is memoized in the current run's store.
#[pyclass(name = "Task")]
pub struct PyTask {
    function: Py<PyAny>,
    name: String,
}

impl PyTask {
    pub(super) fn from_function(function: Py<PyAny>, name: Option<String>) -> PyResult<Self> {
        let name = match name {
            Some(name) => name,
            None => Python::attach(|py| {
                function
                    .bind(py)
                    .getattr("__name__")
                    .ok()
                    .and_then(|value| value.extract::<String>().ok())
            })
            .unwrap_or_else(|| "task".to_string()),
        };
        Ok(Self { function, name })
    }
}

#[pymethods]
impl PyTask {
    #[new]
    #[pyo3(signature = (function, name=None))]
    fn new(function: Py<PyAny>, name: Option<String>) -> PyResult<Self> {
        Self::from_function(function, name)
    }

    /// Task name used as the store namespace.
    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    /// Call the task; results are cached in the checkpoint task cache / run store.
    fn __call__(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let value = py_to_value(input)?;
        let mut hasher = DefaultHasher::new();
        value.to_string().hash(&mut hasher);
        let key = format!("{:x}", hasher.finish());
        let cache_key = format!("{}:{}", self.name, key);
        if let Some(cached) = crate::kernel::context::task_result_get(&cache_key) {
            return value_to_py(py, &cached);
        }
        let store = crate::get_store();
        let namespace = vec!["functional".to_string(), self.name.clone()];
        if let Some(store) = &store
            && let Some(cached) = store
                .get(&namespace, &key)
                .map_err(|error| PyTypeError::new_err(error.to_string()))?
        {
            return value_to_py(py, &cached);
        }
        let result = self.function.call1(py, (input,))?;
        let stored = py_to_value(result.bind(py))?;
        crate::kernel::context::task_result_put(cache_key, stored.clone());
        if let Some(store) = store {
            store
                .put(&namespace, &key, stored)
                .map_err(|error| PyTypeError::new_err(error.to_string()))?;
        }
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!("Task({:?})", self.name)
    }
}

#[cfg(feature = "redis")]
/// Python wrapper around `RedisSaver`.
#[pyclass(name = "RedisSaver")]
pub struct PyRedisSaver {
    pub(crate) inner: Arc<RedisSaver>,
}

#[cfg(feature = "redis")]
#[pymethods]
impl PyRedisSaver {
    #[new]
    fn new(url: String) -> PyResult<Self> {
        let saver =
            RedisSaver::connect(&url).map_err(|error| PyTypeError::new_err(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(saver),
        })
    }

    fn __repr__(&self) -> String {
        "RedisSaver()".to_string()
    }
}

#[cfg(feature = "postgres")]
/// Python wrapper around `PostgresSaver`.
#[pyclass(name = "PostgresSaver")]
pub struct PyPostgresSaver {
    pub(crate) inner: Arc<PostgresSaver>,
}

#[cfg(feature = "postgres")]
#[pymethods]
impl PyPostgresSaver {
    #[new]
    fn new(url: String) -> PyResult<Self> {
        let saver = PostgresSaver::connect(&url)
            .map_err(|error| PyTypeError::new_err(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(saver),
        })
    }

    fn __repr__(&self) -> String {
        "PostgresSaver()".to_string()
    }
}

/// Python wrapper around a long-term memory store (`BaseStore`).
#[pyclass(name = "Store", from_py_object)]
#[derive(Clone)]
pub struct PyStore {
    pub(crate) store: Arc<dyn BaseStore>,
}

impl PyStore {
    pub(super) fn new(store: Arc<dyn BaseStore>) -> Self {
        Self { store }
    }
}

#[pymethods]
impl PyStore {
    fn get(
        &self,
        py: Python<'_>,
        namespace: Vec<String>,
        key: String,
    ) -> PyResult<Option<Py<PyAny>>> {
        let value = self
            .store
            .get(&namespace, &key)
            .map_err(|error| PyTypeError::new_err(error.to_string()))?;
        value.map(|value| value_to_py(py, &value)).transpose()
    }

    fn put(&self, namespace: Vec<String>, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.store
            .put(&namespace, &key, py_to_value(value)?)
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }

    fn delete(&self, namespace: Vec<String>, key: String) -> PyResult<()> {
        self.store
            .delete(&namespace, &key)
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }

    #[pyo3(signature = (namespace_prefix, query=None, limit=None))]
    fn search(
        &self,
        py: Python<'_>,
        namespace_prefix: Vec<String>,
        query: Option<String>,
        limit: Option<usize>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let items: Vec<StoreItem> = self
            .store
            .search(&namespace_prefix, query.as_deref(), limit)
            .map_err(|error| PyTypeError::new_err(error.to_string()))?;
        items
            .iter()
            .map(|item| store_item_to_py(py, item))
            .collect()
    }

    fn namespaces(&self) -> PyResult<Vec<Vec<String>>> {
        self.store
            .namespaces()
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        "Store()".to_string()
    }
}

fn store_item_to_py(py: Python<'_>, item: &StoreItem) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("namespace", item.namespace.clone())?;
    dict.set_item("key", item.key.clone())?;
    dict.set_item("value", value_to_py(py, &item.value)?)?;
    dict.set_item("created_at_millis", item.created_at_millis)?;
    dict.set_item("updated_at_millis", item.updated_at_millis)?;
    Ok(dict.into_any().unbind())
}

/// In-memory long-term memory store, mirroring `langgraph.store.memory.InMemoryStore`.
#[pyclass(name = "InMemoryStore")]
pub struct PyInMemoryStore {
    pub(crate) store: Arc<InMemoryStore>,
}

#[pymethods]
impl PyInMemoryStore {
    #[new]
    fn new() -> Self {
        Self {
            store: Arc::new(InMemoryStore::new()),
        }
    }

    fn len(&self) -> usize {
        self.store.len()
    }

    fn get(
        &self,
        py: Python<'_>,
        namespace: Vec<String>,
        key: String,
    ) -> PyResult<Option<Py<PyAny>>> {
        let value = self
            .store
            .get(&namespace, &key)
            .map_err(|error| PyTypeError::new_err(error.to_string()))?;
        value.map(|value| value_to_py(py, &value)).transpose()
    }

    fn put(&self, namespace: Vec<String>, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.store
            .put(&namespace, &key, py_to_value(value)?)
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }

    fn delete(&self, namespace: Vec<String>, key: String) -> PyResult<()> {
        self.store
            .delete(&namespace, &key)
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }

    #[pyo3(signature = (namespace_prefix, query=None, limit=None))]
    fn search(
        &self,
        py: Python<'_>,
        namespace_prefix: Vec<String>,
        query: Option<String>,
        limit: Option<usize>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let items: Vec<StoreItem> = self
            .store
            .search(&namespace_prefix, query.as_deref(), limit)
            .map_err(|error| PyTypeError::new_err(error.to_string()))?;
        items
            .iter()
            .map(|item| store_item_to_py(py, item))
            .collect()
    }

    fn namespaces(&self) -> PyResult<Vec<Vec<String>>> {
        self.store
            .namespaces()
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        "InMemoryStore()".to_string()
    }
}

/// Resolve a Python `store=` keyword into a kernel store.
pub(super) fn store_from_py(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Arc<dyn BaseStore>>> {
    let Some(obj) = obj else {
        return Ok(None);
    };
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(store) = obj.extract::<PyRef<'_, PyInMemoryStore>>() {
        let store: Arc<dyn BaseStore> = store.store.clone();
        return Ok(Some(store));
    }
    if let Ok(store) = obj.extract::<PyRef<'_, PyStore>>() {
        return Ok(Some(store.store.clone()));
    }
    Err(PyTypeError::new_err(
        "store must be an InMemoryStore or Store object",
    ))
}
