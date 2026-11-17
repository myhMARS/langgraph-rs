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

use super::convert::{normalize_node, py_to_value, to_pyerr, value_to_py};
use crate::Checkpointer;
use crate::InMemoryStore;
#[cfg(feature = "postgres")]
use crate::PostgresSaver;
#[cfg(feature = "redis")]
use crate::RedisSaver;
use crate::kernel::cache::NodeCache;
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

    pub(super) fn goto_targets(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        if self.graph.is_some() {
            return Err(PyTypeError::new_err(
                "Command(graph=...) is only supported for subgraphs",
            ));
        }
        let Some(goto) = &self.goto else {
            return Ok(Vec::new());
        };
        let goto = goto.bind(py);
        if goto.is_none() {
            return Ok(Vec::new());
        }
        if let Ok(node) = goto.extract::<String>() {
            return Ok(vec![node]);
        }
        if let Ok(nodes) = goto.extract::<Vec<String>>() {
            return Ok(nodes);
        }
        Err(PyTypeError::new_err(
            "Command(goto=...) expects a node name or a list of node names",
        ))
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
    pub(super) fn to_policy(&self) -> crate::kernel::cache::CachePolicy {
        let policy = crate::kernel::cache::CachePolicy::new();
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
    pub(super) fn to_policy(&self) -> crate::kernel::timeout::TimeoutPolicy {
        let mut policy = crate::kernel::timeout::TimeoutPolicy::new();
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

#[cfg(feature = "sqlite")]
fn sqlite_path_from_py(object: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(path) = object.extract::<String>() {
        return Ok(path);
    }
    let cursor = object.call_method1("execute", ("PRAGMA database_list",))?;
    let row = cursor.call_method0("fetchone")?;
    let path: Option<String> = row.get_item(2)?.extract()?;
    path.filter(|path| !path.is_empty()).ok_or_else(|| {
        PyTypeError::new_err(
            "SqliteSaver expects a file path or a sqlite3.Connection backed by a file",
        )
    })
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
    fn new(path: &Bound<'_, PyAny>) -> PyResult<Self> {
        let path = sqlite_path_from_py(path)?;
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
    #[cfg(feature = "sqlite")]
    if let Ok(saver) = obj.extract::<PyRef<PyLangGraphSqliteSaver>>() {
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

#[cfg(feature = "postgres")]
fn postgres_url_from_py(object: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(url) = object.extract::<String>() {
        return Ok(url);
    }
    object
        .getattr("info")
        .and_then(|info| info.getattr("dsn"))
        .and_then(|dsn| dsn.extract::<String>())
        .map_err(|_| {
            PyTypeError::new_err(
                "PostgresSaver expects a connection URL or a psycopg connection with info.dsn",
            )
        })
}

#[cfg(feature = "redis")]
fn redis_url_from_py(object: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(url) = object.extract::<String>() {
        return Ok(url);
    }
    let kwargs = object
        .getattr("connection_pool")
        .and_then(|pool| pool.getattr("connection_kwargs"))
        .map_err(|_| {
            PyTypeError::new_err(
                "RedisSaver expects a URL or a redis client with connection_pool.connection_kwargs",
            )
        })?;
    let kwargs = kwargs.cast::<PyDict>()?;
    let get = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> { kwargs.get_item(key) };
    let host = get("host")?
        .and_then(|value| value.extract::<String>().ok())
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port = get("port")?
        .and_then(|value| value.extract::<u16>().ok())
        .unwrap_or(6379);
    let db = get("db")?
        .and_then(|value| value.extract::<u16>().ok())
        .unwrap_or(0);
    let username = get("username")?.and_then(|value| value.extract::<String>().ok());
    let password = get("password")?.and_then(|value| value.extract::<String>().ok());
    let secure = get("ssl")?
        .and_then(|value| value.extract::<bool>().ok())
        .unwrap_or(false);
    let scheme = if secure { "rediss" } else { "redis" };
    let auth = match (username, password) {
        (Some(username), Some(password)) => format!("{username}:{password}@"),
        (None, Some(password)) => format!(":{password}@"),
        _ => String::new(),
    };
    Ok(format!("{scheme}://{auth}{host}:{port}/{db}"))
}

#[cfg(feature = "redis")]
#[pyclass(name = "RedisSaver")]
pub struct PyRedisSaver {
    pub(crate) inner: Arc<RedisSaver>,
}

#[cfg(feature = "redis")]
#[pymethods]
impl PyRedisSaver {
    #[new]
    fn new(url: &Bound<'_, PyAny>) -> PyResult<Self> {
        let url = redis_url_from_py(url)?;
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
    fn new(url: &Bound<'_, PyAny>) -> PyResult<Self> {
        let url = postgres_url_from_py(url)?;
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

/// SQLite-backed long-term memory store (feature `sqlite`).
#[cfg(feature = "sqlite")]
#[pyclass(name = "SqliteStore")]
pub struct PySqliteStore {
    pub(crate) store: Arc<dyn BaseStore>,
}

#[cfg(feature = "sqlite")]
#[pymethods]
impl PySqliteStore {
    #[new]
    #[pyo3(signature = (path=None))]
    fn new(path: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let path = match path {
            Some(path) => sqlite_path_from_py(path)?,
            None => ":memory:".to_string(),
        };
        let store = crate::SqliteStore::open(path)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            store: Arc::new(store),
        })
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
        "SqliteStore()".to_string()
    }
}

/// In-memory node cache object for `compile(cache=...)`.
#[pyclass(name = "InMemoryCache")]
pub struct PyInMemoryCache {
    pub(crate) inner: Arc<crate::InMemoryCache>,
}

#[pymethods]
impl PyInMemoryCache {
    #[new]
    #[pyo3(signature = (max_entries=None))]
    fn new(max_entries: Option<usize>) -> Self {
        let cache = crate::InMemoryCache::new();
        let cache = match max_entries {
            Some(max_entries) => cache.with_max_entries(max_entries),
            None => cache,
        };
        Self {
            inner: Arc::new(cache),
        }
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn clear(&self) -> PyResult<()> {
        self.inner
            .clear()
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("InMemoryCache(len={})", self.inner.len())
    }
}

/// SQLite node cache object for `compile(cache=...)` (feature `sqlite`).
#[cfg(feature = "sqlite")]
#[pyclass(name = "SqliteCache")]
pub struct PySqliteCache {
    pub(crate) inner: Arc<crate::SqliteCache>,
}

#[cfg(feature = "sqlite")]
#[pymethods]
impl PySqliteCache {
    #[new]
    #[pyo3(signature = (path=":memory:", max_entries=None))]
    fn new(path: &str, max_entries: Option<usize>) -> PyResult<Self> {
        let cache = crate::SqliteCache::open(path)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        let cache = match max_entries {
            Some(max_entries) => cache.with_max_entries(max_entries),
            None => cache,
        };
        Ok(Self {
            inner: Arc::new(cache),
        })
    }

    fn clear(&self) -> PyResult<()> {
        self.inner
            .clear()
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        "SqliteCache()".to_string()
    }
}

/// Redis-backed long-term memory store (feature `redis`).
#[cfg(feature = "redis")]
#[pyclass(name = "RedisStore")]
pub struct PyRedisStore {
    pub(crate) store: Arc<dyn BaseStore>,
}

#[cfg(feature = "redis")]
#[pymethods]
impl PyRedisStore {
    #[new]
    fn new(url: &Bound<'_, PyAny>) -> PyResult<Self> {
        let url = redis_url_from_py(url)?;
        let store = crate::RedisStore::connect(&url)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            store: Arc::new(store),
        })
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
}

/// Postgres-backed long-term memory store (feature `postgres`).
#[cfg(feature = "postgres")]
#[pyclass(name = "PostgresStore")]
pub struct PyPostgresStore {
    pub(crate) store: Arc<dyn BaseStore>,
}

#[cfg(feature = "postgres")]
#[pymethods]
impl PyPostgresStore {
    #[new]
    fn new(url: &Bound<'_, PyAny>) -> PyResult<Self> {
        let url = postgres_url_from_py(url)?;
        let store = crate::PostgresStore::connect(&url)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            store: Arc::new(store),
        })
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
}

/// Redis-backed node cache (feature `redis`).
#[cfg(feature = "redis")]
#[pyclass(name = "RedisCache")]
pub struct PyRedisCache {
    pub(crate) inner: Arc<crate::RedisCache>,
}

#[cfg(feature = "redis")]
#[pymethods]
impl PyRedisCache {
    #[new]
    fn new(url: &str) -> PyResult<Self> {
        let cache = crate::RedisCache::connect(url)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(cache),
        })
    }

    fn clear(&self) -> PyResult<()> {
        self.inner
            .clear()
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }
}

/// Postgres-backed node cache (feature `postgres`).
#[cfg(feature = "postgres")]
#[pyclass(name = "PostgresCache")]
pub struct PyPostgresCache {
    pub(crate) inner: Arc<crate::PostgresCache>,
}

#[cfg(feature = "postgres")]
#[pymethods]
impl PyPostgresCache {
    #[new]
    #[pyo3(signature = (url, max_entries=None))]
    fn new(url: &str, max_entries: Option<usize>) -> PyResult<Self> {
        let cache = crate::PostgresCache::connect(url)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        let cache = match max_entries {
            Some(max_entries) => cache.with_max_entries(max_entries),
            None => cache,
        };
        Ok(Self {
            inner: Arc::new(cache),
        })
    }

    fn clear(&self) -> PyResult<()> {
        self.inner
            .clear()
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }
}

/// Resolve a Python `cache=` object into a kernel node cache.
pub(super) fn cache_from_py(
    obj: &Bound<'_, PyAny>,
) -> Option<Arc<dyn crate::kernel::cache::NodeCache>> {
    if let Ok(cache) = obj.extract::<PyRef<'_, PyInMemoryCache>>() {
        return Some(cache.inner.clone());
    }
    #[cfg(feature = "sqlite")]
    if let Ok(cache) = obj.extract::<PyRef<'_, PySqliteCache>>() {
        return Some(cache.inner.clone());
    }
    #[cfg(feature = "redis")]
    if let Ok(cache) = obj.extract::<PyRef<'_, PyRedisCache>>() {
        return Some(cache.inner.clone());
    }
    #[cfg(feature = "postgres")]
    if let Ok(cache) = obj.extract::<PyRef<'_, PyPostgresCache>>() {
        return Some(cache.inner.clone());
    }
    None
}

/// Checkpointer writing LangGraph's SQLite table layout (feature `sqlite`).
#[cfg(feature = "sqlite")]
#[pyclass(name = "LangGraphSqliteSaver")]
pub struct PyLangGraphSqliteSaver {
    inner: Arc<crate::LangGraphSqliteSaver>,
}

#[cfg(feature = "sqlite")]
#[pymethods]
impl PyLangGraphSqliteSaver {
    #[new]
    fn new(path: &Bound<'_, PyAny>) -> PyResult<Self> {
        let path = sqlite_path_from_py(path)?;
        let saver = crate::LangGraphSqliteSaver::open(path)
            .map_err(|error| PyTypeError::new_err(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(saver),
        })
    }
}

/// Read-only LangGraph SQLite checkpoint reader (feature `sqlite`).
#[cfg(feature = "sqlite")]
#[pyclass(name = "LangGraphSqliteReader")]
pub struct PyLangGraphSqliteReader {
    reader: Arc<crate::LangGraphSqliteReader>,
}

#[cfg(feature = "sqlite")]
#[pymethods]
impl PyLangGraphSqliteReader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let reader = crate::LangGraphSqliteReader::open(path)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            reader: Arc::new(reader),
        })
    }

    fn threads(&self) -> PyResult<Vec<String>> {
        self.reader
            .threads()
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }

    #[pyo3(signature = (thread_id, checkpoint_ns=""))]
    fn latest(
        &self,
        py: Python<'_>,
        thread_id: String,
        checkpoint_ns: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        let checkpoint = self
            .reader
            .latest(&thread_id, checkpoint_ns)
            .map_err(|error| PyTypeError::new_err(error.to_string()))?;
        checkpoint
            .map(|checkpoint| {
                let snapshot =
                    crate::StateSnapshot::<crate::JsonState>::from_checkpoint(checkpoint)
                        .map_err(|error| PyTypeError::new_err(error.to_string()))?;
                super::convert::snapshot_to_py(py, &snapshot)
            })
            .transpose()
    }

    #[pyo3(signature = (thread_id, checkpoint_ns=""))]
    fn history(
        &self,
        py: Python<'_>,
        thread_id: String,
        checkpoint_ns: &str,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let checkpoints = self
            .reader
            .history(&thread_id, checkpoint_ns)
            .map_err(|error| PyTypeError::new_err(error.to_string()))?;
        checkpoints
            .into_iter()
            .map(|checkpoint| {
                let snapshot =
                    crate::StateSnapshot::<crate::JsonState>::from_checkpoint(checkpoint)
                        .map_err(|error| PyTypeError::new_err(error.to_string()))?;
                super::convert::snapshot_to_py(py, &snapshot)
            })
            .collect()
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
    #[cfg(feature = "sqlite")]
    if let Ok(store) = obj.extract::<PyRef<'_, PySqliteStore>>() {
        return Ok(Some(store.store.clone()));
    }
    #[cfg(feature = "redis")]
    if let Ok(store) = obj.extract::<PyRef<'_, PyRedisStore>>() {
        return Ok(Some(store.store.clone()));
    }
    #[cfg(feature = "postgres")]
    if let Ok(store) = obj.extract::<PyRef<'_, PyPostgresStore>>() {
        return Ok(Some(store.store.clone()));
    }
    Err(PyTypeError::new_err(
        "store must be an InMemoryStore, Store, SqliteStore, RedisStore or PostgresStore object",
    ))
}
