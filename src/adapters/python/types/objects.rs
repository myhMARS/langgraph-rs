//! Saver, store, cache and task objects.

use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::types::{PyAny, PyDict};

use super::*;
use crate::Checkpointer;
#[cfg(feature = "postgres")]
use crate::PostgresSaver;
#[cfg(feature = "redis")]
use crate::RedisSaver;
use crate::adapters::python::convert::{py_to_value, value_to_py};
use crate::store::StoreItem;

#[pyclass(name = "InMemorySaver", subclass)]
pub struct PyInMemorySaver {
    pub(super) inner: Arc<crate::InMemorySaver>,
}

/// LangGraph's `CheckpointTuple`: the checkpoint, the config addressing it, and its pending writes.
fn checkpoint_tuple_to_py(
    py: Python<'_>,
    checkpoint: &crate::checkpoint::Checkpoint,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    let config = PyDict::new(py);
    let configurable = PyDict::new(py);
    configurable.set_item("thread_id", &checkpoint.thread_id)?;
    configurable.set_item("checkpoint_ns", &checkpoint.namespace)?;
    configurable.set_item("checkpoint_id", &checkpoint.id)?;
    config.set_item("configurable", configurable)?;
    dict.set_item("config", config)?;

    let body = PyDict::new(py);
    body.set_item("v", 4)?;
    body.set_item("id", &checkpoint.id)?;
    body.set_item("ts", "1970-01-01T00:00:00+00:00")?;
    body.set_item(
        "updated_channels",
        value_to_py(py, &serde_json::Value::Array(
            checkpoint
                .updated_channels
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect(),
        ))?,
    )?;
    body.set_item(
        "channel_values",
        value_to_py(py, &serde_json::Value::Object(checkpoint.values.clone()))?,
    )?;
    body.set_item(
        "channel_versions",
        value_to_py(py, &serde_json::Value::Object(checkpoint.channel_versions.clone()))?,
    )?;
    body.set_item(
        "versions_seen",
        value_to_py(
            py,
            &serde_json::to_value(&checkpoint.versions_seen)
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
        )?,
    )?;
    dict.set_item("checkpoint", body)?;

    let metadata = PyDict::new(py);
    metadata.set_item("source", checkpoint.metadata.get("source").and_then(|value| value.as_str()).unwrap_or("loop"))?;
    metadata.set_item("step", checkpoint.step as i64 - 1)?;
    metadata.set_item("writes", py.None())?;
    metadata.set_item("parents", PyDict::new(py))?;
    dict.set_item("metadata", metadata)?;

    match &checkpoint.parent_id {
        Some(parent) => {
            let parent_config = PyDict::new(py);
            let configurable = PyDict::new(py);
            configurable.set_item("thread_id", &checkpoint.thread_id)?;
            configurable.set_item("checkpoint_ns", &checkpoint.namespace)?;
            configurable.set_item("checkpoint_id", parent)?;
            parent_config.set_item("configurable", configurable)?;
            dict.set_item("parent_config", parent_config)?;
        }
        None => dict.set_item("parent_config", py.None())?,
    }
    let writes = pyo3::types::PyList::empty(py);
    for write in &checkpoint.pending_writes {
        writes.append((&write.task_id, &write.channel, value_to_py(py, &write.value)?))?;
    }
    dict.set_item("pending_writes", writes)?;
    Ok(dict.into_any().unbind())
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

    /// LangGraph's `get_tuple(config)`: the newest checkpoint of the addressed thread.
    #[pyo3(signature = (thread_id, checkpoint_ns=None, checkpoint_id=None))]
    fn _checkpoint_tuple(
        &self,
        py: Python<'_>,
        thread_id: &str,
        checkpoint_ns: Option<&str>,
        checkpoint_id: Option<&str>,
    ) -> PyResult<Option<Py<PyAny>>> {
        use crate::Checkpointer as _;
        let namespace = checkpoint_ns.unwrap_or("");
        let checkpoint = match checkpoint_id {
            Some(id) => self
                .inner
                .history_in_shared(thread_id, namespace)
                .map_err(|error| PyTypeError::new_err(error.to_string()))?
                .into_iter()
                .find(|checkpoint| checkpoint.id == id),
            None => self
                .inner
                .latest_in_shared(thread_id, namespace)
                .map_err(|error| PyTypeError::new_err(error.to_string()))?,
        };
        checkpoint
            .map(|checkpoint| checkpoint_tuple_to_py(py, checkpoint.as_ref()))
            .transpose()
    }

    /// LangGraph's `list(config)`: the thread history, newest first.
    #[pyo3(signature = (thread_id, checkpoint_ns=None, before=None, limit=None))]
    fn _checkpoint_tuples(
        &self,
        py: Python<'_>,
        thread_id: &str,
        checkpoint_ns: Option<&str>,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        use crate::Checkpointer as _;
        let checkpoint_ns = checkpoint_ns.unwrap_or("");
        let checkpoints = self
            .inner
            .history_page_in_shared(thread_id, checkpoint_ns, before, limit)
            .map_err(|error| PyTypeError::new_err(error.to_string()))?;
        let list = pyo3::types::PyList::empty(py);
        for checkpoint in checkpoints {
            list.append(checkpoint_tuple_to_py(py, checkpoint.as_ref())?)?;
        }
        Ok(list.into_any().unbind())
    }

    /// LangGraph's `delete_thread(thread_id)`.
    fn delete_thread(&self, thread_id: &str) -> PyResult<()> {
        use crate::Checkpointer as _;
        self.inner
            .delete_thread(thread_id)
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }

    /// LangGraph's channel version counter (`"<counter:032x>.<hash>.<timestamp>"`).
    #[pyo3(signature = (_current=None, _channel=None))]
    fn get_next_version(&self, _current: Option<&Bound<'_, PyAny>>, _channel: Option<&str>) -> String {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        use std::time::{SystemTime, UNIX_EPOCH};
        let value = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        format!("{value:032x}.0.{nanos}")
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
pub(crate) fn checkpointer_from_py(
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
    // A custom saver written against LangGraph's `BaseCheckpointSaver` protocol.
    if obj.hasattr("get_tuple").unwrap_or(false) && obj.hasattr("put").unwrap_or(false) {
        return Ok(Some(Arc::new(
            crate::adapters::python::custom_saver::PySaver::new(obj.clone().unbind()),
        ) as Arc<dyn Checkpointer>));
    }
    Err(PyTypeError::new_err(
        "checkpointer must be a bool, one of the built-in savers, or an object implementing \
         LangGraph's `get_tuple`/`put`/`list`/`put_writes` saver protocol",
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
    pub(crate) fn from_function(function: Py<PyAny>, name: Option<String>) -> PyResult<Self> {
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
        if let Some(cached) = crate::runtime::context::task_result_get(&cache_key) {
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
        crate::runtime::context::task_result_put(cache_key, stored.clone());
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
#[path = "store_objects.rs"]
mod store_objects;
pub(crate) use store_objects::*;
