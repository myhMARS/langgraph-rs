//! Saver, store, cache and task objects.

use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::types::{PyAny, PyDict};

use super::*;
#[cfg(feature = "postgres")]
use crate::PostgresSaver;
#[cfg(feature = "redis")]
use crate::RedisSaver;
use crate::adapters::python::convert::{py_to_value, value_to_py};
use crate::kernel::cache::NodeCache;
use crate::kernel::store::{BaseStore, StoreItem};
use crate::{Checkpointer, InMemoryStore};

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
    pub(crate) fn new(store: Arc<dyn BaseStore>) -> Self {
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
pub(crate) fn cache_from_py(
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
                crate::adapters::python::convert::snapshot_to_py(py, &snapshot)
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
                crate::adapters::python::convert::snapshot_to_py(py, &snapshot)
            })
            .collect()
    }
}

/// Resolve a Python `store=` keyword into a kernel store.
pub(crate) fn store_from_py(
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
