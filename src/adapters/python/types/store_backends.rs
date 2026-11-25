//! Redis/Postgres store and cache objects plus LangGraph SQLite readers.

use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::types::PyAny;

use super::*;
use crate::_internal::cache::NodeCache;
use crate::store::BaseStore;

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

    #[pyo3(signature = (namespace, key, value, index=None, ttl=None))]
    fn put(
        &self,
        namespace: Vec<String>,
        key: String,
        value: &Bound<'_, PyAny>,
        index: Option<&Bound<'_, PyAny>>,
        ttl: Option<f64>,
    ) -> PyResult<()> {
        // `index=` only matters for stores built with a vector index.
        let _ = index;
        store_put(self.store.as_ref(), namespace, key, value, ttl)
    }

    fn delete(&self, namespace: Vec<String>, key: String) -> PyResult<()> {
        self.store
            .delete(&namespace, &key)
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (namespace_prefix, query=None, filter=None, limit=10, offset=0, refresh_ttl=None))]
    fn search(
        &self,
        py: Python<'_>,
        namespace_prefix: Vec<String>,
        query: Option<String>,
        filter: Option<&Bound<'_, PyAny>>,
        limit: usize,
        offset: usize,
        refresh_ttl: Option<bool>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let _ = refresh_ttl;
        store_search(py, self.store.as_ref(), namespace_prefix, query, filter, limit, offset)
    }

    #[allow(dead_code)]
    fn search_legacy(
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

    #[pyo3(signature = (prefix=None, suffix=None, max_depth=None, limit=100, offset=0))]
    fn list_namespaces(
        &self,
        py: Python<'_>,
        prefix: Option<Vec<String>>,
        suffix: Option<Vec<String>>,
        max_depth: Option<usize>,
        limit: usize,
        offset: usize,
    ) -> PyResult<Vec<Py<PyAny>>> {
        store_list_namespaces(
            py,
            self.store.as_ref(),
            prefix,
            suffix,
            max_depth,
            limit,
            offset,
        )
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

    #[pyo3(signature = (namespace, key, value, index=None, ttl=None))]
    fn put(
        &self,
        namespace: Vec<String>,
        key: String,
        value: &Bound<'_, PyAny>,
        index: Option<&Bound<'_, PyAny>>,
        ttl: Option<f64>,
    ) -> PyResult<()> {
        // `index=` only matters for stores built with a vector index.
        let _ = index;
        store_put(self.store.as_ref(), namespace, key, value, ttl)
    }

    fn delete(&self, namespace: Vec<String>, key: String) -> PyResult<()> {
        self.store
            .delete(&namespace, &key)
            .map_err(|error| PyTypeError::new_err(error.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (namespace_prefix, query=None, filter=None, limit=10, offset=0, refresh_ttl=None))]
    fn search(
        &self,
        py: Python<'_>,
        namespace_prefix: Vec<String>,
        query: Option<String>,
        filter: Option<&Bound<'_, PyAny>>,
        limit: usize,
        offset: usize,
        refresh_ttl: Option<bool>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let _ = refresh_ttl;
        store_search(py, self.store.as_ref(), namespace_prefix, query, filter, limit, offset)
    }

    #[allow(dead_code)]
    fn search_legacy(
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

    #[pyo3(signature = (prefix=None, suffix=None, max_depth=None, limit=100, offset=0))]
    fn list_namespaces(
        &self,
        py: Python<'_>,
        prefix: Option<Vec<String>>,
        suffix: Option<Vec<String>>,
        max_depth: Option<usize>,
        limit: usize,
        offset: usize,
    ) -> PyResult<Vec<Py<PyAny>>> {
        store_list_namespaces(
            py,
            self.store.as_ref(),
            prefix,
            suffix,
            max_depth,
            limit,
            offset,
        )
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
) -> Option<Arc<dyn crate::_internal::cache::NodeCache>> {
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
    pub(crate) inner: Arc<crate::LangGraphSqliteSaver>,
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
    // A custom store written against LangGraph's `BaseStore` protocol.
    if let Some(store) = crate::adapters::python::custom_store::wrap_py_store(obj) {
        return Ok(Some(store));
    }
    Err(PyTypeError::new_err(
        "store must be an InMemoryStore, Store, SqliteStore, RedisStore, PostgresStore object or \
         any object implementing LangGraph's `get`/`put`/`search` store protocol",
    ))
}
