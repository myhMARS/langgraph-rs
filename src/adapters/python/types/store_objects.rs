//! Store and cache Python objects.

use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::types::{PyAny, PyDict};

use super::*;
use crate::InMemoryStore;
use crate::adapters::python::convert::{py_to_value, value_to_py};
use crate::kernel::cache::NodeCache;
use crate::kernel::store::BaseStore;

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
#[path = "store_backends.rs"]
mod store_backends;
pub(crate) use store_backends::*;
