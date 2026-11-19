//! Store and cache Python objects.

use std::sync::Arc;

use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::types::{PyAny, PyDict};

use super::*;
use crate::_internal::cache::NodeCache;
use crate::InMemoryStore;
use crate::adapters::python::convert::{py_to_value, value_to_py};
use crate::store::BaseStore;

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
        store_item_by_key(py, self.store.as_ref(), &namespace, &key)
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

    /// LangGraph's spelling: namespaces as tuples.
    fn list_namespaces(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        namespaces_to_py(py, self.store.as_ref())
    }

    fn __repr__(&self) -> String {
        "Store()".to_string()
    }
}

fn store_item_to_py(py: Python<'_>, item: &StoreItem) -> PyResult<Py<PyAny>> {
    let entry = Py::new(
        py,
        PyStoreItem {
            namespace: item.namespace.clone(),
            key: item.key.clone(),
            value: value_to_py(py, &item.value)?,
            created_at_millis: item.created_at_millis,
            updated_at_millis: item.updated_at_millis,
            score: None,
        },
    )?;
    Ok(entry.into_any())
}

/// Read one item by key, wrapped in LangGraph's `Item` shape.
fn store_item_by_key(
    py: Python<'_>,
    store: &dyn BaseStore,
    namespace: &[String],
    key: &str,
) -> PyResult<Option<Py<PyAny>>> {
    let value = store
        .get(namespace, key)
        .map_err(|error| PyTypeError::new_err(error.to_string()))?;
    let now = crate::store::store_now();
    value
        .map(|value| {
            let entry = Py::new(
                py,
                PyStoreItem {
                    namespace: namespace.to_vec(),
                    key: key.to_string(),
                    value: value_to_py(py, &value)?,
                    created_at_millis: now,
                    updated_at_millis: now,
                    score: None,
                },
            )?;
            Ok(entry.into_any())
        })
        .transpose()
}

fn namespaces_to_py(py: Python<'_>, store: &dyn BaseStore) -> PyResult<Vec<Py<PyAny>>> {
    let namespaces = store
        .namespaces()
        .map_err(|error| PyTypeError::new_err(error.to_string()))?;
    namespaces
        .into_iter()
        .map(|namespace| {
            Ok(pyo3::types::PyTuple::new(py, namespace)?.into_any().unbind())
        })
        .collect()
}

/// LangGraph's `Item`: attribute access plus tuple-style unpacking.
#[pyclass(name = "StoreItem")]
pub struct PyStoreItem {
    #[pyo3(get)]
    pub namespace: Vec<String>,
    #[pyo3(get)]
    pub key: String,
    #[pyo3(get)]
    pub value: Py<PyAny>,
    #[pyo3(get)]
    pub created_at_millis: u64,
    #[pyo3(get)]
    pub updated_at_millis: u64,
    #[pyo3(get)]
    pub score: Option<f64>,
}

#[pymethods]
impl PyStoreItem {
    #[getter]
    fn created_at(&self) -> String {
        crate::checkpoint::iso_timestamp(self.created_at_millis)
    }

    #[getter]
    fn updated_at(&self) -> String {
        crate::checkpoint::iso_timestamp(self.updated_at_millis)
    }

    fn __len__(&self) -> usize {
        5
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(name) = key.extract::<String>() {
            return match name.as_str() {
                "namespace" => Ok(pyo3::types::PyList::new(py, self.namespace.clone())?.into_any().unbind()),
                "key" => Ok(self.key.clone().into_py_any(py)?),
                "value" => Ok(self.value.clone_ref(py)),
                "created_at" => Ok(self.created_at().into_py_any(py)?),
                "updated_at" => Ok(self.updated_at().into_py_any(py)?),
                "created_at_millis" => Ok(self.created_at_millis.into_py_any(py)?),
                "updated_at_millis" => Ok(self.updated_at_millis.into_py_any(py)?),
                "score" => Ok(self.score.into_py_any(py)?),
                other => {
                    // Legacy `langgraph_rs` code indexed into the stored value directly.
                    if let Ok(dict) = self.value.bind(py).cast::<PyDict>()
                        && let Some(found) = dict.get_item(other)?
                    {
                        return Ok(found.unbind());
                    }
                    Err(PyKeyError::new_err(other.to_string()))
                }
            };
        }
        let index = key.extract::<isize>()?;
        let field = match index.rem_euclid(5) {
            0 => pyo3::types::PyList::new(py, self.namespace.clone())?.into_any().unbind(),
            1 => self.key.clone().into_py_any(py)?,
            2 => self.value.clone_ref(py),
            3 => self.created_at().into_py_any(py)?,
            4 => self.updated_at().into_py_any(py)?,
            _ => unreachable!(),
        };
        Ok(field)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let namespace: Vec<String> = self.namespace.clone();
        let value = self.value.bind(py).repr()?.to_string();
        Ok(format!(
            "Item(namespace={namespace:?}, key={:?}, value={value}, created_at={:?}, updated_at={:?}, score={:?})",
            self.key,
            self.created_at(),
            self.updated_at(),
            self.score
        ))
    }
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
        store_item_by_key(py, self.store.as_ref(), &namespace, &key)
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

    /// LangGraph's spelling: namespaces as tuples.
    fn list_namespaces(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        namespaces_to_py(py, self.store.as_ref())
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
        store_item_by_key(py, self.store.as_ref(), &namespace, &key)
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

    /// LangGraph's spelling: namespaces as tuples.
    fn list_namespaces(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        namespaces_to_py(py, self.store.as_ref())
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

#[path = "store_backends.rs"]
mod store_backends;
pub(crate) use store_backends::*;
