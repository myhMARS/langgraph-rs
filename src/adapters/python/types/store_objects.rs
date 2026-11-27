//! Store and cache Python objects.

use std::sync::Arc;

use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::types::{PyAny, PyDict, PyTuple};

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

    /// LangGraph's spelling: namespaces as tuples.
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

    fn __repr__(&self) -> String {
        "Store()".to_string()
    }
}

/// `store.put(...)`: shared by every store backend, including the optional TTL.
pub(crate) fn store_put(
    store: &dyn BaseStore,
    namespace: Vec<String>,
    key: String,
    value: &Bound<'_, PyAny>,
    ttl: Option<f64>,
) -> PyResult<()> {
    store
        .put_with_ttl(&namespace, &key, py_to_value(value)?, ttl)
        .map_err(|error| PyTypeError::new_err(error.to_string()))
}

/// `index=` of `put(...)`: `False` -> index nothing, `None` -> the store default, list -> paths.
pub(crate) fn index_paths_from_py(index: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<String>>> {
    let Some(index) = index.filter(|value| !value.is_none()) else {
        return Ok(None);
    };
    if let Ok(flag) = index.extract::<bool>() {
        return Ok(if flag { None } else { Some(Vec::new()) });
    }
    let paths = index
        .extract::<Vec<String>>()
        .map_err(|error| PyTypeError::new_err(format!("put(index=...) must be False, None or a list of paths: {error}")))?;
    Ok(Some(paths))
}

/// Parse `InMemoryStore(index={...})` into an engine index configuration.
pub(crate) fn index_config_from_py(index: &Bound<'_, PyAny>) -> PyResult<crate::store::IndexConfig> {
    let dict = index
        .cast::<PyDict>()
        .map_err(|error| PyTypeError::new_err(format!("index must be a dict: {error}")))?;
    let dims = dict
        .get_item("dims")?
        .map(|value| value.extract::<usize>())
        .transpose()?
        .unwrap_or(0);
    let embed = dict
        .get_item("embed")?
        .ok_or_else(|| PyTypeError::new_err("index must include an `embed` embeddings object or callable"))?;
    let fields = dict
        .get_item("fields")?
        .filter(|value| !value.is_none())
        .map(|value| value.extract::<Vec<String>>())
        .transpose()?;
    Ok(crate::store::IndexConfig {
        dims,
        embed: Arc::new(super::store_embedder::PyEmbedder::new(
            embed.clone().unbind(),
        )),
        fields,
    })
}
/// `store.search(...)`: LangGraph's `query` / `filter` / `limit` / `offset`.
pub(crate) fn store_search(
    py: Python<'_>,
    store: &dyn BaseStore,
    namespace_prefix: Vec<String>,
    query: Option<String>,
    filter: Option<&Bound<'_, PyAny>>,
    limit: usize,
    offset: usize,
) -> PyResult<Vec<Py<PyAny>>> {
    let filter = match filter {
        Some(filter) if !filter.is_none() => {
            let value = py_to_value(filter)?;
            match value {
                serde_json::Value::Object(map) => Some(map),
                _ => {
                    return Err(PyTypeError::new_err(
                        "store.search(filter=...) must be a dict",
                    ));
                }
            }
        }
        _ => None,
    };
    let options = crate::store::SearchOptions {
        query,
        filter,
        limit,
        offset,
    };
    let hits = store
        .search_with(&namespace_prefix, &options)
        .map_err(|error| PyTypeError::new_err(error.to_string()))?;
    hits.into_iter()
        .map(|hit| {
            let entry = Py::new(
                py,
                PyStoreItem {
                    namespace: hit.item.namespace.clone(),
                    key: hit.item.key.clone(),
                    value: value_to_py(py, &hit.item.value)?,
                    created_at_millis: hit.item.created_at_millis,
                    updated_at_millis: hit.item.updated_at_millis,
                    score: hit.score,
                },
            )?;
            Ok(entry.into_any())
        })
        .collect()
}

/// `store.list_namespaces(...)`: prefix / suffix / depth filters plus pagination.
pub(crate) fn store_list_namespaces(
    py: Python<'_>,
    store: &dyn BaseStore,
    prefix: Option<Vec<String>>,
    suffix: Option<Vec<String>>,
    max_depth: Option<usize>,
    limit: usize,
    offset: usize,
) -> PyResult<Vec<Py<PyAny>>> {
    let options = crate::store::NamespaceOptions {
        prefix,
        suffix,
        max_depth,
        limit,
        offset,
    };
    let namespaces = store
        .list_namespaces(&options)
        .map_err(|error| PyTypeError::new_err(error.to_string()))?;
    namespaces
        .into_iter()
        .map(|namespace| Ok(pyo3::types::PyTuple::new(py, namespace)?.into_any().unbind()))
        .collect()
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
/// LangGraph's `Item`: attribute access plus tuple-style unpacking.
#[pyclass(name = "StoreItem")]
pub struct PyStoreItem {
    /// LangGraph exposes the namespace as a tuple.
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
    fn namespace(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, self.namespace.clone())?.into_any().unbind())
    }

    #[getter]
    fn created_at(&self, py: Python<'_>) -> Py<PyAny> {
        datetime_from_millis(py, self.created_at_millis)
    }

    #[getter]
    fn updated_at(&self, py: Python<'_>) -> Py<PyAny> {
        datetime_from_millis(py, self.updated_at_millis)
    }

    fn __len__(&self) -> usize {
        5
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(name) = key.extract::<String>() {
            return match name.as_str() {
                "namespace" => Ok(PyTuple::new(py, self.namespace.clone())?.into_any().unbind()),
                "key" => Ok(self.key.clone().into_py_any(py)?),
                "value" => Ok(self.value.clone_ref(py)),
                "created_at" => Ok(self.created_at(py)),
                "updated_at" => Ok(self.updated_at(py)),
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
            0 => PyTuple::new(py, self.namespace.clone())?.into_any().unbind(),
            1 => self.key.clone().into_py_any(py)?,
            2 => self.value.clone_ref(py),
            3 => self.created_at(py),
            4 => self.updated_at(py),
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
            self.created_at(py).bind(py).str().map(|text| text.to_string()).unwrap_or_default(),
            self.updated_at(py).bind(py).str().map(|text| text.to_string()).unwrap_or_default(),
            self.score
        ))
    }
}

/// `datetime` for a millisecond timestamp (LangGraph reports datetimes on `Item`).
fn datetime_from_millis(py: Python<'_>, millis: u64) -> Py<PyAny> {
    let build = || -> pyo3::PyResult<Py<PyAny>> {
        let module = py.import("datetime")?;
        let class = module.getattr("datetime")?;
        let seconds = millis / 1000;
        let micros = (millis % 1000) * 1000;
        // LangGraph reports timezone-aware UTC timestamps.
        let utc = module.getattr("timezone")?.getattr("utc")?;
        let value = class.call_method1("fromtimestamp", (seconds, utc))?;
        let value = if micros > 0 {
            let delta = module.getattr("timedelta")?.call1((0, 0, micros))?;
            value.call_method1("__add__", (delta,))?
        } else {
            value
        };
        Ok(value.unbind())
    };
    build().unwrap_or_else(|_| py.None())
}

/// In-memory long-term memory store, mirroring `langgraph.store.memory.InMemoryStore`.
#[pyclass(name = "InMemoryStore", subclass)]
pub struct PyInMemoryStore {
    pub(crate) store: Arc<InMemoryStore>,
}

#[pymethods]
impl PyInMemoryStore {
    #[new]
    #[pyo3(signature = (index=None))]
    fn new(index: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut store = InMemoryStore::new();
        if let Some(index) = index.filter(|value| !value.is_none()) {
            let config = index_config_from_py(index)?;
            store = store.with_index(config);
        }
        Ok(Self {
            store: Arc::new(store),
        })
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

    #[pyo3(signature = (namespace, key, value, index=None, ttl=None))]
    fn put(
        &self,
        namespace: Vec<String>,
        key: String,
        value: &Bound<'_, PyAny>,
        index: Option<&Bound<'_, PyAny>>,
        ttl: Option<f64>,
    ) -> PyResult<()> {
        let paths = index_paths_from_py(index)?;
        self.store
            .put_indexed(&namespace, &key, py_to_value(value)?, paths.as_deref(), ttl)
            .map_err(|error| PyTypeError::new_err(error.to_string()))
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

    /// LangGraph's spelling: namespaces as tuples.
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

    /// LangGraph's spelling: namespaces as tuples.
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
