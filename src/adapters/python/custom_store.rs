//! Bridge for long-term stores written in Python (LangGraph's `BaseStore`).
//!
//! LangGraph's protocol is `get(namespace, key)` / `put(namespace, key, value, index=None)` /
//! `delete(namespace, key)` / `search(namespace_prefix, query=..., limit=..., offset=...)` /
//! `list_namespaces(...)`, with namespaces as tuples of strings and items carrying `value` (plus
//! optional timestamps). The engine only needs the value, so missing timestamps fall back to
//! "now".

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::_internal::error::{Error, Result};
use crate::store::{BaseStore, StoreItem};
use crate::Value;

/// A store implemented by a Python object following LangGraph's `BaseStore` protocol.
pub struct PyStore {
    object: Py<PyAny>,
    name: String,
}

impl PyStore {
    pub fn new(object: Py<PyAny>) -> Self {
        let name = Python::attach(|py| {
            object
                .bind(py)
                .get_type()
                .name()
                .map(|name| name.to_string())
                .unwrap_or_else(|_| "custom store".to_string())
        });
        Self { object, name }
    }

    fn namespace(&self, parts: &[String]) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let tuple = PyTuple::new(py, parts.iter())?;
            Ok(tuple.into_any().unbind())
        })
    }

    fn store_error(&self, method: &str, error: PyErr) -> Error {
        Error::other(format!("{} store `{method}` failed: {error}", self.name))
    }

    /// Read the value out of whatever the store returned for an item.
    fn item_value(object: &Bound<'_, PyAny>) -> Option<Value> {
        if object.is_none() {
            return None;
        }
        // LangGraph `Item`s expose `.value`; plain dicts use a "value" key.
        if let Ok(value) = object.getattr("value") {
            return crate::adapters::python::convert::py_to_value(&value).ok();
        }
        if let Ok(dict) = object.cast::<PyDict>()
            && let Ok(Some(value)) = dict.get_item("value")
        {
            return crate::adapters::python::convert::py_to_value(&value).ok();
        }
        crate::adapters::python::convert::py_to_value(object).ok()
    }
}

impl BaseStore for PyStore {
    fn get(&self, namespace: &[String], key: &str) -> Result<Option<Value>> {
        Python::attach(|py| {
            let call = self
                .object
                .bind(py)
                .call_method1("get", (self.namespace(namespace).map_err(|error| self.store_error("get", error))?, key))
                .map_err(|error| self.store_error("get", error))?;
            Ok(Self::item_value(&call))
        })
    }

    fn put(&self, namespace: &[String], key: &str, value: Value) -> Result<()> {
        Python::attach(|py| {
            let namespace = self
                .namespace(namespace)
                .map_err(|error| self.store_error("put", error))?;
            let value = crate::adapters::python::convert::value_to_py(py, &value)
                .map_err(|error| self.store_error("put", error))?;
            self.object
                .bind(py)
                .call_method1("put", (namespace, key, value))
                .map_err(|error| self.store_error("put", error))?;
            Ok(())
        })
    }

    fn delete(&self, namespace: &[String], key: &str) -> Result<()> {
        Python::attach(|py| {
            let namespace = self
                .namespace(namespace)
                .map_err(|error| self.store_error("delete", error))?;
            self.object
                .bind(py)
                .call_method1("delete", (namespace, key))
                .map_err(|error| self.store_error("delete", error))?;
            Ok(())
        })
    }

    fn search(
        &self,
        namespace_prefix: &[String],
        query: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StoreItem>> {
        Python::attach(|py| {
            let prefix = self
                .namespace(namespace_prefix)
                .map_err(|error| self.store_error("search", error))?;
            let kwargs = PyDict::new(py);
            if let Some(query) = query {
                kwargs
                    .set_item("query", query)
                    .map_err(|error| self.store_error("search", error))?;
            }
            if let Some(limit) = limit {
                kwargs
                    .set_item("limit", limit)
                    .map_err(|error| self.store_error("search", error))?;
            }
            let results = self
                .object
                .bind(py)
                .call_method("search", (prefix,), Some(&kwargs))
                .map_err(|error| self.store_error("search", error))?;
            let mut items = Vec::new();
            for entry in results
                .try_iter()
                .map_err(|error| self.store_error("search", error))?
            {
                let entry = entry.map_err(|error| self.store_error("search", error))?;
                let Some(value) = Self::item_value(&entry) else {
                    continue;
                };
                let key = entry
                    .getattr("key")
                    .ok()
                    .and_then(|key| key.extract::<String>().ok())
                    .or_else(|| {
                        entry
                            .cast::<PyDict>()
                            .ok()
                            .and_then(|dict| dict.get_item("key").ok().flatten())
                            .and_then(|key| key.extract::<String>().ok())
                    })
                    .unwrap_or_default();
                let namespace = entry
                    .getattr("namespace")
                    .ok()
                    .and_then(|namespace| {
                        namespace
                            .extract::<Vec<String>>()
                            .ok()
                            .or_else(|| namespace.extract::<String>().ok().map(|one| vec![one]))
                    })
                    .unwrap_or_else(|| namespace_prefix.to_vec());
                let now = crate::store::store_now();
                items.push(StoreItem {
                    namespace,
                    key,
                    value,
                    created_at_millis: now,
                    updated_at_millis: now,
                    expires_at_millis: None,
                });
            }
            Ok(items)
        })
    }

    fn namespaces(&self) -> Result<Vec<Vec<String>>> {
        Python::attach(|py| {
            let object = self.object.bind(py);
            let call = match object.getattr("list_namespaces") {
                Ok(callable) => callable.call0(),
                Err(_) => object.call_method0("namespaces"),
            }
            .map_err(|error| self.store_error("list_namespaces", error))?;
            let mut namespaces = Vec::new();
            for entry in call
                .try_iter()
                .map_err(|error| self.store_error("list_namespaces", error))?
            {
                let entry = entry.map_err(|error| self.store_error("list_namespaces", error))?;
                if let Ok(parts) = entry.extract::<Vec<String>>() {
                    namespaces.push(parts);
                }
            }
            Ok(namespaces)
        })
    }
}

/// Wrap a foreign Python store object, when it looks like one.
pub(crate) fn wrap_py_store(object: &Bound<'_, PyAny>) -> Option<Arc<dyn BaseStore>> {
    let looks_like_store = object.hasattr("get").unwrap_or(false)
        && object.hasattr("put").unwrap_or(false)
        && object.hasattr("search").unwrap_or(false);
    looks_like_store.then(|| Arc::new(PyStore::new(object.clone().unbind())) as Arc<dyn BaseStore>)
}