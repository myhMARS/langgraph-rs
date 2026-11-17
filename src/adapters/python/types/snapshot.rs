//! LangGraph-style `StateSnapshot` object that is also dict-compatible.

use pyo3::exceptions::PyAttributeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator};

/// Mapping-like snapshot returned by `get_state` / `get_state_history`.
///
/// Supports both LangGraph attribute access (`snapshot.values`, `snapshot.next`)
/// and mapping access (`snapshot["values"]`) for backwards compatibility.
#[pyclass(name = "StateSnapshot")]
pub struct PyStateSnapshot {
    pub(crate) data: Py<PyDict>,
}

impl PyStateSnapshot {
    pub(crate) fn dict<'py>(&self, py: Python<'py>) -> &Bound<'py, PyDict> {
        self.data.bind(py)
    }
}

#[pymethods]
impl PyStateSnapshot {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let dict = self.data.bind(py);
        match dict.get_item(key)? {
            Some(value) => Ok(value.unbind()),
            None => Err(pyo3::exceptions::PyKeyError::new_err(
                "snapshot key not found",
            )),
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        default: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        match self.data.bind(py).get_item(key)? {
            Some(value) => Ok(value.unbind()),
            None => Ok(default
                .map(|value| value.clone().unbind())
                .unwrap_or_else(|| py.None())),
        }
    }

    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let dict = self.data.bind(py);
        if let Some(value) = dict.get_item(name)? {
            return Ok(value.unbind());
        }
        Err(PyAttributeError::new_err(format!(
            "StateSnapshot has no attribute {name:?}"
        )))
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.data.bind(py).len()
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        PyIterator::from_object(self.data.bind(py))
    }

    fn __contains__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.data.bind(py).contains(key)
    }

    fn keys(&self, py: Python<'_>) -> Py<PyAny> {
        self.data.bind(py).keys().into_any().unbind()
    }

    fn items(&self, py: Python<'_>) -> Py<PyAny> {
        self.data.bind(py).items().into_any().unbind()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(self.data.bind(py).repr()?.to_string())
    }

    /// LangGraph config shape for this snapshot.
    #[getter]
    fn config(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = self.data.bind(py);
        let thread_id = dict.get_item("thread_id")?.map(|value| value.unbind());
        let checkpoint_id = dict.get_item("checkpoint_id")?.map(|value| value.unbind());
        let checkpoint_ns = dict.get_item("checkpoint_ns")?.map(|value| value.unbind());
        let configurable = PyDict::new(py);
        if let Some(thread_id) = thread_id {
            configurable.set_item("thread_id", thread_id)?;
        }
        if let Some(checkpoint_id) = checkpoint_id {
            configurable.set_item("checkpoint_id", checkpoint_id)?;
        }
        if let Some(checkpoint_ns) = checkpoint_ns {
            configurable.set_item("checkpoint_ns", checkpoint_ns)?;
        }
        let config = PyDict::new(py);
        config.set_item("configurable", configurable)?;
        Ok(config.into_any().unbind())
    }
}
