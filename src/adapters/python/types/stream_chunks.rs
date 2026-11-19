//! Iterable return value for `CompiledGraph.stream`.

use pyo3::prelude::*;
use pyo3::types::PyList;

/// Streamed chunks of one `stream(...)` call.
///
/// LangGraph returns a generator; this object supports the same protocol (`for`, `next()`,
/// `iter()`) and additionally keeps `len()` and indexing for existing `langgraph_rs` code.
#[pyclass(name = "StreamChunks")]
pub struct PyStreamChunks {
    items: Vec<Py<PyAny>>,
    index: usize,
}

impl PyStreamChunks {
    pub(crate) fn new(items: Vec<Py<PyAny>>) -> Self {
        Self { items, index: 0 }
    }
}

#[pymethods]
impl PyStreamChunks {
    /// Return a fresh iterator over the chunks.
    ///
    /// LangGraph's generator is single-use; handing back a new iterator keeps `for` loops
    /// repeatable (what a list-based `langgraph_rs` return value allowed) while `next()`
    /// works on the object itself.
    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyStreamChunks>> {
        Py::new(
            py,
            PyStreamChunks {
                items: self
                    .items
                    .iter()
                    .map(|item| item.clone_ref(py))
                    .collect(),
                index: 0,
            },
        )
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Py<PyAny>> {
        let py = slf.py();
        let item = slf.items.get(slf.index).map(|item| item.clone_ref(py));
        if item.is_some() {
            slf.index += 1;
        }
        item
    }

    fn __len__(&self) -> usize {
        self.items.len()
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
        let len = self.items.len() as isize;
        let resolved = if index < 0 { len + index } else { index };
        if resolved < 0 || resolved >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "stream chunk index out of range",
            ));
        }
        Ok(self.items[resolved as usize].clone_ref(py))
    }

    fn __repr__(&self) -> String {
        format!("StreamChunks({} chunks)", self.items.len())
    }

    /// Materialize every chunk into a list.
    fn to_list(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list = PyList::new(py, self.items.iter().map(|item| item.clone_ref(py)))?;
        Ok(list.into_any().unbind())
    }
}
