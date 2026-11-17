//! Shared graph-description and stream-mode helpers.

use pyo3::types::{PyAny, PyDict};
use serde_json::Value;

use super::*;
use crate::adapters::python::convert::value_to_py;

pub(crate) fn global_step_for(
    history: &[crate::StateSnapshot<crate::JsonState>],
    checkpoint_id: &str,
) -> Option<i64> {
    let total = history.len() as i64;
    let index = history
        .iter()
        .position(|snapshot| snapshot.checkpoint_id == checkpoint_id)? as i64;
    Some(total - 2 - index)
}

pub(crate) fn snapshot_to_py_with_step(
    py: Python<'_>,
    snapshot: &crate::StateSnapshot<crate::JsonState>,
    step: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let value = snapshot_to_py(py, snapshot)?;
    let Some(step) = step else {
        return Ok(value);
    };
    if let Ok(snapshot) = value
        .bind(py)
        .extract::<pyo3::PyRef<'_, crate::adapters::python::types::PyStateSnapshot>>()
    {
        let dict = snapshot.dict(py);
        dict.set_item("step", step)?;
        if let Some(metadata) = dict.get_item("metadata")? {
            if let Ok(metadata) = metadata.cast::<PyDict>() {
                metadata.set_item("step", step)?;
            }
        }
        return Ok(value);
    }
    let dict = value.bind(py).cast::<PyDict>()?;
    dict.set_item("step", step)?;
    if let Some(metadata) = dict.get_item("metadata")? {
        if let Ok(metadata) = metadata.cast::<PyDict>() {
            metadata.set_item("step", step)?;
        }
    }
    Ok(value)
}

/// Graph description object returned by `CompiledGraph.get_graph()`.
#[pyclass(name = "Graph")]
pub struct PyGraph {
    pub(crate) mermaid: String,
    pub(crate) ascii: String,
    pub(crate) spec: Value,
}

#[pymethods]
impl PyGraph {
    /// Mermaid flowchart source.
    fn draw_mermaid(&self) -> String {
        self.mermaid.clone()
    }

    /// Compact ASCII map of the graph.
    fn draw_ascii(&self) -> String {
        self.ascii.clone()
    }

    /// Serializable graph description (nodes, edges, conditional edges).
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_py(py, &self.spec)
    }

    fn __repr__(&self) -> String {
        "Graph()".to_string()
    }
}

pub(crate) fn stream_modes_from_py(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<crate::StreamMode>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let names: Vec<String> = if let Ok(name) = value.extract::<String>() {
        vec![name]
    } else {
        value.extract::<Vec<String>>().map_err(|_| {
            PyTypeError::new_err("stream_mode must be a string or a list of strings")
        })?
    };
    let mut modes = Vec::with_capacity(names.len());
    for name in names {
        let mode = crate::StreamMode::parse(&name)
            .ok_or_else(|| PyTypeError::new_err(format!("unknown stream_mode `{name}`")))?;
        modes.push(mode);
    }
    Ok(Some(modes))
}
