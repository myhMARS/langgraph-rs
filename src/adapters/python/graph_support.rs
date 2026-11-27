//! Shared graph-description and stream-mode helpers.

use pyo3::types::PyAny;
use serde_json::Value;

use super::*;
use crate::adapters::python::convert::value_to_py;

/// Attach child subgraph snapshots to the root snapshot's tasks (`subgraphs=True`).
///
/// LangGraph keys a child namespace by task id (`f"{parent_ns}:{task_id}"`), so the match
/// uses the pending task's id when available and falls back to the node name.
pub(crate) fn embed_subgraph_states(
    py: Python<'_>,
    root: &crate::StateSnapshot<crate::JsonState>,
    subgraphs: &[crate::StateSnapshot<crate::JsonState>],
) -> PyResult<Py<PyAny>> {
    let root_py = snapshot_to_py(py, root)?;
    if let Ok(root_snapshot) = root_py
        .bind(py)
        .extract::<pyo3::PyRef<'_, crate::adapters::python::types::PyStateSnapshot>>()
        && let Ok(Some(tasks)) = root_snapshot.dict(py).get_item("tasks")
        // LangGraph exposes `tasks` as a tuple; older code assumed a list.
        && let Ok(iter) = tasks.try_iter()
    {
        for task in iter {
            let Ok(task) = task else {
                continue;
            };
            let Ok(task_dict) = task.cast::<pyo3::types::PyDict>() else {
                continue;
            };
            let Ok(Some(node)) = task_dict.get_item("node") else {
                continue;
            };
            let Ok(node) = node.extract::<String>() else {
                continue;
            };
            let task_id = task_dict
                .get_item("id")
                .ok()
                .flatten()
                .and_then(|id| id.extract::<String>().ok());
            let child = subgraphs.iter().find(|snapshot| {
                let Some(segment) = snapshot.checkpoint_ns.split('|').next_back() else {
                    return false;
                };
                if let Some(task_id) = task_id.as_deref()
                    && let Some((segment_task, segment_id)) = segment.rsplit_once(':')
                {
                    return segment_id == task_id && segment_task == node;
                }
                segment == node
                    || segment
                        .strip_prefix(&node)
                        .is_some_and(|rest| rest.starts_with(':'))
            });
            if let Some(child) = child {
                task_dict.set_item("state", snapshot_to_py(py, child)?)?;
            }
        }
    }
    Ok(root_py)
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




