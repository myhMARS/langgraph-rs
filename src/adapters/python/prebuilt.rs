//! Python bindings for the prebuilt helpers (`tools_condition`, `ToolNode`,
//! `create_react_agent`).

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::convert::*;
use crate::kernel::state::{JsonState, StateSchema};
use crate::prebuilt::{TOOLS, ToolDefinition, ToolNode, tools_condition as rust_tools_condition};
use crate::{END, Result, Route, StateGraph, reducers};

fn tool_definitions(tools: &Bound<'_, PyAny>) -> PyResult<Vec<ToolDefinition>> {
    let mut entries: Vec<(String, Py<PyAny>)> = Vec::new();
    if let Ok(dict) = tools.cast::<PyDict>() {
        for (name, value) in dict.iter() {
            let name: String = name.extract()?;
            if !value.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "tool `{name}` must be callable"
                )));
            }
            entries.push((name, value.unbind()));
        }
    } else {
        let items = tools.try_iter().map_err(|_| {
            PyTypeError::new_err("tools must be a dict or an iterable of callables")
        })?;
        for (index, item) in items.enumerate() {
            let item = item?;
            if !item.is_callable() {
                return Err(PyTypeError::new_err("every tool must be callable"));
            }
            let name = item
                .getattr("__name__")
                .ok()
                .and_then(|name| name.extract::<String>().ok())
                .unwrap_or_else(|| format!("tool_{index}"));
            entries.push((name, item.unbind()));
        }
    }
    Ok(entries
        .into_iter()
        .map(|(name, function)| {
            ToolDefinition::new(name, move |args| {
                Python::attach(|py| -> Result<serde_json::Value> {
                    let py_args =
                        value_to_py(py, &args).map_err(|error| pyerr_to_error(py, error))?;
                    let result = function
                        .call1(py, (py_args,))
                        .map_err(|error| pyerr_to_error(py, error))?;
                    py_to_value(result.bind(py)).map_err(|error| pyerr_to_error(py, error))
                })
            })
        })
        .collect())
}

/// Executes tool calls from the last message (LangGraph `ToolNode`).
#[pyclass(name = "ToolNode")]
pub struct PyToolNode {
    node: ToolNode,
}

#[pymethods]
impl PyToolNode {
    #[new]
    #[pyo3(signature = (tools, messages_key=None))]
    fn new(tools: &Bound<'_, PyAny>, messages_key: Option<String>) -> PyResult<Self> {
        let mut node = ToolNode::new(tool_definitions(tools)?);
        if let Some(key) = messages_key {
            node = node.messages_key(key);
        }
        Ok(Self { node })
    }

    /// Execute the pending tool calls and return message updates.
    fn __call__(&self, py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let state = state_from_py(Some(state))?;
        let output = self
            .node
            .execute(&state)
            .map_err(|error| to_pyerr(py, error))?;
        let dict = PyDict::new(py);
        for (channel, value) in output.updates_map() {
            dict.set_item(channel, value_to_py(py, value)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "ToolNode()".to_string()
    }
}

/// Route to "tools" when the last message has tool calls, END otherwise.
#[pyfunction]
pub(super) fn tools_condition(state: &Bound<'_, PyAny>) -> PyResult<String> {
    let state = state_from_py(Some(state))?;
    let route = rust_tools_condition(&state);
    Ok(route
        .targets()
        .first()
        .cloned()
        .unwrap_or_else(|| END.to_string()))
}

/// Build a minimal ReAct agent from a Python model callable and tools.
#[pyfunction]
#[pyo3(signature = (model, tools, messages_key=None, checkpointer=None))]
pub(super) fn create_react_agent(
    model: Py<PyAny>,
    tools: &Bound<'_, PyAny>,
    messages_key: Option<String>,
    checkpointer: Option<&Bound<'_, PyAny>>,
) -> PyResult<super::graph::PyCompiledGraph> {
    let mut tool_node = ToolNode::new(tool_definitions(tools)?);
    if let Some(key) = messages_key.clone() {
        tool_node = tool_node.messages_key(key);
    }
    let messages_key = messages_key.unwrap_or_else(|| "messages".to_string());
    let schema = StateSchema::dynamic().with_reducer(&messages_key, reducers::messages());
    let inner = StateGraph::<JsonState>::with_schema(schema)
        .add_node("agent", node_from_python("agent".to_string(), model))
        .add_node(TOOLS, move |state: &JsonState| tool_node.execute(state))
        .set_entry_point("agent")
        .add_conditional_edges("agent", move |state: &JsonState| {
            Ok::<Route, crate::Error>(rust_tools_condition(state))
        })
        .add_edge(TOOLS, "agent");
    let mut options = crate::CompileOptions::new();
    if let Some(checkpointer) = checkpointer {
        if let Some(saver) = super::types::checkpointer_from_py(Some(checkpointer))? {
            options = options.with_checkpointer(saver);
        }
    }
    let compiled = inner
        .compile_with(options)
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(super::graph::PyCompiledGraph::from_inner(compiled))
}
