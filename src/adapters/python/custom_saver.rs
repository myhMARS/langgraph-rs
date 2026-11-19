//! Bridge for checkpointers written in Python (LangGraph's `BaseCheckpointSaver`).
//!
//! LangGraph's protocol is `put(config, checkpoint, metadata, new_versions)` /
//! `put_writes(config, writes, task_id, task_path)` / `get_tuple(config)` /
//! `list(config, ...)`, and its checkpoint payload is a dict. The Rust engine keeps extra
//! scheduling information (pending `next` tasks, deferred routes, goto sources, task results)
//! that LangGraph stores elsewhere, so those fields travel in
//! `metadata["langgraph_rs"]` and are restored on read. A saver that drops unknown metadata
//! keys cannot be used with this engine -- the error says so explicitly.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::_internal::error::{Error, Result};
use crate::_internal::ports::Checkpointer;
use crate::checkpoint::{Checkpoint, PendingWrite};
use crate::Value;

/// Metadata key carrying the fields LangGraph has no slot for.
const PRIVATE_KEY: &str = "langgraph_rs";

/// A checkpointer implemented by a Python object following LangGraph's saver protocol.
pub struct PySaver {
    object: Py<PyAny>,
}

impl PySaver {
    pub fn new(object: Py<PyAny>) -> Self {
        Self { object }
    }


}

fn config_for<'py>(
    py: Python<'py>,
    thread_id: &str,
    namespace: &str,
    checkpoint_id: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let configurable = PyDict::new(py);
    configurable.set_item("thread_id", thread_id)?;
    // LangGraph savers read `checkpoint_ns` directly, so it must always be present (empty for
    // the root graph) even when the checkpoint has no id yet.
    configurable.set_item("checkpoint_ns", namespace)?;
    if let Some(id) = checkpoint_id {
        configurable.set_item("checkpoint_id", id)?;
    }
    let config = PyDict::new(py);
    config.set_item("configurable", configurable)?;
    Ok(config)
}

fn checkpoint_to_py(py: Python<'_>, checkpoint: &Checkpoint) -> PyResult<Py<PyAny>> {
    let payload = PyDict::new(py);
    payload.set_item("v", 1)?;
    payload.set_item("id", &checkpoint.id)?;
    payload.set_item("ts", checkpoint_timestamp(checkpoint))?;
    payload.set_item("channel_values", value_to_py(py, &Value::Object(checkpoint.values.clone()))?)?;
    payload.set_item(
        "channel_versions",
        value_to_py(py, &Value::Object(checkpoint.channel_versions.clone()))?,
    )?;
    let seen = PyDict::new(py);
    for (node, versions) in &checkpoint.versions_seen {
        seen.set_item(node, value_to_py(py, versions)?)?;
    }
    payload.set_item("versions_seen", seen)?;
    payload.set_item("updated_channels", PyList::empty(py))?;
    Ok(payload.into_any().unbind())
}

fn checkpoint_timestamp(checkpoint: &Checkpoint) -> String {
    let millis = checkpoint.created_at_millis;
    let seconds = millis / 1000;
    let sub = millis % 1000;
    format!("{seconds}.{sub:03}")
}

/// Everything LangGraph's checkpoint dict cannot hold.
fn private_metadata(py: Python<'_>, checkpoint: &Checkpoint) -> PyResult<Py<PyAny>> {
    let extras = serde_json::json!({
        "next": checkpoint.next,
        "deferred_routes": checkpoint.deferred_routes,
        "resume_tasks": checkpoint.resume_tasks,
        "goto_sources": checkpoint.goto_sources,
        "completed_tasks": checkpoint.completed_tasks,
        "interrupted": checkpoint.interrupted,
        "paused_before": checkpoint.paused_before,
        "parent_id": checkpoint.parent_id,
        "created_at_millis": checkpoint.created_at_millis,
        "interrupts": checkpoint.interrupts,
        "task_results": checkpoint.task_results,
        "pending_writes": checkpoint.pending_writes,
        "deferred": checkpoint.deferred,
        "namespace": checkpoint.namespace,
        "thread_id": checkpoint.thread_id,
        "step": checkpoint.step,
    });
    value_to_py(py, &extras)
}

fn metadata_to_py(py: Python<'_>, checkpoint: &Checkpoint) -> PyResult<Py<PyAny>> {
    let metadata = PyDict::new(py);
    let source = checkpoint
        .metadata
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or("loop");
    metadata.set_item("source", source)?;
    metadata.set_item("step", checkpoint.step)?;
    metadata.set_item("parents", PyDict::new(py))?;
    metadata.set_item(PRIVATE_KEY, private_metadata(py, checkpoint)?)?;
    Ok(metadata.into_any().unbind())
}

fn checkpoint_from_py(py: Python<'_>, tuple: &Bound<'_, PyAny>, thread: &str) -> Result<Checkpoint> {
    let checkpoint_obj = tuple
        .get_item(1)
        .map_err(|error| saver_error(py, "get_tuple", error))?;
    let metadata_obj = tuple
        .get_item(2)
        .map_err(|error| saver_error(py, "get_tuple", error))?;
    let payload = py_to_json(&checkpoint_obj)?;
    let metadata = py_to_json(&metadata_obj)?;
    let extras = metadata
        .get(PRIVATE_KEY)
        .cloned()
        .ok_or_else(|| {
            Error::Checkpoint(
                "the Python saver dropped the metadata `langgraph_rs` key, which carries the \
                 scheduling state this engine needs; use a saver that stores metadata verbatim"
                    .to_string(),
            )
        })?;
    let field = |name: &str| extras.get(name).cloned().unwrap_or(Value::Null);
    let parse = |name: &str| -> Result<Value> { Ok(field(name)) };
    let to_vec = |name: &str| -> Result<Vec<Value>> {
        Ok(field(name).as_array().cloned().unwrap_or_default())
    };
    let to_map = |name: &str| -> Result<serde_json::Map<String, Value>> {
        Ok(field(name).as_object().cloned().unwrap_or_default())
    };
    let queue = |name: &str| -> Result<Vec<crate::_internal::node::Task>> {
        let items = to_vec(name)?;
        serde_json::from_value(Value::Array(items))
            .map_err(|error| Error::Checkpoint(format!("saved `{name}` is invalid: {error}")))
    };

    Ok(Checkpoint {
        id: payload
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        thread_id: field("thread_id")
            .as_str()
            .unwrap_or(thread)
            .to_string(),
        step: field("step")
            .as_u64()
            .map(|step| step as usize)
            .or_else(|| metadata.get("step").and_then(Value::as_u64).map(|s| s as usize))
            .unwrap_or_default(),
        values: payload
            .get("channel_values")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default(),
        next: queue("next")?,
        deferred_routes: queue("deferred_routes")?,
        resume_tasks: queue("resume_tasks")?,
        goto_sources: to_vec("goto_sources")?
            .iter()
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect(),
        completed_tasks: queue("completed_tasks")?,
        interrupted: field("interrupted").as_bool().unwrap_or(false),
        paused_before: field("paused_before").as_bool().unwrap_or(false),
        parent_id: field("parent_id").as_str().map(str::to_string),
        created_at_millis: field("created_at_millis").as_u64().unwrap_or_default(),
        interrupts: serde_json::from_value(parse("interrupts")?).unwrap_or_default(),
        metadata: metadata
            .as_object()
            .cloned()
            .map(|mut map| {
                map.remove(PRIVATE_KEY);
                map
            })
            .unwrap_or_default(),
        namespace: field("namespace").as_str().unwrap_or_default().to_string(),
        task_results: to_map("task_results")?,
        channel_versions: payload
            .get("channel_versions")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default(),
        versions_seen: payload
            .get("versions_seen")
            .and_then(Value::as_object)
            .map(|map| {
                map.iter()
                    .map(|(node, versions)| (node.clone(), Arc::new(versions.clone())))
                    .collect()
            })
            .unwrap_or_default(),
        updated_channels: Vec::new(),
        pending_writes: serde_json::from_value(parse("pending_writes")?).unwrap_or_default(),
        deferred: queue("deferred")?,
    })
}

fn saver_error(_py: Python<'_>, method: &str, error: PyErr) -> Error {
    Error::Checkpoint(format!("saved tuples returned by `{method}` are malformed: {error}"))
}

fn py_to_json(obj: &Bound<'_, PyAny>) -> Result<Value> {
    crate::adapters::python::convert::py_to_value(obj)
        .map_err(|error| Error::Checkpoint(format!("cannot read saver payload: {error}")))
}

fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    crate::adapters::python::convert::value_to_py(py, value)
}

impl Checkpointer for PySaver {
    fn put(&self, checkpoint: Checkpoint) -> Result<()> {
        Python::attach(|py| {
            let config = config_for(
                py,
                &checkpoint.thread_id,
                &checkpoint.namespace,
                Some(&checkpoint.id),
            )
            .map_err(|error| Error::Checkpoint(error.to_string()))?;
            let payload = checkpoint_to_py(py, &checkpoint)
                .map_err(|error| Error::Checkpoint(error.to_string()))?;
            let metadata = metadata_to_py(py, &checkpoint)
                .map_err(|error| Error::Checkpoint(error.to_string()))?;
            let versions = value_to_py(py, &Value::Object(checkpoint.channel_versions.clone()))
                .map_err(|error| Error::Checkpoint(error.to_string()))?;
            let put = self
                .object
                .bind(py)
                .getattr("put")
                .map_err(|error| saver_error(py, "put", error))?;
            put.call1((config, payload, metadata, versions))
                .map_err(|error| saver_error(py, "put", error))?;
            for write in &checkpoint.pending_writes {
                self.put_write(py, &checkpoint, write)?;
            }
            Ok(())
        })
    }

    fn get(&self, thread_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        Python::attach(|py| {
            let config = config_for(py, thread_id, "", Some(checkpoint_id))
                .map_err(|error| Error::Checkpoint(error.to_string()))?;
            let tuple = self
                .object
                .bind(py)
                .getattr("get_tuple")
                .and_then(|callable| callable.call1((config,)))
                .map_err(|error| saver_error(py, "get_tuple", error))?;
            if tuple.is_none() {
                return Ok(None);
            }
            checkpoint_from_py(py, &tuple, thread_id).map(Some)
        })
    }

    fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        self.history(thread_id).map(|mut all| all.drain(..).next())
    }

    fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        Python::attach(|py| {
            let config = config_for(py, thread_id, "", None)
                .map_err(|error| Error::Checkpoint(error.to_string()))?;
            let iterator = self
                .object
                .bind(py)
                .getattr("list")
                .and_then(|callable| callable.call1((config,)))
                .map_err(|error| saver_error(py, "list", error))?;
            let mut out = Vec::new();
            for item in iterator
                .try_iter()
                .map_err(|error| saver_error(py, "list", error))?
            {
                let item = item.map_err(|error| saver_error(py, "list", error))?;
                out.push(checkpoint_from_py(py, &item, thread_id)?);
            }
            Ok(out)
        })
    }

    fn is_in_memory(&self) -> bool {
        false
    }
}

impl PySaver {
    fn put_write(&self, py: Python<'_>, checkpoint: &Checkpoint, write: &PendingWrite) -> Result<()> {
        let config = config_for(
            py,
            &checkpoint.thread_id,
            &checkpoint.namespace,
            Some(&checkpoint.id),
        )
        .map_err(|error| Error::Checkpoint(error.to_string()))?;
        let writes = PyList::empty(py);
        let pair = pyo3::types::PyTuple::new(
            py,
            [
                value_to_py(py, &Value::String(write.channel.clone()))
                    .map_err(|error| Error::Checkpoint(error.to_string()))?,
                value_to_py(py, &write.value)
                    .map_err(|error| Error::Checkpoint(error.to_string()))?,
            ],
        )
        .map_err(|error| Error::Checkpoint(error.to_string()))?;
        writes
            .append(pair)
            .map_err(|error| Error::Checkpoint(error.to_string()))?;
        let put = self
            .object
            .bind(py)
            .getattr("put_writes")
            .map_err(|error| saver_error(py, "put_writes", error))?;
        put.call1((config, writes, write.task_id.clone()))
            .map_err(|error| saver_error(py, "put_writes", error))?;
        Ok(())
    }
}