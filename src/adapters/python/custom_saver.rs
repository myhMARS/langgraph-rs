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

/// LangGraph-shaped `(checkpoint, metadata, channel_versions)` payload.
///
/// LangGraph's scheduler reads `__root__`, `branch:to:<node>` and `__interrupt__` out of the
/// channel values, so they are added here -- the same shape the LangGraph-schema SQLite saver
/// writes, which is what makes cross-engine resume work.
fn langgraph_payload(
    py: Python<'_>,
    checkpoint: &Checkpoint,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let mut document = crate::interop::checkpoint_to_langgraph(checkpoint);
    let lg_id = langgraph_id(checkpoint);
    let interrupt_entries = langgraph_interrupt_entries(&lg_id, checkpoint);
    if let Some(object) = document.get_mut("checkpoint").and_then(Value::as_object_mut) {
        object.insert("id".to_string(), Value::String(lg_id.clone()));
        // LangGraph 1.2 writes v4 checkpoints; keep the payload version in step with it.
        object.insert("v".to_string(), serde_json::json!(4));
        let mut root = checkpoint.values.clone();
        if !checkpoint.interrupts.is_empty() {
            let interrupts = checkpoint
                .interrupts
                .iter()
                .zip(&interrupt_entries)
                .map(|(interrupt, (id, _, _))| {
                    serde_json::json!({
                        "__langgraph_type__": "Interrupt",
                        "value": interrupt.value,
                        "id": id,
                        "response_schema": interrupt.response_schema,
                    })
                })
                .collect::<Vec<_>>();
            root.insert("__interrupt__".to_string(), Value::Array(interrupts));
        }
        let mut channel_values = serde_json::Map::new();
        channel_values.insert("__root__".to_string(), Value::Object(root.clone()));
        for (channel, value) in &root {
            channel_values.insert(channel.clone(), value.clone());
        }
        let mut versions = object
            .get("channel_versions")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let version = Value::String(crate::checkpoint::version_string(checkpoint.step));
        for channel in root.keys() {
            versions
                .entry(channel.clone())
                .or_insert_with(|| version.clone());
        }
        let mut sends = Vec::new();
        for task in &checkpoint.next {
            if let Some(input) = &task.input {
                sends.push(serde_json::json!({
                    "__langgraph_type__": "Send",
                    "node": task.node,
                    "input": input,
                }));
            } else {
                channel_values.insert(format!("branch:to:{}", task.node), Value::Null);
                versions.insert(format!("branch:to:{}", task.node), version.clone());
            }
        }
        if !sends.is_empty() {
            channel_values.insert("__pregel_tasks".to_string(), Value::Array(sends));
            versions.insert("__pregel_tasks".to_string(), version.clone());
        }
        versions.insert("__root__".to_string(), version);
        // LangGraph only re-runs a pending node when `versions_seen` does not already cover the
        // channels it reads, so pending nodes are dropped from the map.
        if let Some(seen) = object.get_mut("versions_seen").and_then(Value::as_object_mut) {
            for task in &checkpoint.next {
                seen.remove(&task.node);
            }
        }
        let mut updated = channel_values
            .keys()
            .map(|channel| Value::String(channel.clone()))
            .collect::<Vec<_>>();
        updated.sort_by_key(Value::to_string);
        object.insert("updated_channels".to_string(), Value::Array(updated));
        object.insert("channel_values".to_string(), Value::Object(channel_values));
        object.insert("channel_versions".to_string(), Value::Object(versions));
    }
    if let Some(metadata) = document.get_mut("metadata").and_then(Value::as_object_mut) {
        // LangGraph's metadata step matches the checkpoint step, like the SQLite saver.
        metadata.insert("step".to_string(), serde_json::json!(checkpoint.step as i64));
        metadata
            .entry("source".to_string())
            .or_insert_with(|| Value::String("loop".to_string()));
        metadata
            .entry("parents".to_string())
            .or_insert_with(|| serde_json::json!({}));
        metadata.insert(
            PRIVATE_KEY.to_string(),
            serde_json::to_value(checkpoint).unwrap_or(Value::Null),
        );
    }
    let checkpoint_value = document.get("checkpoint").cloned().unwrap_or(Value::Null);
    let metadata_value = document.get("metadata").cloned().unwrap_or(Value::Null);
    let versions = checkpoint_value
        .get("channel_versions")
        .cloned()
        .unwrap_or(Value::Null);
    Ok((
        value_to_py(py, &checkpoint_value)?,
        value_to_py(py, &metadata_value)?,
        value_to_py(py, &versions)?,
    ))
}

fn checkpoint_from_py(py: Python<'_>, tuple: &Bound<'_, PyAny>, thread: &str) -> Result<Checkpoint> {
    let payload = py_to_json(
        &tuple
            .get_item(1)
            .map_err(|error| saver_error(py, "get_tuple", error))?,
    )?;
    let metadata = py_to_json(
        &tuple
            .get_item(2)
            .map_err(|error| saver_error(py, "get_tuple", error))?,
    )?;
    // Checkpoints written by LangGraph itself carry no engine sidecar; rebuild the scheduling
    // state from its `branch:to:` channels and `__interrupt__` pending writes instead.
    let Some(extras) = metadata.get(PRIVATE_KEY).cloned() else {
        return checkpoint_from_langgraph_tuple(tuple, thread, &payload, &metadata);
    };
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
        // `__root__` / `branch:to:*` / `__pregel_tasks` are LangGraph scheduling markers, not
        // user channels; the LangGraph-schema reader filters them the same way.
        values: payload
            .get("channel_values")
            .and_then(Value::as_object)
            .map(|channels| {
                channels
                    .iter()
                    .filter(|(channel, value)| {
                        !matches!(
                            channel.as_str(),
                            "__root__" | "__start__" | "__end__" | "__interrupt__" | "__pregel_tasks"
                        ) && !channel.starts_with("branch:to:")
                            && !value.is_null()
                    })
                    .map(|(channel, value)| (channel.clone(), value.clone()))
                    .collect()
            })
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
            task_writes: to_map("task_writes")?,
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

/// Convert a LangGraph-written checkpoint tuple into this engine's checkpoint.
fn checkpoint_from_langgraph_tuple(
    tuple: &Bound<'_, PyAny>,
    thread: &str,
    payload: &Value,
    metadata: &Value,
) -> Result<Checkpoint> {
    let channel_values = payload
        .get("channel_values")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut values = serde_json::Map::new();
    for root in ["__root__", "__start__"] {
        if let Some(root) = channel_values.get(root).and_then(Value::as_object) {
            for (key, value) in root {
                if key != "__interrupt__" {
                    values.insert(key.clone(), value.clone());
                }
            }
        }
    }
    let mut branch_nodes: Vec<(String, String)> = Vec::new();
    for (channel, value) in &channel_values {
        if let Some(node) = channel.strip_prefix("branch:to:") {
            branch_nodes.push((node.to_string(), String::new()));
            continue;
        }
        if matches!(channel.as_str(), "__root__" | "__start__" | "__end__" | "__interrupt__")
            || value.is_null()
        {
            continue;
        }
        values.insert(channel.clone(), value.clone());
    }
    let mut pending_writes: Vec<Value> = Vec::new();
    let mut interrupts: Vec<Value> = Vec::new();
    let mut write_task_ids: Vec<String> = Vec::new();
    if let Ok(writes) = tuple.get_item(4)
        && !writes.is_none()
        && let Ok(items) = writes.try_iter()
    {
        for item in items.flatten() {
            let Ok(raw) = py_to_json(&item) else {
                continue;
            };
            let Some(parts) = raw.as_array() else {
                continue;
            };
            let task_id = parts
                .first()
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let channel = parts.get(1).and_then(Value::as_str).unwrap_or_default();
            let value = parts.get(2).cloned().unwrap_or(Value::Null);
            if channel == "__interrupt__" {
                write_task_ids.push(task_id.clone());
                let entries = match &value {
                    Value::Array(items) => items.clone(),
                    other => vec![other.clone()],
                };
                for entry in entries {
                    let object = entry.as_object();
                    interrupts.push(serde_json::json!({
                        "node": "",
                        "value": object
                            .and_then(|map| map.get("value"))
                            .cloned()
                            .unwrap_or(Value::Null),
                        "id": object
                            .and_then(|map| map.get("id"))
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                        "response_schema": object
                            .and_then(|map| map.get("response_schema"))
                            .cloned()
                            .unwrap_or(Value::Null),
                        "task_id": task_id,
                        "static_pause": false,
                    }));
                }
                continue;
            }
            if channel.starts_with("branch:to:") {
                continue;
            }
            pending_writes.push(serde_json::json!({
                "task_id": task_id,
                "node": "",
                "channel": channel,
                "value": value,
                "replace": false,
            }));
        }
    }
    if branch_nodes.is_empty() {
        for task_id in &write_task_ids {
            branch_nodes.push((String::new(), task_id.clone()));
        }
    }
    let mut next: Vec<crate::_internal::node::Task> = Vec::new();
    for (index, (node, task_id)) in branch_nodes.iter().enumerate() {
        let id = if task_id.is_empty() {
            write_task_ids
                .get(index)
                .cloned()
                .unwrap_or_else(|| format!("task-{index}"))
        } else {
            task_id.clone()
        };
        next.push(crate::_internal::node::Task {
            id,
            node: node.clone(),
            input: None,
            resume: None,
            resume_history: Vec::new(),
            error: None,
            timeout_millis: None,
            interrupted: !interrupts.is_empty(),
            triggers: Vec::new(),
            push_index: None,
            push: false,
        });
    }
    for (index, interrupt) in interrupts.iter_mut().enumerate() {
        let task = next.get(index).or_else(|| next.first()).cloned();
        if let (Some(task), Some(object)) = (task, interrupt.as_object_mut()) {
            object.insert("node".to_string(), Value::String(task.node.clone()));
        }
    }
    for (index, write) in pending_writes.iter_mut().enumerate() {
        let task = next.get(index).or_else(|| next.first()).cloned();
        if let (Some(task), Some(object)) = (task, write.as_object_mut()) {
            object.insert("node".to_string(), Value::String(task.node.clone()));
        }
    }
    let step = metadata
        .get("step")
        .and_then(Value::as_i64)
        .map(|step| step.max(0) as usize)
        .unwrap_or_default();
    let parent_id = tuple
        .get_item(3)
        .ok()
        .filter(|parent| !parent.is_none())
        .and_then(|parent| {
            parent
                .get_item("configurable")
                .ok()
                .and_then(|configurable| configurable.get_item("checkpoint_id").ok())
        })
        .and_then(|id| id.extract::<String>().ok());
    let checkpoint_id = payload
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let mut checkpoint = Checkpoint::new(checkpoint_id, thread.to_string(), step, values);
    checkpoint.parent_id = parent_id;
    checkpoint.metadata = metadata.as_object().cloned().unwrap_or_default();
    checkpoint.channel_versions = payload
        .get("channel_versions")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    checkpoint.versions_seen = payload
        .get("versions_seen")
        .and_then(Value::as_object)
        .map(|map| {
            map.iter()
                .map(|(node, versions)| (node.clone(), std::sync::Arc::new(versions.clone())))
                .collect()
        })
        .unwrap_or_default();
    checkpoint.interrupted = !interrupts.is_empty();
    checkpoint.interrupts = serde_json::from_value(Value::Array(interrupts)).unwrap_or_default();
    checkpoint.pending_writes =
        serde_json::from_value(Value::Array(pending_writes)).unwrap_or_default();
    checkpoint.next = next;
    Ok(checkpoint)
}
/// UUIDv6-style, time-ordered checkpoint id.
///
/// LangGraph picks the latest checkpoint by comparing ids, so ours must sort by wall clock the
/// same way its `uuid6` values do -- otherwise a thread written by both engines would resume from
/// the wrong checkpoint.
fn langgraph_id(checkpoint: &Checkpoint) -> String {
    // UUIDv6 timestamps are 100ns ticks since 1582-10-15.
    // Sub-millisecond ticks carry the step so checkpoints of one thread never tie.
    let ticks = checkpoint
        .created_at_millis
        .saturating_mul(10_000)
        .saturating_add((checkpoint.step as u64).min(9_999))
        .wrapping_add(0x01b2_1dd2_1381_4000);
    let time_high = ((ticks >> 28) & 0xffff_ffff) as u32;
    let time_mid = ((ticks >> 12) & 0xffff) as u16;
    let time_low = (ticks & 0x0fff) as u16;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in checkpoint.id.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let clock = (((hash >> 48) & 0x3fff) as u16) | 0x8000;
    let node = hash & 0xffff_ffff_ffff;
    format!("{time_high:08x}{time_mid:04x}6{time_low:03x}{clock:04x}{node:012x}")
}

fn sha1_uuid_like(parts: &[&[u8]]) -> String {
    use sha1::{Digest, Sha1};

    let mut hasher = Sha1::new();
    for part in parts {
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

fn id_bytes(checkpoint_id: &str) -> Vec<u8> {
    // Dashes are not part of the value, so strip them before grouping nibbles.
    let hex: String = checkpoint_id
        .chars()
        .filter(char::is_ascii_hexdigit)
        .collect();
    hex.as_bytes()
        .chunks(2)
        .filter_map(|pair| {
            std::str::from_utf8(pair)
                .ok()
                .and_then(|pair| u8::from_str_radix(pair, 16).ok())
        })
        .collect()
}

fn langgraph_task_id(checkpoint_id: &str, node: &str, step: usize) -> String {
    let bytes = id_bytes(checkpoint_id);
    let trigger = format!("branch:to:{node}");
    let step = (step + 1).to_string();
    sha1_uuid_like(&[
        &bytes,
        node.as_bytes(),
        step.as_bytes(),
        node.as_bytes(),
        b"__pregel_pull",
        trigger.as_bytes(),
    ])
}

fn langgraph_send_task_id(checkpoint_id: &str, node: &str, step: usize, index: usize) -> String {
    let bytes = id_bytes(checkpoint_id);
    let step = (step + 1).to_string();
    let index = index.to_string();
    sha1_uuid_like(&[
        &bytes,
        node.as_bytes(),
        step.as_bytes(),
        node.as_bytes(),
        b"__pregel_push",
        index.as_bytes(),
    ])
}

/// `(interrupt id, task id, task path)` triples LangGraph expects for a paused checkpoint.
fn langgraph_interrupt_entries(checkpoint_id: &str, checkpoint: &Checkpoint) -> Vec<(String, String, String)> {
    let mut send_queues: std::collections::BTreeMap<String, std::collections::VecDeque<(usize, String)>> =
        std::collections::BTreeMap::new();
    for (index, task) in checkpoint
        .next
        .iter()
        .filter(|task| task.input.is_some())
        .enumerate()
    {
        let task_id = langgraph_send_task_id(checkpoint_id, &task.node, checkpoint.step, index);
        send_queues
            .entry(task.node.clone())
            .or_default()
            .push_back((index, task_id));
    }
    let mut answered: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    checkpoint
        .interrupts
        .iter()
        .map(|interrupt| {
            if let Some(queue) = send_queues.get_mut(&interrupt.node)
                && let Some((index, task_id)) = queue.pop_front()
            {
                let key = format!("{}:{}", interrupt.node, task_id);
                let id = format!("{:032x}", xxhash_rust::xxh3::xxh3_128(key.as_bytes()));
                return (id, task_id, format!("~__pregel_push, {index:010}, 0000000000"));
            }
            let count = *answered.entry(interrupt.node.clone()).or_insert(0);
            *answered.get_mut(&interrupt.node).expect("counter") += 1;
            let task_id = langgraph_task_id(checkpoint_id, &interrupt.node, checkpoint.step);
            let namespace = format!("{}:{}", interrupt.node, task_id);
            let key = if count == 0 {
                namespace
            } else {
                format!("{namespace}|{count}")
            };
            let id = format!("{:032x}", xxhash_rust::xxh3::xxh3_128(key.as_bytes()));
            (id, task_id, format!("~__pregel_pull, {}", interrupt.node))
        })
        .collect()
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
            let lg_id = langgraph_id(&checkpoint);
            let config = config_for(
                py,
                &checkpoint.thread_id,
                &checkpoint.namespace,
                Some(&lg_id),
            )
            .map_err(|error| Error::Checkpoint(error.to_string()))?;
            let (payload, metadata, versions) = langgraph_payload(py, &checkpoint)
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
            // LangGraph derives pending tasks from these writes, so a pause it can resume needs
            // the `__interrupt__` entries keyed by its own task ids.
            self.put_langgraph_writes(py, &checkpoint, &lg_id)?;
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
            Some(&langgraph_id(checkpoint)),
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

    /// Publish LangGraph-shaped writes so its scheduler sees the pending tasks and interrupts.
    fn put_langgraph_writes(
        &self,
        py: Python<'_>,
        checkpoint: &Checkpoint,
        lg_id: &str,
    ) -> Result<()> {
        let config = config_for(
            py,
            &checkpoint.thread_id,
            &checkpoint.namespace,
            Some(lg_id),
        )
        .map_err(|error| Error::Checkpoint(error.to_string()))?;
        let put = self
            .object
            .bind(py)
            .getattr("put_writes")
            .map_err(|error| saver_error(py, "put_writes", error))?;
        let interrupt_class = py
            .import("langgraph.types")
            .and_then(|module| module.getattr("Interrupt"))
            .ok();
        let entries = langgraph_interrupt_entries(lg_id, checkpoint);
        for (interrupt, (interrupt_id, task_id, task_path)) in
            checkpoint.interrupts.iter().zip(entries)
        {
            let value = match &interrupt_class {
                Some(class) => class
                    .call1((
                        value_to_py(py, &interrupt.value)
                            .map_err(|error| Error::Checkpoint(error.to_string()))?,
                        interrupt_id,
                    ))
                    .map_err(|error| saver_error(py, "put_writes", error))?
                    .unbind(),
                None => value_to_py(
                    py,
                    &serde_json::json!({ "value": interrupt.value, "id": interrupt_id }),
                )
                .map_err(|error| Error::Checkpoint(error.to_string()))?,
            };
            let list =
                PyList::new(py, [value]).map_err(|error| Error::Checkpoint(error.to_string()))?;
            let entry = pyo3::types::PyTuple::new(
                py,
                [
                    value_to_py(py, &Value::String("__interrupt__".to_string()))
                        .map_err(|error| Error::Checkpoint(error.to_string()))?,
                    list.into_any().unbind(),
                ],
            )
            .map_err(|error| Error::Checkpoint(error.to_string()))?;
            let writes =
                PyList::new(py, [entry]).map_err(|error| Error::Checkpoint(error.to_string()))?;
            put.call1((
                config.clone(),
                writes,
                task_id.clone(),
                task_path.clone(),
            ))
            .map_err(|error| saver_error(py, "put_writes", error))?;
        }
        Ok(())
    }
}