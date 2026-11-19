//! Conversion between langgraph-rs checkpoints and LangGraph's checkpoint-tuple JSON.
//!
//! LangGraph persists checkpoints as a tuple `(config, checkpoint, metadata, parent_config,
//! pending_writes)`. This module exports our [`Checkpoint`] in that JSON shape and imports
//! such documents back. The conversion is JSON-level: LangGraph stores channel values with
//! its msgpack-based serializer, so direct database interoperability is not guaranteed.
//! A `metadata.langgraph_rs` extension keeps our pending tasks/interrupt state so that
//! round-tripping through this module is lossless.

use serde_json::{Map, Value, json};

use crate::checkpoint::{Checkpoint, now_millis};

pub mod msgpack;
use crate::_internal::error::{Error, Result};

fn generated_version(checkpoint: &Checkpoint, channel: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    checkpoint.id.hash(&mut hasher);
    checkpoint.thread_id.hash(&mut hasher);
    checkpoint.step.hash(&mut hasher);
    channel.hash(&mut hasher);
    let hash = hasher.finish();
    format!("{:032}.0.{}", hash, hash.rotate_left(17))
}

/// Export a checkpoint as a LangGraph-shaped checkpoint tuple document.
pub fn checkpoint_to_langgraph(checkpoint: &Checkpoint) -> Value {
    let channel_versions = if checkpoint.channel_versions.is_empty() {
        checkpoint
            .values
            .keys()
            .map(|channel| {
                (
                    channel.clone(),
                    Value::String(generated_version(checkpoint, channel)),
                )
            })
            .collect::<Map<String, Value>>()
    } else {
        checkpoint.channel_versions.clone()
    };
    let updated_channels = if checkpoint.updated_channels.is_empty() {
        checkpoint
            .values
            .keys()
            .map(|channel| Value::String(channel.clone()))
            .collect::<Vec<_>>()
    } else {
        checkpoint
            .updated_channels
            .iter()
            .map(|channel| Value::String(channel.clone()))
            .collect::<Vec<_>>()
    };
    let ts = format_rfc3339(checkpoint.created_at_millis);
    let source = checkpoint
        .metadata
        .get("source")
        .cloned()
        .unwrap_or_else(|| Value::String("loop".to_string()));
    let parent_config = checkpoint.parent_id.as_ref().map(|parent| {
        json!({
            "configurable": {
                "thread_id": checkpoint.thread_id,
                "checkpoint_ns": checkpoint.namespace,
                "checkpoint_id": parent,
            }
        })
    });
    let mut metadata = Map::new();
    metadata.insert("source".to_string(), source);
    // LangGraph's `metadata["step"]` is one less than our internal counter: the input
    // checkpoint is -1 there and 0 here, so the first superstep is 0 vs 1.
    metadata.insert("step".to_string(), json!(checkpoint.step as i64 - 1));
    metadata.insert("parents".to_string(), json!({}));
    metadata.insert("thread_id".to_string(), json!(checkpoint.thread_id));
    metadata.insert("checkpoint_ns".to_string(), json!(checkpoint.namespace));
    metadata.insert("created_at".to_string(), json!(ts));
    metadata.insert(
        "langgraph_rs".to_string(),
        json!({
            "next": checkpoint.next,
            "interrupted": checkpoint.interrupted,
            "paused_before": checkpoint.paused_before,
            "interrupts": checkpoint.interrupts,
            "metadata": checkpoint.metadata,
            "created_at_millis": checkpoint.created_at_millis,
            "parent_id": checkpoint.parent_id,
            "task_results": checkpoint.task_results,
        }),
    );
    json!({
        "config": {
            "configurable": {
                "thread_id": checkpoint.thread_id,
                "checkpoint_ns": checkpoint.namespace,
                "checkpoint_id": checkpoint.id,
            }
        },
        "checkpoint": {
            // Exports stay on v1: LangGraph derives task ids with SHA-1 for v1, matching
            // the ids the compatible saver writes, so a Python reader recomputes the same
            // ids we stored. (v4/xxh3 ids are used internally for new checkpoints.)
            "v": 1,
            "id": checkpoint.id,
            "ts": ts,
            "channel_values": checkpoint.values,
            "channel_versions": channel_versions,
            "versions_seen": checkpoint.versions_seen,
            "updated_channels": updated_channels,
        },
        "metadata": metadata,
        "parent_config": parent_config,
        "deferred": checkpoint.deferred,
        "pending_writes": checkpoint
            .pending_writes
            .iter()
            .map(|write| {
                json!({
                    "task_id": write.task_id,
                    "node": write.node,
                    "channel": write.channel,
                    "value": write.value,
                    "replace": write.replace,
                })
            })
            .collect::<Vec<_>>(),
    })
}

/// Import a LangGraph-shaped checkpoint tuple document (or a bare checkpoint object).
pub fn checkpoint_from_langgraph(value: &Value) -> Result<Checkpoint> {
    let object = value
        .as_object()
        .ok_or_else(|| Error::other("checkpoint document must be a JSON object"))?;
    let checkpoint = object
        .get("checkpoint")
        .or_else(|| object.get("channel_values").map(|_| value))
        .unwrap_or(value);
    let metadata = object
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let configurable = object
        .get("config")
        .and_then(|config| config.get("configurable"))
        .and_then(Value::as_object);
    let thread_id = metadata
        .get("thread_id")
        .and_then(Value::as_str)
        .or_else(|| {
            configurable
                .and_then(|config| config.get("thread_id"))
                .and_then(Value::as_str)
        })
        .ok_or_else(|| Error::other("checkpoint document has no thread_id"))?
        .to_string();
    let step = metadata
        .get("step")
        .and_then(Value::as_i64)
        .or_else(|| checkpoint.get("step").and_then(Value::as_i64))
        .map(|step| (step + 1).max(0) as usize)
        .unwrap_or_default();
    let namespace = metadata
        .get("checkpoint_ns")
        .and_then(Value::as_str)
        .or_else(|| {
            configurable
                .and_then(|config| config.get("checkpoint_ns"))
                .and_then(Value::as_str)
        })
        .unwrap_or_default()
        .to_string();
    let id = checkpoint
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| {
            configurable
                .and_then(|config| config.get("checkpoint_id"))
                .and_then(Value::as_str)
        })
        .map(str::to_owned)
        .unwrap_or_else(|| crate::checkpoint::new_checkpoint_id(&thread_id, step));
    let raw_values = checkpoint
        .get("channel_values")
        .or_else(|| checkpoint.get("values"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    // LangGraph stores pending interrupts in `channel_values.__interrupt__`; they are not
    // state channels, so decode them and keep them out of `values`.
    let mut channel_interrupts: Vec<crate::_internal::error::Interrupt> = Vec::new();
    let mut collect_channel_interrupts = |entries: &[Value]| {
        for entry in entries {
            let Some(entry) = entry.as_object() else {
                continue;
            };
            let Some(id) = entry.get("id").and_then(Value::as_str) else {
                continue;
            };
            if channel_interrupts.iter().any(|known| known.id == id) {
                continue;
            }
            channel_interrupts.push(crate::_internal::error::Interrupt {
                node: String::new(),
                value: entry.get("value").cloned().unwrap_or(Value::Null),
                id: id.to_string(),
                response_schema: entry
                    .get("response_schema")
                    .filter(|schema| !schema.is_null())
                    .cloned(),
                task_id: None,
                static_pause: false,
            });
        }
    };
    for source in [
        &raw_values,
        raw_values
            .get("__root__")
            .and_then(Value::as_object)
            .unwrap_or(&Map::new()),
    ] {
        if let Some(entries) = source.get("__interrupt__").and_then(Value::as_array) {
            collect_channel_interrupts(entries);
        }
    }
    // `__root__` holds the whole state for root-schema graphs: flatten it instead of
    // dropping it, then strip the remaining control channels.
    let mut values = Map::new();
    if let Some(root) = raw_values.get("__root__").and_then(Value::as_object) {
        for (channel, value) in root {
            if !channel.starts_with("__") {
                values.insert(channel.clone(), value.clone());
            }
        }
    }
    for (channel, value) in raw_values {
        if channel == "__root__" || channel.starts_with("__") {
            continue;
        }
        values.insert(channel, value);
    }
    let parent_id = object
        .get("parent_config")
        .and_then(|parent| parent.get("configurable"))
        .and_then(|config| config.get("checkpoint_id"))
        .and_then(Value::as_str)
        .map(str::to_owned);

    let extra = metadata.get("langgraph_rs");
    let created_at_millis = extra
        .and_then(|extra| extra.get("created_at_millis"))
        .and_then(Value::as_u64)
        .or_else(|| {
            checkpoint
                .get("ts")
                .and_then(Value::as_str)
                .and_then(parse_rfc3339)
        })
        .unwrap_or_else(now_millis);
    let mut checkpoint_metadata = extra
        .and_then(|extra| extra.get("metadata"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if checkpoint_metadata.is_empty()
        && let Some(source) = metadata.get("source").and_then(Value::as_str)
    {
        checkpoint_metadata.insert("source".to_string(), json!(source));
    }
    let next = match extra.and_then(|extra| extra.get("next")) {
        Some(next) => serde_json::from_value(next.clone())
            .map_err(|error| Error::other(format!("invalid next tasks: {error}")))?,
        None => Vec::new(),
    };
    let interrupts: Vec<crate::_internal::error::Interrupt> =
        match extra.and_then(|extra| extra.get("interrupts")) {
            Some(interrupts) => serde_json::from_value(interrupts.clone())
                .map_err(|error| Error::other(format!("invalid interrupts: {error}")))?,
            None => Vec::new(),
        };
    let task_results = extra
        .and_then(|extra| extra.get("task_results"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    // Hints for rebuilding pending tasks: LangGraph stores interrupts as `__interrupt__`
    // writes, which the state filter below drops, so collect task id -> node first.
    let mut task_hints: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    // Real LangGraph checkpoints persist interrupts as `__interrupt__` pending writes, so
    // they are decoded here before the state filter drops the control channel.
    let mut foreign_interrupts: Vec<crate::_internal::error::Interrupt> = Vec::new();
    let mut foreign_errors: Vec<(String, String, String)> = Vec::new();
    if let Some(writes) = object.get("pending_writes").and_then(Value::as_array) {
        for write in writes {
            let (task_id, node, channel, value, replace) = if let Some(write) = write.as_object() {
                (
                    write.get("task_id").and_then(Value::as_str),
                    write.get("node").and_then(Value::as_str),
                    write.get("channel").and_then(Value::as_str),
                    write.get("value").cloned(),
                    write
                        .get("replace")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                )
            } else if let Some(tuple) = write.as_array() {
                (
                    tuple.first().and_then(Value::as_str),
                    tuple.get(3).and_then(Value::as_str),
                    tuple.get(1).and_then(Value::as_str),
                    tuple.get(2).cloned(),
                    false,
                )
            } else {
                continue;
            };
            let Some(task_id) = task_id else { continue };
            if let Some(node) = node
                && !node.is_empty()
            {
                task_hints.insert(task_id.to_string(), node.to_string());
            }
            let _ = replace;
            if channel == Some("__error__") {
                let message = match value {
                    Some(Value::Object(map)) => map
                        .get("message")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_else(|| Value::Object(map).to_string()),
                    Some(Value::String(text)) => text,
                    Some(other) => other.to_string(),
                    None => String::new(),
                };
                foreign_errors.push((
                    task_id.to_string(),
                    node.unwrap_or_default().to_string(),
                    message,
                ));
                continue;
            }
            if channel != Some("__interrupt__") {
                continue;
            }
            let Some(value) = value else { continue };
            let entries: Vec<&Value> = match &value {
                Value::Array(items) => items.iter().collect(),
                other => vec![other],
            };
            for entry in entries {
                let Some(entry) = entry.as_object() else {
                    continue;
                };
                foreign_interrupts.push(crate::_internal::error::Interrupt {
                    node: node.unwrap_or_default().to_string(),
                    value: entry.get("value").cloned().unwrap_or(Value::Null),
                    id: entry
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    response_schema: entry
                        .get("response_schema")
                        .filter(|schema| !schema.is_null())
                        .cloned(),
                    task_id: Some(task_id.to_string().into_boxed_str()),
                    static_pause: false,
                });
            }
        }
    }
    let mut interrupts = interrupts;
    for foreign in channel_interrupts {
        if !interrupts.iter().any(|known| known.id == foreign.id) {
            interrupts.push(foreign);
        }
    }
    for foreign in foreign_interrupts {
        if !interrupts.iter().any(|known| known.id == foreign.id) {
            interrupts.push(foreign);
        }
    }
    let pending_writes: Vec<crate::checkpoint::PendingWrite> = object
        .get("pending_writes")
        .and_then(Value::as_array)
        .map(|writes| {
            writes
                .iter()
                .filter_map(|write| {
                    if let Some(write) = write.as_object() {
                        let task_id = write.get("task_id")?.as_str()?.to_string();
                        let node = write
                            .get("node")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let channel = write.get("channel")?.as_str()?.to_string();
                        // LangGraph reserves `__`-prefixed control channels (RESUME,
                        // INTERRUPT, ERROR, NULL_TASK_ID writes). They drive the source
                        // engine's own loop and are not state, so replaying them as
                        // updates would either fail or pollute the imported state.
                        if channel.starts_with("__") {
                            return None;
                        }
                        let value = write.get("value").cloned().unwrap_or(Value::Null);
                        let replace = write
                            .get("replace")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);
                        Some(crate::checkpoint::PendingWrite {
                            task_id,
                            node,
                            channel,
                            value,
                            replace,
                        })
                    } else {
                        let tuple = write.as_array()?;
                        let task_id = tuple.first()?.as_str()?.to_string();
                        let channel = tuple.get(1)?.as_str()?.to_string();
                        // LangGraph's real `PendingWrite` is a tuple; reserved control
                        // channels must be dropped here too, not only in the object branch.
                        if channel.starts_with("__") {
                            return None;
                        }
                        let value = tuple.get(2).cloned().unwrap_or(Value::Null);
                        Some(crate::checkpoint::PendingWrite {
                            task_id,
                            node: String::new(),
                            channel,
                            value,
                            replace: false,
                        })
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    // Real LangGraph checkpoints carry no `next`; the pending tasks must be reconstructed
    // from the interrupts and pending writes, otherwise a resume would silently finish.
    let mut next = next;
    if next.is_empty()
        && (!interrupts.is_empty() || !pending_writes.is_empty() || !foreign_errors.is_empty())
    {
        let mut seen = std::collections::HashSet::new();
        for interrupt in &interrupts {
            let node = interrupt.node.clone();
            let id = interrupt
                .task_id
                .as_deref()
                .map(str::to_owned)
                .unwrap_or_else(|| crate::_internal::node::new_task_id(&node));
            if seen.insert(id.clone()) {
                let mut task = crate::_internal::node::Task::new(node);
                task.id = id;
                task.interrupted = true;
                next.push(task);
            }
        }
        // Only interrupts (and their task ids) mark a task as still pending; plain state
        // writes belong to tasks that already finished.
        for interrupt in &interrupts {
            let node = if interrupt.node.is_empty() {
                interrupt
                    .task_id
                    .as_deref()
                    .and_then(|task_id| task_hints.get(task_id).cloned())
                    .unwrap_or_default()
            } else {
                interrupt.node.clone()
            };
            if node.is_empty() {
                continue;
            }
            let id = interrupt
                .task_id
                .as_deref()
                .map(str::to_owned)
                .unwrap_or_else(|| crate::_internal::node::new_task_id(&node));
            if !seen.insert(id.clone()) {
                continue;
            }
            let mut task = crate::_internal::node::Task::new(node);
            task.id = id;
            task.interrupted = true;
            next.push(task);
        }
        // A failed task keeps its slot so the error handler can resume it.
        for (task_id, node, message) in &foreign_errors {
            let node = if node.is_empty() {
                task_hints.get(task_id).cloned().unwrap_or_default()
            } else {
                node.clone()
            };
            if node.is_empty() || !seen.insert(task_id.clone()) {
                continue;
            }
            let mut task = crate::_internal::node::Task::new(node.clone());
            task.id = task_id.clone();
            task.error = Some(crate::_internal::error::NodeFailure {
                node,
                message: message.clone(),
                attempts: 1,
            });
            next.push(task);
        }
    }
    let interrupted = extra
        .and_then(|extra| extra.get("interrupted"))
        .and_then(Value::as_bool)
        .unwrap_or(!interrupts.is_empty())
        || next.iter().any(|task| task.interrupted);
    Ok(Checkpoint {
        id,
        thread_id,
        step,
        values,
        next,
        deferred_routes: Vec::new(),
        resume_tasks: Vec::new(),
        goto_sources: Vec::new(),
        completed_tasks: Vec::new(),
        interrupted,
        paused_before: extra
            .and_then(|extra| extra.get("paused_before"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        parent_id: extra
            .and_then(|extra| extra.get("parent_id"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or(parent_id),
        created_at_millis,
        interrupts,
        metadata: checkpoint_metadata,
        namespace,
        task_results,
        channel_versions: extra
            .and_then(|extra| extra.get("channel_versions"))
            .or_else(|| checkpoint.get("channel_versions"))
            .or_else(|| object.get("channel_versions"))
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default(),
        versions_seen: extra
            .and_then(|extra| extra.get("versions_seen"))
            .or_else(|| checkpoint.get("versions_seen"))
            .or_else(|| object.get("versions_seen"))
            .and_then(Value::as_object)
            .map(|seen| {
                seen.iter()
                    .map(|(node, versions)| (node.clone(), std::sync::Arc::new(versions.clone())))
                    .collect()
            })
            .unwrap_or_default(),
        updated_channels: extra
            .and_then(|extra| extra.get("updated_channels"))
            .or_else(|| checkpoint.get("updated_channels"))
            .or_else(|| object.get("updated_channels"))
            .and_then(Value::as_array)
            .map(|channels| {
                channels
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        pending_writes,
        deferred: object
            .get("deferred")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default(),
    })
}

fn format_rfc3339(millis: u64) -> String {
    let seconds = (millis / 1000) as i64;
    let millis_part = millis % 1000;
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis_part:03}Z")
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let month_position = (5 * doy + 2) / 153;
    let day = doy - (153 * month_position + 2) / 5 + 1;
    let month = if month_position < 10 {
        month_position + 3
    } else {
        month_position - 9
    };
    (
        if month <= 2 { year + 1 } else { year },
        month as u32,
        day as u32,
    )
}

fn parse_rfc3339(value: &str) -> Option<u64> {
    let bytes = value.as_bytes();
    if bytes.len() < 20 {
        return None;
    }
    let year: i64 = value.get(0..4)?.parse().ok()?;
    let month: u32 = value.get(5..7)?.parse().ok()?;
    let day: u32 = value.get(8..10)?.parse().ok()?;
    let hour: i64 = value.get(11..13)?.parse().ok()?;
    let minute: i64 = value.get(14..16)?.parse().ok()?;
    let second: i64 = value.get(17..19)?.parse().ok()?;
    let millis = if value.len() > 20 {
        value
            .get(20..23)
            .and_then(|fraction| fraction.parse::<u64>().ok())
            .unwrap_or_default()
    } else {
        0
    };
    let days = days_from_civil(year, month, day);
    let seconds = days * 86_400 + hour * 3_600 + minute * 60 + second;
    Some((seconds.max(0) as u64) * 1000 + millis)
}

fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = month as i64;
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Encode the LangGraph-shaped checkpoint tuple as standard MessagePack bytes.
#[cfg(feature = "msgpack")]
pub fn checkpoint_to_langgraph_msgpack(checkpoint: &Checkpoint) -> Result<Vec<u8>> {
    Ok(msgpack::encode(&checkpoint_to_langgraph(checkpoint)))
}

/// Decode MessagePack bytes produced by LangGraph's `JsonPlusSerializer` (JSON-shaped
/// documents) into a checkpoint.
#[cfg(feature = "msgpack")]
pub fn checkpoint_from_langgraph_msgpack(bytes: &[u8]) -> Result<Checkpoint> {
    checkpoint_from_langgraph(&msgpack::decode(bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_roundtrip() {
        let millis = 1_763_328_000_123u64;
        let text = format_rfc3339(millis);
        assert_eq!(parse_rfc3339(&text), Some(millis));
    }
}
