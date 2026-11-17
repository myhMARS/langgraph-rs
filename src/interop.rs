//! Conversion between langgraph-rs checkpoints and LangGraph's checkpoint-tuple JSON.
//!
//! LangGraph persists checkpoints as a tuple `(config, checkpoint, metadata, parent_config,
//! pending_writes)`. This module exports our [`Checkpoint`] in that JSON shape and imports
//! such documents back. The conversion is JSON-level: LangGraph stores channel values with
//! its msgpack-based serializer, so direct database interoperability is not guaranteed.
//! A `metadata.langgraph_rs` extension keeps our pending tasks/interrupt state so that
//! round-tripping through this module is lossless.

use serde_json::{Map, Value, json};

use crate::kernel::checkpoint::{Checkpoint, now_millis};

pub mod msgpack;
use crate::kernel::error::{Error, Result};

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
    metadata.insert("step".to_string(), json!(checkpoint.step));
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
        .and_then(Value::as_u64)
        .or_else(|| checkpoint.get("step").and_then(Value::as_u64))
        .unwrap_or_default() as usize;
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
        .unwrap_or_else(|| crate::kernel::checkpoint::new_checkpoint_id(&thread_id, step));
    let values = checkpoint
        .get("channel_values")
        .or_else(|| checkpoint.get("values"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
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
    let interrupts = match extra.and_then(|extra| extra.get("interrupts")) {
        Some(interrupts) => serde_json::from_value(interrupts.clone())
            .map_err(|error| Error::other(format!("invalid interrupts: {error}")))?,
        None => Vec::new(),
    };
    let task_results = extra
        .and_then(|extra| extra.get("task_results"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    Ok(Checkpoint {
        id,
        thread_id,
        step,
        values,
        next,
        interrupted: extra
            .and_then(|extra| extra.get("interrupted"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
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
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default(),
        versions_seen: extra
            .and_then(|extra| extra.get("versions_seen"))
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default(),
        updated_channels: extra
            .and_then(|extra| extra.get("updated_channels"))
            .and_then(Value::as_array)
            .map(|channels| {
                channels
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        pending_writes: object
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
                            let value = write.get("value").cloned().unwrap_or(Value::Null);
                            let replace = write
                                .get("replace")
                                .and_then(Value::as_bool)
                                .unwrap_or(false);
                            Some(crate::kernel::checkpoint::PendingWrite {
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
                            let value = tuple.get(2).cloned().unwrap_or(Value::Null);
                            Some(crate::kernel::checkpoint::PendingWrite {
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
