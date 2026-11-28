//! Channel semantics: how node writes are merged into a state field.
//!
//! A channel is the unit of state. Every node returns a map of `channel -> value`
//! writes and the channel decides how a write is folded into the value produced by
//! the previous superstep. This is the Rust equivalent of LangGraph's
//! `Annotated[..., reducer]` state annotations.

use std::fmt;
use std::sync::Arc;

use serde::Serialize;
use serde_json::Value;

use crate::{Error, Result};

/// Merge strategy for a single state channel.
#[derive(Clone)]
pub enum Channel {
    /// Keep the most recent write. Concurrent writes in the same superstep are rejected.
    LastValue,
    /// Keep the most recent write and allow concurrent writes (last one wins).
    AnyValue,
    /// Keep the write for the current superstep only; cleared before the next step.
    EphemeralValue,
    /// Like [`Channel::LastValue`] but the value is never written to checkpoints.
    UntrackedValue,
    /// Accumulate writer names until every expected name arrived, then become `true`.
    NamedBarrierValue {
        /// Names that must write before the barrier opens.
        names: Vec<String>,
    },
    /// Fold deltas with a reducer and snapshot the accumulated value (LangGraph `DeltaChannel`).
    Delta {
        /// Folding function applied to every delta.
        reducer: Reducer,
        /// Accepted for API compatibility; this engine always snapshots the full value.
        snapshot_frequency: usize,
    },
    /// Keep the last write but only publish it when the graph finishes.
    LastValueAfterFinish,
    /// Collect writes into a JSON array (LangGraph `Topic`).
    Topic {
        /// Keep values across supersteps (`true`) or only the current superstep (`false`).
        accumulate: bool,
    },
    /// Fold writes with a user supplied binary reducer (e.g. message list merging).
    Reducer(Reducer),
}

impl fmt::Debug for Channel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Channel::LastValue => f.write_str("LastValue"),
            Channel::AnyValue => f.write_str("AnyValue"),
            Channel::EphemeralValue => f.write_str("EphemeralValue"),
            Channel::UntrackedValue => f.write_str("UntrackedValue"),
            Channel::NamedBarrierValue { names } => write!(f, "NamedBarrierValue({names:?})"),
            Channel::Delta {
                snapshot_frequency, ..
            } => write!(f, "Delta(snapshot_frequency={snapshot_frequency})"),
            Channel::LastValueAfterFinish => f.write_str("LastValueAfterFinish"),
            Channel::Topic { accumulate } => write!(f, "Topic(accumulate={accumulate})"),
            Channel::Reducer(_) => f.write_str("Reducer(..)"),
        }
    }
}

impl Channel {
    /// Build a reducer channel from a folding function.
    ///
    /// The reducer receives the current channel value and the incoming write and
    /// returns the merged value.
    pub fn reducer<F>(reducer: F) -> Self
    where
        F: Fn(&Value, &Value) -> Result<Value> + Send + Sync + 'static,
    {
        Channel::Reducer(Reducer::new(reducer))
    }

    /// Build a reducer channel from a prebuilt [`Reducer`].
    pub fn with_reducer(reducer: Reducer) -> Self {
        Channel::Reducer(reducer)
    }

    /// Topic channel: `accumulate=true` keeps values across supersteps (LangGraph
    /// `Topic(accumulate=True)`), `false` keeps only the current superstep's values.
    pub fn topic(accumulate: bool) -> Self {
        Channel::Topic { accumulate }
    }

    /// Last write wins and concurrent writes are allowed (`AnyValue`).
    pub fn any_value() -> Self {
        Channel::AnyValue
    }

    /// Value is visible for one superstep and then cleared (`EphemeralValue`).
    pub fn ephemeral_value() -> Self {
        Channel::EphemeralValue
    }

    /// Value is never persisted to checkpoints (`UntrackedValue`).
    pub fn untracked_value() -> Self {
        Channel::UntrackedValue
    }

    /// Open after every name in `names` wrote to the channel (`NamedBarrierValue`).
    pub fn named_barrier<I, T>(names: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Channel::NamedBarrierValue {
            names: names.into_iter().map(Into::into).collect(),
        }
    }

    /// The value a channel holds before anything writes to it, when it has one.
    ///
    /// LangGraph's aggregating channels start from a default (`[]` for `operator.add` on a list),
    /// which shows up in snapshots and in what nodes read.
    pub fn default_value(&self) -> Option<Value> {
        match self {
            Channel::Reducer(reducer) | Channel::Delta { reducer, .. } => {
                reducer.default_value().cloned()
            }
            Channel::Topic { .. } => Some(Value::Array(Vec::new())),
            _ => None,
        }
    }

    /// Whether the channel value is dropped before the next superstep.
    pub(crate) fn is_ephemeral(&self) -> bool {
        matches!(self, Channel::EphemeralValue)
    }

    /// Whether the channel value is excluded from checkpoints.
    pub(crate) fn is_untracked(&self) -> bool {
        matches!(self, Channel::UntrackedValue)
    }

    /// Delta-accumulating channel: writes are folded by `reducer`.
    pub fn delta(reducer: Reducer) -> Self {
        Channel::Delta {
            reducer,
            snapshot_frequency: 1,
        }
    }

    /// Set the delta snapshot frequency (stored for API compatibility).
    pub fn with_snapshot_frequency(mut self, frequency: usize) -> Self {
        if let Channel::Delta {
            snapshot_frequency, ..
        } = &mut self
        {
            *snapshot_frequency = frequency.max(1);
        }
        self
    }

    /// Last-value channel whose writes are published only when the graph finishes.
    pub fn last_value_after_finish() -> Self {
        Channel::LastValueAfterFinish
    }

    /// Whether writes to this channel are held until the graph finishes.
    pub(crate) fn is_finish_value(&self) -> bool {
        matches!(self, Channel::LastValueAfterFinish)
    }

    /// Whether this channel is a named barrier (LangGraph `NamedBarrierValue`).
    pub(crate) fn is_named_barrier(&self) -> bool {
        matches!(self, Channel::NamedBarrierValue { .. })
    }

    /// Whether the channel value belongs in a checkpoint.
    pub fn persists(&self) -> bool {
        !self.is_ephemeral() && !self.is_untracked()
    }

    /// Merge `incoming` writes into `current`.
    /// Apply owned writes, moving the winning value when the channel keeps only one.
    ///
    /// `apply` receives `&[Value]` and must clone the winner; the executor owns the writes,
    /// so this variant avoids that copy (it matters for large channel values).
    pub(crate) fn apply_owned(
        &self,
        current: Option<&Value>,
        mut incoming: Vec<Value>,
    ) -> Result<Option<Value>> {
        match self {
            Channel::LastValue
            | Channel::AnyValue
            | Channel::EphemeralValue
            | Channel::UntrackedValue => Ok(incoming.pop().or_else(|| current.cloned())),
            _ => self.apply(current, &incoming),
        }
    }

    pub(crate) fn apply(
        &self,
        current: Option<&Value>,
        incoming: &[Value],
    ) -> Result<Option<Value>> {
        match self {
            Channel::LastValue
            | Channel::AnyValue
            | Channel::EphemeralValue
            | Channel::UntrackedValue => Ok(incoming.last().cloned().or_else(|| current.cloned())),
            Channel::NamedBarrierValue { names } => {
                let mut seen: Vec<String> = match current {
                    None | Some(Value::Null) | Some(Value::Bool(false)) => Vec::new(),
                    Some(Value::Bool(true)) => {
                        // Re-arm when a new round of writes arrives after the barrier opened.
                        Vec::new()
                    }
                    Some(Value::Array(values)) => values
                        .iter()
                        .filter_map(|value| value.as_str().map(str::to_owned))
                        .collect(),
                    Some(other) => {
                        return Err(Error::other(format!(
                            "channel with `NamedBarrierValue` semantics expects an array, found `{other}`"
                        )));
                    }
                };
                for value in incoming {
                    let name = value.as_str().ok_or_else(|| {
                        Error::other(
                            "named barrier writes must be channel values containing the writer name",
                        )
                    })?;
                    if !names.iter().any(|candidate| candidate == name) {
                        return Err(Error::other(format!(
                            "named barrier received unknown writer `{name}`"
                        )));
                    }
                    if !seen.iter().any(|existing| existing == name) {
                        seen.push(name.to_string());
                    }
                }
                let complete = names
                    .iter()
                    .all(|name| seen.iter().any(|seen| seen == name));
                if complete {
                    Ok(Some(Value::Bool(true)))
                } else {
                    Ok(Some(Value::Array(
                        seen.into_iter().map(Value::String).collect(),
                    )))
                }
            }
            Channel::Topic { accumulate } => {
                let mut accumulated = if *accumulate {
                    match current {
                        None | Some(Value::Null) => Vec::new(),
                        Some(Value::Array(values)) => values.clone(),
                        Some(other) => {
                            return Err(Error::other(format!(
                                "channel with `Topic` semantics expects an array, found `{other}`"
                            )));
                        }
                    }
                } else {
                    Vec::new()
                };
                for value in incoming {
                    match value {
                        Value::Array(values) => accumulated.extend(values.iter().cloned()),
                        other => accumulated.push(other.clone()),
                    }
                }
                Ok(Some(Value::Array(accumulated)))
            }
            Channel::LastValueAfterFinish => {
                Ok(incoming.last().cloned().or_else(|| current.cloned()))
            }
            Channel::Delta { reducer, .. } | Channel::Reducer(reducer) => {
                let Some(first) = incoming.first() else {
                    return Ok(current.cloned());
                };
                // LangGraph's `BinaryOperatorAggregate` starts from the channel's empty value
                // (`[]` for `add_messages`, `0` for `add`) and folds *every* write through the
                // reducer — that is what lets a reducer de-duplicate the submitted input. A channel
                // without a declared default uses the shape of the first write as that empty value.
                let start = current
                    .cloned()
                    .or_else(|| reducer.default_value().cloned())
                    .or_else(|| empty_like(first));
                // Values with no JSON form (sets, datetimes, Python objects) arrive as a stash
                // marker, so there is no meaningful empty value: the first write initializes the
                // channel, exactly like LangGraph's `BinaryOperatorAggregate` without a default.
                let Some(mut accumulated) = start else {
                    let mut accumulated = first.clone();
                    for value in &incoming[1..] {
                        accumulated = (reducer.0.as_ref())(&accumulated, value)?;
                    }
                    return Ok(Some(accumulated));
                };
                for value in incoming {
                    accumulated = (reducer.0.as_ref())(&accumulated, value)?;
                }
                Ok(Some(accumulated))
            }
        }
    }
}

/// The empty value of the kind of collection/scalar `value` holds.
///
/// Used when a reducer channel has no declared default: LangGraph folds from the annotation's empty
/// value, and the submitted value's shape is the closest equivalent available here.
fn empty_like(value: &Value) -> Option<Value> {
    match value {
        Value::Array(_) => Some(Value::Array(Vec::new())),
        // A stash marker (`{"__langgraph_rs_object__": id}`, used for values with no JSON form)
        // must not be mistaken for a mapping: there is no empty value to fold from.
        Value::Object(map) if map.contains_key("__langgraph_rs_object__") => None,
        Value::Object(_) => Some(Value::Object(serde_json::Map::new())),
        Value::Number(number) => Some(if number.is_f64() {
            Value::from(0.0)
        } else {
            Value::from(0)
        }),
        Value::String(_) => Some(Value::from("")),
        Value::Null | Value::Bool(_) => None,
    }
}

/// Folding function stored by a [`Reducer`].
pub(crate) type ReducerFn = Arc<dyn Fn(&Value, &Value) -> Result<Value> + Send + Sync>;

/// A user supplied binary reducer.
#[derive(Clone)]
pub struct Reducer(pub(crate) ReducerFn, pub(crate) Option<Value>);

impl Reducer {
    /// The channel's starting value, when the reducer has one (`[]` for append, `0` for add).
    pub(crate) fn default_value(&self) -> Option<&Value> {
        self.1.as_ref()
    }

    /// Attach the channel's starting value (LangGraph's `BinaryOperatorAggregate` default).
    pub fn with_default(mut self, value: Value) -> Self {
        self.1 = Some(value);
        self
    }
}

impl fmt::Debug for Reducer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Reducer(..)")
    }
}

impl Reducer {
    /// Wrap a folding function into a reducer.
    pub fn new<F>(reducer: F) -> Self
    where
        F: Fn(&Value, &Value) -> Result<Value> + Send + Sync + 'static,
    {
        Reducer(Arc::new(reducer), None)
    }
}

/// A removal marker for `add_messages`-style reducers (LangGraph `RemoveMessage`).
///
/// Serializes to `{"type": "remove", "id": ...}`; the `messages` reducer removes the
/// message with the same id instead of merging it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoveMessage {
    /// Id of the message to remove.
    pub id: String,
}

impl RemoveMessage {
    /// Create a removal marker.
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }

    /// JSON representation understood by the `messages` reducer.
    pub fn to_value(&self) -> Value {
        serde_json::json!({ "type": "remove", "id": self.id })
    }
}

impl From<RemoveMessage> for Value {
    fn from(message: RemoveMessage) -> Self {
        message.to_value()
    }
}

impl Serialize for RemoveMessage {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("RemoveMessage", 2)?;
        state.serialize_field("type", "remove")?;
        state.serialize_field("id", &self.id)?;
        state.end()
    }
}

/// Ready made reducers for the most common state shapes.
pub mod reducers {
    use super::*;

    /// The id of a stored message.
    ///
    /// Messages arrive either as LangChain constructor dicts (their fields under `kwargs`) or as
    /// plain mappings, so both shapes are checked.
    fn message_id(message: &serde_json::Value) -> Option<&str> {
        let object = message.as_object()?;
        object
            .get("id")
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                object
                    .get("kwargs")
                    .and_then(serde_json::Value::as_object)
                    .and_then(|kwargs| kwargs.get("id"))
                    .and_then(serde_json::Value::as_str)
            })
    }

    /// Concatenate arrays: `[a, b] + [c] == [a, b, c]`.
    ///
    /// This is the Rust equivalent of `operator.add` on message lists.
    pub fn append() -> Reducer {
        Reducer::new(|current, incoming| {
            let mut accumulated = match current {
                Value::Array(values) => values.clone(),
                Value::Null => Vec::new(),
                other => {
                    return Err(Error::other(format!(
                        "reducer `append` expected an array, found `{other}`"
                    )));
                }
            };
            match incoming {
                Value::Array(values) => accumulated.extend(values.iter().cloned()),
                other => {
                    return Err(Error::other(format!(
                        "reducer `append` expected an array write, found `{other}`"
                    )));
                }
            }
            Ok(Value::Array(accumulated))
        })
        .with_default(Value::Array(Vec::new()))
    }

    /// Numerically add values.
    pub fn add() -> Reducer {
        Reducer::new(|current, incoming| {
            let left = current.as_f64().unwrap_or(0.0);
            let right = incoming.as_f64().unwrap_or(0.0);
            let sum = left + right;
            if sum.fract() == 0.0 && sum.abs() < i64::MAX as f64 {
                Ok(Value::from(sum as i64))
            } else {
                Ok(Value::from(sum))
            }
        })
    }


/// LangGraph's sentinel id that clears the whole message list.
const REMOVE_ALL_MESSAGES: &str = "__remove_all__";

/// The id a `RemoveMessage` marker asks to delete, when `message` is one.
///
/// `Some(Some(id))` removes that id, `Some(None)` is a marker without an id, and `None` means the
/// value is an ordinary message write.
fn removal_id(message: &Value) -> Option<Option<String>> {
    let object = message.as_object()?;
    // Our own marker, or LangChain's serialized `RemoveMessage` (its fields live under `kwargs`
    // and the class is recorded in `id`).
    let kind = object.get("type").and_then(Value::as_str);
    let kwargs = object.get("kwargs").and_then(Value::as_object);
    let class_name = object
        .get("id")
        .and_then(Value::as_array)
        .and_then(|path| path.last())
        .and_then(Value::as_str);
    let remove = kind == Some("remove")
        || kwargs
            .and_then(|kwargs| kwargs.get("type"))
            .and_then(Value::as_str)
            == Some("remove")
        || class_name == Some("RemoveMessage")
        || object
            .get("__remove__")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    remove.then(|| message_id(message).map(str::to_owned))
}

/// Give every incoming message an id, so a later write that echoes it dedupes by id.
///
/// Engine-side writes (`ToolNode` output, dict input, `update_state`) never pass through the Python
/// encoder, so ids have to be assigned here as well as there.
fn ensure_message_ids(incoming: &mut [serde_json::Value]) {
    for message in incoming.iter_mut() {
        // Only message-like objects carry `content`; plain state values are left alone.
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        let has_content = object.contains_key("content")
            || object
                .get("kwargs")
                .and_then(serde_json::Value::as_object)
                .map(|kwargs| kwargs.contains_key("content"))
                .unwrap_or(false);
        if !has_content {
            continue;
        }
        let filled = if object.contains_key("kwargs") {
            let kwargs = object
                .get_mut("kwargs")
                .and_then(serde_json::Value::as_object_mut);
            match kwargs {
                Some(kwargs) => {
                    if kwargs.get("id").map(serde_json::Value::is_string).unwrap_or(false) {
                        continue;
                    }
                    kwargs.insert("id".to_string(), serde_json::Value::String(new_message_id()));
                    true
                }
                None => false,
            }
        } else {
            if object.get("id").map(serde_json::Value::is_string).unwrap_or(false) {
                continue;
            }
            object.insert("id".to_string(), serde_json::Value::String(new_message_id()));
            true
        };
        let _ = filled;
    }
}

/// A fresh message id (`uuid4`, dashed like LangGraph's).
fn new_message_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or_default()
        ^ (COUNTER.fetch_add(1, Ordering::Relaxed) << 17)
        ^ std::process::id() as u64;
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let a = next();
    let b = next();
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        (a >> 32) as u32,
        ((a >> 16) & 0xffff) as u16,
        (a & 0xffff) as u16,
        ((b >> 48) & 0xffff) as u16,
        b & 0xffff_ffff_ffff
    )
}

    /// Merge chat-message style arrays by `id`.
    ///
    /// Mirrors LangGraph's `add_messages`: `RemoveMessage` markers are collected first and applied
    /// after the loop (so remove-then-add keeps its position and a repeated removal is a no-op),
    /// `REMOVE_ALL_MESSAGES` clears the list, messages without an id get one, and a write that shares
    /// an id replaces the existing message in place.
    pub fn messages() -> Reducer {
        Reducer::new(|current, incoming| {
            let mut accumulated = match current {
                Value::Array(values) => values.clone(),
                Value::Null => Vec::new(),
                other => {
                    return Err(Error::other(format!(
                        "reducer `messages` expected an array, found `{other}`"
                    )));
                }
            };
            let mut incoming: Vec<Value> = match incoming {
                Value::Array(values) => values.clone(),
                other => {
                    return Err(Error::other(format!(
                        "reducer `messages` expected an array write, found `{other}`"
                    )));
                }
            };
            // LangGraph gives every stored message an id, so an echoed list dedupes.
            ensure_message_ids(&mut incoming);
            let mut removed: Vec<String> = Vec::new();
            let mut remove_all = false;
            let mut writes: Vec<Value> = Vec::new();
            for message in incoming.iter().cloned() {
                match removal_id(&message) {
                    Some(Some(id)) if id == REMOVE_ALL_MESSAGES => remove_all = true,
                    Some(Some(id)) => {
                        if !removed.contains(&id) {
                            removed.push(id);
                        }
                    }
                    Some(None) => {}
                    None => writes.push(message),
                }
            }
            if remove_all {
                accumulated.clear();
                removed.clear();
            }
            // Writes apply first, so a message re-added under a removed id survives at that
            // position (LangGraph only deletes the messages the update did not replace).
            let mut replaced: Vec<String> = Vec::new();
            let mut index: std::collections::HashMap<String, usize> = accumulated
                .iter()
                .enumerate()
                .filter_map(|(index, message)| {
                    message_id(message).map(|id| (id.to_string(), index))
                })
                .collect();
            for message in writes {
                match message_id(&message) {
                    Some(id) => match index.get(id).copied() {
                        Some(position) => {
                            replaced.push(id.to_string());
                            accumulated[position] = message;
                        }
                        None => {
                            index.insert(id.to_string(), accumulated.len());
                            accumulated.push(message);
                        }
                    },
                    None => accumulated.push(message),
                }
            }
            for id in &removed {
                if replaced.iter().any(|kept| kept == id) {
                    continue;
                }
                let before = accumulated.len();
                accumulated.retain(|candidate| message_id(candidate) != Some(id.as_str()));
                if accumulated.len() == before {
                    // LangGraph refuses to delete an id that is not there.
                    return Err(Error::validation(format!(
                        "Attempting to delete a message with an ID that doesn't exist ('{id}')"
                    )));
                }
            }
            Ok(Value::Array(accumulated))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn last_value_keeps_the_latest_write() {
        let channel = Channel::LastValue;
        let merged = channel
            .apply(Some(&json!(1)), &[json!(2), json!(3)])
            .unwrap();
        assert_eq!(merged, Some(json!(3)));
    }

    #[test]
    fn named_barrier_opens_after_all_names() {
        let channel = Channel::named_barrier(["a", "b"]);
        let current = json!(["a"]);
        let opened = channel.apply(Some(&current), &[json!("b")]).unwrap();
        assert_eq!(opened, Some(json!(true)));
        let pending = channel.apply(None, &[json!("a")]).unwrap();
        assert_eq!(pending, Some(json!(["a"])));
    }

    #[test]
    fn topic_appends_writes() {
        let channel = Channel::topic(true);
        let merged = channel
            .apply(Some(&json!(["a"])), &[json!("b"), json!("c")])
            .unwrap();
        assert_eq!(merged, Some(json!(["a", "b", "c"])));
    }

    #[test]
    fn reducer_initializes_with_the_first_write() {
        let channel = Channel::with_reducer(reducers::append());
        let merged = channel.apply(None, &[json!([1, 2])]).unwrap();
        assert_eq!(merged, Some(json!([1, 2])));

        let merged = channel.apply(Some(&json!([1, 2])), &[json!([3])]).unwrap();
        assert_eq!(merged, Some(json!([1, 2, 3])));
    }

    #[test]
    fn messages_reducer_replaces_by_id() {
        let channel = Channel::with_reducer(reducers::messages());
        let current = json!([{ "id": "a", "content": "old" }]);
        let write = json!([{ "id": "a", "content": "new" }, { "id": "b", "content": "hi" }]);
        let merged = channel.apply(Some(&current), &[write]).unwrap().unwrap();
        assert_eq!(
            merged,
            json!([
                { "id": "a", "content": "new" },
                { "id": "b", "content": "hi" }
            ])
        );
    }
    #[test]
    fn named_barrier_rearms_on_new_writes() {
        let channel = Channel::named_barrier(["a", "b"]);
        let opened = channel.apply(None, &[json!("a"), json!("b")]).unwrap();
        assert_eq!(opened, Some(json!(true)));
        let pending = channel.apply(Some(&json!(true)), &[json!("a")]).unwrap();
        assert_eq!(pending, Some(json!(["a"])));
        let reopened = channel.apply(Some(&json!(["a"])), &[json!("b")]).unwrap();
        assert_eq!(reopened, Some(json!(true)));
    }
}
