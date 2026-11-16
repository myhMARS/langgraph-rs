//! Channel semantics: how node writes are merged into a state field.
//!
//! A channel is the unit of state. Every node returns a map of `channel -> value`
//! writes and the channel decides how a write is folded into the value produced by
//! the previous superstep. This is the Rust equivalent of LangGraph's
//! `Annotated[..., reducer]` state annotations.

use std::fmt;
use std::sync::Arc;

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
    /// Append every write to a JSON array (useful for event logs / pub-sub style fields).
    Topic,
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
            Channel::Topic => f.write_str("Topic"),
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

    /// Whether the channel value belongs in a checkpoint.
    pub fn persists(&self) -> bool {
        !self.is_ephemeral() && !self.is_untracked()
    }

    /// Merge `incoming` writes into `current`.
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
                        return Ok(Some(Value::Bool(true)));
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
            Channel::Topic => {
                let mut accumulated = match current {
                    None | Some(Value::Null) => Vec::new(),
                    Some(Value::Array(values)) => values.clone(),
                    Some(other) => {
                        return Err(Error::other(format!(
                            "channel with `Topic` semantics expects an array, found `{other}`"
                        )));
                    }
                };
                accumulated.extend(incoming.iter().cloned());
                Ok(Some(Value::Array(accumulated)))
            }
            Channel::LastValueAfterFinish => {
                Ok(incoming.last().cloned().or_else(|| current.cloned()))
            }
            Channel::Delta { reducer, .. } | Channel::Reducer(reducer) => {
                let Some(first) = incoming.first() else {
                    return Ok(current.cloned());
                };
                let mut accumulated = current.cloned().unwrap_or_else(|| first.clone());
                // When there is no current value the first write already initializes the
                // channel, exactly like LangGraph's `BinaryOperatorAggregate`.
                let skip = usize::from(current.is_none());
                for value in &incoming[skip..] {
                    accumulated = (reducer.0.as_ref())(&accumulated, value)?;
                }
                Ok(Some(accumulated))
            }
        }
    }
}

/// Folding function stored by a [`Reducer`].
pub(crate) type ReducerFn = Arc<dyn Fn(&Value, &Value) -> Result<Value> + Send + Sync>;

/// A user supplied binary reducer.
#[derive(Clone)]
pub struct Reducer(pub(crate) ReducerFn);

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
        Reducer(Arc::new(reducer))
    }
}

/// Ready made reducers for the most common state shapes.
pub mod reducers {
    use super::*;

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

    /// Merge chat-message style arrays by `id`.
    ///
    /// Messages without an `id` are appended; messages that share an `id` with an
    /// existing message replace it in place. This mirrors LangGraph's
    /// `add_messages` reducer.
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
            let incoming = match incoming {
                Value::Array(values) => values.clone(),
                other => {
                    return Err(Error::other(format!(
                        "reducer `messages` expected an array write, found `{other}`"
                    )));
                }
            };
            for message in incoming {
                let id = message
                    .as_object()
                    .and_then(|object| object.get("id"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                match id {
                    Some(id) => {
                        let existing = accumulated.iter().position(|candidate| {
                            candidate
                                .as_object()
                                .and_then(|object| object.get("id"))
                                .and_then(Value::as_str)
                                == Some(id.as_str())
                        });
                        match existing {
                            Some(index) => accumulated[index] = message,
                            None => accumulated.push(message),
                        }
                    }
                    None => accumulated.push(message),
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
        let channel = Channel::Topic;
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
}
