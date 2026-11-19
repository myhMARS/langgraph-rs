//! Typed graph state and its channel schema.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

use crate::channels::Channel;
use crate::{Error, Result};

/// Declares every state channel and how writes to it are merged.
///
/// # Example
///
/// ```
/// use langgraph_rs::{StateSchema, reducers};
///
/// let schema = StateSchema::new()
///     .last_value("question")
///     .topic("events")
///     .with_reducer("messages", reducers::append());
/// assert_eq!(schema.names().count(), 3);
/// ```
#[derive(Clone, Debug, Default)]
pub struct StateSchema {
    channels: IndexMap<String, Channel>,
    allow_implicit_channels: bool,
    input_channels: Option<std::collections::HashSet<String>>,
    output_channels: Option<std::collections::HashSet<String>>,
    private_channels: std::collections::HashSet<String>,
    managed_values: bool,
}

impl StateSchema {
    /// Create an empty, strict schema: writes to undeclared channels fail validation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a schema that accepts any channel and treats it as [`Channel::LastValue`].
    ///
    /// This is used by [`JsonState`](crate::JsonState) and keeps ad-hoc graphs convenient.
    pub fn dynamic() -> Self {
        Self {
            channels: IndexMap::new(),
            allow_implicit_channels: true,
            ..Self::default()
        }
    }

    /// Restrict the channels accepted as graph input (LangGraph input schema).
    pub fn with_input_channels<I, T>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.input_channels = Some(names.into_iter().map(Into::into).collect());
        self
    }

    /// Restrict the channels returned to the caller (LangGraph output schema).
    pub fn with_output_channels<I, T>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.output_channels = Some(names.into_iter().map(Into::into).collect());
        self
    }

    /// Inject LangGraph managed values (`remaining_steps`, `is_last_step`) into node state.
    ///
    /// The injected channels are private: they are visible to nodes but never persisted or
    /// returned to the caller.
    pub fn with_managed_values(mut self) -> Self {
        self.managed_values = true;
        self.private_channels.insert("remaining_steps".to_string());
        self.private_channels.insert("is_last_step".to_string());
        self
    }

    /// Whether managed values are injected before each superstep.
    pub(crate) fn managed_values(&self) -> bool {
        self.managed_values
    }

    /// Mark channels as internal: never returned and never persisted.
    pub fn with_private_channels<I, T>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.private_channels = names.into_iter().map(Into::into).collect();
        self
    }

    /// Keep only declared input channels (private channels are never inputs).
    pub(crate) fn filter_input(&self, values: Map<String, Value>) -> Map<String, Value> {
        values
            .into_iter()
            .filter(|(name, _)| {
                !self.private_channels.contains(name)
                    && self
                        .input_channels
                        .as_ref()
                        .is_none_or(|allowed| allowed.contains(name.as_str()))
            })
            .collect()
    }

    /// Filter final values through the output schema and drop private channels.
    pub(crate) fn output_values(&self, values: &Map<String, Value>) -> Map<String, Value> {
        values
            .iter()
            .filter(|(name, _)| {
                let name: &str = name.as_str();
                !self.private_channels.contains(name)
                    && self
                        .output_channels
                        .as_ref()
                        .is_none_or(|allowed| allowed.contains(name))
            })
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }

    /// `true` when output filtering would change values.
    pub(crate) fn has_output_filter(&self) -> bool {
        self.output_channels.is_some() || !self.private_channels.is_empty()
    }

    /// Add a channel with an explicit merge strategy.
    pub fn with_channel(mut self, name: impl Into<String>, channel: Channel) -> Self {
        self.channels.insert(name.into(), channel);
        self
    }

    /// Declare a plain last-write-wins channel.
    pub fn last_value(self, name: impl Into<String>) -> Self {
        self.with_channel(name, Channel::LastValue)
    }

    /// Declare a topic channel (current-superstep values, LangGraph default).
    pub fn topic(self, name: impl Into<String>) -> Self {
        self.with_channel(name, Channel::topic(false))
    }

    /// Declare an appending topic channel that keeps values across supersteps.
    pub fn topic_accumulate(self, name: impl Into<String>) -> Self {
        self.with_channel(name, Channel::topic(true))
    }

    /// Declare a last-write-wins channel that tolerates concurrent writes.
    pub fn any_value(self, name: impl Into<String>) -> Self {
        self.with_channel(name, Channel::AnyValue)
    }

    /// Declare a channel whose value is cleared before the next superstep.
    pub fn ephemeral_value(self, name: impl Into<String>) -> Self {
        self.with_channel(name, Channel::EphemeralValue)
    }

    /// Declare a channel that is never written to checkpoints.
    pub fn untracked_value(self, name: impl Into<String>) -> Self {
        self.with_channel(name, Channel::UntrackedValue)
    }

    /// Declare a delta-accumulating channel folded by `reducer`.
    pub fn delta(self, name: impl Into<String>, reducer: crate::Reducer) -> Self {
        self.with_channel(name, Channel::delta(reducer))
    }

    /// Declare a channel whose writes become visible only when the graph finishes.
    pub fn last_value_after_finish(self, name: impl Into<String>) -> Self {
        self.with_channel(name, Channel::LastValueAfterFinish)
    }

    /// Whether writes to `name` are held until the graph finishes.
    pub(crate) fn is_finish_value(&self, name: &str) -> bool {
        self.channels
            .get(name)
            .is_some_and(Channel::is_finish_value)
    }

    /// Declare a topic channel that only keeps the current superstep's values.
    pub fn topic_current(self, name: impl Into<String>) -> Self {
        self.with_channel(name, Channel::topic(false))
    }

    /// Declare a barrier channel that opens after `names` wrote to it.
    pub fn named_barrier<I, T>(self, name: impl Into<String>, names: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.with_channel(name, Channel::named_barrier(names))
    }

    /// Whether values of `name` belong in a checkpoint.
    pub(crate) fn persists(&self, name: &str) -> bool {
        if self.private_channels.contains(name) {
            return false;
        }
        self.channels
            .get(name)
            .map(Channel::persists)
            .unwrap_or(true)
    }

    /// Drop every ephemeral channel from `values`; returns whether anything changed.
    pub(crate) fn clear_ephemeral(&self, values: &mut Map<String, Value>) -> bool {
        let mut changed = false;
        for (name, channel) in &self.channels {
            if channel.is_ephemeral() && values.remove(name).is_some() {
                changed = true;
            }
        }
        changed
    }

    /// Drop non-accumulating topic values that this superstep did not rewrite.
    ///
    /// LangGraph calls `Topic.update` after the superstep's writes are applied: a topic
    /// written this step keeps its values, an untouched one is emptied. Values written in
    /// superstep S are therefore still visible to the tasks of S+1.
    pub(crate) fn clear_step_topics(
        &self,
        values: &mut Map<String, Value>,
        written: &std::collections::HashSet<String>,
    ) -> bool {
        let mut changed = false;
        for (name, channel) in &self.channels {
            if matches!(
                channel,
                crate::channels::Channel::Topic { accumulate: false }
            ) && !written.contains(name)
                && values.remove(name).is_some()
            {
                changed = true;
            }
        }
        changed
    }

    /// Clone of `values` without untracked/ephemeral channels (for checkpoints).
    pub(crate) fn persisted_values(&self, values: &Map<String, Value>) -> Map<String, Value> {
        values
            .iter()
            .filter(|(name, _)| self.persists(name))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }

    /// Declare a channel folded by a user supplied reducer.
    pub fn reducer<F>(self, name: impl Into<String>, reducer: F) -> Self
    where
        F: Fn(&Value, &Value) -> Result<Value> + Send + Sync + 'static,
    {
        self.with_channel(name, Channel::reducer(reducer))
    }

    /// Declare a channel folded by a prebuilt [`Reducer`](crate::Reducer).
    pub fn with_reducer(self, name: impl Into<String>, reducer: crate::Reducer) -> Self {
        self.with_channel(name, Channel::Reducer(reducer))
    }

    /// Look up a channel definition.
    pub fn channel(&self, name: &str) -> Option<&Channel> {
        self.channels.get(name)
    }

    /// Returns `true` when the channel is declared.
    pub fn contains(&self, name: &str) -> bool {
        self.channels.contains_key(name)
    }

    /// Iterate over declared channel names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.channels.keys().map(String::as_str)
    }

    /// Number of declared channels.
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// Returns `true` when no channel is declared.
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /// Returns `true` when undeclared channels are accepted as last-value channels.
    pub fn allows_implicit_channels(&self) -> bool {
        self.allow_implicit_channels
    }

    /// Resolve a channel, or `Ok(None)` when implicit channels are allowed.
    pub(crate) fn resolve(&self, name: &str) -> Result<Option<&Channel>> {
        match self.channels.get(name) {
            Some(channel) => Ok(Some(channel)),
            None if self.allow_implicit_channels => Ok(None),
            None => Err(Error::UnknownChannel(name.to_string())),
        }
    }
}

/// Convenience state with a single `messages` channel merged by `reducers::messages`.
///
/// Equivalent to LangGraph's `MessagesState`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MessagesState {
    /// Chat messages merged by `id` (removals supported through `RemoveMessage`).
    pub messages: Vec<Value>,
}

impl State for MessagesState {
    fn schema() -> StateSchema {
        StateSchema::new().with_reducer("messages", crate::reducers::messages())
    }
}

/// A graph state.
///
/// Implement this trait for any `Serialize + DeserializeOwned + Default` type and
/// describe its channels in [`State::schema`]. The engine converts the typed value to
/// JSON channels internally, so state can be checkpointed and streamed as plain JSON.
pub trait State: Clone + Default + Serialize + DeserializeOwned + Send + Sync + 'static {
    /// Channel schema for this state type.
    fn schema() -> StateSchema;
}

/// A schemaless JSON state backed by a map.
///
/// Useful for graphs whose channels are only known at runtime. The graph must be
/// built with [`StateGraph::with_schema`](crate::StateGraph::with_schema) to declare
/// reducers; undeclared channels behave as last-value channels.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct JsonState(pub Map<String, Value>);

impl JsonState {
    /// Create an empty JSON state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read a channel value.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key)
    }

    /// Read and deserialize a channel value.
    pub fn get_as<T: DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.0
            .get(key)
            .and_then(|value| serde_json::from_value(value.clone()).ok())
    }

    /// Insert a channel value.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) {
        self.0.insert(key.into(), value);
    }

    /// Builder variant of [`JsonState::insert`].
    pub fn with(mut self, key: impl Into<String>, value: Value) -> Self {
        self.insert(key, value);
        self
    }

    /// Iterate over the channels.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.0.iter()
    }
}

impl From<Map<String, Value>> for JsonState {
    fn from(map: Map<String, Value>) -> Self {
        JsonState(map)
    }
}

impl From<JsonState> for Map<String, Value> {
    fn from(state: JsonState) -> Self {
        state.0
    }
}

impl State for JsonState {
    fn schema() -> StateSchema {
        StateSchema::dynamic()
    }
}

/// Serialize a typed state into its channel values.
pub(crate) fn state_to_values<S: Serialize>(state: &S) -> Result<Map<String, Value>> {
    match serde_json::to_value(state)? {
        Value::Object(values) => Ok(values),
        other => Err(Error::NonObjectState(other.to_string())),
    }
}

std::thread_local! {
    static DEFAULT_STATE_VALUES: std::cell::RefCell<
        std::collections::HashMap<std::any::TypeId, Map<String, Value>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

fn default_state_values<S: DeserializeOwned + Default + Serialize + 'static>()
-> Result<Map<String, Value>> {
    let type_id = std::any::TypeId::of::<S>();
    if let Some(cached) = DEFAULT_STATE_VALUES.with(|cache| cache.borrow().get(&type_id).cloned()) {
        return Ok(cached);
    }
    let defaults = match serde_json::to_value(S::default())? {
        Value::Object(defaults) => defaults,
        _ => Map::new(),
    };
    DEFAULT_STATE_VALUES.with(|cache| {
        cache.borrow_mut().insert(type_id, defaults.clone());
    });
    Ok(defaults)
}

/// Deserialize channel values back into the typed state.
pub(crate) fn values_to_state<S: DeserializeOwned + Default + Serialize + 'static>(
    values: &Map<String, Value>,
) -> Result<S> {
    // Fill channels missing from the checkpoint (untracked/ephemeral values) with the
    // defaults from `S::default()` so state types keep strict field requirements.
    let mut merged = default_state_values::<S>()?;
    merged.extend(values.clone());
    Ok(serde_json::from_value(Value::Object(merged))?)
}

/// A single channel write produced by a node.
#[derive(Clone, Debug)]
pub(crate) struct ChannelUpdate {
    pub(crate) channel: String,
    pub(crate) value: Value,
    pub(crate) node: String,
}

/// Merge all writes produced by a superstep into `values`.
pub(crate) fn apply_updates(
    schema: &StateSchema,
    values: &mut Map<String, Value>,
    updates: Vec<ChannelUpdate>,
) -> Result<()> {
    // A single write per superstep is the common case: skip building the grouping maps.
    if updates.len() == 1 {
        let update = updates.into_iter().next().expect("length checked");
        let Some(definition) = schema.resolve(&update.channel)? else {
            // Dynamic schemas accept implicit channels; the write becomes a `last_value`.
            values.insert(update.channel, update.value);
            return Ok(());
        };
        if let Channel::Topic { accumulate: true } = definition {
            let mut accumulated = match values.remove(&update.channel) {
                Some(Value::Array(items)) => items,
                _ => Vec::new(),
            };
            match update.value {
                Value::Array(items) => accumulated.extend(items),
                other => accumulated.push(other),
            }
            values.insert(update.channel, Value::Array(accumulated));
            return Ok(());
        }
        let current = values.get(&update.channel);
        if let Some(merged) = definition.apply_owned(current, vec![update.value])? {
            values.insert(update.channel, merged);
        }
        return Ok(());
    }
    let mut grouped: IndexMap<String, (Vec<Value>, Vec<String>)> = IndexMap::new();
    for update in updates {
        let entry = grouped.entry(update.channel).or_default();
        entry.0.push(update.value);
        entry.1.push(update.node);
    }

    for (channel, (incoming, nodes)) in grouped {
        match schema.resolve(&channel)? {
            Some(definition) => {
                // Accumulating topics append in place instead of cloning the whole
                // accumulated array on every merge.
                if let Channel::Topic { accumulate: true } = definition {
                    let mut accumulated = match values.remove(&channel) {
                        Some(Value::Array(items)) => items,
                        _ => Vec::new(),
                    };
                    for value in incoming {
                        match value {
                            Value::Array(items) => accumulated.extend(items),
                            other => accumulated.push(other),
                        }
                    }
                    values.insert(channel, Value::Array(accumulated));
                    continue;
                }
                if matches!(
                    definition,
                    Channel::LastValue | Channel::EphemeralValue | Channel::UntrackedValue
                ) && incoming.len() > 1
                {
                    return Err(Error::ConcurrentUpdate {
                        channel,
                        nodes: nodes.join(", "),
                    });
                }
                let current = values.get(&channel);
                if let Some(merged) = definition.apply_owned(current, incoming)? {
                    values.insert(channel, merged);
                }
            }
            None => {
                if incoming.len() > 1 {
                    return Err(Error::ConcurrentUpdate {
                        channel,
                        nodes: nodes.join(", "),
                    });
                }
                if let Some(value) = incoming.into_iter().next() {
                    values.insert(channel, value);
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn last_value_channel_rejects_concurrent_writes() {
        let schema = StateSchema::new().last_value("value");
        let mut values = Map::new();
        let updates = vec![
            ChannelUpdate {
                channel: "value".into(),
                value: json!(1),
                node: "a".into(),
            },
            ChannelUpdate {
                channel: "value".into(),
                value: json!(2),
                node: "b".into(),
            },
        ];
        let error = apply_updates(&schema, &mut values, updates).unwrap_err();
        assert!(matches!(error, Error::ConcurrentUpdate { .. }));
    }

    #[test]
    fn unknown_channel_is_rejected_for_strict_schemas() {
        let schema = StateSchema::new();
        let mut values = Map::new();
        let updates = vec![ChannelUpdate {
            channel: "missing".into(),
            value: json!(true),
            node: "a".into(),
        }];
        let error = apply_updates(&schema, &mut values, updates).unwrap_err();
        assert!(matches!(error, Error::UnknownChannel(_)));
    }

    #[test]
    fn dynamic_schema_accepts_implicit_channels() {
        let schema = StateSchema::dynamic();
        let mut values = Map::new();
        apply_updates(
            &schema,
            &mut values,
            vec![ChannelUpdate {
                channel: "anything".into(),
                value: json!(42),
                node: "a".into(),
            }],
        )
        .unwrap();
        assert_eq!(values.get("anything"), Some(&json!(42)));
    }

    #[test]
    fn typed_state_round_trips_through_channels() {
        #[derive(Clone, Default, Serialize, Deserialize, PartialEq, Debug)]
        struct Demo {
            value: i64,
        }
        let state = Demo { value: 7 };
        let values = state_to_values(&state).unwrap();
        assert_eq!(values.get("value"), Some(&json!(7)));
        assert_eq!(values_to_state::<Demo>(&values).unwrap(), state);
    }
}
