//! Typed graph state and its channel schema.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

use crate::kernel::channel::Channel;
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
        }
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

    /// Declare an appending channel.
    pub fn topic(self, name: impl Into<String>) -> Self {
        self.with_channel(name, Channel::Topic)
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

/// Deserialize channel values back into the typed state.
pub(crate) fn values_to_state<S: DeserializeOwned>(values: &Map<String, Value>) -> Result<S> {
    Ok(serde_json::from_value(Value::Object(values.clone()))?)
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
    let mut grouped: IndexMap<String, (Vec<Value>, Vec<String>)> = IndexMap::new();
    for update in updates {
        let entry = grouped.entry(update.channel).or_default();
        entry.0.push(update.value);
        entry.1.push(update.node);
    }

    for (channel, (incoming, nodes)) in grouped {
        match schema.resolve(&channel)? {
            Some(definition) => {
                if matches!(definition, Channel::LastValue) && incoming.len() > 1 {
                    return Err(Error::ConcurrentUpdate {
                        channel,
                        nodes: nodes.join(", "),
                    });
                }
                let current = values.get(&channel).cloned();
                if let Some(merged) = definition.apply(current.as_ref(), &incoming)? {
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
