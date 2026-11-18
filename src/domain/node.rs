//! Node outputs, dynamic fan-out and routing primitives.

use std::fmt;
use std::sync::Arc;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Result;
use crate::kernel::error::NodeFailure;
use crate::kernel::usage::Usage;

/// Execution function of a graph node.
pub type NodeFn<S> = Arc<dyn Fn(&S) -> Result<NodeOutput> + std::marker::Send + std::marker::Sync>;

/// Raw parts of a [`NodeOutput`]: updates, replacements, goto targets and sends.
pub(crate) type OutputParts = (
    IndexMap<String, Value>,
    IndexMap<String, Value>,
    Vec<String>,
    Vec<Send>,
);

/// Writes and control flow returned by a node.
///
/// A node normally just returns channel writes:
///
/// ```
/// use langgraph_rs::NodeOutput;
/// let output = NodeOutput::new().set("answer", "42");
/// assert_eq!(output.updates_map().len(), 1);
/// ```
///
/// Use [`NodeOutput::goto`] to jump to specific nodes (the equivalent of LangGraph's
/// `Command(goto=...)`) and [`NodeOutput::send`] to fan out dynamically.
#[derive(Clone, Debug, Default)]
pub struct NodeOutput {
    updates: IndexMap<String, Value>,
    replaces: IndexMap<String, Value>,
    goto: Vec<String>,
    sends: Vec<Send>,
    usage: Option<Usage>,
    parent: bool,
}

impl NodeOutput {
    /// Create an empty output.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a channel write.
    pub fn update(mut self, channel: impl Into<String>, value: Value) -> Self {
        self.updates.insert(channel.into(), value);
        self
    }

    /// Set a channel write from any serializable value; panics if serialization fails.
    pub fn set(self, channel: impl Into<String>, value: impl Serialize) -> Self {
        match self.try_set(channel, value) {
            Ok(output) => output,
            Err(error) => panic!("NodeOutput::set failed: {error}"),
        }
    }

    /// Fallible variant of [`NodeOutput::set`].
    pub fn try_set(mut self, channel: impl Into<String>, value: impl Serialize) -> Result<Self> {
        self.updates
            .insert(channel.into(), serde_json::to_value(value)?);
        Ok(self)
    }

    /// Add several channel writes.
    pub fn updates<I, K>(mut self, updates: I) -> Self
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        for (channel, value) in updates {
            self.updates.insert(channel.into(), value);
        }
        self
    }

    /// Jump to a node regardless of the static outgoing edges.
    pub fn goto(mut self, node: impl Into<String>) -> Self {
        self.goto.push(node.into());
        self
    }

    /// Route to a node of the **parent** graph (`Command(graph=PARENT)`).
    ///
    /// Only meaningful inside a subgraph node: the child graph aborts, its writes are
    /// applied to the parent state and the parent schedules the target node.
    pub fn goto_parent(mut self, node: impl Into<String>) -> Self {
        self.parent = true;
        self.goto.push(node.into());
        self
    }

    pub(crate) fn is_parent(&self) -> bool {
        self.parent
    }

    /// Add a dynamic fan-out task.
    pub fn send(mut self, send: Send) -> Self {
        self.sends.push(send);
        self
    }

    /// Write directly to a channel, bypassing its reducer (LangGraph `Overwrite`).
    pub fn overwrite(self, channel: impl Into<String>, value: Value) -> Self {
        self.replace(channel, value)
    }

    /// Replace a channel value directly, bypassing its reducer.
    ///
    /// This is what subgraph nodes use to publish the subgraph's final state: the
    /// parent channel is overwritten instead of being folded again.
    pub fn replace(mut self, channel: impl Into<String>, value: Value) -> Self {
        self.replaces.insert(channel.into(), value);
        self
    }

    /// Replacements attached with [`NodeOutput::replace`].
    pub fn replaces_map(&self) -> &IndexMap<String, Value> {
        &self.replaces
    }

    /// Attach consumption information for this node execution.
    ///
    /// Equivalent to calling [`record_usage`](crate::record_usage) from inside the node,
    /// but keeps the usage next to the return value.
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Consumption information attached with [`NodeOutput::with_usage`].
    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    /// Returns `true` when the node produced no writes and no control flow.
    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
            && self.replaces.is_empty()
            && self.goto.is_empty()
            && self.sends.is_empty()
    }

    /// Channel writes produced by this node.
    pub fn updates_map(&self) -> &IndexMap<String, Value> {
        &self.updates
    }

    /// Explicit jumps requested by this node.
    pub fn goto_targets(&self) -> &[String] {
        &self.goto
    }

    /// Dynamic fan-out tasks requested by this node.
    pub fn sends(&self) -> &[Send] {
        &self.sends
    }

    pub(crate) fn into_parts(self) -> OutputParts {
        (self.updates, self.replaces, self.goto, self.sends)
    }

    /// Rebuild an output from a cached node output.
    pub(crate) fn from_cached(cached: crate::kernel::cache::CachedOutput) -> Self {
        let mut output = NodeOutput::new().updates(cached.updates);
        for (channel, value) in cached.replaces {
            output = output.replace(channel, value);
        }
        for target in cached.goto {
            output = output.goto(target);
        }
        for send in cached.sends {
            output = output.send(send);
        }
        if let Some(usage) = cached.usage {
            output = output.with_usage(usage);
        }
        output
    }
}

impl From<IndexMap<String, Value>> for NodeOutput {
    fn from(updates: IndexMap<String, Value>) -> Self {
        NodeOutput {
            updates,
            ..NodeOutput::default()
        }
    }
}

impl From<Map<String, Value>> for NodeOutput {
    fn from(updates: Map<String, Value>) -> Self {
        NodeOutput {
            updates: updates.into_iter().collect(),
            ..NodeOutput::default()
        }
    }
}

/// A dynamic fan-out task produced by [`NodeOutput::send`].
///
/// The optional `arg` replaces the state the target node receives, which makes
/// map/reduce style fan-out possible.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Send {
    /// Target node.
    pub node: String,
    /// State override for the target node (must be a JSON object). `None` forwards the
    /// current state unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arg: Option<Value>,
    /// Per-task timeout in milliseconds (overrides the node's `TimeoutPolicy`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_millis: Option<u64>,
}

impl Send {
    /// Send the current state to `node`.
    pub fn to(node: impl Into<String>) -> Self {
        Send {
            node: node.into(),
            arg: None,
            timeout_millis: None,
        }
    }

    /// Send a custom input to `node`.
    pub fn with_arg(node: impl Into<String>, arg: Value) -> Self {
        Send {
            node: node.into(),
            arg: Some(arg),
            timeout_millis: None,
        }
    }

    /// Attach a per-task run timeout.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout_millis = Some(timeout.as_millis() as u64);
        self
    }
}

/// Destination(s) returned by a conditional edge.
///
/// ```
/// use langgraph_rs::Route;
/// assert!(Route::end().is_end());
/// assert_eq!(Route::node("next").targets(), ["next"]);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Route {
    targets: Vec<String>,
    sends: Vec<Send>,
}

impl Route {
    /// An empty route: stop the graph.
    pub fn end() -> Self {
        Self::default()
    }

    /// Route to a single node.
    pub fn node(node: impl Into<String>) -> Self {
        Self {
            targets: vec![node.into()],
            sends: Vec::new(),
        }
    }

    /// Route to several nodes.
    pub fn nodes<I, T>(nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Self {
            targets: nodes.into_iter().map(Into::into).collect(),
            sends: Vec::new(),
        }
    }

    /// Dispatch dynamic [`Send`] tasks from a router.
    pub fn sends<I>(sends: I) -> Self
    where
        I: IntoIterator<Item = Send>,
    {
        Self {
            targets: Vec::new(),
            sends: sends.into_iter().collect(),
        }
    }

    /// Dispatch a single [`Send`] task from a router.
    pub fn send(send: Send) -> Self {
        Self::sends([send])
    }

    /// Append dynamic [`Send`] tasks to this route.
    pub fn with_sends<I>(mut self, sends: I) -> Self
    where
        I: IntoIterator<Item = Send>,
    {
        self.sends.extend(sends);
        self
    }

    /// Destinations of this route.
    pub fn targets(&self) -> &[String] {
        &self.targets
    }

    /// Dynamic Send tasks dispatched by this route.
    pub fn sent_tasks(&self) -> &[Send] {
        &self.sends
    }

    /// Returns `true` when the route terminates the graph.
    pub fn is_end(&self) -> bool {
        self.targets.is_empty() && self.sends.is_empty()
    }

    pub(crate) fn into_parts(self) -> (Vec<String>, Vec<Send>) {
        (self.targets, self.sends)
    }
}

impl From<&str> for Route {
    fn from(node: &str) -> Self {
        Route::node(node)
    }
}

impl From<String> for Route {
    fn from(node: String) -> Self {
        Route::node(node)
    }
}

impl From<Vec<String>> for Route {
    fn from(nodes: Vec<String>) -> Self {
        Route::nodes(nodes)
    }
}

impl From<&[&str]> for Route {
    fn from(nodes: &[&str]) -> Self {
        Route::nodes(nodes.iter().copied())
    }
}

/// A mapping from router keys to concrete node targets.
///
/// Conditional edges can return symbolic keys (`"retry"`, `"done"`) that are resolved
/// through a path map, mirroring LangGraph's `path_map` argument.
///
/// ```
/// use langgraph_rs::PathMap;
/// let map = PathMap::from([("ok", "finish"), ("retry", "agent")]);
/// assert_eq!(map.get("ok"), Some(&vec!["finish".to_string()]));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PathMap(IndexMap<String, Vec<String>>);

impl PathMap {
    /// Create an empty path map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mapping.
    pub fn insert<I, T>(mut self, key: impl Into<String>, targets: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.0
            .insert(key.into(), targets.into_iter().map(Into::into).collect());
        self
    }

    /// Look up the targets for a router key.
    pub fn get(&self, key: &str) -> Option<&Vec<String>> {
        self.0.get(key)
    }

    /// Iterate over `(key, targets)` entries.
    pub(crate) fn entries(&self) -> impl Iterator<Item = (&String, &Vec<String>)> {
        self.0.iter()
    }

    /// Returns `true` when the map has no entry.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<const N: usize> From<[(&str, &str); N]> for PathMap {
    fn from(entries: [(&str, &str); N]) -> Self {
        let mut map = PathMap::new();
        for (key, target) in entries {
            map = map.insert(key, [target]);
        }
        map
    }
}

impl From<Vec<(&str, &str)>> for PathMap {
    fn from(entries: Vec<(&str, &str)>) -> Self {
        let mut map = PathMap::new();
        for (key, target) in entries {
            map = map.insert(key, [target]);
        }
        map
    }
}

impl From<IndexMap<String, Vec<String>>> for PathMap {
    fn from(entries: IndexMap<String, Vec<String>>) -> Self {
        PathMap(entries)
    }
}

/// Router function stored by a [`ConditionalEdge`].
pub(crate) type RouteFn<S> =
    Arc<dyn Fn(&S) -> Result<Route> + std::marker::Send + std::marker::Sync>;

/// A conditional edge: a pure router evaluated against the current state.
pub struct ConditionalEdge<S> {
    pub(crate) router: RouteFn<S>,
    pub(crate) path_map: Option<PathMap>,
}

impl<S> ConditionalEdge<S> {
    /// Build a conditional edge.
    pub fn new<F>(router: F) -> Self
    where
        F: Fn(&S) -> Result<Route> + std::marker::Send + std::marker::Sync + 'static,
    {
        Self {
            router: Arc::new(router),
            path_map: None,
        }
    }

    /// Attach a path map and return the edge.
    pub fn with_path_map(mut self, path_map: impl Into<PathMap>) -> Self {
        self.path_map = Some(path_map.into());
        self
    }

    /// Evaluate the router and resolve symbolic keys into node names and Send tasks.
    pub(crate) fn resolve(&self, state: &S, source: &str) -> Result<(Vec<String>, Vec<Send>)> {
        let route = match (self.router.as_ref())(state) {
            Ok(route) => route,
            Err(crate::Error::Interrupt(interrupt)) => {
                return Err(crate::Error::Interrupt(interrupt));
            }
            Err(error) => return Err(crate::Error::routing(source, error)),
        };

        let (targets, sends) = route.into_parts();
        match &self.path_map {
            None => Ok((targets, sends)),
            Some(path_map) => {
                let mut resolved = Vec::new();
                for key in targets {
                    match path_map.get(&key) {
                        Some(mapped) => resolved.extend(mapped.iter().cloned()),
                        None => {
                            return Err(crate::Error::routing(
                                source,
                                format!(
                                    "router returned `{key}` which is not present in the path map"
                                ),
                            ));
                        }
                    }
                }
                Ok((resolved, sends))
            }
        }
    }
}

impl<S> Clone for ConditionalEdge<S> {
    fn clone(&self) -> Self {
        Self {
            router: self.router.clone(),
            path_map: self.path_map.clone(),
        }
    }
}

impl<S> fmt::Debug for ConditionalEdge<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConditionalEdge")
            .field("path_map", &self.path_map)
            .finish_non_exhaustive()
    }
}

/// Generate a stable-enough task id for scheduling and `update_state(task_id=...)`.
pub(crate) fn new_task_id(node: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    format!("{node}-{millis}-{sequence}")
}

/// A concrete unit of scheduled work.
///
/// Tasks are stored inside checkpoints so an interrupted graph can be resumed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Task {
    /// Stable task id (also used by `update_state(task_id=...)`).
    #[serde(default)]
    pub id: String,
    /// Node to execute.
    pub node: String,
    /// Optional state override (used by [`Send`] fan-out).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Map<String, Value>>,
    /// Value handed to the node's next `interrupt()` call when resuming.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume: Option<Value>,
    /// Resume values already consumed by earlier interrupts of this task, in call order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resume_history: Vec<Value>,
    /// Failure that scheduled this task (set by error routing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<NodeFailure>,
    /// Per-task timeout in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_millis: Option<u64>,
    /// Whether this task paused on `interrupt()` in the checkpoint that stores it.
    #[serde(default)]
    pub interrupted: bool,
}

impl Task {
    /// Create a task that runs `node` against the current state.
    pub fn new(node: impl Into<String>) -> Self {
        let node = node.into();
        Task {
            id: new_task_id(&node),
            node,
            input: None,
            resume: None,
            resume_history: Vec::new(),
            error: None,
            timeout_millis: None,
            interrupted: false,
        }
    }

    /// Create a task with an explicit state override.
    pub fn with_input(node: impl Into<String>, input: Map<String, Value>) -> Self {
        let node = node.into();
        Task {
            id: new_task_id(&node),
            node,
            input: Some(input),
            resume: None,
            resume_history: Vec::new(),
            error: None,
            timeout_millis: None,
            interrupted: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_output_accumulates_writes_and_control_flow() {
        let output = NodeOutput::new()
            .set("a", 1)
            .update("b", serde_json::json!(true))
            .goto("next")
            .send(Send::to("worker"));
        assert_eq!(output.updates_map().len(), 2);
        assert_eq!(output.goto_targets(), ["next"]);
        assert_eq!(output.sends().len(), 1);
        assert!(!output.is_empty());
    }

    #[test]
    fn path_map_resolves_symbolic_keys() {
        let map = PathMap::from([("ok", "finish"), ("retry", "agent")]);
        assert_eq!(map.get("retry"), Some(&vec!["agent".to_string()]));
        assert!(map.get("missing").is_none());
    }

    #[test]
    fn conditional_edge_resolves_path_map() {
        #[derive(Default)]
        struct State {
            retry: bool,
        }
        let edge = ConditionalEdge::new(|state: &State| {
            Ok(if state.retry {
                Route::node("retry")
            } else {
                Route::node("ok")
            })
        })
        .with_path_map([("retry", "agent"), ("ok", "finish")]);

        assert_eq!(
            edge.resolve(&State { retry: true }, "router").unwrap().0,
            vec!["agent".to_string()]
        );
    }
}
