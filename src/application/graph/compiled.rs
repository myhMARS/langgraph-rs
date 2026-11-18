//! The immutable, runnable graph.

use std::collections::HashSet;
use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::{Map, Value};

use super::Entry;
use crate::kernel::cache::{CachePolicy, NodeCache};
use crate::kernel::error::Result;
use crate::kernel::node::{ConditionalEdge, NodeFn, Task};
use crate::kernel::ports::{Checkpointer, TraceSink};
use crate::kernel::retry::RetryPolicy;
use crate::kernel::state::{State, StateSchema};
use crate::kernel::store::BaseStore;
use crate::kernel::timeout::TimeoutPolicy;

/// A graph that is ready to run.
///
/// Build one with [`StateGraph::compile`](crate::StateGraph::compile). The graph is
/// immutable afterwards and can be shared between threads.
pub struct CompiledGraph<S: State> {
    pub(crate) schema: StateSchema,
    pub(crate) nodes: IndexMap<String, NodeFn<S>>,
    pub(crate) edges: IndexMap<String, Vec<String>>,
    pub(crate) conditional: IndexMap<String, Vec<ConditionalEdge<S>>>,
    pub(crate) entry: Entry<S>,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer>>,
    pub(crate) trace_sinks: Vec<Arc<dyn TraceSink>>,
    pub(crate) retries: IndexMap<String, RetryPolicy>,
    pub(crate) error_edges: IndexMap<String, String>,
    pub(crate) store: Option<Arc<dyn BaseStore>>,
    pub(crate) node_metadata: indexmap::IndexMap<String, serde_json::Map<String, Value>>,
    pub(crate) timeouts: indexmap::IndexMap<String, TimeoutPolicy>,
    pub(crate) deferred: std::collections::HashSet<String>,
    pub(crate) destinations: indexmap::IndexMap<String, Vec<String>>,
    pub(crate) cache: Option<Arc<dyn NodeCache>>,
    pub(crate) cache_policies: IndexMap<String, CachePolicy>,
    pub(crate) recursion_limit: usize,
    pub(crate) interrupt_before: HashSet<String>,
    pub(crate) interrupt_after: HashSet<String>,
    pub(crate) name: Option<String>,
    pub(crate) context_schema: Option<Value>,
}

impl<S: State> Clone for CompiledGraph<S> {
    fn clone(&self) -> Self {
        Self {
            schema: self.schema.clone(),
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            conditional: self.conditional.clone(),
            entry: self.entry.clone(),
            checkpointer: self.checkpointer.clone(),
            trace_sinks: self.trace_sinks.clone(),
            retries: self.retries.clone(),
            error_edges: self.error_edges.clone(),
            store: self.store.clone(),
            node_metadata: self.node_metadata.clone(),
            timeouts: self.timeouts.clone(),
            deferred: self.deferred.clone(),
            destinations: self.destinations.clone(),
            cache: self.cache.clone(),
            cache_policies: self.cache_policies.clone(),
            recursion_limit: self.recursion_limit,
            interrupt_before: self.interrupt_before.clone(),
            interrupt_after: self.interrupt_after.clone(),
            name: self.name.clone(),
            context_schema: self.context_schema.clone(),
        }
    }
}

impl<S: State> CompiledGraph<S> {
    /// Channel schema of the graph.
    pub fn schema(&self) -> &StateSchema {
        &self.schema
    }

    /// Registered node names in insertion order.
    pub fn node_names(&self) -> impl Iterator<Item = &str> {
        self.nodes.keys().map(String::as_str)
    }

    /// Name given at compile time, when any.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// JSON schema configured for `RunConfig::context`, if any.
    pub fn context_schema(&self) -> Option<&Value> {
        self.context_schema.as_ref()
    }

    /// Serializable graph description (nodes, static edges, conditional edges).
    pub fn graph_spec(&self) -> Value {
        let nodes: Vec<Value> = self
            .nodes
            .keys()
            .map(|node| {
                let mut spec = serde_json::Map::new();
                spec.insert("id".to_string(), Value::String(node.clone()));
                if self.deferred.contains(node) {
                    spec.insert("deferred".to_string(), Value::Bool(true));
                }
                if let Some(metadata) = self.node_metadata.get(node)
                    && !metadata.is_empty()
                {
                    spec.insert("metadata".to_string(), Value::Object(metadata.clone()));
                }
                Value::Object(spec)
            })
            .collect();
        let edges: Vec<Value> = self
            .edges
            .iter()
            .flat_map(|(source, targets)| {
                targets.iter().map(move |target| {
                    serde_json::json!({
                        "source": source,
                        "target": target,
                        "conditional": false,
                    })
                })
            })
            .collect();
        let conditional: Vec<Value> = self
            .conditional
            .keys()
            .map(|source| serde_json::json!({ "source": source, "conditional": true }))
            .collect();
        serde_json::json!({
            "name": self.name,
            "nodes": nodes,
            "edges": edges,
            "conditional_edges": conditional,
        })
    }

    /// Returns `true` when checkpointing is enabled.
    pub fn has_checkpointer(&self) -> bool {
        self.checkpointer.is_some()
    }

    /// The configured checkpointer, if any.
    pub fn checkpointer(&self) -> Option<&Arc<dyn Checkpointer>> {
        self.checkpointer.as_ref()
    }

    /// Export the latest checkpoint of a thread as LangGraph-shaped JSON.
    pub fn export_checkpoint(&self, thread_id: &str) -> Result<Option<Value>> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or(crate::Error::MissingCheckpointer("export_checkpoint"))?;
        Ok(checkpointer
            .latest(thread_id)?
            .map(|checkpoint| crate::interop::checkpoint_to_langgraph(&checkpoint)))
    }

    /// Import a LangGraph-shaped checkpoint document as a checkpoint of `thread_id`.
    pub fn import_checkpoint(&self, thread_id: &str, document: &Value) -> Result<()> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or(crate::Error::MissingCheckpointer("import_checkpoint"))?;
        let mut checkpoint = crate::interop::checkpoint_from_langgraph(document)?;
        checkpoint.thread_id = thread_id.to_string();
        checkpointer.put(checkpoint)
    }

    /// Default superstep budget.
    pub fn recursion_limit(&self) -> usize {
        self.recursion_limit
    }

    /// Render the graph as a Mermaid `graph TD` diagram.
    ///
    /// Static edges are solid arrows, conditional edges are dashed arrows through a
    /// router node (with path-map labels), error edges are labelled `on error`, and node
    /// labels include retry policies and interrupt hooks.
    ///
    /// ```
    /// use langgraph_rs::{NodeOutput, State, StateGraph, StateSchema};
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Clone, Default, Serialize, Deserialize)]
    /// struct Demo {
    ///     value: i64,
    /// }
    ///
    /// impl State for Demo {
    ///     fn schema() -> StateSchema {
    ///         StateSchema::new().last_value("value")
    ///     }
    /// }
    ///
    /// let graph = StateGraph::<Demo>::new()
    ///     .add_node("work", |_state: &Demo| Ok(NodeOutput::new()))
    ///     .set_entry_point("work")
    ///     .set_finish_point("work")
    ///     .compile()
    ///     .unwrap();
    /// let mermaid = graph.to_mermaid();
    /// assert!(mermaid.contains("graph TD"));
    /// assert!(mermaid.contains("START --> n_work"));
    /// ```
    /// Render the graph as a compact ASCII map.
    pub fn to_ascii(&self) -> String {
        crate::application::graph::visualize::render_ascii(
            crate::application::graph::visualize::GraphLayout {
                nodes: self.nodes.keys().map(String::as_str).collect(),
                edges: &self.edges,
                conditional: &self.conditional,
                entry: &self.entry,
                retries: &self.retries,
                error_edges: &self.error_edges,
                interrupt_before: &self.interrupt_before,
                interrupt_after: &self.interrupt_after,
            },
        )
    }

    pub fn to_mermaid(&self) -> String {
        crate::application::graph::visualize::render_mermaid(
            crate::application::graph::visualize::GraphLayout {
                nodes: self.nodes.keys().map(String::as_str).collect(),
                edges: &self.edges,
                conditional: &self.conditional,
                entry: &self.entry,
                retries: &self.retries,
                error_edges: &self.error_edges,
                interrupt_before: &self.interrupt_before,
                interrupt_after: &self.interrupt_after,
            },
        )
    }

    /// Render the graph as a Graphviz DOT document.
    pub fn to_dot(&self) -> String {
        crate::application::graph::visualize::render_dot(
            crate::application::graph::visualize::GraphLayout {
                nodes: self.nodes.keys().map(String::as_str).collect(),
                edges: &self.edges,
                conditional: &self.conditional,
                entry: &self.entry,
                retries: &self.retries,
                error_edges: &self.error_edges,
                interrupt_before: &self.interrupt_before,
                interrupt_after: &self.interrupt_after,
            },
        )
    }

    /// Render the graph as a PNG image using Graphviz `dot`.
    pub fn draw_png(&self) -> Result<Vec<u8>> {
        crate::application::graph::visualize::dot_to_png(&self.to_dot())
    }

    /// Run the graph to completion and return the final typed state.
    ///
    /// `input` is the initial state of a fresh run. Use
    /// [`CompiledGraph::invoke_updates`] when only a state patch is available.
    /// Read the latest checkpoint of a thread.
    pub(crate) fn node(&self, name: &str) -> Option<&NodeFn<S>> {
        self.nodes.get(name)
    }

    pub(crate) fn edges_of(&self, node: &str) -> Option<&[String]> {
        self.edges.get(node).map(Vec::as_slice)
    }

    pub(crate) fn conditional_of(&self, node: &str) -> Option<&[ConditionalEdge<S>]> {
        self.conditional.get(node).map(Vec::as_slice)
    }

    /// Trace sink adapters installed through the kernel.
    pub(crate) fn trace_sinks(&self) -> &[Arc<dyn TraceSink>] {
        &self.trace_sinks
    }

    /// Retry policy of a node, if one was configured.
    pub fn retry_policy(&self, node: &str) -> Option<&RetryPolicy> {
        self.retries.get(node)
    }

    /// Error-handler node for a node, if one was configured.
    pub fn error_handler(&self, node: &str) -> Option<&String> {
        self.error_edges.get(node)
    }

    /// List threads that have at least one checkpoint.
    pub fn destinations_of(&self, node: &str) -> Option<&[String]> {
        self.destinations.get(node).map(Vec::as_slice)
    }

    /// Whether the node is deferred until the graph has no regular work left.
    pub fn is_deferred(&self, node: &str) -> bool {
        self.deferred.contains(node)
    }

    /// Timeout policy of one node, when configured.
    pub fn timeout_of(&self, node: &str) -> Option<&TimeoutPolicy> {
        self.timeouts.get(node)
    }

    /// Metadata attached to nodes (owner, tags, ...).
    pub fn node_metadata(&self) -> &indexmap::IndexMap<String, serde_json::Map<String, Value>> {
        &self.node_metadata
    }

    /// JSON input schema attached to a node, when present.
    pub fn node_input_schema(&self, node: &str) -> Option<&Value> {
        self.node_metadata
            .get(node)
            .and_then(|metadata| metadata.get("input_schema"))
    }

    /// Metadata of one node.
    pub fn node_metadata_for(&self, node: &str) -> Option<&serde_json::Map<String, Value>> {
        self.node_metadata.get(node)
    }

    /// Long-term memory store, when the graph was compiled with one.
    pub(crate) fn store(&self) -> Option<&Arc<dyn BaseStore>> {
        self.store.as_ref()
    }

    /// Cache backend, when node caching is enabled.
    pub(crate) fn cache(&self) -> Option<&Arc<dyn NodeCache>> {
        self.cache.as_ref()
    }

    /// Cache policy of a node, if caching is enabled for it.
    pub(crate) fn cache_policy(&self, node: &str) -> Option<&CachePolicy> {
        self.cache_policies.get(node)
    }

    pub(crate) fn interrupts_before(&self, node: &str) -> bool {
        self.interrupt_before.contains(node)
    }

    pub(crate) fn interrupts_after(&self, node: &str) -> bool {
        self.interrupt_after.contains(node)
    }

    pub(crate) fn entry_tasks(&self, state: &S) -> Result<Vec<Task>> {
        crate::application::pregel::scheduling::entry_tasks(&self.entry, state)
    }
}

#[path = "run_api.rs"]
mod run_api;
