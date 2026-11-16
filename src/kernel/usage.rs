//! Per-node consumption tracking.
//!
//! Nodes can report what they consumed (`record_usage(...)` or
//! [`NodeOutput::with_usage`]) and the engine records how long each execution took.
//! A run produces a [`RunReport`] with one [`NodeExecution`] per executed node, plus
//! aggregated totals per node and per run.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Token / cost accounting reported by one node execution.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Model or provider that produced the consumption (e.g. `"gpt-4o"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Prompt / input tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Completion / output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Cost in US dollars, if the node can estimate it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    /// Free-form metadata (tool calls, request ids, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
}

impl Usage {
    /// Create an empty usage record.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the input token count.
    pub fn with_input_tokens(mut self, tokens: u64) -> Self {
        self.input_tokens = Some(tokens);
        self
    }

    /// Set the output token count.
    pub fn with_output_tokens(mut self, tokens: u64) -> Self {
        self.output_tokens = Some(tokens);
        self
    }

    /// Set the estimated cost in US dollars.
    pub fn with_cost_usd(mut self, cost: f64) -> Self {
        self.cost_usd = Some(cost);
        self
    }

    /// Attach a metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.metadata
            .get_or_insert_with(Map::new)
            .insert(key.into(), value.into());
        self
    }

    /// Total tokens (`input + output`), when either side is known.
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.unwrap_or_default() + self.output_tokens.unwrap_or_default()
    }

    /// Returns `true` when nothing was reported.
    pub fn is_empty(&self) -> bool {
        self.model.is_none()
            && self.input_tokens.is_none()
            && self.output_tokens.is_none()
            && self.cost_usd.is_none()
            && self.metadata.is_none()
    }

    /// Fold another usage record into this one.
    pub fn merge(&mut self, other: &Usage) {
        if self.model.is_none() {
            self.model = other.model.clone();
        }
        if let Some(tokens) = other.input_tokens {
            self.input_tokens = Some(self.input_tokens.unwrap_or_default() + tokens);
        }
        if let Some(tokens) = other.output_tokens {
            self.output_tokens = Some(self.output_tokens.unwrap_or_default() + tokens);
        }
        if let Some(cost) = other.cost_usd {
            self.cost_usd = Some(self.cost_usd.unwrap_or_default() + cost);
        }
        if let Some(metadata) = &other.metadata {
            let merged = self.metadata.get_or_insert_with(Map::new);
            for (key, value) in metadata.clone() {
                merged.insert(key, value);
            }
        }
    }
}

/// One recorded node execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeExecution {
    /// Node that ran.
    pub node: String,
    /// Superstep the node ran in.
    pub step: usize,
    /// Wall-clock duration of the node call in milliseconds.
    pub duration_millis: u64,
    /// Consumption reported by the node, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Consumption and timing report of a graph run.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunReport {
    /// One entry per node execution, in execution order.
    #[serde(default)]
    pub executions: Vec<NodeExecution>,
    /// Wall-clock duration of the whole run in milliseconds.
    #[serde(default)]
    pub total_duration_millis: u64,
}

impl RunReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one node execution.
    pub fn record(&mut self, execution: NodeExecution) {
        self.executions.push(execution);
    }

    /// All node executions in execution order.
    pub fn executions(&self) -> &[NodeExecution] {
        &self.executions
    }

    /// Executions of one node, in execution order.
    pub fn executions_for<'a>(&'a self, node: &'a str) -> impl Iterator<Item = &'a NodeExecution> {
        self.executions
            .iter()
            .filter(move |entry| entry.node == node)
    }

    /// Aggregated usage per node name.
    pub fn usage_by_node(&self) -> BTreeMap<String, Usage> {
        let mut totals: BTreeMap<String, Usage> = BTreeMap::new();
        for execution in &self.executions {
            if let Some(usage) = &execution.usage {
                totals
                    .entry(execution.node.clone())
                    .or_default()
                    .merge(usage);
            }
        }
        totals
    }

    /// Aggregated usage of the whole run.
    pub fn total_usage(&self) -> Usage {
        let mut total = Usage::new();
        for execution in &self.executions {
            if let Some(usage) = &execution.usage {
                total.merge(usage);
            }
        }
        total
    }

    /// Total tokens consumed by the run.
    pub fn total_tokens(&self) -> u64 {
        self.total_usage().total_tokens()
    }

    /// Total estimated cost of the run in US dollars.
    pub fn total_cost_usd(&self) -> f64 {
        self.total_usage().cost_usd.unwrap_or_default()
    }
}

/// Result of a graph run: final state plus a consumption report.
#[derive(Clone, Debug)]
pub struct RunOutcome<S> {
    /// Final graph state.
    pub state: S,
    /// Per-node consumption and timing report.
    pub report: RunReport,
}

impl<S> RunOutcome<S> {
    /// Unwrap the final state.
    pub fn into_state(self) -> S {
        self.state
    }

    /// Consumption report of the run.
    pub fn report(&self) -> &RunReport {
        &self.report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn usage_aggregates_tokens_and_cost() {
        let mut total = Usage::new();
        total.merge(
            &Usage::new()
                .with_model("gpt-4o")
                .with_input_tokens(10)
                .with_output_tokens(5)
                .with_cost_usd(0.02),
        );
        total.merge(&Usage::new().with_input_tokens(1).with_output_tokens(2));
        assert_eq!(total.total_tokens(), 18);
        assert_eq!(total.cost_usd, Some(0.02));
        assert_eq!(total.model.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn report_groups_executions_by_node() {
        let mut report = RunReport::new();
        report.record(NodeExecution {
            node: "agent".into(),
            step: 0,
            duration_millis: 12,
            usage: Some(
                Usage::new()
                    .with_input_tokens(100)
                    .with_output_tokens(20)
                    .with_metadata("tool", json!("search")),
            ),
        });
        report.record(NodeExecution {
            node: "agent".into(),
            step: 1,
            duration_millis: 7,
            usage: Some(Usage::new().with_input_tokens(30)),
        });
        report.record(NodeExecution {
            node: "judge".into(),
            step: 2,
            duration_millis: 3,
            usage: None,
        });
        report.total_duration_millis = 22;

        let by_node = report.usage_by_node();
        assert_eq!(by_node["agent"].total_tokens(), 150);
        assert_eq!(
            by_node["agent"].metadata.as_ref().unwrap()["tool"],
            json!("search")
        );
        assert_eq!(report.total_tokens(), 150);
        assert_eq!(report.executions_for("agent").count(), 2);
    }
}
