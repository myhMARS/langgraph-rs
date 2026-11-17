//! Shared result types for the sync and async Pregel loops.
//!
//! Both runtimes compute the same kind of transition after a superstep. Keeping the
//! shape in one place is the first step towards one loop with pluggable execution
//! backends.

use std::collections::HashSet;

use serde_json::Value;

use crate::kernel::error::{Interrupt, NodeFailure};
use crate::kernel::node::{NodeOutput, Task};
use crate::kernel::state::ChannelUpdate;
use crate::kernel::usage::Usage;

/// Everything one superstep's node outcomes contribute to the run.
#[derive(Debug, Default)]
pub(crate) struct ProcessedResults {
    pub(crate) channel_updates: Vec<ChannelUpdate>,
    pub(crate) replacements: Vec<(String, Value)>,
    pub(crate) next_tasks: Vec<Task>,
    pub(crate) goto_targets: Vec<(String, String)>,
    pub(crate) executed: Vec<String>,
    pub(crate) goto_nodes: HashSet<String>,
}

/// Merge per-attempt usage into one entry for a node execution.
pub(crate) fn merge_usage(usage: Vec<Usage>) -> Option<Usage> {
    let mut combined: Option<Usage> = None;
    for item in usage {
        combined.get_or_insert_with(Usage::new).merge(&item);
    }
    combined
}

/// Successful result of one node execution.
#[derive(Debug)]
pub(crate) struct TaskResult {
    pub(crate) node: String,
    pub(crate) output: NodeOutput,
    pub(crate) custom: Vec<Value>,
    pub(crate) messages: Vec<Value>,
    pub(crate) usage: Vec<Usage>,
    pub(crate) duration_millis: u64,
    pub(crate) attempts: u32,
    pub(crate) cached: bool,
    pub(crate) input: Option<Value>,
}

/// Permanent failure of one node execution.
#[derive(Debug)]
pub(crate) struct TaskFailure {
    pub(crate) failure: NodeFailure,
    pub(crate) usage: Vec<Usage>,
    pub(crate) duration_millis: u64,
    pub(crate) input: Option<Value>,
}

/// Outcome of one scheduled task.
pub(crate) enum TaskOutcome {
    Ok(Box<TaskResult>),
    Interrupted(Interrupt),
    Failed(Box<TaskFailure>),
}
