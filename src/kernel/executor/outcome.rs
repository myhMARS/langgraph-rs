//! Shared result types for the sync and async Pregel loops.
//!
//! Both runtimes compute the same kind of transition after a superstep. Keeping the
//! shape in one place is the first step towards one loop with pluggable execution
//! backends.

use std::collections::HashSet;

use serde_json::Value;

use crate::kernel::node::Task;
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
