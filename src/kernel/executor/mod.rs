//! The superstep execution engine.
//!
//! [`execute`] is the only loop in the crate. It delegates:
//!
//! - node execution to [`superstep`],
//! - edge evaluation / scheduling to [`scheduling`],
//! - persistence to [`checkpointing`],
//!
//! and owns the per-step merge + control flow, which keeps each file focused on one
//! concern.

pub(crate) mod backend;
pub(crate) mod checkpointing;
pub(crate) mod outcome;
pub(crate) mod pregel;
pub(crate) mod scheduling;
pub(crate) mod start;
pub(crate) mod step;
mod superstep;

use serde_json::{Map, Value, json};

use self::checkpointing::CheckpointWriter;
use crate::kernel::error::Result;
use crate::kernel::graph::{CompiledGraph, Durability, RunConfig};
use crate::kernel::state::{ChannelUpdate, State, apply_updates, values_to_state};
use crate::kernel::stream::StreamChunk;
use crate::kernel::usage::RunOutcome;

/// Where a run starts from.
pub(crate) enum StartInput {
    /// Full channel values (from a typed input state).
    Values(Map<String, Value>),
    /// Partial writes applied over the default state, starting at explicit nodes
    /// (LangGraph `invoke(Command(goto=...))`).
    Gateways {
        values: Map<String, Value>,
        nodes: Vec<String>,
        sends: Vec<crate::kernel::node::Send>,
    },
    /// Partial writes applied over the default state.
    Updates(Map<String, Value>),
    /// Continue a checkpointed thread.
    Resume {
        thread_id: String,
        checkpoint_id: Option<String>,
        resume: Option<Value>,
        updates: Option<Map<String, Value>>,
        goto: Vec<String>,
    },
}

/// Execute a graph and return its final state together with the consumption report.
pub(crate) fn execute<S: State>(
    graph: &CompiledGraph<S>,
    start: StartInput,
    config: &RunConfig,
    sink: &mut dyn FnMut(StreamChunk),
) -> Result<RunOutcome<S>> {
    execute_with_emit(graph, start, config, sink, true)
}

/// Execute without building stream chunks (invoke/run paths).
pub(crate) fn execute_quiet<S: State>(
    graph: &CompiledGraph<S>,
    start: StartInput,
    config: &RunConfig,
    sink: &mut dyn FnMut(StreamChunk),
) -> Result<RunOutcome<S>> {
    execute_with_emit(graph, start, config, sink, false)
}

fn execute_with_emit<S: State>(
    graph: &CompiledGraph<S>,
    start: StartInput,
    config: &RunConfig,
    sink: &mut dyn FnMut(StreamChunk),
    emit: bool,
) -> Result<RunOutcome<S>> {
    let checkpoint_writer = if config.durability == Durability::Async {
        graph
            .checkpointer()
            .map(|checkpointer| CheckpointWriter::start(checkpointer.clone()))
    } else {
        None
    };
    let backend = backend::SyncBackend {
        graph,
        writer: checkpoint_writer.as_ref(),
        max_concurrency: config.max_concurrency,
    };
    let mut adapt = |chunk| sink(chunk);
    pregel::block_on(pregel::run_loop(&backend, start, config, &mut adapt, emit))
}

/// Inject managed values (`remaining_steps`, `is_last_step`) before a superstep.
pub(crate) fn apply_managed_values<S: State>(
    schema: &crate::kernel::state::StateSchema,
    values: &mut Map<String, Value>,
    state: &mut S,
    step: usize,
    recursion_limit: usize,
) -> Result<()> {
    if !schema.managed_values() {
        return Ok(());
    }
    let remaining = recursion_limit.saturating_sub(step + 1);
    values.insert("remaining_steps".to_string(), json!(remaining));
    values.insert("is_last_step".to_string(), json!(remaining <= 1));
    *state = values_to_state::<S>(values)?;
    Ok(())
}

pub(crate) fn apply_pending_writes(
    schema: &crate::kernel::state::StateSchema,
    values: &mut Map<String, Value>,
    writes: &[crate::kernel::checkpoint::PendingWrite],
) -> Result<()> {
    if writes.is_empty() {
        return Ok(());
    }
    let updates: Vec<ChannelUpdate> = writes
        .iter()
        .filter(|write| !write.replace && !schema.is_finish_value(&write.channel))
        .map(|write| ChannelUpdate {
            channel: write.channel.clone(),
            value: write.value.clone(),
            node: write.node.clone(),
        })
        .collect();
    apply_updates(schema, values, updates)?;
    for write in writes {
        if write.replace || schema.is_finish_value(&write.channel) {
            schema.resolve(&write.channel)?;
            values.insert(write.channel.clone(), write.value.clone());
        }
    }
    Ok(())
}
