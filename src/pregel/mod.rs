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
pub(crate) mod fresh;
pub(crate) mod outcome;
pub(crate) mod pause;
pub(crate) mod run_loop;
pub(crate) mod scheduling;
pub(crate) mod start;
pub(crate) mod step;
mod superstep;

use serde_json::{Map, Value, json};

use self::checkpointing::CheckpointWriter;
use crate::_internal::error::Result;
use crate::_internal::state::{ChannelUpdate, State, apply_updates, values_to_state};
use crate::_internal::stream::StreamChunk;
use crate::_internal::usage::RunOutcome;
use crate::graph::{CompiledGraph, Durability, RunConfig};

/// Where a run starts from.
pub(crate) enum StartInput {
    /// Full channel values (from a typed input state).
    Values(Map<String, Value>),
    /// Partial writes applied over the default state, starting at explicit nodes
    /// (LangGraph `invoke(Command(goto=...))`).
    Gateways {
        values: Map<String, Value>,
        nodes: Vec<String>,
        sends: Vec<crate::_internal::node::Send>,
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
        sends: Vec<crate::_internal::node::Send>,
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
    // The async writer exists to keep slow savers off the run thread; an in-memory saver
    // cannot block, so spawning a thread and cloning every checkpoint into it is pure
    // overhead (it dominated short runs).
    let checkpoint_writer = if config.durability == Durability::Async {
        graph
            .checkpointer()
            .filter(|checkpointer| !checkpointer.is_in_memory())
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
    run_loop::block_on(run_loop::run_loop(
        &backend, start, config, &mut adapt, emit,
    ))
}

/// Inject managed values (`remaining_steps`, `is_last_step`) before a superstep.
pub(crate) fn apply_managed_values<S: State>(
    schema: &crate::_internal::state::StateSchema,
    values: &mut Map<String, Value>,
    state: &mut S,
    managed_step: usize,
    recursion_limit: usize,
) -> Result<()> {
    if !schema.managed_values() {
        return Ok(());
    }
    // LangGraph reports `stop - step` where `step` is the 1-based tick counter, so the
    // first superstep (our 0-based `managed_step`) leaves `limit - 1`.
    let remaining = recursion_limit.saturating_sub(managed_step.saturating_add(1));
    values.insert("remaining_steps".to_string(), json!(remaining));
    // LangGraph: `is_last_step` is exactly one superstep before the limit.
    values.insert("is_last_step".to_string(), json!(remaining == 1));
    *state = values_to_state::<S>(values)?;
    Ok(())
}

pub(crate) fn apply_pending_writes(
    schema: &crate::_internal::state::StateSchema,
    values: &mut Map<String, Value>,
    writes: &[crate::checkpoint::PendingWrite],
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
