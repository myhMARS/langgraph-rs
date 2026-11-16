//! # langgraph-rs
//!
//! A Rust port of the core runtime of [LangGraph](https://github.com/langchain-ai/langgraph):
//! a stateful, graph-based execution engine for orchestrating LLM agents and workflows.
//!
//! The engine runs a [`StateGraph`] in **supersteps**. Each superstep executes the
//! currently scheduled nodes (in parallel when there is more than one), merges their
//! writes into the graph state through [`Channel`] merge policies, evaluates outgoing
//! edges and schedules the next superstep until no work is left or [`END`] is reached.
//!
//! ## Features
//!
//! - typed or JSON state with `LastValue`, `Topic` and custom reducer channels
//! - static edges, conditional routing, symbolic path maps and multi-target fan-out
//! - parallel supersteps executed on scoped threads
//! - [`Send`] dynamic fan-out for map/reduce workflows
//! - checkpointing via the [`Checkpointer`] trait with an in-memory saver, thread
//!   history and time travel
//! - human-in-the-loop interrupts ([`interrupt`] / [`CompiledGraph::resume`]) plus
//!   `interrupt_before` / `interrupt_after` hooks
//! - streaming of values, per-node updates, custom events, interrupts, checkpoints and
//!   per-node usage records
//! - per-node consumption tracking ([`record_usage`], [`Usage`], [`RunReport`]): tokens,
//!   cost, wall-clock duration and aggregated totals per node and per run
//! - optional Python bindings behind the `python` feature (PyO3), exposing the same engine
//!   to Python callables
//!
//! ## Quick start
//!
//! ```
//! use langgraph_rs::{NodeOutput, Route, RunConfig, State, StateGraph, StateSchema};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Clone, Default, Serialize, Deserialize)]
//! struct Counter {
//!     value: i64,
//! }
//!
//! impl State for Counter {
//!     fn schema() -> StateSchema {
//!         StateSchema::new().last_value("value")
//!     }
//! }
//!
//! let graph = StateGraph::<Counter>::new()
//!     .add_node("increment", |state: &Counter| {
//!         Ok(NodeOutput::new().set("value", state.value + 1))
//!     })
//!     .set_entry_point("increment")
//!     .add_conditional_edges("increment", |state: &Counter| {
//!         Ok(if state.value < 3 {
//!             Route::node("increment")
//!         } else {
//!             Route::end()
//!         })
//!     })
//!     .compile()
//!     .unwrap();
//!
//! let output = graph.invoke(&Counter::default(), &RunConfig::new()).unwrap();
//! assert_eq!(output.value, 3);
//! ```
//!
//! ## Execution model
//!
//! ```text
//!  input ──► [ superstep 1: run nodes ] ──► merge channels ──► route edges ──► [ superstep 2 ] ──► ... ──► END
//!                     │                          │
//!                     └── checkpoint ────────────┘
//! ```
//!
//! A node is any `Fn(&State) -> Result<NodeOutput>`; it returns channel writes instead
//! of mutating state, which is what makes checkpointing, resuming and time travel
//! possible.

mod channel;
mod checkpoint;
mod command;
mod context;
mod error;
mod executor;
mod graph;
mod node;
mod state;
mod stream;
mod usage;

pub use channel::{Channel, Reducer, reducers};
pub use checkpoint::{Checkpoint, Checkpointer, InMemorySaver, MemorySaver};
pub use command::Command;
pub use context::{current_node, custom, interrupt, push_custom, record_tokens, record_usage};
pub use error::{Error, Interrupt, Result};
pub use graph::{
    CompileOptions, CompiledGraph, DEFAULT_RECURSION_LIMIT, RunConfig, StateGraph, StateSnapshot,
    Targets,
};
pub use node::{ConditionalEdge, NodeFn, NodeOutput, PathMap, Route, Send, Task};
pub use state::{JsonState, State, StateSchema};
pub use stream::StreamChunk;
pub use usage::{NodeExecution, RunOutcome, RunReport, Usage};

pub use indexmap::IndexMap;
pub use serde_json::{Value, json};

/// Reserved node name used as the source of entry-point edges.
pub const START: &str = "__start__";
/// Reserved node name that terminates a run.
pub const END: &str = "__end__";
