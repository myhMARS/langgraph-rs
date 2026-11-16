//! # langgraph-rs
//!
//! A Rust port of the core runtime of LangGraph: a stateful, graph-based execution
//! engine for orchestrating LLM agents and workflows.
//!
//! ## State
//!
//! State is split into channels. A [`Channel`] defines how a node write is merged into
//! the value produced by the previous superstep, mirroring LangGraph's
//! `Annotated[..., reducer]` annotations.

mod channel;
mod error;
mod state;

pub use channel::{Channel, Reducer, reducers};
pub use error::{Error, Interrupt, Result};
pub use state::{JsonState, State, StateSchema};

pub use indexmap::IndexMap;
pub use serde_json::{Value, json};

/// Reserved node name used as the source of entry-point edges.
pub const START: &str = "__start__";
/// Reserved node name that terminates a run.
pub const END: &str = "__end__";