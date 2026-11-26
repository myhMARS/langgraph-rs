//! Graph lifecycle transitions (pause and resume) recorded by a run.
//!
//! LangGraph's loop pushes these onto a queue that the streaming layer drains once per
//! iteration, which keeps the callback dispatch out of the engine. The bindings do the same: the
//! run records what happened, and the Python facade dispatches the events after the call.

use std::sync::{Arc, Mutex};

/// Which transition happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifecycleKind {
    /// The graph paused (dynamic `interrupt()` or a static breakpoint).
    Interrupt,
    /// The graph resumed from a persisted checkpoint.
    Resume,
}

impl LifecycleKind {
    /// LangGraph's callback name for this kind.
    pub fn as_str(self) -> &'static str {
        match self {
            LifecycleKind::Interrupt => "interrupt",
            LifecycleKind::Resume => "resume",
        }
    }
}

/// One recorded transition.
#[derive(Clone, Debug)]
pub struct LifecycleEvent {
    /// Interrupt or resume.
    pub kind: LifecycleKind,
    /// LangGraph loop status (`pending`, `interrupt_before`, ...).
    pub status: String,
    /// Checkpoint the run paused on or resumed from.
    pub checkpoint_id: String,
    /// Checkpoint namespace of the graph that transitioned.
    pub namespace: String,
    /// Interrupt payloads, for `kind = Interrupt`.
    pub interrupts: Vec<crate::Interrupt>,
}

impl LifecycleEvent {
    /// A pause recorded at `checkpoint_id` with the given interrupts.
    pub fn interrupt(
        status: impl Into<String>,
        checkpoint_id: impl Into<String>,
        namespace: impl Into<String>,
        interrupts: Vec<crate::Interrupt>,
    ) -> Self {
        Self {
            kind: LifecycleKind::Interrupt,
            status: status.into(),
            checkpoint_id: checkpoint_id.into(),
            namespace: namespace.into(),
            interrupts,
        }
    }

    /// A resume recorded at `checkpoint_id`.
    pub fn resume(
        status: impl Into<String>,
        checkpoint_id: impl Into<String>,
        namespace: impl Into<String>,
    ) -> Self {
        Self {
            kind: LifecycleKind::Resume,
            status: status.into(),
            checkpoint_id: checkpoint_id.into(),
            namespace: namespace.into(),
            interrupts: Vec::new(),
        }
    }
}

/// Queue a run appends its transitions to; the facade drains it once the call returns.
pub type LifecycleSink = Arc<Mutex<Vec<LifecycleEvent>>>;

/// Record one transition on the sink, when the caller installed one.
pub fn record(sink: Option<&LifecycleSink>, event: LifecycleEvent) {
    if let Some(sink) = sink
        && let Ok(mut events) = sink.lock()
    {
        events.push(event);
    }
}
