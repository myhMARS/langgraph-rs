//! Cooperative run control (LangGraph `RunControl` basics).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Abort flag shared with a running graph.
///
/// The synchronous engine checks it between supersteps. Async graphs additionally race node
/// futures against an abort notification, so `abort()` cancels an in-flight await safely.
#[derive(Clone, Debug)]
pub struct RunControl {
    aborted: Arc<AtomicBool>,
    #[cfg(feature = "async")]
    sender: tokio::sync::watch::Sender<bool>,
}

impl Default for RunControl {
    fn default() -> Self {
        Self {
            aborted: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "async")]
            sender: tokio::sync::watch::channel(false).0,
        }
    }
}

impl RunControl {
    /// Create an unaborted control handle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request that the run stops at the next superstep boundary.
    pub fn abort(&self) {
        self.aborted.store(true, Ordering::SeqCst);
        #[cfg(feature = "async")]
        {
            let _ = self.sender.send(true);
        }
    }

    /// Returns `true` once [`RunControl::abort`] was called.
    pub fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::SeqCst)
    }
}

#[cfg(feature = "async")]
impl RunControl {
    /// Wait until [`RunControl::abort`] is called.
    pub async fn wait_aborted(&self) {
        let mut receiver = self.sender.subscribe();
        if *receiver.borrow() {
            return;
        }
        let _ = receiver.changed().await;
    }
}
