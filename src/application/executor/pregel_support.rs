//! Blocking driver and checkpooint/stream helper functions for the Pregel loop.

use std::collections::HashMap;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use serde_json::Value;

use crate::kernel::executor::backend::PregelBackend;
use crate::kernel::state::State;
use crate::kernel::stream::StreamChunk;

pub(crate) fn block_on<F: std::future::Future>(future: F) -> F::Output {
    struct ThreadWake(std::thread::Thread);
    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

pub(crate) fn drain_collector<S: State, B: PregelBackend<S>, F: FnMut(StreamChunk)>(
    backend: &B,
    setup: &crate::kernel::executor::step::RunSetup,
    sink: &mut F,
) {
    if let Some(collector) = backend.collector().or_else(|| setup.collector().cloned()) {
        let chunks: Vec<StreamChunk> = collector
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .drain(..)
            .collect();
        for chunk in chunks {
            sink(chunk);
        }
    }
}

pub(crate) fn finish_pending(
    finish_updates: &HashMap<String, Value>,
    schema: &crate::kernel::state::StateSchema,
) -> Vec<crate::kernel::checkpoint::PendingWrite> {
    finish_updates
        .iter()
        .filter(|(channel, _)| schema.persists(channel))
        .map(|(channel, value)| crate::kernel::checkpoint::PendingWrite {
            task_id: "__finish__".to_string(),
            node: "__finish__".to_string(),
            channel: channel.clone(),
            value: value.clone(),
            replace: false,
        })
        .collect()
}
