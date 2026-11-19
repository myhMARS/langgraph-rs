//! Reusable OS-thread pool for parallel supersteps.
//!
//! Spawning a scoped thread per task costs 50-80 us on Windows, which dwarfs the work a
//! typical node does. The pool is process-wide (so threads are reused across runs) and
//! only accepts `'static` jobs; `run_superstep` therefore hands each worker an owned
//! snapshot (`Arc`-shared graph and tasks, cloned state) instead of borrowing the run.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};

type Job = Box<dyn FnOnce() + Send + 'static>;

thread_local! {
    /// `true` while the current thread runs a pool job. Nested runs (a node invoking a
    /// child graph) must not queue on the same pool: the worker would block waiting for
    /// jobs that may never get a free thread, so those runs fall back to scoped threads.
    static ON_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Marks the current thread as a pool worker until dropped.
pub(crate) struct WorkerGuard {
    previous: bool,
}

impl Drop for WorkerGuard {
    fn drop(&mut self) {
        ON_WORKER.with(|flag| flag.set(self.previous));
    }
}

pub(crate) fn enter_worker() -> WorkerGuard {
    let previous = ON_WORKER.with(|flag| flag.replace(true));
    WorkerGuard { previous }
}

pub(crate) fn on_worker() -> bool {
    ON_WORKER.with(std::cell::Cell::get)
}

pub(crate) struct ThreadPool {
    sender: mpsc::Sender<Job>,
    size: usize,
}

impl ThreadPool {
    fn new(size: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));
        for index in 0..size {
            let receiver = Arc::clone(&receiver);
            let spawned = std::thread::Builder::new()
                .name(format!("langgraph-worker-{index}"))
                .stack_size(512 * 1024)
                .spawn(move || {
                    loop {
                        let job = {
                            let guard = match receiver.lock() {
                                Ok(guard) => guard,
                                Err(poisoned) => poisoned.into_inner(),
                            };
                            guard.recv()
                        };
                        match job {
                            // A panicking node must not take the worker down: the job body
                            // records its own failure, so a stray panic is contained here.
                            Ok(job) => {
                                let _ = catch_unwind(AssertUnwindSafe(job));
                            }
                            Err(_) => break,
                        }
                    }
                });
            if spawned.is_err() {
                break;
            }
        }
        Self { sender, size }
    }

    /// Process-wide pool, sized from the available parallelism.
    pub(crate) fn global() -> Option<&'static ThreadPool> {
        static POOL: OnceLock<Option<ThreadPool>> = OnceLock::new();
        POOL.get_or_init(|| {
            let size = std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(4)
                .clamp(2, 16);
            Some(ThreadPool::new(size))
        })
        .as_ref()
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    pub(crate) fn submit(&self, job: Job) -> bool {
        self.sender.send(job).is_ok()
    }
}