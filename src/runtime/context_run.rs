//! Run and superstep context: thread-local guards, managed values, recursion.

use super::*;

/// Sink that receives stream chunks the moment a node produces them.
pub(crate) type LiveSink = std::sync::Arc<std::sync::Mutex<dyn FnMut(StreamChunk) + Send>>;

thread_local! {
    /// Sink installed by a caller (for example the Python binding's live stream) for the next run
    /// started on this thread; `install_run` consumes it.
    static PENDING_LIVE_SINK: std::cell::RefCell<Option<LiveSink>> =
        const { std::cell::RefCell::new(None) };
}

/// Messages channels of the latest compiled run.
///
/// Conversion to Python happens both inside a run (node input) and after it (results, snapshots)
/// and, for async graphs, on a different thread than the one that installed the scope — so the
/// names live in a process-wide registry instead of the run context.
static LAST_MESSAGES_CHANNELS: std::sync::LazyLock<std::sync::Mutex<Vec<String>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

/// Install `sink` for the next run started on this thread.
pub(crate) fn set_pending_live_sink(sink: LiveSink) {
    PENDING_LIVE_SINK.with(|slot| {
        *slot.borrow_mut() = Some(sink);
    });
}

/// Take (and clear) the pending live sink.
pub(crate) fn take_pending_live_sink() -> Option<LiveSink> {
    PENDING_LIVE_SINK.with(|slot| slot.borrow_mut().take())
}

/// The live sink of the current run, when one is installed.
pub(crate) fn current_live_sink() -> Option<LiveSink> {
    CURRENT_RUN.with(|slot| slot.borrow().as_ref().and_then(|run| run.live_sink.clone()))
}

/// Execution context of the current run.
#[derive(Clone, Default)]
pub(crate) struct RunContext {
    pub(crate) thread_id: Option<String>,
    pub(crate) namespace: String,
    pub(crate) store: Option<std::sync::Arc<dyn BaseStore>>,
    pub(crate) context: Option<Value>,
    pub(crate) step: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) managed_step: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) recursion_limit: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) step_active: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Emits node-produced chunks immediately when a caller asked for live output.
    pub(crate) live_sink: Option<LiveSink>,
    /// `RunnableConfig["tags"]` of the current run, exposed through `get_config()`.
    pub(crate) tags: Vec<String>,
    /// `RunnableConfig["metadata"]` of the current run.
    pub(crate) metadata: serde_json::Map<String, serde_json::Value>,
    /// Channels holding LangChain message lists (`add_messages`).
    pub(crate) messages_channels: Vec<String>,
}

/// RAII guard restoring the previous run context when the engine returns.
pub(crate) struct RunGuard {
    previous: Option<RunContext>,
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CURRENT_RUN.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

/// Install the run context for the duration of a graph execution.
/// Whether message channels are handed back to Python as LangChain message objects.
///
/// LangGraph always returns message objects; the extension is migrating to that representation, and
/// the switch keeps the old dictionary shape available until the whole surface is ready.
static MESSAGES_AS_OBJECTS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Enable or disable message-object conversion.
pub(crate) fn set_messages_as_objects(enabled: bool) {
    MESSAGES_AS_OBJECTS.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Whether message channels should be converted to LangChain message objects.
pub(crate) fn messages_as_objects() -> bool {
    MESSAGES_AS_OBJECTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Record which channels hold LangChain message lists.
pub(crate) fn set_messages_channels(channels: Vec<String>) {
    if let Ok(mut known) = LAST_MESSAGES_CHANNELS.lock() {
        *known = channels.clone();
    }
    CURRENT_RUN.with(|slot| {
        if let Some(run) = slot.borrow_mut().as_mut() {
            run.messages_channels = channels;
        }
    });
}

/// Whether `name` is an `add_messages` channel of the running graph.
pub(crate) fn is_messages_channel(name: &str) -> bool {
    let in_run = CURRENT_RUN
        .try_with(|slot| {
            slot.borrow()
                .as_ref()
                .is_some_and(|run| run.messages_channels.iter().any(|item| item == name))
        })
        .unwrap_or(false);
    in_run
        || LAST_MESSAGES_CHANNELS
            .lock()
            .is_ok_and(|known| known.iter().any(|item| item == name))
}

/// Tags and metadata of the active run.
pub(crate) fn user_config() -> (
    Vec<String>,
    serde_json::Map<String, serde_json::Value>,
) {
    CURRENT_RUN.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|run| (run.tags.clone(), run.metadata.clone()))
            .unwrap_or_default()
    })
}

/// Record the caller's `tags` / `metadata` on the active run (used by `get_config()`).
pub(crate) fn set_user_config(
    tags: Vec<String>,
    metadata: serde_json::Map<String, serde_json::Value>,
) {
    CURRENT_RUN.with(|slot| {
        if let Some(run) = slot.borrow_mut().as_mut() {
            run.tags = tags;
            run.metadata = metadata;
        }
    });
}

pub(crate) fn install_run(
    thread_id: Option<String>,
    namespace: String,
    store: Option<std::sync::Arc<dyn BaseStore>>,
    context: Option<Value>,
    live_sink: Option<LiveSink>,
) -> RunGuard {
    let live_sink = live_sink.or_else(take_pending_live_sink).or_else(current_live_sink);
    let previous = CURRENT_RUN.with(|slot| {
        slot.borrow_mut().replace(RunContext {
            tags: Vec::new(),
            metadata: serde_json::Map::new(),
            messages_channels: Vec::new(),
            thread_id,
            namespace,
            store,
            context,
            step: Default::default(),
            managed_step: Default::default(),
            recursion_limit: Default::default(),
            step_active: Default::default(),
            live_sink,
        })
    });
    RunGuard { previous }
}

/// Build a run context value for task-local scoping.
#[cfg(feature = "async")]
pub(crate) fn run_context(
    thread_id: Option<String>,
    namespace: String,
    store: Option<std::sync::Arc<dyn BaseStore>>,
    context: Option<Value>,
) -> RunContext {
    RunContext {
        tags: Vec::new(),
        metadata: serde_json::Map::new(),
        messages_channels: Vec::new(),
        thread_id,
        namespace,
        store,
        context,
        step: Default::default(),
        managed_step: Default::default(),
        recursion_limit: Default::default(),
        step_active: Default::default(),
        live_sink: None,
    }
}

/// Scope an async run with a task-local copy of its run context.
#[cfg(feature = "async")]
pub(crate) async fn with_async_run<F: std::future::Future>(
    run: RunContext,
    future: F,
) -> F::Output {
    let results = RefCell::new(Map::new());
    let writes = RefCell::new(Map::new());
    ASYNC_TASK_WRITES.scope(writes, ASYNC_TASK_RESULTS.scope(results, ASYNC_RUN.scope(run, future))).await
}

/// Scope an async node future with its task identity for context accessors.
#[cfg(feature = "async")]
pub(crate) async fn with_async_node<F: std::future::Future>(
    node: String,
    task_id: Option<String>,
    future: F,
) -> F::Output {
    ASYNC_NODE.scope((node, task_id), future).await
}

pub(crate) fn current_run() -> Option<RunContext> {
    #[cfg(feature = "async")]
    if let Ok(run) = ASYNC_RUN.try_with(|run| run.clone()) {
        return Some(run);
    }
    CURRENT_RUN.with(|slot| slot.borrow().clone())
}

/// Long-term memory store of the current run, when the graph was compiled with one.
pub fn get_store() -> Option<std::sync::Arc<dyn BaseStore>> {
    current_run().and_then(|run| run.store.clone())
}

/// User context passed to the current run via `RunConfig::with_context`.
pub fn current_context() -> Option<Value> {
    current_run().and_then(|run| run.context.clone())
}

/// Runtime information (thread, namespace, store, context) of the current run.
pub fn current_runtime() -> Option<Runtime> {
    current_run().map(|run| Runtime {
        thread_id: run.thread_id.clone(),
        namespace: run.namespace.clone(),
        store: run.store.clone(),
        context: run.context.clone(),
    })
}

#[cfg(feature = "async")]
pub(crate) fn has_sync_task() -> bool {
    CURRENT_TASK.with(|slot| slot.borrow().is_some())
}

/// Capture the current run context for a scoped worker thread.
pub(crate) fn capture_run_context() -> Option<RunContext> {
    current_run()
}

/// Install a captured run context on the current thread.
pub(crate) fn install_run_context(run: RunContext) -> RunGuard {
    let previous = CURRENT_RUN.with(|slot| slot.borrow_mut().replace(run));
    RunGuard { previous }
}

/// Current superstep context `(step, managed_step, recursion_limit)`, when inside a step.
///
/// The managed step is captured too so worker threads reinstall the same per-run
/// counter instead of falling back to the absolute checkpoint step.
pub(crate) fn capture_step_context() -> Option<(usize, usize, usize)> {
    CURRENT_STEP.with(|slot| *slot.borrow())
}

/// RAII guard restoring the previous step context.
type AsyncStepPrevious = (
    std::sync::Arc<std::sync::atomic::AtomicUsize>,
    std::sync::Arc<std::sync::atomic::AtomicUsize>,
    std::sync::Arc<std::sync::atomic::AtomicUsize>,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
    usize,
    usize,
    usize,
    bool,
);

pub(crate) struct StepGuard {
    previous: Option<(usize, usize, usize)>,
    async_previous: Option<AsyncStepPrevious>,
}

impl Drop for StepGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CURRENT_STEP.with(|slot| {
            *slot.borrow_mut() = previous;
        });
        if let Some((step, managed, limit, active, old_step, old_managed, old_limit, old_active)) =
            self.async_previous.take()
        {
            step.store(old_step, std::sync::atomic::Ordering::Relaxed);
            managed.store(old_managed, std::sync::atomic::Ordering::Relaxed);
            limit.store(old_limit, std::sync::atomic::Ordering::Relaxed);
            active.store(old_active, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Install the current superstep number and recursion limit for managed values.
pub(crate) fn install_step(step: usize, managed_step: usize, recursion_limit: usize) -> StepGuard {
    let previous = CURRENT_STEP.with(|slot| {
        slot.borrow_mut()
            .replace((step, managed_step, recursion_limit))
    });
    let async_previous = current_run().map(|run| {
        let old = (
            run.step.load(std::sync::atomic::Ordering::Relaxed),
            run.managed_step.load(std::sync::atomic::Ordering::Relaxed),
            run.recursion_limit
                .load(std::sync::atomic::Ordering::Relaxed),
            run.step_active.load(std::sync::atomic::Ordering::Relaxed),
        );
        run.step.store(step, std::sync::atomic::Ordering::Relaxed);
        run.managed_step
            .store(managed_step, std::sync::atomic::Ordering::Relaxed);
        run.recursion_limit
            .store(recursion_limit, std::sync::atomic::Ordering::Relaxed);
        run.step_active
            .store(true, std::sync::atomic::Ordering::Relaxed);
        (
            run.step,
            run.managed_step,
            run.recursion_limit,
            run.step_active,
            old.0,
            old.1,
            old.2,
            old.3,
        )
    });
    StepGuard {
        previous,
        async_previous,
    }
}

/// Remaining supersteps before the recursion limit (LangGraph `RemainingSteps`).
///
/// Returns `None` outside a run. The value is `recursion_limit - step`, so the first
/// superstep of a graph compiled with a limit of 25 reports 24.
pub fn remaining_steps() -> Option<usize> {
    if let Some(run) = current_run()
        && run.step_active.load(std::sync::atomic::Ordering::Relaxed)
    {
        let step = run.managed_step.load(std::sync::atomic::Ordering::Relaxed);
        let limit = run
            .recursion_limit
            .load(std::sync::atomic::Ordering::Relaxed);
        return Some(limit.saturating_sub(step.saturating_add(1)));
    }
    CURRENT_STEP.with(|slot| {
        slot.borrow()
            .map(|(_, managed, limit)| limit.saturating_sub(managed.saturating_add(1)))
    })
}

/// Whether the current superstep is the last one before the recursion limit.
///
/// Mirrors LangGraph's `IsLastStep` managed value: `true` when at most one superstep is
/// left (`remaining_steps() <= 1`).
pub fn is_last_step() -> bool {
    remaining_steps() == Some(1)
}
