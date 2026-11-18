//! Per-run task result cache and subgraph stream collector.

use super::*;

/// RAII guard restoring the previous subgraph chunk collector.
pub(crate) struct CollectorGuard {
    previous: Option<Arc<Mutex<Vec<StreamChunk>>>>,
}

impl Drop for CollectorGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        SUBGRAPH_COLLECTOR.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

/// Install a collector that subgraph nodes push their chunks into.
pub(crate) fn install_subgraph_collector(
    collector: Arc<Mutex<Vec<StreamChunk>>>,
) -> CollectorGuard {
    let previous = SUBGRAPH_COLLECTOR.with(|slot| slot.borrow_mut().replace(collector));
    CollectorGuard { previous }
}

/// Collector installed by the running executor, when subgraph streaming is enabled.
pub(crate) fn current_subgraph_collector() -> Option<Arc<Mutex<Vec<StreamChunk>>>> {
    SUBGRAPH_COLLECTOR.with(|slot| slot.borrow().clone())
}

/// Read a memoized functional-API task result for `key`.
pub(crate) fn task_result_get(key: &str) -> Option<Value> {
    #[cfg(feature = "async")]
    if let Ok(value) = ASYNC_TASK_RESULTS.try_with(|slot| slot.borrow().get(key).cloned()) {
        return value;
    }
    TASK_RESULTS.with(|slot| slot.borrow().get(key).cloned())
}

/// Store a functional-API task result under `key`.
pub(crate) fn task_result_put(key: String, value: Value) {
    #[cfg(feature = "async")]
    if ASYNC_TASK_RESULTS
        .try_with(|slot| {
            slot.borrow_mut().insert(key.clone(), value.clone());
        })
        .is_ok()
    {
        return;
    }
    TASK_RESULTS.with(|slot| {
        slot.borrow_mut().insert(key, value);
    });
}

/// Reset the task-result cache at the start of a fresh run.
pub(crate) fn clear_task_results() {
    #[cfg(feature = "async")]
    if ASYNC_TASK_RESULTS
        .try_with(|slot| slot.borrow_mut().clear())
        .is_ok()
    {
        return;
    }
    TASK_RESULTS.with(|slot| slot.borrow_mut().clear());
}

/// Restore memoized task results from a resumed checkpoint.
pub(crate) fn load_task_results(results: &Map<String, Value>) {
    #[cfg(feature = "async")]
    if ASYNC_TASK_RESULTS
        .try_with(|slot| {
            let mut slot = slot.borrow_mut();
            slot.clear();
            slot.extend(results.clone());
        })
        .is_ok()
    {
        return;
    }
    TASK_RESULTS.with(|slot| {
        let mut slot = slot.borrow_mut();
        slot.clear();
        slot.extend(results.clone());
    });
}

/// Snapshot of the memoized task results for checkpoint persistence.
pub(crate) fn snapshot_task_results() -> Map<String, Value> {
    #[cfg(feature = "async")]
    if let Ok(results) = ASYNC_TASK_RESULTS.try_with(|slot| slot.borrow().clone()) {
        return results;
    }
    TASK_RESULTS.with(|slot| slot.borrow().clone())
}
