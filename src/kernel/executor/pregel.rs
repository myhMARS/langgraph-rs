//! One Pregel loop shared by the sync and async runtimes.
//!
//! All superstep, checkpoint and interrupt decisions live here. A [`PregelBackend`]
//! supplies graph metadata, node-batch execution and checkpoint persistence; the sync
//! runtime drives this loop with `futures::executor::block_on`, the async runtime
//! awaits it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::Instant;

use serde_json::Value;

use crate::kernel::context;
use crate::kernel::error::{Error, Interrupt, Result};
use crate::kernel::executor::backend::PregelBackend;
use crate::kernel::executor::checkpointing::save_checkpoint_via;
use crate::kernel::executor::outcome::{
    TaskOutcome, apply_finish_updates, apply_processed_updates, collect_pending_writes,
    debug_chunk, finish, interrupt_after_node, interrupt_before_pause, interrupts_from,
    merge_interrupted_tasks, process_outcomes, should_save_after_step, should_save_terminal,
    values_chunk,
};
use crate::kernel::executor::scheduling::{collect_next_nodes, schedule_next_tasks};
use crate::kernel::executor::{StartInput, start, step};
use crate::kernel::graph::RunConfig;
use crate::kernel::node::Task;
use crate::kernel::state::{State, values_to_state};
use crate::kernel::stream::StreamChunk;
use crate::kernel::usage::RunOutcome;

/// Execute a graph through the shared Pregel loop.
pub(crate) async fn run_loop<S: State, B: PregelBackend<S>, F: FnMut(StreamChunk)>(
    backend: &B,
    start: StartInput,
    config: &RunConfig,
    sink: &mut F,
) -> Result<RunOutcome<S>> {
    let started = Instant::now();
    let mut report = crate::kernel::usage::RunReport::new();
    let recursion_limit = config
        .recursion_limit
        .unwrap_or_else(|| backend.recursion_limit());
    let thread_id = config.thread_id.clone();
    let namespace = config.checkpoint_ns.clone().unwrap_or_default();
    let durability = config.durability;
    let store = config.store.clone().or_else(|| backend.store());
    let setup = step::begin_run(
        thread_id.clone(),
        namespace.clone(),
        store,
        config.context.clone(),
        backend.context_schema(),
        config.stream_subgraphs,
        backend.collector(),
    )?;

    if thread_id.is_some() && !backend.has_checkpointer() {
        return Err(Error::MissingCheckpointer("a run bound to a thread_id"));
    }

    let fresh = matches!(
        &start,
        StartInput::Values(_) | StartInput::Updates(_) | StartInput::Gateways { .. }
    );
    let mut parent: Option<String> = None;
    let mut skip_interrupt_before = false;
    let (mut values, mut state, mut tasks, mut step) = if let Some((values, state, tasks)) =
        start::resolve_fresh_start(
            &start,
            backend.schema(),
            |node| backend.node_exists(node),
            |state| backend.entry_tasks(state),
        )? {
        (values, state, tasks, 0usize)
    } else {
        let StartInput::Resume {
            thread_id: resume_thread,
            checkpoint_id,
            resume,
            updates,
        } = start
        else {
            unreachable!("fresh start inputs were resolved above");
        };
        let checkpoint = backend
            .load_checkpoint(&resume_thread, checkpoint_id.as_deref(), &namespace)
            .await?
            .ok_or_else(|| {
                Error::Checkpoint(format!("no checkpoint found for thread `{resume_thread}`"))
            })?;
        parent = Some(checkpoint.id.clone());
        skip_interrupt_before = checkpoint.paused_before;
        let (values, state, tasks) =
            start::resolve_resume_start(&checkpoint, &resume, updates, backend.schema())?;
        (values, state, tasks, checkpoint.step)
    };

    if tasks.is_empty() {
        return finish(state, backend.schema(), report, started);
    }

    if fresh {
        context::clear_task_results();
        if let Some(tid) = thread_id.as_deref() {
            parent = backend
                .load_checkpoint(tid, None, &namespace)
                .await?
                .map(|checkpoint| checkpoint.id);
        }
        // LangGraph records an input checkpoint before the first superstep.
        save_checkpoint_via(
            backend,
            thread_id.as_deref(),
            &namespace,
            &mut parent,
            &values,
            &tasks,
            step,
            false,
            false,
            Vec::new(),
            Vec::new(),
            &[],
            sink,
        )
        .await?;
        save_checkpoint_via(
            backend,
            thread_id.as_deref(),
            &namespace,
            &mut parent,
            &values,
            &tasks,
            step,
            false,
            false,
            Vec::new(),
            Vec::new(),
            &[],
            sink,
        )
        .await?;
    }

    let mut pending_deferred: Vec<Task> = Vec::new();
    let mut finish_updates: HashMap<String, Value> = HashMap::new();
    let mut iteration = 0usize;
    loop {
        let prepared = step::begin_step(
            config.control.as_ref(),
            backend.schema(),
            &mut values,
            &mut state,
            step,
            recursion_limit,
            iteration,
            &tasks,
            skip_interrupt_before,
            |node| backend.interrupts_before(node),
        )?;
        iteration = prepared.iteration;
        let _step_guard = prepared.guard;
        let paused = prepared.paused;
        skip_interrupt_before = false;
        if let Some(interrupted) = paused {
            let (interrupted_interrupt, chunk) = interrupt_before_pause(&interrupted);
            save_checkpoint_via(
                backend,
                thread_id.as_deref(),
                &namespace,
                &mut parent,
                &values,
                &tasks,
                step,
                true,
                true,
                vec![interrupted_interrupt.clone()],
                Vec::new(),
                &[],
                sink,
            )
            .await?;
            sink(chunk);
            return Err(Error::Interrupt(interrupted_interrupt));
        }

        let outcomes = backend
            .run_superstep(&state, &tasks, step, config.control.as_ref())
            .await?;
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

        let interrupts = interrupts_from(&outcomes);
        if !interrupts.is_empty() {
            if interrupts.len() == 1 {
                let interrupted = interrupts.first().expect("one interrupt").clone();
                let interrupted_ids: HashSet<String> = tasks
                    .iter()
                    .zip(&outcomes)
                    .filter(|(_, outcome)| matches!(outcome, TaskOutcome::Interrupted(_)))
                    .map(|(task, _)| task.id.clone())
                    .collect();
                let pending_writes = collect_pending_writes(&tasks, &outcomes);
                let successful: Vec<TaskOutcome> = outcomes
                    .into_iter()
                    .filter(|outcome| !matches!(outcome, TaskOutcome::Interrupted(_)))
                    .collect();
                let processed = process_outcomes(backend, successful, step, &mut report, sink)?;
                let next_tasks = merge_interrupted_tasks(
                    &tasks,
                    |task| interrupted_ids.contains(&task.id),
                    processed.next_tasks,
                    true,
                );
                save_checkpoint_via(
                    backend,
                    thread_id.as_deref(),
                    &namespace,
                    &mut parent,
                    &values,
                    &next_tasks,
                    step,
                    true,
                    false,
                    vec![interrupted.clone()],
                    pending_writes,
                    &[],
                    sink,
                )
                .await?;
                sink(StreamChunk::Interrupt {
                    node: interrupted.node.clone(),
                    value: interrupted.value.clone(),
                });
                return Err(Error::Interrupt(interrupted));
            }
            let interrupted_nodes: HashSet<String> =
                interrupts.iter().map(|i| i.node.clone()).collect();
            let successful: Vec<TaskOutcome> = outcomes
                .into_iter()
                .filter(|outcome| !matches!(outcome, TaskOutcome::Interrupted(_)))
                .collect();
            let processed = process_outcomes(backend, successful, step, &mut report, sink)?;
            apply_processed_updates(
                backend.schema(),
                &mut values,
                processed.channel_updates,
                processed.replacements,
                &mut finish_updates,
            )?;
            sink(values_chunk(step, &values));
            let next_tasks = merge_interrupted_tasks(
                &tasks,
                |task| interrupted_nodes.contains(&task.node),
                processed.next_tasks,
                false,
            );
            save_checkpoint_via(
                backend,
                thread_id.as_deref(),
                &namespace,
                &mut parent,
                &values,
                &next_tasks,
                step,
                true,
                false,
                interrupts.clone(),
                Vec::new(),
                &[],
                sink,
            )
            .await?;
            for interrupt in &interrupts {
                sink(StreamChunk::Interrupt {
                    node: interrupt.node.clone(),
                    value: interrupt.value.clone(),
                });
            }
            return Err(Error::Interrupt(
                interrupts.into_iter().next().expect("interrupt"),
            ));
        }

        let processed = process_outcomes(backend, outcomes, step, &mut report, sink)?;
        let channel_updates = processed.channel_updates;
        let replacements = processed.replacements;
        let next_tasks = processed.next_tasks;
        let goto_targets = processed.goto_targets;
        let executed = processed.executed;
        let goto_nodes = processed.goto_nodes;

        apply_processed_updates(
            backend.schema(),
            &mut values,
            channel_updates,
            replacements,
            &mut finish_updates,
        )?;
        state = values_to_state::<S>(&values)?;
        sink(values_chunk(step, &values));
        if context::has_parent_command() {
            return finish(state, backend.schema(), report, started);
        }

        let (next_nodes, router_sends) = collect_next_nodes(
            &state,
            &executed,
            &goto_nodes,
            &goto_targets,
            |node| backend.edges_of(node),
            |node| backend.conditional_of(node),
            |node| backend.node_exists(node),
            |node| PregelBackend::destinations_of(backend, node),
        )?;
        let next_tasks = schedule_next_tasks(
            next_nodes,
            router_sends,
            next_tasks,
            |node| backend.node_exists(node),
            |node| PregelBackend::destinations_of(backend, node),
            |node| backend.is_deferred(node),
            &mut pending_deferred,
        )?;

        let scheduled: Vec<String> = next_tasks.iter().map(|task| task.node.clone()).collect();
        sink(debug_chunk(step, &executed, &scheduled));

        step += 1;
        let interrupt_after =
            interrupt_after_node(&executed, |node| backend.interrupts_after(node));
        if should_save_after_step(durability, interrupt_after.is_some()) {
            save_checkpoint_via(
                backend,
                thread_id.as_deref(),
                &namespace,
                &mut parent,
                &values,
                &next_tasks,
                step,
                interrupt_after.is_some(),
                false,
                interrupt_after
                    .iter()
                    .map(|node| Interrupt::new(node.clone(), Value::Null))
                    .collect(),
                Vec::new(),
                &executed,
                sink,
            )
            .await?;
        }
        if let Some(node) = interrupt_after {
            sink(StreamChunk::Interrupt {
                node: node.clone(),
                value: Value::Null,
            });
            return Err(Error::Interrupt(Interrupt::new(node, Value::Null)));
        }

        if next_tasks.is_empty() {
            if !finish_updates.is_empty() {
                apply_finish_updates(&mut values, &mut finish_updates);
                state = values_to_state::<S>(&values)?;
            }
            if should_save_terminal(durability, thread_id.is_some()) {
                save_checkpoint_via(
                    backend,
                    thread_id.as_deref(),
                    &namespace,
                    &mut parent,
                    &values,
                    &[],
                    step,
                    false,
                    false,
                    Vec::new(),
                    Vec::new(),
                    &executed,
                    sink,
                )
                .await?;
            }
            return finish(state, backend.schema(), report, started);
        }
        // Ephemeral channels are visible for exactly one superstep.
        backend.clear_ephemeral(&mut values);
        state = values_to_state::<S>(&values)?;
        tasks = next_tasks;
    }
}

/// Drive a future to completion, parking the current thread while it is pending.
///
/// Unlike utures::executor::block_on this supports nested runs (a subgraph executes
/// another graph on the same thread).
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
