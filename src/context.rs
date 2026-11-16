//! Dynamic node context: human-in-the-loop interrupts and custom stream writes.
//!
//! The engine installs a thread-local task context while a node is running. Nodes can
//! then call [`interrupt`] or [`custom`] without threading a context argument through
//! every signature, which keeps node closures identical to their LangGraph counterparts.

use std::cell::RefCell;

use serde::Serialize;
use serde_json::Value;

use crate::{Error, Interrupt, Result};

thread_local! {
    static CURRENT_TASK: RefCell<Option<TaskContext>> = const { RefCell::new(None) };
}

#[derive(Debug, Default)]
pub(crate) struct TaskContext {
    node: String,
    resume_value: Option<Value>,
    resume_consumed: bool,
    custom: Vec<Value>,
}

impl TaskContext {
    pub(crate) fn new(node: impl Into<String>, resume_value: Option<Value>) -> Self {
        Self {
            node: node.into(),
            resume_value,
            resume_consumed: false,
            custom: Vec::new(),
        }
    }
}

/// Install `context` for the duration of `run`, returning its result and any custom
/// stream writes the node produced.
pub(crate) fn with_task<R>(context: TaskContext, run: impl FnOnce() -> R) -> (R, Vec<Value>) {
    let previous = CURRENT_TASK.with(|slot| slot.borrow_mut().replace(context));
    let result = run();
    let context = CURRENT_TASK.with(|slot| {
        let mut slot = slot.borrow_mut();
        std::mem::replace(&mut *slot, previous)
    });
    let custom = context.map(|context| context.custom).unwrap_or_default();
    (result, custom)
}

/// Pause the graph and wait for human input.
///
/// On the first execution this returns [`Error::Interrupt`] carrying `value`; the
/// engine checkpoints the state and unwinds. When the graph is resumed with
/// [`CompiledGraph::resume`](crate::CompiledGraph::resume), the node is re-executed and
/// this same call returns the value supplied by the caller instead.
///
/// ```no_run
/// use langgraph_rs::{interrupt, NodeOutput};
/// # fn example(state: &()) -> langgraph_rs::Result<NodeOutput> {
/// let answer: String = interrupt(serde_json::json!({ "question": "approve?" }))?
///     .as_str()
///     .unwrap_or("no")
///     .to_string();
/// Ok(NodeOutput::new().set("answer", answer))
/// # }
/// ```
pub fn interrupt(value: Value) -> Result<Value> {
    CURRENT_TASK.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(context) = slot.as_mut() else {
            return Err(Error::other(
                "interrupt() can only be called from inside a running graph node",
            ));
        };
        if !context.resume_consumed {
            if let Some(value) = context.resume_value.take() {
                context.resume_consumed = true;
                return Ok(value);
            }
        }
        Err(Error::Interrupt(Interrupt {
            node: context.node.clone(),
            value,
        }))
    })
}

/// Emit a custom value into the graph stream from inside a node.
pub fn custom(value: impl Serialize) -> Result<()> {
    push_custom(serde_json::to_value(value)?)
}

/// Emit an already serialized custom value into the graph stream.
pub fn push_custom(value: Value) -> Result<()> {
    CURRENT_TASK.with(|slot| match slot.borrow_mut().as_mut() {
        Some(context) => {
            context.custom.push(value);
            Ok(())
        }
        None => Err(Error::other(
            "custom() can only be called from inside a running graph node",
        )),
    })
}

/// Name of the node that is currently executing, if any.
pub fn current_node() -> Option<String> {
    CURRENT_TASK.with(|slot| slot.borrow().as_ref().map(|context| context.node.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn interrupt_without_resume_returns_the_payload() {
        let (result, _custom) = with_task(TaskContext::new("ask", None), || {
            interrupt(json!({ "question": "approve?" }))
        });
        let error = result.unwrap_err();
        assert_eq!(
            error.interrupt().map(|interrupt| interrupt.node.as_str()),
            Some("ask")
        );
        assert_eq!(
            error.interrupt().map(|interrupt| interrupt.value.clone()),
            Some(json!({ "question": "approve?" }))
        );
    }

    #[test]
    fn interrupt_with_resume_value_returns_it() {
        let (result, _custom) = with_task(TaskContext::new("ask", Some(json!("yes"))), || {
            interrupt(json!({ "question": "approve?" }))
        });
        assert_eq!(result.unwrap(), json!("yes"));
    }

    #[test]
    fn custom_values_are_collected() {
        let (result, custom) = with_task(TaskContext::new("emit", None), || {
            custom(json!({ "progress": 0.5 })).unwrap();
            push_custom(json!("done")).unwrap();
            current_node()
        });
        assert_eq!(result.as_deref(), Some("emit"));
        assert_eq!(custom, vec![json!({ "progress": 0.5 }), json!("done")]);
    }

    #[test]
    fn interrupt_outside_a_node_is_an_error() {
        assert!(interrupt(json!("nope")).is_err());
    }
}
