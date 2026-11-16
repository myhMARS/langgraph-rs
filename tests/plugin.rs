//! Microkernel plugin and hexagonal port tests.

use std::sync::{Arc, Mutex};

use langgraph_rs::{
    Channel, CompileOptions, Kernel, NodeExecution, NodeOutput, Plugin, RunConfig, State,
    StateGraph, StateSchema, TraceSink,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct PluginState {
    total: i64,
}

impl State for PluginState {
    fn schema() -> StateSchema {
        // `double_sum` is registered by the `DoubleSum` plugin below.
        Kernel::builder()
            .with_plugin(DoubleSum)
            .expect("plugin installs")
            .build()
            .schema([("total".to_string(), "double_sum".to_string())])
            .expect("plugin schema resolves")
    }
}

/// A downstream microkernel plugin: a reducer that adds `incoming * 2`.
struct DoubleSum;

impl Plugin for DoubleSum {
    fn name(&self) -> &'static str {
        "double-sum"
    }

    fn install(&self, kernel: &mut Kernel) -> langgraph_rs::Result<()> {
        kernel.register_channel("double_sum", || {
            Channel::reducer(|current, incoming| {
                let left = current.as_i64().unwrap_or_default();
                let right = incoming.as_i64().unwrap_or_default();
                Ok(serde_json::json!(left + right * 2))
            })
        });
        Ok(())
    }
}

#[test]
fn plugins_can_register_channel_policies() {
    let graph = StateGraph::<PluginState>::new()
        .add_node("one", |_state: &PluginState| {
            Ok(NodeOutput::new().set("total", 1))
        })
        .add_node("two", |_state: &PluginState| {
            Ok(NodeOutput::new().set("total", 2))
        })
        .add_edge(langgraph_rs::START, "one")
        .add_edge("one", "two")
        .set_finish_point("two")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&PluginState::default(), &RunConfig::new())
        .unwrap();
    // default state starts at 0; the reducer folds 1 then 2 -> 0 + 1*2 + 2*2 = 6
    assert_eq!(output.total, 6);
}

#[test]
fn builtin_kernel_exposes_the_default_channel_specs() {
    let kernel = Kernel::builtin();
    let specs: Vec<&str> = kernel.channel_specs().collect();
    for expected in ["last_value", "topic", "append", "add", "messages"] {
        assert!(specs.contains(&expected), "missing {expected} in {specs:?}");
    }
    assert!(kernel.resolve_channel("append").is_some());
    assert!(kernel.resolve_channel("does-not-exist").is_none());
}

#[test]
fn unknown_plugin_specs_are_rejected() {
    let error = Kernel::builtin()
        .schema([("x".to_string(), "nope".to_string())])
        .unwrap_err();
    assert!(error.to_string().contains("unknown channel plugin `nope`"));
}

#[derive(Clone, Default)]
struct RecordingSink(Arc<Mutex<Vec<NodeExecution>>>);

impl TraceSink for RecordingSink {
    fn record(&self, execution: &NodeExecution) {
        self.0.lock().unwrap().push(execution.clone());
    }
}

#[test]
fn trace_sinks_receive_every_node_execution() {
    let sink = Arc::new(RecordingSink::default());
    let kernel = Kernel::builder().with_trace_sink(sink.clone()).build();

    let graph = StateGraph::<PluginState>::new()
        .add_node("first", |_state: &PluginState| {
            langgraph_rs::record_tokens(10, 2)?;
            Ok(NodeOutput::new().set("total", 10))
        })
        .add_node("second", |_state: &PluginState| {
            Ok(NodeOutput::new().set("total", 1))
        })
        .set_entry_point("first")
        .add_edge("first", "second")
        .set_finish_point("second")
        .compile_with(CompileOptions::new().with_kernel(kernel))
        .unwrap();

    graph
        .invoke(&PluginState::default(), &RunConfig::new())
        .unwrap();

    let executions = sink.0.lock().unwrap().clone();
    assert_eq!(executions.len(), 2);
    assert_eq!(executions[0].node, "first");
    assert_eq!(executions[0].usage.as_ref().unwrap().total_tokens(), 12);
    assert_eq!(executions[1].node, "second");
}

#[test]
fn closure_trace_sinks_work_through_the_blanket_impl() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let captured = recorded.clone();
    let sink: Arc<dyn TraceSink> = Arc::new(move |execution: &NodeExecution| {
        captured.lock().unwrap().push(execution.node.clone());
    });
    let kernel = Kernel::builder().with_trace_sink(sink).build();

    let graph = StateGraph::<PluginState>::new()
        .add_node("only", |_state: &PluginState| Ok(NodeOutput::new()))
        .set_entry_point("only")
        .set_finish_point("only")
        .compile_with(CompileOptions::new().with_kernel(kernel))
        .unwrap();
    graph
        .invoke(&PluginState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(*recorded.lock().unwrap(), vec!["only".to_string()]);
}
