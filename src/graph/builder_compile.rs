//! Graph validation and compilation.
/// LangGraph's wording for a graph without an entry point.
const ENTRYPOINT_MESSAGE: &str =
    "Graph must have an entrypoint: add at least one edge from START to another node";



use super::*;

impl<S: State> StateGraph<S> {
    pub fn compile(self) -> Result<CompiledGraph<S>> {
        self.compile_with(CompileOptions::default())
    }

    /// Validate the definition and compile it with checkpointing / interrupt options.
    pub fn compile_with(self, options: CompileOptions) -> Result<CompiledGraph<S>> {
        self.validate()?;

        for node in options
            .interrupt_before
            .iter()
            .chain(options.interrupt_after.iter())
        {
            if !self.nodes.contains_key(node) {
                return Err(Error::validation(format!(
                    "interrupt hook references unknown node `{node}`"
                )));
            }
        }

        for node in self.retries.keys() {
            if !self.nodes.contains_key(node) {
                return Err(Error::validation(format!(
                    "retry policy references unknown node `{node}`"
                )));
            }
        }
        for (source, handler) in &self.error_edges {
            self.require_node("error edge source", source)?;
            self.require_node("error handler", handler)?;
        }
        for node in self.cache_policies.keys() {
            self.require_node("cache policy", node)?;
        }
        for node in self.timeouts.keys() {
            self.require_node("timeout policy", node)?;
        }
        for (node, destinations) in &self.destinations {
            self.require_node("destination list", node)?;
            for target in destinations {
                if target == END {
                    continue;
                }
                if target == START {
                    return Err(Error::validation(format!(
                        "`{START}` is not a valid destination of `{node}`"
                    )));
                }
                self.require_node("destination", target)?;
            }
        }
        for node in &self.deferred {
            self.require_node("deferred node", node)?;
            // LangGraph lets a deferred node have ordinary outgoing edges: they are followed once
            // the deferred task finally runs.
        }

        // Subgraphs compiled without their own checkpointer adopt the parent's saver, like
        // LangGraph (which passes the checkpointer through the run config).
        if let Some(saver) = &options.checkpointer {
            for hook in &self.subgraph_hooks {
                hook(saver);
            }
        }

        let entry = match self.conditional_entry {
            Some(edge) => Entry::Conditional(edge),
            None => Entry::Static(self.entry.clone()),
        };

        Ok(CompiledGraph {
            schema: self.schema,
            nodes: self.nodes,
            edges: self.edges,
            conditional: self.conditional,
            entry,
            checkpointer: options.checkpointer,
            trace_sinks: options.kernel.trace_sinks().to_vec(),
            retries: self.retries,
            error_edges: self.error_edges,
            cache: self.cache,
            cache_policies: self.cache_policies,
            store: options.store.or(self.store),
            node_metadata: self.node_metadata,
            timeouts: self.timeouts,
            deferred: self.deferred,
            destinations: self.destinations,
            recursion_limit: options
                .recursion_limit
                .unwrap_or_else(crate::graph::options::default_recursion_limit),
            interrupt_before: options.interrupt_before.into_iter().collect(),
            interrupt_after: options.interrupt_after.into_iter().collect(),
            name: options.name,
            context_schema: options.context_schema,
            inherited_saver: Arc::new(std::sync::OnceLock::new()),
            subgraphs: self.subgraphs,
            subgraph_hooks: self.subgraph_hooks,
        })
    }

    fn validate(&self) -> Result<()> {
        // `add_edge(START, END)` yields a graph without nodes that still compiles (LangGraph
        // returns the input unchanged), so only a graph with neither nodes nor edges fails here.
        if self.nodes.is_empty() && self.entry.is_empty() {
            return Err(Error::validation(ENTRYPOINT_MESSAGE));
        }
        if !self.duplicate_nodes.is_empty() {
            // LangGraph reports the first duplicate by name.
            return Err(Error::validation(format!(
                "Node `{}` already present.",
                self.duplicate_nodes[0]
            )));
        }
        for reserved in [START, END] {
            if self.nodes.contains_key(reserved) {
                return Err(Error::validation(format!(
                    "`{reserved}` is a reserved node name"
                )));
            }
        }

        if self.conditional_entry.is_some() && !self.entry.is_empty() {
            return Err(Error::validation(
                "a graph cannot combine a static entry point with a conditional entry point",
            ));
        }
        // LangGraph accepts `add_edge(START, END)`: the graph runs and returns its input unchanged,
        // so an entry of `END` counts as an entry point instead of failing validation.
        let entry_ends = self.entry.iter().any(|target| target == END);
        let real_entry = self.entry.iter().any(|target| target != END);
        if self.conditional_entry.is_none() && !real_entry && !entry_ends {
            return Err(Error::validation(ENTRYPOINT_MESSAGE));
        }

        for name in &self.entry {
            if name == END {
                continue;
            }
            self.require_node("entry point", name)?;
        }

        for (source, targets) in &self.edges {
            self.require_node("edge source", source)?;
            for target in targets {
                self.require_target(source, target)?;
            }
        }

        if let Some(edge) = &self.conditional_entry
            && let Some(path_map) = &edge.path_map
        {
            for (key, targets) in path_map.entries() {
                for target in targets {
                    if target == START {
                        return Err(Error::validation(format!(
                            "conditional entry point maps key `{key}` to `{START}`, which is not a valid destination"
                        )));
                    }
                    self.require_target(START, target)?;
                }
            }
        }

        for (source, edges) in &self.conditional {
            self.require_node("conditional edge source", source)?;
            for edge in edges {
                if let Some(path_map) = &edge.path_map {
                    for (key, targets) in path_map.entries() {
                        for target in targets {
                            if target == START {
                                return Err(Error::validation(format!(
                                    "conditional edge `{source}` maps key `{key}` to `{START}`, which is not a valid destination"
                                )));
                            }
                            self.require_target(source, target)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn require_node(&self, what: &str, name: &str) -> Result<()> {
        if name == START || name == END {
            return Err(Error::validation(format!(
                "`{name}` cannot be used as a {what}"
            )));
        }
        if !self.nodes.contains_key(name) {
            // LangGraph models `add_edge(START, x)` as an edge, so an unknown target reads that way.
            if what == "entry point" {
                return Err(Error::validation(format!(
                    "Found edge ending at unknown node `{name}`"
                )));
            }
            return Err(Error::validation(format!(
                "{what} `{name}` is not a registered node"
            )));
        }
        Ok(())
    }

    fn require_target(&self, source: &str, target: &str) -> Result<()> {
        if target == END {
            return Ok(());
        }
        if !self.nodes.contains_key(target) {
            // LangGraph names the node and, for a branch, the branch key.
            if self.conditional.contains_key(source) {
                return Err(Error::validation(format!(
                    "At '{source}' node, 'condition' branch found unknown target '{target}'"
                )));
            }
            return Err(Error::validation(format!(
                "Found edge ending at unknown node `{target}`"
            )));
        }
        if target == START {
            return Err(Error::validation(format!(
                "edge `{source}` targets `{START}`, which is not a valid destination"
            )));
        }
        if !self.nodes.contains_key(target) {
            return Err(Error::validation(format!(
                "edge `{source}` targets unknown node `{target}`"
            )));
        }
        Ok(())
    }
}
