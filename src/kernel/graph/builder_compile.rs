//! Graph validation and compilation.

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
            if self.entry.contains(node) {
                return Err(Error::validation(format!(
                    "deferred node `{node}` cannot be an entry point"
                )));
            }
            if self
                .edges
                .get(node)
                .is_some_and(|targets| !targets.is_empty())
            {
                return Err(Error::validation(format!(
                    "deferred node `{node}` cannot have outgoing edges"
                )));
            }
            if self
                .conditional
                .get(node)
                .is_some_and(|edges| !edges.is_empty())
            {
                return Err(Error::validation(format!(
                    "deferred node `{node}` cannot have conditional edges"
                )));
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
            recursion_limit: options.recursion_limit.unwrap_or(DEFAULT_RECURSION_LIMIT),
            interrupt_before: options.interrupt_before.into_iter().collect(),
            interrupt_after: options.interrupt_after.into_iter().collect(),
            name: options.name,
            context_schema: options.context_schema,
        })
    }

    fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(Error::validation("the graph has no nodes"));
        }
        if !self.duplicate_nodes.is_empty() {
            return Err(Error::validation(format!(
                "nodes registered more than once: {}",
                self.duplicate_nodes.join(", ")
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
        if self.conditional_entry.is_none() && self.entry.is_empty() {
            return Err(Error::validation(
                "no entry point: call set_entry_point(...) or add_edge(START, ...)",
            ));
        }

        for name in &self.entry {
            self.require_node("entry point", name)?;
        }

        for (source, targets) in &self.edges {
            self.require_node("edge source", source)?;
            for target in targets {
                self.require_target(source, target)?;
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
