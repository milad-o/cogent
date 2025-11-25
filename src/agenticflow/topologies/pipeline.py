"""Pipeline topology pattern.

Agents process sequentially in a defined order,
each transforming or enriching the result.
"""

from typing import Any, Sequence

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from agenticflow.topologies.base import BaseTopology
from agenticflow.agents import Agent


class PipelineTopology(BaseTopology):
    """Pipeline pattern: sequential agent processing.

    Agents process in a fixed order, with each agent's output
    becoming the next agent's input. Ideal for:
    - Multi-stage processing (extract -> transform -> load)
    - Content pipelines (research -> write -> edit -> review)
    - Sequential refinement workflows

    Example:
        >>> researcher = Agent(AgentConfig(name="researcher"))
        >>> writer = Agent(AgentConfig(name="writer"))
        >>> editor = Agent(AgentConfig(name="editor"))
        >>> topology = PipelineTopology(
        ...     config=TopologyConfig(name="content-pipeline"),
        ...     agents=[researcher, writer, editor],
        ...     stages=["researcher", "writer", "editor"],
        ... )
        >>> result = await topology.run("Create a technical article")
    """

    def __init__(
        self,
        *args: Any,
        stages: Sequence[str] | None = None,
        allow_skip: bool = False,
        allow_repeat: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize pipeline topology.

        Args:
            *args: Arguments passed to BaseTopology.
            stages: Ordered list of agent names defining the pipeline.
                   If not provided, uses agent order from initialization.
            allow_skip: Whether stages can skip downstream stages.
            allow_repeat: Whether a stage can loop back to previous stages.
            **kwargs: Keyword arguments passed to BaseTopology.
        """
        super().__init__(*args, **kwargs)

        if stages:
            # Validate all stages are in agents
            for stage in stages:
                if stage not in self.agents:
                    raise ValueError(f"Stage '{stage}' not in agents")
            self.stages = list(stages)
        else:
            # Use agent order
            self.stages = list(self.agents.keys())

        self.allow_skip = allow_skip
        self.allow_repeat = allow_repeat

    def _build_graph(self) -> CompiledStateGraph:
        """Build sequential pipeline graph.

        Structure:
            stage_1 -> stage_2 -> stage_3 -> ... -> END
        """
        builder = StateGraph(dict)

        # Add all stage nodes
        for stage_name in self.stages:
            builder.add_node(
                stage_name,
                self._create_pipeline_node(self.agents[stage_name], stage_name),
            )

        # Connect stages sequentially
        for i, stage_name in enumerate(self.stages[:-1]):
            next_stage = self.stages[i + 1]

            if self.allow_skip or self.allow_repeat:
                # Conditional routing
                builder.add_conditional_edges(
                    stage_name,
                    lambda s, ns=next_stage: self._pipeline_route(s, ns),
                    self._get_stage_targets(i),
                )
            else:
                # Simple sequential
                builder.add_edge(stage_name, next_stage)

        # Last stage goes to END
        last_stage = self.stages[-1]
        if self.allow_skip or self.allow_repeat:
            builder.add_conditional_edges(
                last_stage,
                lambda s: self._final_route(s),
                {
                    **({name: name for name in self.stages} if self.allow_repeat else {}),
                    "FINISH": END,
                },
            )
        else:
            builder.add_edge(last_stage, END)

        # Start with first stage
        builder.set_entry_point(self.stages[0])

        return builder.compile(checkpointer=self.checkpointer)

    def _get_stage_targets(self, current_index: int) -> dict[str, str]:
        """Get valid targets from current stage.

        Args:
            current_index: Index of current stage.

        Returns:
            Dict mapping target names to node names.
        """
        targets: dict[str, str] = {}

        # Can always go forward
        if current_index < len(self.stages) - 1:
            for name in self.stages[current_index + 1:]:
                targets[name] = name

        # Can skip to end
        if self.allow_skip:
            targets["FINISH"] = END

        # Can go back
        if self.allow_repeat:
            for name in self.stages[:current_index]:
                targets[name] = name

        return targets

    def _create_pipeline_node(
        self, agent: Agent, stage_name: str
    ) -> Any:
        """Create pipeline stage node.

        Pipeline nodes receive the accumulated context from
        previous stages and add their contribution.
        """
        base_node = self._create_agent_node(agent)

        async def pipeline_node(state: dict[str, Any]) -> dict[str, Any]:
            """Process pipeline stage."""
            # Track stage in context
            context = state.get("context", {})
            context["current_stage"] = stage_name
            context["stage_index"] = self.stages.index(stage_name)
            context["total_stages"] = len(self.stages)

            # Get previous stage results
            results = state.get("results", [])
            if results:
                context["previous_result"] = results[-1]

            state_with_context = {**state, "context": context}

            # Run base node
            result = await base_node(state_with_context)

            # Track stage completion
            result["completed_stages"] = state.get("completed_stages", []) + [stage_name]
            result["current_stage"] = stage_name

            return result

        return pipeline_node

    def _pipeline_route(self, state: dict[str, Any], default_next: str) -> str:
        """Route to next stage with skip/repeat logic.

        Args:
            state: Current state.
            default_next: Default next stage in sequence.

        Returns:
            Next stage name or "FINISH".
        """
        # Check for explicit next stage
        next_stage = state.get("next_stage")
        if next_stage:
            if next_stage == "FINISH":
                return "FINISH"
            if next_stage in self.stages:
                # Validate skip/repeat permissions
                current_idx = self.stages.index(state.get("current_stage", ""))
                target_idx = self.stages.index(next_stage)

                if target_idx > current_idx + 1 and not self.allow_skip:
                    return default_next
                if target_idx < current_idx and not self.allow_repeat:
                    return default_next

                return next_stage

        return default_next

    def _final_route(self, state: dict[str, Any]) -> str:
        """Route from final stage."""
        if self.allow_repeat:
            # Check if repeat requested
            next_stage = state.get("next_stage")
            if next_stage and next_stage in self.stages:
                return next_stage

        return "FINISH"

    def _route(self, state: dict[str, Any]) -> str:
        """General route implementation."""
        return state.get("next_stage", "FINISH")
