"""Mesh topology pattern.

All agents can communicate with all other agents,
enabling flexible peer-to-peer coordination.
"""

from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agenticflow.topologies.base import BaseTopology


class MeshTopology(BaseTopology):
    """Mesh pattern: all agents can communicate with each other.

    Every agent can hand off to any other agent based on
    the task requirements. This is ideal for:
    - Collaborative problem solving
    - Dynamic task routing
    - Peer review workflows

    Example:
        >>> analyst = Agent(AgentConfig(name="analyst"))
        >>> reviewer = Agent(AgentConfig(name="reviewer"))
        >>> editor = Agent(AgentConfig(name="editor"))
        >>> topology = MeshTopology(
        ...     config=TopologyConfig(name="collaborative-team"),
        ...     agents=[analyst, reviewer, editor],
        ... )
        >>> result = await topology.run("Analyze and refine this report")
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize mesh topology."""
        super().__init__(*args, **kwargs)
        self.agent_names = list(self.agents.keys())

    def _build_graph(self) -> CompiledStateGraph:
        """Build fully connected mesh graph.

        Structure:
            router -> agent_a <-> agent_b <-> agent_c -> ...
            any agent -> END (when complete)
        """
        builder = StateGraph(dict)

        # Add router node to select first agent
        builder.add_node("router", self._router_node)

        # Add all agent nodes with handoff capability
        for name in self.agent_names:
            builder.add_node(name, self._create_mesh_agent_node(self.agents[name]))

        # Router routes to any agent
        builder.add_conditional_edges(
            "router",
            self._initial_route,
            {name: name for name in self.agent_names},
        )

        # Each agent can route to any other agent or END
        for name in self.agent_names:
            targets = {
                other: other for other in self.agent_names if other != name
            }
            targets["FINISH"] = END
            builder.add_conditional_edges(name, self._route, targets)

        # Start with router
        builder.set_entry_point("router")

        return builder.compile(checkpointer=self.checkpointer)

    async def _router_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Initial router to select first agent."""
        task = state.get("task", "")

        # Simple heuristic: pick first agent or based on task keywords
        selected = self._select_initial_agent(task)

        return {
            **state,
            "next_agent": selected,
            "iteration": 0,
        }

    def _select_initial_agent(self, task: str) -> str:
        """Select initial agent based on task.

        Override this for custom selection logic.

        Args:
            task: The task description.

        Returns:
            Name of the agent to start with.
        """
        task_lower = task.lower()

        # Check if any agent name is mentioned in task
        for name in self.agent_names:
            if name.lower() in task_lower:
                return name

        # Default to first agent
        return self.agent_names[0] if self.agent_names else ""

    def _create_mesh_agent_node(
        self, agent: Any
    ) -> Any:
        """Create mesh agent node with handoff capability.

        Mesh agents can explicitly hand off to other agents
        by mentioning them in their response.
        """
        base_node = self._create_agent_node(agent)

        async def mesh_node(state: dict[str, Any]) -> dict[str, Any]:
            """Process and determine next handoff."""
            result = await base_node(state)

            # Parse agent's response for handoff hints
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                content = last_message.get("content", "")

                # Check for handoff mentions
                next_agent = self._parse_handoff(content, agent.config.name)
                result["next_agent"] = next_agent

                # Check for completion
                if self._is_complete(content):
                    result["completed"] = True

            return result

        return mesh_node

    def _parse_handoff(self, content: str, current_agent: str) -> str | None:
        """Parse agent response for handoff target.

        Args:
            content: Agent's response content.
            current_agent: Name of current agent.

        Returns:
            Target agent name or None.
        """
        content_lower = content.lower()

        # Check for explicit handoff mentions
        handoff_keywords = ["hand off to", "pass to", "delegate to", "send to"]
        for keyword in handoff_keywords:
            if keyword in content_lower:
                # Find which agent is mentioned after the keyword
                for name in self.agent_names:
                    if name != current_agent and name.lower() in content_lower:
                        return name

        # Check for implicit mentions
        for name in self.agent_names:
            if name != current_agent and name.lower() in content_lower:
                return name

        return None

    def _is_complete(self, content: str) -> bool:
        """Check if task is complete based on response.

        Args:
            content: Agent's response content.

        Returns:
            True if task appears complete.
        """
        completion_signals = [
            "task complete",
            "finished",
            "done",
            "final answer",
            "here is the result",
        ]
        content_lower = content.lower()
        return any(signal in content_lower for signal in completion_signals)

    def _initial_route(self, state: dict[str, Any]) -> str:
        """Route from initial router."""
        return state.get("next_agent", self.agent_names[0])

    def _route(self, state: dict[str, Any]) -> str:
        """Route based on agent's handoff decision."""
        # Check limits
        if state.get("iteration", 0) >= self.config.max_iterations:
            return "FINISH"

        if state.get("completed"):
            return "FINISH"

        # Get handoff target
        next_agent = state.get("next_agent")
        if next_agent and next_agent in self.agent_names:
            return next_agent

        return "FINISH"
