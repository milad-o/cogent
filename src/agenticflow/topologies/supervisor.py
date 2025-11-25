"""Supervisor topology pattern.

One supervisor agent coordinates multiple worker agents,
delegating tasks and synthesizing results.
"""

from typing import Any, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessage

from agenticflow.topologies.base import BaseTopology, TopologyState


class SupervisorTopology(BaseTopology):
    """Supervisor pattern: one coordinator, multiple workers.

    The supervisor receives tasks, delegates to appropriate workers,
    and synthesizes final results. This is ideal for:
    - Task decomposition and distribution
    - Quality control and result aggregation
    - Hierarchical decision making

    Example:
        >>> supervisor = Agent(AgentConfig(name="supervisor", role=AgentRole.COORDINATOR))
        >>> worker1 = Agent(AgentConfig(name="researcher", role=AgentRole.WORKER))
        >>> worker2 = Agent(AgentConfig(name="writer", role=AgentRole.WORKER))
        >>> topology = SupervisorTopology(
        ...     config=TopologyConfig(name="content-team"),
        ...     agents=[supervisor, worker1, worker2],
        ...     supervisor_name="supervisor",
        ... )
        >>> result = await topology.run("Write a blog post about AI")
    """

    def __init__(
        self,
        *args: Any,
        supervisor_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize supervisor topology.

        Args:
            *args: Arguments passed to BaseTopology.
            supervisor_name: Name of the supervisor agent. If not provided,
                            first agent is used as supervisor.
            **kwargs: Keyword arguments passed to BaseTopology.
        """
        super().__init__(*args, **kwargs)

        if supervisor_name:
            if supervisor_name not in self.agents:
                raise ValueError(f"Supervisor '{supervisor_name}' not in agents")
            self.supervisor_name = supervisor_name
        else:
            # First agent is supervisor
            self.supervisor_name = next(iter(self.agents))

        self.worker_names = [
            name for name in self.agents if name != self.supervisor_name
        ]

    def _build_graph(self) -> CompiledStateGraph:
        """Build supervisor graph.

        Structure:
            supervisor -> (worker selection) -> worker -> supervisor -> ...
            supervisor -> END (when complete)
        """
        # Define state schema
        builder = StateGraph(dict)

        # Add supervisor node
        builder.add_node("supervisor", self._supervisor_node)

        # Add worker nodes
        for worker_name in self.worker_names:
            builder.add_node(
                worker_name,
                self._create_agent_node(self.agents[worker_name]),
            )

        # Add routing from supervisor
        builder.add_conditional_edges(
            "supervisor",
            self._route,
            {
                **{name: name for name in self.worker_names},
                "FINISH": END,
            },
        )

        # Workers always return to supervisor
        for worker_name in self.worker_names:
            builder.add_edge(worker_name, "supervisor")

        # Start with supervisor
        builder.set_entry_point("supervisor")

        # Compile with checkpointer if available
        return builder.compile(checkpointer=self.checkpointer)

    async def _supervisor_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Supervisor decision node.

        The supervisor analyzes the task and results, then decides
        which worker to delegate to next or whether to finish.
        """
        supervisor = self.agents[self.supervisor_name]

        # Build supervisor prompt
        task = state.get("task", "")
        results = state.get("results", [])
        iteration = state.get("iteration", 0)

        # Format worker results
        results_text = ""
        if results:
            for r in results:
                results_text += f"\n- {r['agent']}: {r.get('thought', '')[:200]}..."

        context = {
            "workers": self.worker_names,
            "results_so_far": results_text,
            "iteration": iteration,
            "max_iterations": self.config.max_iterations,
        }

        # Supervisor thinks about next action
        thought = await supervisor.think(
            f"Task: {task}\n\nAvailable workers: {', '.join(self.worker_names)}\n"
            f"Results so far:{results_text or ' None yet'}\n\n"
            f"Decide: delegate to a worker OR finish if task is complete.",
            context,
        )

        # Parse supervisor decision
        next_agent = self._parse_supervisor_decision(thought)

        return {
            "messages": state.get("messages", [])
            + [
                {
                    "role": "assistant",
                    "content": thought,
                    "agent": self.supervisor_name,
                    "decision": next_agent,
                }
            ],
            "current_agent": self.supervisor_name,
            "next_worker": next_agent,
            "iteration": iteration + 1,
            "completed": next_agent == "FINISH",
        }

    def _parse_supervisor_decision(self, thought: str) -> str:
        """Parse supervisor's thought to extract next worker or FINISH.

        Args:
            thought: Supervisor's reasoning.

        Returns:
            Worker name or "FINISH".
        """
        thought_lower = thought.lower()

        # Check for finish signals
        finish_signals = ["finish", "complete", "done", "final", "end"]
        for signal in finish_signals:
            if signal in thought_lower:
                return "FINISH"

        # Check for worker mentions
        for worker_name in self.worker_names:
            if worker_name.lower() in thought_lower:
                return worker_name

        # Default to first worker if unclear
        if self.worker_names:
            return self.worker_names[0]

        return "FINISH"

    def _route(self, state: dict[str, Any]) -> str:
        """Route based on supervisor's decision."""
        # Check iteration limit
        if state.get("iteration", 0) >= self.config.max_iterations:
            return "FINISH"

        # Check if completed
        if state.get("completed"):
            return "FINISH"

        # Get supervisor's decision
        next_worker = state.get("next_worker", "FINISH")

        if next_worker in self.worker_names:
            return next_worker

        return "FINISH"
