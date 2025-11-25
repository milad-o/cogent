"""Hierarchical topology pattern.

Tree-structured agent organization with managers and teams,
enabling nested delegation and aggregation.
"""

from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from agenticflow.topologies.base import BaseTopology
from agenticflow.agents import Agent


@dataclass
class TeamNode:
    """Represents a node in the hierarchy tree.

    Attributes:
        manager: Name of the manager agent for this team.
        members: Names of direct team members (workers or sub-managers).
        parent: Name of parent manager (None for root).
    """

    manager: str
    members: list[str] = field(default_factory=list)
    parent: str | None = None


class HierarchicalTopology(BaseTopology):
    """Hierarchical pattern: tree-structured teams.

    Agents are organized in a tree with managers coordinating
    their teams. Ideal for:
    - Large organizations with clear reporting lines
    - Divide-and-conquer problem solving
    - Parallel team execution with aggregation

    Example:
        >>> ceo = Agent(AgentConfig(name="ceo"))
        >>> eng_lead = Agent(AgentConfig(name="eng_lead"))
        >>> dev1 = Agent(AgentConfig(name="dev1"))
        >>> dev2 = Agent(AgentConfig(name="dev2"))
        >>> topology = HierarchicalTopology(
        ...     config=TopologyConfig(name="org-chart"),
        ...     agents=[ceo, eng_lead, dev1, dev2],
        ...     hierarchy={
        ...         "ceo": ["eng_lead"],
        ...         "eng_lead": ["dev1", "dev2"],
        ...     },
        ...     root="ceo",
        ... )
        >>> result = await topology.run("Build a new feature")
    """

    def __init__(
        self,
        *args: Any,
        hierarchy: dict[str, list[str]] | None = None,
        root: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize hierarchical topology.

        Args:
            *args: Arguments passed to BaseTopology.
            hierarchy: Dict mapping manager names to their direct reports.
            root: Name of the root manager. Required if hierarchy provided.
            **kwargs: Keyword arguments passed to BaseTopology.
        """
        super().__init__(*args, **kwargs)

        if hierarchy:
            if not root:
                raise ValueError("root must be specified with hierarchy")
            if root not in self.agents:
                raise ValueError(f"Root '{root}' not in agents")

            self.hierarchy = hierarchy
            self.root = root
            self._validate_hierarchy()
        else:
            # Build flat hierarchy with first agent as root
            agent_names = list(self.agents.keys())
            self.root = agent_names[0] if agent_names else ""
            self.hierarchy = {
                self.root: agent_names[1:] if len(agent_names) > 1 else []
            }

        # Build team nodes
        self.teams = self._build_teams()

    def _validate_hierarchy(self) -> None:
        """Validate hierarchy structure."""
        all_mentioned = {self.root}
        for manager, members in self.hierarchy.items():
            if manager not in self.agents:
                raise ValueError(f"Manager '{manager}' not in agents")
            for member in members:
                if member not in self.agents:
                    raise ValueError(f"Member '{member}' not in agents")
                all_mentioned.add(member)

        # Check all agents are in hierarchy
        for agent_name in self.agents:
            if agent_name not in all_mentioned:
                raise ValueError(
                    f"Agent '{agent_name}' not in hierarchy"
                )

    def _build_teams(self) -> dict[str, TeamNode]:
        """Build team node structures from hierarchy."""
        teams: dict[str, TeamNode] = {}

        # Find parent for each manager
        parent_map: dict[str, str | None] = {self.root: None}
        for manager, members in self.hierarchy.items():
            for member in members:
                if member in self.hierarchy:  # member is also a manager
                    parent_map[member] = manager

        # Create team nodes
        for manager, members in self.hierarchy.items():
            teams[manager] = TeamNode(
                manager=manager,
                members=members,
                parent=parent_map.get(manager),
            )

        return teams

    def _get_leaves(self, manager: str) -> list[str]:
        """Get all leaf agents under a manager.

        Args:
            manager: Manager name.

        Returns:
            List of leaf (worker) agent names.
        """
        if manager not in self.hierarchy:
            return [manager]

        leaves = []
        for member in self.hierarchy[manager]:
            if member in self.hierarchy:
                leaves.extend(self._get_leaves(member))
            else:
                leaves.append(member)
        return leaves

    def _build_graph(self) -> CompiledStateGraph:
        """Build hierarchical graph.

        Structure:
            root_manager -> sub_managers / workers -> aggregate -> ...
        """
        builder = StateGraph(dict)

        # Add manager nodes
        for manager in self.hierarchy:
            builder.add_node(
                f"{manager}_delegate",
                self._create_delegation_node(manager),
            )
            builder.add_node(
                f"{manager}_aggregate",
                self._create_aggregation_node(manager),
            )

        # Add worker nodes (non-managers)
        all_members = set()
        for members in self.hierarchy.values():
            all_members.update(members)

        workers = [m for m in all_members if m not in self.hierarchy]
        for worker in workers:
            builder.add_node(
                worker,
                self._create_agent_node(self.agents[worker]),
            )

        # Connect graph
        self._connect_hierarchy(builder)

        # Start with root delegation
        builder.set_entry_point(f"{self.root}_delegate")

        return builder.compile(checkpointer=self.checkpointer)

    def _connect_hierarchy(self, builder: StateGraph) -> None:
        """Connect hierarchy nodes in the graph.

        Args:
            builder: StateGraph builder.
        """
        for manager, members in self.hierarchy.items():
            delegate_node = f"{manager}_delegate"
            aggregate_node = f"{manager}_aggregate"

            # Delegation routes to members
            member_targets = {}
            for member in members:
                if member in self.hierarchy:
                    # Route to sub-manager's delegation
                    member_targets[member] = f"{member}_delegate"
                else:
                    # Route to worker
                    member_targets[member] = member

            builder.add_conditional_edges(
                delegate_node,
                lambda s, m=members: self._delegation_route(s, m),
                member_targets,
            )

            # Members route back to aggregation
            for member in members:
                if member in self.hierarchy:
                    # Sub-manager aggregation goes to parent aggregation
                    builder.add_edge(f"{member}_aggregate", aggregate_node)
                else:
                    # Worker goes to manager's aggregation
                    builder.add_edge(member, aggregate_node)

            # Aggregation routes up or ends
            if manager == self.root:
                builder.add_edge(aggregate_node, END)
            # else: handled by parent's member connections

    def _create_delegation_node(self, manager: str) -> Any:
        """Create delegation node for a manager.

        The manager analyzes the task and delegates to team members.
        """
        agent = self.agents[manager]
        members = self.hierarchy.get(manager, [])

        async def delegation_node(state: dict[str, Any]) -> dict[str, Any]:
            """Manager delegates to team."""
            task = state.get("task", "")
            context = state.get("context", {})
            context["manager"] = manager
            context["team_members"] = members

            # Manager decides how to delegate
            thought = await agent.think(
                f"Task: {task}\n\n"
                f"Your team: {', '.join(members)}\n"
                f"Decide how to delegate this task to your team.",
                context,
            )

            # Parse delegation
            delegations = self._parse_delegations(thought, members)

            return {
                **state,
                "context": context,
                "delegations": delegations,
                "current_manager": manager,
                "messages": state.get("messages", [])
                + [
                    {
                        "role": "assistant",
                        "content": thought,
                        "agent": manager,
                        "type": "delegation",
                    }
                ],
            }

        return delegation_node

    def _create_aggregation_node(self, manager: str) -> Any:
        """Create aggregation node for a manager.

        The manager synthesizes results from team members.
        """
        agent = self.agents[manager]

        async def aggregation_node(state: dict[str, Any]) -> dict[str, Any]:
            """Manager aggregates team results."""
            results = state.get("results", [])
            task = state.get("task", "")

            # Format team results
            team_results = "\n".join(
                f"- {r['agent']}: {r.get('thought', '')[:300]}"
                for r in results
                if r.get("agent") in self.hierarchy.get(manager, [])
            )

            # Manager synthesizes
            thought = await agent.think(
                f"Task: {task}\n\n"
                f"Team results:\n{team_results}\n\n"
                f"Synthesize and provide your conclusion.",
                state.get("context", {}),
            )

            return {
                **state,
                "results": results
                + [
                    {
                        "agent": manager,
                        "thought": thought,
                        "type": "aggregation",
                    }
                ],
                "messages": state.get("messages", [])
                + [
                    {
                        "role": "assistant",
                        "content": thought,
                        "agent": manager,
                        "type": "aggregation",
                    }
                ],
            }

        return aggregation_node

    def _parse_delegations(
        self, thought: str, members: list[str]
    ) -> dict[str, str]:
        """Parse manager's delegation decisions.

        Args:
            thought: Manager's reasoning.
            members: Available team members.

        Returns:
            Dict mapping member to their assigned subtask.
        """
        delegations: dict[str, str] = {}
        thought_lower = thought.lower()

        for member in members:
            if member.lower() in thought_lower:
                # Extract context around member mention
                idx = thought_lower.find(member.lower())
                context = thought[max(0, idx - 20): idx + len(member) + 100]
                delegations[member] = context.strip()

        # Default: all members get the same task
        if not delegations:
            for member in members:
                delegations[member] = "process the task"

        return delegations

    def _delegation_route(
        self, state: dict[str, Any], members: list[str]
    ) -> str:
        """Route from delegation to first member.

        For simplicity, we process members sequentially.
        A more advanced implementation could parallelize.
        """
        # Find first unprocessed member
        processed = state.get("processed_members", [])
        for member in members:
            if member not in processed:
                return member

        # All processed - shouldn't reach here
        return members[0] if members else END

    def _route(self, state: dict[str, Any]) -> str:
        """General route implementation."""
        return state.get("next_node", END)
