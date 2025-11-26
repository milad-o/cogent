"""Custom topology with explicit agent relationships.

Allows defining arbitrary connections between agents for
flexible coordination patterns and accurate visualization.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence

from langgraph.types import Command

from agenticflow.agent import Agent
from agenticflow.topologies.base import BaseTopology, TopologyConfig, TopologyState


@dataclass
class Edge:
    """An edge (connection) between two agents.

    Attributes:
        source: Name of the source agent.
        target: Name of the target agent.
        label: Optional label for the edge (shown in diagrams).
        condition: Optional condition description for when this edge is taken.
        bidirectional: Whether the edge goes both ways.
    """

    source: str
    target: str
    label: str = ""
    condition: str = ""
    bidirectional: bool = False

    def __post_init__(self) -> None:
        """Validate edge."""
        if not self.source or not self.target:
            raise ValueError("Edge must have both source and target")

    def reversed(self) -> "Edge":
        """Get reversed edge (for bidirectional edges)."""
        return Edge(
            source=self.target,
            target=self.source,
            label=self.label,
            condition=self.condition,
            bidirectional=False,
        )


@dataclass
class CustomTopologyConfig(TopologyConfig):
    """Configuration for custom topology.

    Extends TopologyConfig with edge definitions.
    """

    edges: list[Edge] = field(default_factory=list)
    entry_point: str | None = None  # First agent to execute
    default_routing: str = "sequential"  # sequential, round_robin, or manual


class CustomTopology(BaseTopology):
    """Custom topology with explicit agent relationships.

    Allows defining arbitrary connections between agents,
    making it suitable for:
    - Complex coordination patterns not covered by standard topologies
    - Hybrid patterns (e.g., supervisor with some mesh connections)
    - Accurate Mermaid diagram generation

    Example:
        >>> edges = [
        ...     Edge("Gateway", "Validator"),
        ...     Edge("Validator", "Processor", condition="valid"),
        ...     Edge("Validator", "Gateway", condition="invalid"),
        ...     Edge("Processor", "Storage"),
        ... ]
        >>> topology = CustomTopology(
        ...     config=CustomTopologyConfig(name="pipeline", edges=edges),
        ...     agents=[gateway, validator, processor, storage],
        ... )
    """

    def __init__(
        self,
        config: TopologyConfig | CustomTopologyConfig,
        agents: Sequence[Agent],
        edges: list[Edge] | None = None,
        entry_point: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize custom topology.

        Args:
            config: Topology configuration.
            agents: List of agents in the topology.
            edges: List of edges defining connections.
                Can also be passed via CustomTopologyConfig.
            entry_point: Name of first agent to execute.
            **kwargs: Additional arguments for BaseTopology.
        """
        super().__init__(config=config, agents=agents, **kwargs)

        # Get edges from config or parameter
        if isinstance(config, CustomTopologyConfig):
            self._edges = config.edges if edges is None else edges
            self._entry_point = config.entry_point if entry_point is None else entry_point
        else:
            self._edges = edges or []
            self._entry_point = entry_point

        # If no entry point specified, use first agent
        if self._entry_point is None and agents:
            self._entry_point = agents[0].config.name

        # Build adjacency list for routing
        self._adjacency: dict[str, list[str]] = {}
        for edge in self._edges:
            if edge.source not in self._adjacency:
                self._adjacency[edge.source] = []
            self._adjacency[edge.source].append(edge.target)

            if edge.bidirectional:
                if edge.target not in self._adjacency:
                    self._adjacency[edge.target] = []
                self._adjacency[edge.target].append(edge.source)

    @property
    def edges(self) -> list[Edge]:
        """Get the edges in this topology."""
        return self._edges

    @property
    def entry_point(self) -> str | None:
        """Get the entry point agent name."""
        return self._entry_point

    def _build_graph(self) -> None:
        """Build the LangGraph state graph from edges."""
        from langgraph.graph import END, START

        # Add all agent nodes
        for agent in self._agents:
            self._graph.add_node(agent.config.name, self._create_agent_node(agent))

        # Set entry point
        if self._entry_point and self._entry_point in self.agents:
            self._graph.add_edge(START, self._entry_point)

        # Add edges based on configuration
        for agent_name in self.agents:
            targets = self._adjacency.get(agent_name, [])
            if targets:
                # Use conditional routing if multiple targets
                if len(targets) == 1:
                    self._graph.add_edge(agent_name, targets[0])
                else:
                    self._graph.add_conditional_edges(
                        agent_name,
                        self._route,
                        {t: t for t in targets} | {"__end__": END},
                    )
            else:
                # No outgoing edges, go to END
                self._graph.add_edge(agent_name, END)

    def _route(self, state: TopologyState) -> str:
        """Route to next agent based on state.

        Default implementation uses round-robin or first available.
        Override for custom routing logic.
        """
        current = state.current_agent
        if not current:
            return "__end__"

        targets = self._adjacency.get(current, [])
        if not targets:
            return "__end__"

        # Simple: return first target
        # Override this method for complex routing
        return targets[0]

    def add_edge(
        self,
        source: str,
        target: str,
        label: str = "",
        condition: str = "",
        bidirectional: bool = False,
    ) -> "CustomTopology":
        """Add an edge to the topology.

        Args:
            source: Source agent name.
            target: Target agent name.
            label: Optional edge label.
            condition: Optional condition description.
            bidirectional: Whether edge goes both ways.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If source or target agent doesn't exist.
        """
        if source not in self.agents:
            raise ValueError(f"Source agent '{source}' not found")
        if target not in self.agents:
            raise ValueError(f"Target agent '{target}' not found")

        edge = Edge(source, target, label, condition, bidirectional)
        self._edges.append(edge)

        # Update adjacency
        if source not in self._adjacency:
            self._adjacency[source] = []
        self._adjacency[source].append(target)

        if bidirectional:
            if target not in self._adjacency:
                self._adjacency[target] = []
            self._adjacency[target].append(source)

        return self

    def get_edges_from(self, agent_name: str) -> list[Edge]:
        """Get all edges originating from an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            List of edges from this agent.
        """
        return [e for e in self._edges if e.source == agent_name]

    def get_edges_to(self, agent_name: str) -> list[Edge]:
        """Get all edges pointing to an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            List of edges to this agent.
        """
        return [e for e in self._edges if e.target == agent_name]

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        agents: Sequence[Agent],
        **kwargs: Any,
    ) -> "CustomTopology":
        """Create topology from dictionary configuration.

        Args:
            config: Dictionary with 'name', 'edges', etc.
            agents: List of agents.
            **kwargs: Additional arguments.

        Returns:
            CustomTopology instance.

        Example:
            >>> config = {
            ...     "name": "my-flow",
            ...     "edges": [
            ...         {"source": "A", "target": "B"},
            ...         {"source": "B", "target": "C", "label": "success"},
            ...     ],
            ...     "entry_point": "A",
            ... }
            >>> topology = CustomTopology.from_dict(config, agents)
        """
        edges = [
            Edge(
                source=e["source"],
                target=e["target"],
                label=e.get("label", ""),
                condition=e.get("condition", ""),
                bidirectional=e.get("bidirectional", False),
            )
            for e in config.get("edges", [])
        ]

        topology_config = CustomTopologyConfig(
            name=config.get("name", "custom"),
            description=config.get("description", ""),
            edges=edges,
            entry_point=config.get("entry_point"),
        )

        return cls(config=topology_config, agents=agents, **kwargs)
