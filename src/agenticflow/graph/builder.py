"""Graph builder for LangGraph workflows.

Provides a fluent API for constructing agent graphs
with proper node and edge configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence, TYPE_CHECKING

from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from agenticflow.graph.state import AgentGraphState

if TYPE_CHECKING:
    from agenticflow.agents import Agent


class EdgeType(Enum):
    """Types of edges in the graph."""

    DIRECT = "direct"  # Always go to target
    CONDITIONAL = "conditional"  # Route based on state
    PARALLEL = "parallel"  # Fan out to multiple


@dataclass
class NodeConfig:
    """Configuration for a graph node.

    Attributes:
        name: Node name.
        func: Node function or callable.
        metadata: Additional node metadata.
    """

    name: str
    func: Callable[[dict[str, Any]], dict[str, Any] | Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeConfig:
    """Configuration for a graph edge.

    Attributes:
        source: Source node name.
        target: Target node name (for direct edges).
        edge_type: Type of edge.
        condition: Routing function (for conditional).
        targets: Map of condition results to targets.
    """

    source: str
    target: str | None = None
    edge_type: EdgeType = EdgeType.DIRECT
    condition: Callable[[dict[str, Any]], str] | None = None
    targets: dict[str, str] | None = None


class GraphBuilder:
    """Fluent builder for LangGraph StateGraphs.

    Simplifies graph construction with a chainable API.

    Example:
        >>> builder = GraphBuilder("my-graph")
        >>> builder.add_agent_node("researcher", researcher_agent)
        >>> builder.add_agent_node("writer", writer_agent)
        >>> builder.add_edge("researcher", "writer")
        >>> builder.add_edge_to_end("writer")
        >>> builder.set_entry("researcher")
        >>> graph = builder.compile()
    """

    def __init__(
        self,
        name: str,
        state_class: type = AgentGraphState,
    ) -> None:
        """Initialize graph builder.

        Args:
            name: Graph name.
            state_class: State schema class.
        """
        self.name = name
        self.state_class = state_class
        self._nodes: dict[str, NodeConfig] = {}
        self._edges: list[EdgeConfig] = []
        self._entry_point: str | None = None
        self._checkpointer: BaseCheckpointSaver | None = None

    def add_node(
        self,
        name: str,
        func: Callable[[dict[str, Any]], dict[str, Any] | Any],
        **metadata: Any,
    ) -> "GraphBuilder":
        """Add a node to the graph.

        Args:
            name: Node name.
            func: Node function.
            **metadata: Additional metadata.

        Returns:
            Self for chaining.
        """
        self._nodes[name] = NodeConfig(
            name=name,
            func=func,
            metadata=metadata,
        )
        return self

    def add_agent_node(
        self,
        name: str,
        agent: "Agent",
        **metadata: Any,
    ) -> "GraphBuilder":
        """Add an agent as a node.

        Creates a node function that invokes the agent.

        Args:
            name: Node name.
            agent: Agent instance.
            **metadata: Additional metadata.

        Returns:
            Self for chaining.
        """

        async def agent_node(state: dict[str, Any]) -> dict[str, Any]:
            """Process state through agent."""
            task = state.get("task", "")
            context = state.get("context", {})

            # Agent thinks
            thought = await agent.think(task, context)

            # Return state update
            from langchain_core.messages import AIMessage

            return {
                "messages": [AIMessage(content=thought)],
                "current_agent": name,
                "results": [{"agent": name, "thought": thought}],
                "iteration": 1,
            }

        return self.add_node(name, agent_node, agent=agent.config.name, **metadata)

    def add_edge(
        self,
        source: str,
        target: str,
    ) -> "GraphBuilder":
        """Add a direct edge between nodes.

        Args:
            source: Source node.
            target: Target node.

        Returns:
            Self for chaining.
        """
        self._edges.append(EdgeConfig(
            source=source,
            target=target,
            edge_type=EdgeType.DIRECT,
        ))
        return self

    def add_conditional_edge(
        self,
        source: str,
        condition: Callable[[dict[str, Any]], str],
        targets: dict[str, str],
    ) -> "GraphBuilder":
        """Add a conditional edge with routing.

        Args:
            source: Source node.
            condition: Function that returns target key.
            targets: Map of condition results to node names.

        Returns:
            Self for chaining.
        """
        self._edges.append(EdgeConfig(
            source=source,
            edge_type=EdgeType.CONDITIONAL,
            condition=condition,
            targets=targets,
        ))
        return self

    def add_edge_to_end(self, source: str) -> "GraphBuilder":
        """Add edge from node to END.

        Args:
            source: Source node.

        Returns:
            Self for chaining.
        """
        self._edges.append(EdgeConfig(
            source=source,
            target="__end__",
            edge_type=EdgeType.DIRECT,
        ))
        return self

    def set_entry(self, node: str) -> "GraphBuilder":
        """Set the entry point node.

        Args:
            node: Entry node name.

        Returns:
            Self for chaining.
        """
        self._entry_point = node
        return self

    def with_checkpointer(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> "GraphBuilder":
        """Configure checkpointing.

        Args:
            checkpointer: Checkpointer instance. If None, uses MemorySaver.

        Returns:
            Self for chaining.
        """
        self._checkpointer = checkpointer or MemorySaver()
        return self

    def compile(self) -> CompiledStateGraph:
        """Compile the graph.

        Returns:
            Compiled LangGraph StateGraph.

        Raises:
            ValueError: If graph is invalid.
        """
        if not self._nodes:
            raise ValueError("Graph has no nodes")

        if not self._entry_point:
            # Use first node as entry
            self._entry_point = next(iter(self._nodes))

        # Build StateGraph
        builder = StateGraph(dict)

        # Add nodes
        for name, config in self._nodes.items():
            builder.add_node(name, config.func)

        # Add edges
        for edge in self._edges:
            if edge.edge_type == EdgeType.DIRECT:
                if edge.target == "__end__":
                    builder.add_edge(edge.source, END)
                else:
                    builder.add_edge(edge.source, edge.target)
            elif edge.edge_type == EdgeType.CONDITIONAL:
                if edge.condition and edge.targets:
                    # Convert targets to include END
                    targets = {}
                    for key, target in edge.targets.items():
                        if target == "__end__":
                            targets[key] = END
                        else:
                            targets[key] = target
                    builder.add_conditional_edges(
                        edge.source,
                        edge.condition,
                        targets,
                    )

        # Set entry point
        builder.set_entry_point(self._entry_point)

        # Compile with checkpointer
        return builder.compile(checkpointer=self._checkpointer)

    def visualize(self) -> str:
        """Generate DOT visualization of the graph.

        Returns:
            DOT format string.
        """
        lines = ["digraph G {"]
        lines.append("  rankdir=TB;")

        # Add nodes
        for name in self._nodes:
            lines.append(f'  "{name}" [shape=box];')

        # Add edges
        for edge in self._edges:
            if edge.edge_type == EdgeType.DIRECT:
                target = edge.target or "END"
                lines.append(f'  "{edge.source}" -> "{target}";')
            elif edge.edge_type == EdgeType.CONDITIONAL:
                if edge.targets:
                    for label, target in edge.targets.items():
                        lines.append(
                            f'  "{edge.source}" -> "{target}" [label="{label}"];'
                        )

        lines.append("}")
        return "\n".join(lines)
