"""Core topology primitives - simple, clean, native Python.

This module provides the foundation for multi-agent coordination patterns.
No external dependencies - just pure async Python.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agenticflow.flow.delegation import DelegationMixin

if TYPE_CHECKING:
    from ..agent import Agent
    from ..memory import TeamMemory
    from ..graph import GraphView


class TopologyType(str, Enum):
    """Available coordination patterns."""

    SUPERVISOR = "supervisor"  # One coordinator, multiple workers
    PIPELINE = "pipeline"  # Sequential: A → B → C
    MESH = "mesh"  # All-to-all collaboration
    HIERARCHICAL = "hierarchical"  # Tree structure with levels


@dataclass
class AgentConfig:
    """Configuration for an agent in a topology.
    
    Args:
        agent: The agent instance
        name: Optional custom name (defaults to agent.name)
        role: Role description for this agent in the topology
        can_delegate: Delegation policy - list of agent names this agent can delegate to
        can_reply: Whether agent can reply to delegated requests
        metadata: Additional metadata
    """

    agent: Agent
    name: str | None = None
    role: str | None = None
    can_delegate: list[str] | bool | None = None
    can_reply: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = getattr(self.agent, "name", "agent")
        
        # Legacy support: can_delegate_to → can_delegate
        if hasattr(self, 'can_delegate_to') and not self.can_delegate:
            self.can_delegate = getattr(self, 'can_delegate_to')


@dataclass
class TopologyConfig:
    """Configuration for a topology."""

    name: str = "Topology"
    """Display name for the topology."""

    description: str = ""
    """Description of what the topology does."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional configuration metadata."""


@dataclass
class TopologyResult:
    """Result from topology execution."""

    output: str
    """Final synthesized output."""

    agent_outputs: dict[str, str] = field(default_factory=dict)
    """Individual outputs from each agent."""

    execution_order: list[str] = field(default_factory=list)
    """Order in which agents were invoked."""

    rounds: int = 1
    """Number of coordination rounds (for mesh/iterative patterns)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional execution metadata."""

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return bool(self.output)


class BaseTopology(ABC, DelegationMixin):
    """Base class for all coordination patterns.

    A topology defines how multiple agents work together to accomplish a task.
    Each topology implements a different coordination strategy.
    
    Inherits from DelegationMixin to provide A2A delegation capabilities
    across all topology types.

    Example:
        >>> from agenticflow import Agent, ChatModel
        >>> from agenticflow.topologies import Supervisor, AgentConfig
        >>>
        >>> model = ChatModel(provider="openai", model="gpt-4o-mini")
        >>> researcher = Agent(name="researcher", model=model)
        >>> writer = Agent(name="writer", model=model)
        >>>
        >>> topology = Supervisor(
        ...     coordinator=AgentConfig(
        ...         agent=researcher,
        ...         role="coordinator",
        ...         can_delegate=["writer"]  # Enable delegation
        ...     ),
        ...     workers=[AgentConfig(
        ...         agent=writer,
        ...         role="content writer",
        ...         can_reply=True  # Can respond to delegated tasks
        ...     )]
        ... )
        >>> result = await topology.run("Write a blog post about AI")
    """

    topology_type: TopologyType
    
    def __post_init__(self) -> None:
        """Configure delegation for all agents in the topology."""
        # Apply delegation configuration to all agents
        for agent_config in self.get_agents():
            if agent_config.can_delegate or agent_config.can_reply:
                self.configure_delegation(
                    agent_config.agent,
                    can_delegate=agent_config.can_delegate,
                    can_reply=agent_config.can_reply,
                )

    @abstractmethod
    async def run(
        self,
        task: str,
        *,
        team_memory: TeamMemory | None = None,
    ) -> TopologyResult:
        """Execute the topology with the given task.

        Args:
            task: The task or query to process.
            team_memory: Optional TeamMemory for agents to share state.

        Returns:
            TopologyResult with outputs from all agents and final synthesis.
        """

    @abstractmethod
    def get_agents(self) -> list[AgentConfig]:
        """Get all agents in this topology."""

    async def stream(
        self,
        task: str,
        *,
        team_memory: TeamMemory | None = None,
    ):
        """Stream execution updates as they happen.

        Yields status updates and partial results during execution.
        Default implementation just yields the final result.

        Args:
            task: The task to process.
            team_memory: Optional TeamMemory for agents to share state.

        Yields:
            Dict with 'type' (status/output/error) and 'data'.
        """
        yield {"type": "status", "data": f"Starting {self.topology_type.value} execution"}
        result = await self.run(task, team_memory=team_memory)
        yield {"type": "output", "data": result.output}
        yield {"type": "complete", "data": result}

    # ─────────────────────────────────────────────────────────────────────
    # Visualization (Graph API)
    # ─────────────────────────────────────────────────────────────────────

    def get_agents_dict(self) -> dict[str, "Agent"]:
        """Get agents as a dict by name for visualization.
        
        Used by graph builders to access agents by name.
        Concrete topologies may override if they have specific structure.
        """
        return {cfg.name: cfg.agent for cfg in self.get_agents() if cfg.name}

    @property
    def config(self) -> TopologyConfig:
        """Get topology configuration. Override in subclasses."""
        return TopologyConfig(name=f"{self.topology_type.value.title()} Topology")

    def graph(
        self,
        *,
        show_tools: bool = True,
    ) -> "GraphView":
        """Get a graph visualization of this topology.

        Returns a GraphView that provides a unified interface for
        rendering to Mermaid, Graphviz, or ASCII formats.

        Args:
            show_tools: Whether to show agent tools in the diagram.

        Returns:
            GraphView instance for rendering.

        Example:
            >>> # Get Mermaid code
            >>> print(topology.graph().mermaid())

            >>> # Get ASCII for terminal
            >>> print(topology.graph().ascii())

            >>> # Get Graphviz DOT
            >>> print(topology.graph().dot())

            >>> # Save as PNG
            >>> topology.graph().save("topology.png")
        """
        from agenticflow.graph import GraphView

        return GraphView.from_topology(self, show_tools=show_tools)

    def _repr_html_(self) -> str:
        """IPython/Jupyter HTML representation."""
        return self.graph().html()

