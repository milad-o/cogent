"""Core topology primitives - simple, clean, native Python.

This module provides the foundation for multi-agent coordination patterns.
No external dependencies - just pure async Python.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agent import Agent
    from ..memory import TeamMemory


class TopologyType(str, Enum):
    """Available coordination patterns."""

    SUPERVISOR = "supervisor"  # One coordinator, multiple workers
    PIPELINE = "pipeline"  # Sequential: A → B → C
    MESH = "mesh"  # All-to-all collaboration
    HIERARCHICAL = "hierarchical"  # Tree structure with levels


@dataclass
class AgentConfig:
    """Configuration for an agent in a topology."""

    agent: Agent
    name: str | None = None
    role: str | None = None
    can_delegate_to: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = getattr(self.agent, "name", "agent")


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


class BaseTopology(ABC):
    """Base class for all coordination patterns.

    A topology defines how multiple agents work together to accomplish a task.
    Each topology implements a different coordination strategy.

    Example:
        >>> from agenticflow import Agent, ChatModel
        >>> from agenticflow.topologies import Supervisor, AgentConfig
        >>>
        >>> model = ChatModel(provider="openai", model="gpt-4o-mini")
        >>> researcher = Agent(name="researcher", model=model)
        >>> writer = Agent(name="writer", model=model)
        >>>
        >>> topology = Supervisor(
        ...     coordinator=AgentConfig(agent=researcher, role="coordinator"),
        ...     workers=[AgentConfig(agent=writer, role="content writer")]
        ... )
        >>> result = await topology.run("Write a blog post about AI")
    """

    topology_type: TopologyType

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
