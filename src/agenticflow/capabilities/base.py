"""
Base capability class for composable agent tools.

A Capability is a pluggable module that adds tools to any agent.
It encapsulates related functionality (memory, search, execution)
and exposes it through native tools.

Example:
    ```python
    class MyCapability(BaseCapability):
        @property
        def name(self) -> str:
            return "my_capability"

        @property
        def tools(self) -> list[BaseTool]:
            return [self._my_tool()]

        def _my_tool(self):
            @tool
            def do_something(x: str) -> str:
                '''Do something useful.'''
                return f"Did: {x}"
            return do_something
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agenticflow.tools.base import BaseTool

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent


class BaseCapability(ABC):
    """
    Base class for agent capabilities.

    A capability is a self-contained module that:
    1. Provides tools for the agent to use
    2. May maintain internal state (graphs, caches, connections)
    3. Can be initialized/shutdown with the agent lifecycle

    Capabilities are composable - an agent can have multiple
    capabilities, and each capability's tools are automatically
    registered with the agent.

    Attributes:
        agent: The agent this capability is attached to (set on init)
    """

    _agent: Agent | None = None

    @property
    def agent(self) -> Agent | None:
        """The agent this capability is attached to."""
        return self._agent

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name for this capability.

        Used for logging, debugging, and capability lookup.
        Should be lowercase with underscores (e.g., "knowledge_graph").
        """
        ...

    @property
    def description(self) -> str:
        """
        Human-readable description of what this capability does.

        Override to provide a custom description.
        """
        return f"{self.name} capability"

    @property
    @abstractmethod
    def tools(self) -> list[BaseTool]:
        """
        Tools this capability provides to the agent.

        These tools are automatically registered when the capability
        is attached to an agent. Tools should be created fresh each
        time (not cached) to ensure proper closure over capability state.

        Returns:
            List of BaseTool instances.
        """
        ...

    async def initialize(self, agent: Agent) -> None:
        """
        Called when capability is attached to an agent.

        Override to perform setup like:
        - Database connections
        - Loading cached state
        - Subscribing to events

        Args:
            agent: The agent this capability is being attached to.
        """
        self._agent = agent

    async def shutdown(self) -> None:
        """
        Called when capability is detached or agent shuts down.

        Override to perform cleanup like:
        - Closing connections
        - Persisting state
        - Unsubscribing from events
        """
        self._agent = None

    def to_dict(self) -> dict[str, Any]:
        """Convert capability info to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "tool_count": len(self.tools),
            "tools": [t.name for t in self.tools],
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, tools={len(self.tools)})"
