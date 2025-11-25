"""Base topology classes and interfaces.

Provides abstract base for multi-agent coordination patterns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
from enum import Enum

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command, interrupt

from agenticflow.agents import Agent
from agenticflow.events import EventBus
from agenticflow.memory import MemoryManager


class HandoffStrategy(Enum):
    """How agents hand off work to each other."""

    COMMAND = "command"  # Use LangGraph Command for explicit handoffs
    INTERRUPT = "interrupt"  # Use interrupt for human-in-the-loop
    AUTOMATIC = "automatic"  # Let the topology decide routing
    BROADCAST = "broadcast"  # Send to all agents


@dataclass
class TopologyConfig:
    """Configuration for a multi-agent topology."""

    name: str
    description: str = ""
    max_iterations: int = 100
    handoff_strategy: HandoffStrategy = HandoffStrategy.AUTOMATIC
    enable_memory: bool = True
    enable_checkpointing: bool = True
    recursion_limit: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyState:
    """Shared state across all agents in a topology.

    This is the state type for LangGraph's StateGraph.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    current_agent: str | None = None
    task: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    completed: bool = False
    error: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "messages": self.messages,
            "current_agent": self.current_agent,
            "task": self.task,
            "context": self.context,
            "iteration": self.iteration,
            "completed": self.completed,
            "error": self.error,
            "results": self.results,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopologyState":
        """Create state from dictionary."""
        return cls(
            messages=data.get("messages", []),
            current_agent=data.get("current_agent"),
            task=data.get("task", ""),
            context=data.get("context", {}),
            iteration=data.get("iteration", 0),
            completed=data.get("completed", False),
            error=data.get("error"),
            results=data.get("results", []),
        )


class BaseTopology(ABC):
    """Abstract base class for multi-agent topologies.

    Subclasses implement specific coordination patterns like
    supervisor, mesh, pipeline, or hierarchical.
    """

    def __init__(
        self,
        config: TopologyConfig,
        agents: Sequence[Agent],
        event_bus: EventBus | None = None,
        memory_manager: MemoryManager | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        """Initialize topology.

        Args:
            config: Topology configuration.
            agents: List of agents in the topology.
            event_bus: Optional event bus for publishing events.
            memory_manager: Optional memory manager for persistence.
            checkpointer: Optional checkpointer for state persistence.
        """
        self.config = config
        self.agents = {agent.config.name: agent for agent in agents}
        self.event_bus = event_bus
        self.memory_manager = memory_manager
        self.checkpointer = checkpointer
        self._graph: CompiledStateGraph | None = None

    @property
    def graph(self) -> CompiledStateGraph:
        """Get or build the compiled graph."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    @abstractmethod
    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph StateGraph for this topology.

        Returns:
            Compiled StateGraph ready for execution.
        """
        ...

    @abstractmethod
    def _route(self, state: dict[str, Any]) -> str | list[str]:
        """Determine next node(s) based on current state.

        Args:
            state: Current topology state.

        Returns:
            Name of next node or list of parallel nodes.
        """
        ...

    def _create_agent_node(self, agent: Agent) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create a graph node function for an agent.

        Args:
            agent: The agent to wrap.

        Returns:
            Node function that processes state and returns updates.
        """

        async def node(state: dict[str, Any]) -> dict[str, Any]:
            """Process state through agent."""
            # Publish event
            if self.event_bus:
                await self.event_bus.publish(
                    "topology.agent.start",
                    {
                        "topology": self.config.name,
                        "agent": agent.config.name,
                        "iteration": state.get("iteration", 0),
                    },
                )

            # Build messages for agent
            messages = state.get("messages", [])
            task = state.get("task", "")
            context = state.get("context", {})

            # Let agent think about the task
            thought = await agent.think(task, context)

            # Add thought to messages
            new_messages = messages + [
                {
                    "role": "assistant",
                    "content": thought,
                    "agent": agent.config.name,
                }
            ]

            # Determine result
            result = {
                "messages": new_messages,
                "current_agent": agent.config.name,
                "iteration": state.get("iteration", 0) + 1,
                "results": state.get("results", [])
                + [
                    {
                        "agent": agent.config.name,
                        "thought": thought,
                    }
                ],
            }

            # Publish completion event
            if self.event_bus:
                await self.event_bus.publish(
                    "topology.agent.complete",
                    {
                        "topology": self.config.name,
                        "agent": agent.config.name,
                        "thought_length": len(thought),
                    },
                )

            return result

        return node

    def handoff(
        self,
        target: str,
        state_update: dict[str, Any] | None = None,
        *,
        resume_value: Any = None,
    ) -> Command:
        """Create a handoff command to another agent.

        Uses LangGraph's Command for explicit routing.

        Args:
            target: Name of target agent or node.
            state_update: Optional state updates to apply.
            resume_value: Optional value to resume with after interrupt.

        Returns:
            LangGraph Command for handoff.
        """
        update = state_update or {}
        update["current_agent"] = target

        return Command(
            goto=target,
            update=update,
            resume=resume_value,
        )

    def request_human_input(
        self,
        question: str,
        state: dict[str, Any],
    ) -> Any:
        """Request human input using LangGraph interrupt.

        Args:
            question: Question to ask human.
            state: Current state for context.

        Returns:
            Human's response after resumption.
        """
        return interrupt(
            {
                "question": question,
                "context": state.get("context", {}),
                "current_agent": state.get("current_agent"),
            }
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> TopologyState:
        """Run the topology on a task.

        Args:
            task: The task to process.
            context: Optional initial context.
            thread_id: Optional thread ID for checkpointing.

        Returns:
            Final topology state.
        """
        from agenticflow.core import generate_id

        thread_id = thread_id or generate_id("thread")

        # Publish start event
        if self.event_bus:
            await self.event_bus.publish(
                "topology.run.start",
                {
                    "topology": self.config.name,
                    "task": task,
                    "thread_id": thread_id,
                },
            )

        # Initial state
        initial_state = {
            "messages": [],
            "current_agent": None,
            "task": task,
            "context": context or {},
            "iteration": 0,
            "completed": False,
            "error": None,
            "results": [],
        }

        # Run graph
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.config.recursion_limit,
        }

        final_state = None
        async for state in self.graph.astream(initial_state, config):
            final_state = state
            if self.event_bus:
                await self.event_bus.publish(
                    "topology.run.step",
                    {
                        "topology": self.config.name,
                        "state": state,
                    },
                )

        # Publish completion
        if self.event_bus:
            await self.event_bus.publish(
                "topology.run.complete",
                {
                    "topology": self.config.name,
                    "thread_id": thread_id,
                    "iterations": final_state.get("iteration", 0) if final_state else 0,
                },
            )

        return TopologyState.from_dict(final_state or initial_state)

    async def resume(
        self,
        thread_id: str,
        human_input: Any,
    ) -> TopologyState:
        """Resume execution after an interrupt.

        Args:
            thread_id: Thread ID to resume.
            human_input: Human's response to the interrupt.

        Returns:
            Final topology state after resumption.
        """
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.config.recursion_limit,
        }

        # Resume with Command
        command = Command(resume=human_input)

        final_state = None
        async for state in self.graph.astream(command, config):
            final_state = state

        return TopologyState.from_dict(final_state or {})
