"""Graph runner for executing LangGraph workflows.

Provides a unified interface for running graphs with
streaming, checkpointing, and observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, TYPE_CHECKING

from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agenticflow.core import generate_id

if TYPE_CHECKING:
    from agenticflow.events import EventBus
    from agenticflow.observability import Tracer


class StreamMode(Enum):
    """Streaming modes for graph execution."""

    VALUES = "values"  # Stream full state after each step
    UPDATES = "updates"  # Stream only state updates
    DEBUG = "debug"  # Stream detailed debug info


@dataclass
class RunConfig:
    """Configuration for graph execution.

    Attributes:
        thread_id: Thread ID for checkpointing.
        recursion_limit: Max recursion depth.
        stream_mode: How to stream results.
        timeout_seconds: Execution timeout.
        metadata: Additional run metadata.
    """

    thread_id: str | None = None
    recursion_limit: int = 50
    stream_mode: StreamMode = StreamMode.VALUES
    timeout_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a graph execution.

    Attributes:
        final_state: Final state after execution.
        thread_id: Thread ID used.
        steps: Number of steps executed.
        interrupted: Whether execution was interrupted.
        error: Error message if failed.
    """

    final_state: dict[str, Any]
    thread_id: str
    steps: int = 0
    interrupted: bool = False
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if run was successful."""
        return self.error is None and not self.interrupted


class GraphRunner:
    """Runner for executing LangGraph workflows.

    Provides a high-level interface for running graphs
    with streaming, checkpointing, and event integration.

    Example:
        >>> runner = GraphRunner(graph)
        >>> result = await runner.run(
        ...     {"task": "Write a story"},
        ...     config=RunConfig(thread_id="story-1"),
        ... )
        >>> print(result.final_state)
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        event_bus: "EventBus | None" = None,
        tracer: "Tracer | None" = None,
    ) -> None:
        """Initialize runner.

        Args:
            graph: Compiled LangGraph to run.
            event_bus: Optional event bus for events.
            tracer: Optional tracer for observability.
        """
        self.graph = graph
        self.event_bus = event_bus
        self.tracer = tracer

    async def run(
        self,
        initial_state: dict[str, Any],
        config: RunConfig | None = None,
    ) -> RunResult:
        """Run the graph to completion.

        Args:
            initial_state: Initial state for execution.
            config: Run configuration.

        Returns:
            Execution result.
        """
        config = config or RunConfig()
        thread_id = config.thread_id or generate_id("thread")

        # Publish start event
        if self.event_bus:
            await self.event_bus.publish(
                "graph.run.start",
                {
                    "thread_id": thread_id,
                    "initial_state": initial_state,
                },
            )

        run_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": config.recursion_limit,
        }

        final_state = None
        steps = 0
        error = None
        interrupted = False

        try:
            async for state in self.graph.astream(
                initial_state,
                run_config,
                stream_mode=config.stream_mode.value,
            ):
                final_state = state
                steps += 1

                # Publish step event
                if self.event_bus:
                    await self.event_bus.publish(
                        "graph.run.step",
                        {
                            "thread_id": thread_id,
                            "step": steps,
                            "state": state,
                        },
                    )

        except Exception as e:
            error = str(e)
            if "interrupt" in error.lower():
                interrupted = True
                error = None

        # Publish completion event
        if self.event_bus:
            await self.event_bus.publish(
                "graph.run.complete",
                {
                    "thread_id": thread_id,
                    "steps": steps,
                    "success": error is None,
                    "interrupted": interrupted,
                },
            )

        return RunResult(
            final_state=final_state or initial_state,
            thread_id=thread_id,
            steps=steps,
            interrupted=interrupted,
            error=error,
        )

    async def stream(
        self,
        initial_state: dict[str, Any],
        config: RunConfig | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream graph execution step by step.

        Args:
            initial_state: Initial state.
            config: Run configuration.

        Yields:
            State after each step.
        """
        config = config or RunConfig()
        thread_id = config.thread_id or generate_id("thread")

        run_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": config.recursion_limit,
        }

        async for state in self.graph.astream(
            initial_state,
            run_config,
            stream_mode=config.stream_mode.value,
        ):
            yield state

    async def resume(
        self,
        thread_id: str,
        human_input: Any,
        config: RunConfig | None = None,
    ) -> RunResult:
        """Resume execution after interrupt.

        Args:
            thread_id: Thread ID to resume.
            human_input: Human's response.
            config: Run configuration.

        Returns:
            Execution result.
        """
        config = config or RunConfig()

        # Publish resume event
        if self.event_bus:
            await self.event_bus.publish(
                "graph.run.resume",
                {
                    "thread_id": thread_id,
                    "human_input": human_input,
                },
            )

        run_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": config.recursion_limit,
        }

        # Resume with Command
        command = Command(resume=human_input)

        final_state = None
        steps = 0
        error = None
        interrupted = False

        try:
            async for state in self.graph.astream(
                command,
                run_config,
                stream_mode=config.stream_mode.value,
            ):
                final_state = state
                steps += 1

        except Exception as e:
            error = str(e)
            if "interrupt" in error.lower():
                interrupted = True
                error = None

        return RunResult(
            final_state=final_state or {},
            thread_id=thread_id,
            steps=steps,
            interrupted=interrupted,
            error=error,
        )

    async def get_state(self, thread_id: str) -> dict[str, Any] | None:
        """Get current state for a thread.

        Args:
            thread_id: Thread ID.

        Returns:
            Current state or None.
        """
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state = await self.graph.aget_state(config)
            return state.values if state else None
        except Exception:
            return None

    async def update_state(
        self,
        thread_id: str,
        updates: dict[str, Any],
        as_node: str | None = None,
    ) -> None:
        """Update state for a thread.

        Args:
            thread_id: Thread ID.
            updates: State updates to apply.
            as_node: Node to attribute update to.
        """
        config = {"configurable": {"thread_id": thread_id}}
        await self.graph.aupdate_state(config, updates, as_node=as_node)

    def get_history(self, thread_id: str) -> list[dict[str, Any]]:
        """Get execution history for a thread.

        Args:
            thread_id: Thread ID.

        Returns:
            List of historical states.
        """
        config = {"configurable": {"thread_id": thread_id}}

        history = []
        for state in self.graph.get_state_history(config):
            history.append({
                "values": state.values,
                "next": state.next,
                "created_at": state.created_at,
            })
        return history
