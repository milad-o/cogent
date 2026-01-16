"""Streaming support for ReactiveFlow.

This module extends ReactiveFlow with real-time streaming capabilities,
allowing agents to yield tokens as they process events rather than
returning complete responses.

Leverages existing agent streaming infrastructure (agent.run(stream=True))
and wraps it with reactive flow context (event names, agent names, etc.).

Benefits:
- Real-time feedback during long-running agent operations
- Better UX with progressive output display
- Lower perceived latency
- Ability to cancel in-flight operations

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.flow.reactive import ReactiveFlow
    from agenticflow.flow.triggers import react_to
    from agenticflow.models import ChatModel

    # Create agents with streaming-capable models
    researcher = Agent(
        name="researcher",
        model=ChatModel(model="gpt-4o"),
        system_prompt="You research topics thoroughly.",
    )

    writer = Agent(
        name="writer",
        model=ChatModel(model="gpt-4o"),
        system_prompt="You write engaging content.",
    )

    # Create reactive flow
    flow = ReactiveFlow()
    flow.register(researcher, [react_to("task.created")])
    flow.register(writer, [react_to("researcher.completed")])

    # Stream execution - tokens arrive in real-time
    async for chunk in flow.run_streaming("Research quantum computing"):
        print(f"[{chunk.agent_name}] {chunk.content}", end="", flush=True)

        if chunk.is_final:
            print()  # Newline after agent completes
    ```
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agenticflow.agent.streaming import StreamChunk as AgentStreamChunk


@dataclass
class ReactiveStreamChunk:
    """
    A streaming chunk from a reactive agent execution.

    Extends the agent's StreamChunk with reactive-flow context like
    which agent is streaming, what event triggered it, and whether
    this is the final chunk from this reaction.

    Attributes:
        agent_name: Name of the agent currently streaming.
        event_id: ID of the event that triggered this agent.
        event_name: Name/type of the triggering event.
        content: The streamed text content.
        delta: The incremental text added (same as content for compatibility).
        is_final: True if this is the last chunk from this reaction.
        metadata: Additional context (round number, total events, etc.).
        finish_reason: Why streaming stopped (if is_final=True).
    """

    agent_name: str
    """Name of the agent generating this chunk."""

    event_id: str
    """ID of the event that triggered this agent."""

    event_name: str
    """Name/type of the event that triggered this agent."""

    content: str
    """The text content of this streaming chunk."""

    delta: str
    """Incremental text added (same as content)."""

    is_final: bool = False
    """True if this is the last chunk from this reaction."""

    metadata: dict[str, Any] | None = None
    """Additional context about the streaming execution."""

    finish_reason: str | None = None
    """Reason streaming stopped (stop, length, tool_calls, etc.)."""

    @classmethod
    def from_agent_chunk(
        cls,
        chunk: AgentStreamChunk,
        agent_name: str,
        event_id: str,
        event_name: str,
        **metadata: Any,
    ) -> ReactiveStreamChunk:
        """Create ReactiveStreamChunk from agent's StreamChunk."""
        return cls(
            agent_name=agent_name,
            event_id=event_id,
            event_name=event_name,
            content=chunk.content,
            delta=chunk.content,
            is_final=chunk.finish_reason is not None,
            finish_reason=chunk.finish_reason,
            metadata=metadata or {},
        )


# Backward compatibility - some code may expect "StreamChunk"
StreamChunk = ReactiveStreamChunk


async def _stream_agent_execution(
    agent: Any,
    task: str,
    event: Any,
    agent_name: str,
    event_id: str,
    event_name: str,
    **context: Any,
) -> AsyncIterator[ReactiveStreamChunk]:
    """
    Execute agent with streaming enabled and yield ReactiveStreamChunks.

    This is an internal helper that wraps agent execution, enabling
    streaming mode and converting agent StreamChunks to ReactiveStreamChunks
    with full event context.

    Args:
        agent: The agent to execute.
        task: The task/prompt for the agent.
        event: The triggering event.
        agent_name: Name of the agent (for chunk metadata).
        event_id: ID of triggering event (for chunk metadata).
        event_name: Name of triggering event (for chunk metadata).
        **context: Additional context to pass to agent.

    Yields:
        ReactiveStreamChunk: Streaming chunks with reactive context.
    """
    # Import here to avoid circular dependency
    from agenticflow.agent.streaming import StreamChunk as AgentStreamChunk

    # Build task with event context
    if hasattr(event, "data") and isinstance(event.data, dict):
        event_data = event.data
    else:
        event_data = {}

    # Merge event data with additional context
    full_context = {**event_data, **context}

    # Stream agent execution
    async for agent_chunk in agent.think(task, stream=True, **full_context):
        if isinstance(agent_chunk, AgentStreamChunk):
            yield ReactiveStreamChunk.from_agent_chunk(
                chunk=agent_chunk,
                agent_name=agent_name,
                event_id=event_id,
                event_name=event_name,
            )
        else:
            # Fallback for non-StreamChunk responses (shouldn't happen in stream mode)
            content = str(agent_chunk) if agent_chunk else ""
            yield ReactiveStreamChunk(
                agent_name=agent_name,
                event_id=event_id,
                event_name=event_name,
                content=content,
                delta=content,
                is_final=True,
                finish_reason="stop",
            )
