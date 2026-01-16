"""Agent-to-Agent (A2A) communication for ReactiveFlow.

This module provides direct agent-to-agent communication with request/response
semantics, enabling agents to delegate tasks and receive replies.

Example:
    ```python
    from agenticflow import Agent, ReactiveFlow
    from agenticflow.reactive import react_to

    # Delegating agent
    @react_to("user.request")
    async def coordinator(event, context):
        # Delegate to specialist agent
        result = await context.delegate_to(
            agent_name="specialist",
            task="analyze data",
            data={"dataset": event.data["file"]},
            wait=True  # Wait for response
        )
        return f"Analysis complete: {result}"

    # Specialist agent that handles delegated tasks
    @react_to("agent.request", condition=lambda e: e.data.get("to_agent") == "specialist")
    async def specialist(event, context):
        task = event.data["task"]
        # Process task...
        return {"result": "analysis done"}
    ```
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

from agenticflow.events import Event


@dataclass
class AgentRequest:
    """Request from one agent to another.

    Attributes:
        from_agent: Name of the requesting agent
        to_agent: Name of the target agent
        task: Description of the task to perform
        data: Additional data for the task
        correlation_id: Unique ID to correlate request and response
        reply_to: Event name to emit with the response
        timeout_ms: Maximum time to wait for response (milliseconds)
        metadata: Additional metadata for the request
    """

    from_agent: str
    to_agent: str
    task: str
    data: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    reply_to: str = field(default="agent.response")
    timeout_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_event(self) -> Event:
        """Convert request to an event for the target agent."""
        return Event(
            name="agent.request",
            data={
                "from_agent": self.from_agent,
                "to_agent": self.to_agent,
                "task": self.task,
                "data": self.data,
                "correlation_id": self.correlation_id,
                "reply_to": self.reply_to,
                "timeout_ms": self.timeout_ms,
                **self.metadata,
            },
            correlation_id=self.correlation_id,
        )


@dataclass
class AgentResponse:
    """Response from one agent to another.

    Attributes:
        from_agent: Name of the responding agent
        to_agent: Name of the original requester
        result: The response data
        correlation_id: ID linking response to original request
        success: Whether the task completed successfully
        error: Error message if task failed
        metadata: Additional metadata for the response
    """

    from_agent: str
    to_agent: str
    result: Any
    correlation_id: str
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_event(self, event_name: str = "agent.response") -> Event:
        """Convert response to an event for the requester."""
        return Event(
            name=event_name,
            data={
                "from_agent": self.from_agent,
                "to_agent": self.to_agent,
                "result": self.result,
                "correlation_id": self.correlation_id,
                "success": self.success,
                "error": self.error,
                **self.metadata,
            },
            correlation_id=self.correlation_id,
        )


def create_request(
    from_agent: str,
    to_agent: str,
    task: str,
    data: dict[str, Any] | None = None,
    **kwargs: Any,
) -> AgentRequest:
    """Create an agent request.

    Args:
        from_agent: Name of the requesting agent
        to_agent: Name of the target agent
        task: Description of the task
        data: Additional data for the task
        **kwargs: Additional arguments for AgentRequest

    Returns:
        AgentRequest object
    """
    return AgentRequest(
        from_agent=from_agent,
        to_agent=to_agent,
        task=task,
        data=data or {},
        **kwargs,
    )


def create_response(
    from_agent: str,
    to_agent: str,
    correlation_id: str,
    result: Any,
    success: bool = True,
    error: str | None = None,
    **kwargs: Any,
) -> AgentResponse:
    """Create an agent response.

    Args:
        from_agent: Name of the responding agent
        to_agent: Name of the original requester
        correlation_id: ID linking to original request
        result: The response data
        success: Whether task completed successfully
        error: Error message if task failed
        **kwargs: Additional arguments for AgentResponse

    Returns:
        AgentResponse object
    """
    return AgentResponse(
        from_agent=from_agent,
        to_agent=to_agent,
        correlation_id=correlation_id,
        result=result,
        success=success,
        error=error,
        **kwargs,
    )
