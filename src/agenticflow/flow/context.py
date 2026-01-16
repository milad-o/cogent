"""Execution context for flow orchestration.

This module provides execution context for event-driven flows,
enabling agent-to-agent communication and shared state management.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agenticflow.events import Event

if TYPE_CHECKING:
    from agenticflow.flow.a2a import AgentResponse


@dataclass
class FlowContext:
    """Context passed to reactors during event processing.

    FlowContext provides access to flow state, event emission,
    and inter-reactor communication.

    Attributes:
        flow_id: Current flow execution ID
        event: The event being processed
        data: Shared context data dictionary
        history: List of previous events in the flow
        original_task: The original task that started the flow
        emit: Function to emit new events

    Example:
        ```python
        async def my_reactor(event: Event, ctx: FlowContext) -> Event | None:
            # Access shared data
            previous_result = ctx.data.get("last_result")

            # Process and store result
            result = process(event.data)
            ctx.data["last_result"] = result

            # Emit a new event
            await ctx.emit("processing.done", {"result": result})

            return Event.done(result)
        ```
    """

    flow_id: str
    event: Event
    data: dict[str, Any] = field(default_factory=dict)
    history: list[Event] = field(default_factory=list)
    original_task: str | None = None
    _emit_fn: Any = None  # Callable[[str, dict], Awaitable[None]]

    async def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Emit a new event into the flow.

        Args:
            event_type: Type of event to emit
            data: Event data dictionary
        """
        if self._emit_fn:
            await self._emit_fn(event_type, data or {})

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to context data."""
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like setting of context data."""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context data with default."""
        return self.data.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in context data."""
        return key in self.data


@dataclass
class ExecutionContext:
    """Execution context for agents with A2A communication.

    This context is passed to agents during flow execution
    and provides methods for agents to delegate tasks to other agents.

    .. note::
        For new code, prefer using ``FlowContext`` which provides
        a cleaner API for event emission. ExecutionContext is maintained
        for backward compatibility.

    Attributes:
        current_agent: Name of the currently executing agent
        event: The event that triggered this agent
        task: The original task/prompt
        data: Shared context data dictionary
        pending_responses: Dict tracking responses for correlation IDs
        event_queue: Queue for publishing new events
    """

    current_agent: str
    event: Event | None = None
    task: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    pending_responses: dict[str, asyncio.Future[AgentResponse]] = field(default_factory=dict)
    event_queue: asyncio.Queue[Event] | None = None

    async def delegate_to(
        self,
        agent_name: str,
        task: str,
        *,
        data: dict[str, Any] | None = None,
        wait: bool = True,
        timeout_ms: int | None = None,
    ) -> Any | None:
        """Delegate a task to another agent.

        Args:
            agent_name: Name of the agent to delegate to
            task: Description of the task to perform
            data: Additional data for the task
            wait: Whether to wait for a response
            timeout_ms: Maximum time to wait for response (milliseconds)

        Returns:
            Result from the delegated agent if wait=True, else None

        Example:
            ```python
            async def coordinator(event, context):
                # Delegate to specialist and wait for result
                result = await context.delegate_to(
                    agent_name="data_analyst",
                    task="analyze this dataset",
                    data={"file": event.data["path"]},
                    wait=True
                )
                return f"Analysis: {result}"
            ```
        """
        from agenticflow.flow.a2a import create_request

        # Create request
        request = create_request(
            from_agent=self.current_agent,
            to_agent=agent_name,
            task=task,
            data=data,
            timeout_ms=timeout_ms,
        )

        # If waiting for response, set up future
        response_future: asyncio.Future[AgentResponse] | None = None
        if wait:
            response_future = asyncio.Future()
            self.pending_responses[request.correlation_id] = response_future

        # Publish request event
        if self.event_queue is not None:
            await self.event_queue.put(request.to_event())

        # Wait for response if requested
        if wait and response_future is not None:
            try:
                timeout_sec = timeout_ms / 1000 if timeout_ms else None
                response = await asyncio.wait_for(response_future, timeout=timeout_sec)

                if response.success:
                    return response.result
                else:
                    raise RuntimeError(f"Agent {agent_name} failed: {response.error}")
            except TimeoutError:
                self.pending_responses.pop(request.correlation_id, None)
                raise TimeoutError(
                    f"Agent {agent_name} did not respond within {timeout_ms}ms"
                )

        return None

    def reply(self, result: Any, success: bool = True, error: str | None = None) -> AgentResponse:
        """Create a response to an agent request.

        This is used by agents that receive delegated tasks to send results back.

        Args:
            result: The response data
            success: Whether the task completed successfully
            error: Error message if task failed

        Returns:
            AgentResponse object ready to be converted to an event

        Example:
            ```python
            @react_to("agent.request")
            async def specialist(event, context):
                # Process the delegated task
                task = event.data["task"]
                result = perform_analysis(task)

                # Send response back
                response = context.reply(result)
                # Framework will automatically emit the response event
                return response.result
            ```
        """
        from agenticflow.flow.a2a import create_response

        # Extract request details from event
        request_data = self.event.data

        return create_response(
            from_agent=self.current_agent,
            to_agent=request_data.get("from_agent", ""),
            correlation_id=request_data.get("correlation_id", self.event.correlation_id or ""),
            result=result,
            success=success,
            error=error,
        )

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to context data."""
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like setting of context data."""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context data with default."""
        return self.data.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in context data."""
        return key in self.data


# Backward compatibility aliases
ReactiveContext = ExecutionContext
Context = FlowContext  # Preferred alias
