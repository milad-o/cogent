"""Execution context for flow orchestration.

This module provides execution context for both reactive and imperative flows,
enabling agent-to-agent communication and shared state management.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from agenticflow.events import Event
from agenticflow.reactive.a2a import AgentRequest, AgentResponse, create_request


@dataclass
class ExecutionContext:
    """Execution context for agents with A2A communication.
    
    This context is passed to agents during flow execution (reactive or imperative)
    and provides methods for agents to delegate tasks to other agents.
    
    Attributes:
        current_agent: Name of the currently executing agent
        event: The event that triggered this agent (for reactive flows)
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
            except asyncio.TimeoutError:
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
        from agenticflow.reactive.a2a import create_response

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


# Backward compatibility alias for reactive flows
ReactiveContext = ExecutionContext
