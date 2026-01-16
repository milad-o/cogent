"""Agent reactor - wraps an Agent as a reactor for event-driven flows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.events import Event
from agenticflow.reactors.base import BaseReactor

if TYPE_CHECKING:
    from agenticflow.agent import Agent
    from agenticflow.flow.context import Context


class AgentReactor(BaseReactor):
    """Wraps an Agent as a reactor for event-driven flows.

    The AgentReactor extracts task/prompt from incoming events,
    runs the agent, and emits the result as an event.

    Example:
        ```python
        from agenticflow import Agent
        from agenticflow.agent.reactor import AgentReactor

        agent = Agent(name="researcher", model=model)
        reactor = AgentReactor(agent)

        # Or use shorthand (Flow auto-wraps agents)
        flow = Flow()
        flow.register(agent, on="task.created")  # Auto-wrapped
        ```
    """

    def __init__(
        self,
        agent: Agent,
        *,
        task_key: str = "task",
        context_key: str = "context",
        emit_name: str | None = None,
    ) -> None:
        """Initialize agent reactor.

        Args:
            agent: The agent to wrap
            task_key: Key in event.data containing the task/prompt
            context_key: Key in event.data containing additional context
            emit_name: Event name to emit (default: "agent.done")
        """
        super().__init__(agent.name)
        self._agent = agent
        self._task_key = task_key
        self._context_key = context_key
        self._emit_name = emit_name or "agent.done"

    @property
    def agent(self) -> Agent:
        """The wrapped agent."""
        return self._agent

    async def handle(
        self,
        event: Event,
        ctx: Context,
    ) -> Event:
        """Run the agent with event data as input."""
        # Extract task from event
        task = self._extract_task(event, ctx)

        # Build context from event and flow history
        agent_context = self._build_context(event, ctx)

        # Run the agent
        try:
            # If context is available, add it to the prompt
            full_prompt = f"{agent_context}\n\nTask: {task}" if agent_context else task

            result = await self._agent.run(full_prompt)

            return Event(
                name=self._emit_name,
                source=self.name,
                data={
                    "output": result,
                    "task": task,
                    "agent": self.name,
                },
                correlation_id=event.correlation_id,
                metadata={
                    "flow_id": ctx.flow_id,
                    "event_id": event.id,
                },
            )

        except Exception as e:
            return Event(
                name="agent.error",
                source=self.name,
                data={
                    "error": str(e),
                    "task": task,
                    "agent": self.name,
                    "exception_type": type(e).__name__,
                },
                correlation_id=event.correlation_id,
            )

    def _extract_task(self, event: Event, ctx: Context) -> str:
        """Extract the task/prompt from the event."""
        data = event.data

        # Try configured key
        if self._task_key in data:
            return str(data[self._task_key])

        # Try common keys
        for key in ["task", "prompt", "message", "query", "input", "output"]:
            if key in data:
                return str(data[key])

        # Fallback: use original task from context
        if ctx.original_task:
            return ctx.original_task

        # Last resort: stringify data
        return str(data)

    def _build_context(self, event: Event, ctx: Context) -> str:
        """Build context string from event and flow history."""
        parts = []

        # Add explicit context from event
        if self._context_key in event.data:
            parts.append(str(event.data[self._context_key]))

        # Add previous agent outputs from flow history
        for prev_event in ctx.history:
            if prev_event.name == "agent.done" and prev_event.source != self.name:
                output = prev_event.data.get("output", "")
                if output:
                    parts.append(f"[{prev_event.source}]: {output}")

        return "\n\n".join(parts) if parts else ""

    def __repr__(self) -> str:
        return f"AgentReactor({self._agent.name!r})"


def wrap_agent(agent: Agent, **kwargs: Any) -> AgentReactor:
    """Wrap an agent as a reactor.

    Args:
        agent: The agent to wrap
        **kwargs: Additional AgentReactor options

    Returns:
        AgentReactor wrapping the agent
    """
    return AgentReactor(agent, **kwargs)
