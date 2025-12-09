"""Event-driven flow orchestration.

This module provides the EventFlow class - the main orchestrator for
event-driven multi-agent systems.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agenticflow.core.utils import generate_id, now_utc
from agenticflow.reactive.core import (
    AgentTriggerConfig,
    Reaction,
    Trigger,
    TriggerBuilder,
)
from agenticflow.observability.bus import EventBus
from agenticflow.observability.event import Event, EventType
from agenticflow.observability.observer import Observer

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent


@dataclass(frozen=True, slots=True, kw_only=True)
class EventFlowConfig:
    """
    Configuration for event-driven flow execution.

    Attributes:
        max_rounds: Maximum event processing rounds (prevents infinite loops)
        max_concurrent_agents: How many agents can run in parallel
        event_timeout: Timeout for waiting on events (seconds)
        enable_history: Whether to record all events for debugging
        stop_on_idle: Stop when no more events to process
        stop_events: Event types that signal flow completion
    """

    max_rounds: int = 100
    max_concurrent_agents: int = 10
    event_timeout: float = 30.0
    enable_history: bool = True
    stop_on_idle: bool = True
    stop_events: frozenset[str] = frozenset({"flow.completed", "flow.failed"})


@dataclass(kw_only=True)
class EventFlowResult:
    """
    Result of event-driven flow execution.

    Contains the final output, execution history, and metrics.
    """

    output: str
    """Final output from the flow (last agent output or aggregated)."""

    events_processed: int = 0
    """Total number of events processed."""

    reactions: list[Reaction] = field(default_factory=list)
    """All agent reactions that occurred."""

    event_history: list[Event] = field(default_factory=list)
    """Full event history if enabled."""

    final_event: Event | None = None
    """The event that terminated the flow."""

    execution_time_ms: float = 0.0
    """Total execution time in milliseconds."""


class EventFlow:
    """
    Event-driven multi-agent orchestrator.

    Unlike static topologies (supervisor, pipeline, mesh), EventFlow uses
    a reactive model where agents subscribe to events and respond dynamically.

    The flow:
    1. Start with an initial event (e.g., "task.created")
    2. Agents with matching triggers are activated
    3. Agents produce output and optionally emit new events
    4. New events trigger more agents
    5. Flow ends when stop event occurs or no more triggers match

    Example:
        ```python
        # Define agents with triggers
        researcher = Agent(
            name="researcher",
            model=model,
            system_prompt="Research topics thoroughly.",
        )
        researcher_triggers = AgentTriggerConfig(
            triggers=[on("task.created").when(needs_research).build()]
        )

        writer = Agent(
            name="writer",
            model=model,
            system_prompt="Write engaging content.",
        )
        writer_triggers = AgentTriggerConfig(
            triggers=[on("researcher.completed").build()]
        )

        # Create flow
        flow = EventFlow()
        flow.register(researcher, researcher_triggers)
        flow.register(writer, writer_triggers)

        # Run with initial event
        result = await flow.run(
            task="Write about quantum computing",
            initial_event="task.created",
        )
        ```
    """

    def __init__(
        self,
        *,
        config: EventFlowConfig | None = None,
        event_bus: EventBus | None = None,
        observer: Observer | None = None,
    ) -> None:
        """
        Initialize the event flow.

        Args:
            config: Flow configuration
            event_bus: Shared event bus (creates new one if not provided)
            observer: Optional observer for monitoring and tracing
        """
        self.config = config or EventFlowConfig()
        self.bus = event_bus or EventBus()
        self.observer = observer
        self._agents: dict[str, tuple[Agent, AgentTriggerConfig]] = {}
        self._pending_events: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False
        self._stop_event: Event | None = None

        # Attach observer to the event bus
        if self.observer:
            self.observer.attach(self.bus)

    def register(
        self,
        agent: Agent,
        triggers: AgentTriggerConfig | list[Trigger | TriggerBuilder] | None = None,
    ) -> None:
        """
        Register an agent with its triggers.

        Args:
            agent: The agent to register
            triggers: Trigger configuration or list of triggers
        """
        if triggers is None:
            trigger_config = AgentTriggerConfig()
        elif isinstance(triggers, list):
            trigger_config = AgentTriggerConfig()
            for t in triggers:
                trigger_config.add_trigger(t)
        else:
            trigger_config = triggers

        self._agents[agent.name] = (agent, trigger_config)

    def unregister(self, agent_name: str) -> None:
        """Remove an agent from the flow."""
        self._agents.pop(agent_name, None)

    def _observe(
        self,
        event_type: EventType,
        data: dict[str, Any],
    ) -> None:
        """Emit an observability event if observer is attached."""
        if self.observer:
            event = Event(
                id=generate_id(),
                type=event_type,
                timestamp=now_utc(),
                data=data,
            )
            # Use the internal handler directly (observer is attached to bus
            # but reactive events are emitted directly for synchronous handling)
            self.observer._handle_event(event)

    async def run(
        self,
        task: str,
        *,
        initial_event: str = "task.created",
        initial_data: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> EventFlowResult:
        """
        Execute the event-driven flow.

        Args:
            task: The task/prompt to execute
            initial_event: Event type to emit at start
            initial_data: Additional data for initial event
            context: Shared context available to all agents

        Returns:
            EventFlowResult with output and execution details
        """
        import time

        start_time = time.perf_counter()

        # Initialize state
        self._running = True
        self._stop_event = None
        reactions: list[Reaction] = []
        events_processed = 0
        last_output = ""
        context = context or {}

        # Observe: user input received
        self._observe(
            EventType.USER_INPUT,
            {"content": task, "source": "reactive_flow"},
        )

        # Observe: flow started
        self._observe(
            EventType.REACTIVE_FLOW_STARTED,
            {
                "task": task[:200],
                "initial_event": initial_event,
                "agents": list(self._agents.keys()),
                "config": {
                    "max_rounds": self.config.max_rounds,
                    "max_concurrent": self.config.max_concurrent_agents,
                },
            },
        )

        # Emit initial event
        initial = Event(
            id=generate_id(),
            type=EventType.TASK_CREATED if initial_event == "task.created" else EventType.CUSTOM,
            timestamp=now_utc(),
            data={
                "event_name": initial_event,
                "task": task,
                **(initial_data or {}),
            },
        )
        await self._pending_events.put(initial)
        await self.bus.publish(initial)

        self._observe(
            EventType.REACTIVE_EVENT_EMITTED,
            {"event_name": initial_event, "event_id": initial.id},
        )

        # Process events until done
        rounds = 0
        error: Exception | None = None

        try:
            while self._running and rounds < self.config.max_rounds:
                rounds += 1

                self._observe(
                    EventType.REACTIVE_ROUND_STARTED,
                    {"round": rounds, "pending_events": self._pending_events.qsize()},
                )

                # Get next event (with timeout)
                try:
                    event = await asyncio.wait_for(
                        self._pending_events.get(),
                        timeout=self.config.event_timeout,
                    )
                except asyncio.TimeoutError:
                    if self.config.stop_on_idle:
                        break
                    continue

                events_processed += 1
                event_name = event.data.get("event_name", event.type.value)

                self._observe(
                    EventType.REACTIVE_EVENT_PROCESSED,
                    {"event_name": event_name, "event_id": event.id, "round": rounds},
                )

                # Check for stop events
                if event_name in self.config.stop_events:
                    self._stop_event = event
                    break

                # Find and execute matching agents
                agent_reactions = await self._process_event(
                    event=event,
                    task=task,
                    context=context,
                )
                reactions.extend(agent_reactions)

                # Track last successful output
                for reaction in agent_reactions:
                    if reaction.output and not reaction.error:
                        last_output = reaction.output

                self._observe(
                    EventType.REACTIVE_ROUND_COMPLETED,
                    {
                        "round": rounds,
                        "reactions": len(agent_reactions),
                        "total_reactions": len(reactions),
                    },
                )

                # If no agents matched and queue is empty, we're done
                if not agent_reactions and self._pending_events.empty():
                    if self.config.stop_on_idle:
                        break

        except Exception as e:
            error = e
            self._observe(
                EventType.REACTIVE_FLOW_FAILED,
                {"error": str(e), "rounds": rounds, "events_processed": events_processed},
            )
            raise

        finally:
            self._running = False

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Observe: flow completed
        if not error:
            self._observe(
                EventType.REACTIVE_FLOW_COMPLETED,
                {
                    "output_length": len(last_output),
                    "events_processed": events_processed,
                    "reactions": len(reactions),
                    "rounds": rounds,
                    "execution_time_ms": elapsed_ms,
                },
            )
            # Note: Not emitting OUTPUT_GENERATED here - REACTIVE_FLOW_COMPLETED
            # already contains output info, and each agent emits its own output

        return EventFlowResult(
            output=last_output,
            events_processed=events_processed,
            reactions=reactions,
            event_history=self.bus._event_history.copy() if self.config.enable_history else [],
            final_event=self._stop_event,
            execution_time_ms=elapsed_ms,
        )

    async def _process_event(
        self,
        event: Event,
        task: str,
        context: dict[str, Any],
    ) -> list[Reaction]:
        """
        Process a single event through all matching agents.

        Args:
            event: The event to process
            task: Original task
            context: Shared context

        Returns:
            List of reactions from triggered agents
        """
        reactions: list[Reaction] = []

        # Find all agents with matching triggers
        matching: list[tuple[Agent, Trigger]] = []
        for agent, trigger_config in self._agents.values():
            for trigger in trigger_config.get_matching_triggers(event):
                matching.append((agent, trigger))

        if not matching:
            event_name = event.data.get("event_name", event.type.value)
            self._observe(
                EventType.REACTIVE_NO_MATCH,
                {"event_name": event_name, "event_id": event.id},
            )
            return reactions

        # Execute agents (parallel up to max_concurrent)
        semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)

        async def run_agent(agent: Agent, trigger: Trigger) -> Reaction:
            async with semaphore:
                return await self._execute_agent(
                    agent=agent,
                    trigger=trigger,
                    event=event,
                    task=task,
                    context=context,
                )

        tasks = [run_agent(agent, trigger) for agent, trigger in matching]
        reactions = await asyncio.gather(*tasks)

        return list(reactions)

    async def _execute_agent(
        self,
        agent: Agent,
        trigger: Trigger,
        event: Event,
        task: str,
        context: dict[str, Any],
    ) -> Reaction:
        """
        Execute a single agent in response to an event.

        Args:
            agent: The agent to execute
            trigger: The trigger that activated it
            event: The triggering event
            task: Original task
            context: Shared context

        Returns:
            Reaction describing what happened
        """
        emitted: list[str] = []
        output: str | None = None
        error: str | None = None
        event_name = event.data.get("event_name", event.type.value)

        # Observe: agent triggered
        self._observe(
            EventType.REACTIVE_AGENT_TRIGGERED,
            {
                "agent": agent.name,
                "trigger_event": event_name,
                "trigger_on": str(trigger.on),
                "event_id": event.id,
            },
        )

        try:
            # Build prompt from event context
            prompt = self._build_prompt(event, task, context)

            # Execute agent - run() returns the output directly (string)
            result = await agent.run(prompt)
            # Handle both string return and object with .output attribute
            output = result.output if hasattr(result, "output") else str(result)

            # Observe: agent completed
            self._observe(
                EventType.REACTIVE_AGENT_COMPLETED,
                {
                    "agent": agent.name,
                    "output_length": len(output) if output else 0,
                    "trigger_event": event_name,
                },
            )

            # Emit completion event
            _, trigger_config = self._agents[agent.name]
            if trigger_config.auto_emit_completion:
                completion_event = f"{agent.name}.completed"
                await self._emit_event(
                    completion_event,
                    {"agent": agent.name, "output": output, "trigger_event": event.id},
                )
                emitted.append(completion_event)

            # Emit trigger-specified event
            if trigger.emits:
                await self._emit_event(
                    trigger.emits,
                    {"agent": agent.name, "output": output, "trigger_event": event.id},
                )
                emitted.append(trigger.emits)

        except Exception as e:
            error = str(e)

            # Observe: agent failed
            self._observe(
                EventType.REACTIVE_AGENT_FAILED,
                {
                    "agent": agent.name,
                    "error": error,
                    "trigger_event": event_name,
                },
            )

            # Emit error event
            _, trigger_config = self._agents[agent.name]
            if trigger_config.emit_on_error:
                error_event = f"{agent.name}.error"
                await self._emit_event(
                    error_event,
                    {"agent": agent.name, "error": error, "trigger_event": event.id},
                )
                emitted.append(error_event)

        return Reaction(
            agent_name=agent.name,
            trigger=trigger,
            event=event,
            output=output,
            emitted_events=emitted,
            error=error,
        )

    def _build_prompt(
        self,
        event: Event,
        task: str,
        context: dict[str, Any],
    ) -> str:
        """Build prompt for agent from event context."""
        parts = [f"Task: {task}"]

        # Add event context
        if event.data:
            if "output" in event.data:
                parts.append(f"\nPrevious agent output:\n{event.data['output']}")
            if "task" in event.data and event.data["task"] != task:
                parts.append(f"\nSub-task: {event.data['task']}")

        # Add shared context
        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            parts.append(f"\nContext:\n{context_str}")

        return "\n".join(parts)

    async def _emit_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Emit an event to the bus and pending queue."""
        event = Event(
            id=generate_id(),
            type=EventType.CUSTOM,
            timestamp=now_utc(),
            data={"event_name": event_name, **data},
        )
        await self._pending_events.put(event)
        await self.bus.publish(event)

    async def emit(self, event_name: str, data: dict[str, Any] | None = None) -> None:
        """
        Manually emit an event into the flow.

        Useful for external triggers like webhooks, timers, or user input.

        Args:
            event_name: Name/type of the event
            data: Event data
        """
        await self._emit_event(event_name, data or {})

    def stop(self) -> None:
        """Stop the flow gracefully."""
        self._running = False

    @property
    def agents(self) -> list[str]:
        """Get names of all registered agents."""
        return list(self._agents.keys())

    @property
    def is_running(self) -> bool:
        """Check if flow is currently executing."""
        return self._running
