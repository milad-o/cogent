"""Event-driven flow orchestration (ReactiveFlow).

This module provides the ReactiveFlow class - the main orchestrator for
event-driven multi-agent systems where agents react to events.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from agenticflow.flow.base import BaseFlow, FlowResult
from agenticflow.reactive.core import (
    AgentTriggerConfig,
    Reaction,
    Trigger,
    TriggerBuilder,
)
from agenticflow.reactive.skills import Skill, SkillBuilder
from agenticflow.events.bus import EventBus as CoreEventBus
from agenticflow.events.event import Event as CoreEvent
from agenticflow.observability.trace_record import TraceType
from agenticflow.observability.observer import Observer

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent
    from agenticflow.reactive.checkpointer import Checkpointer, FlowState


@dataclass(frozen=True, slots=True, kw_only=True)
class ReactiveFlowConfig:
    """
    Configuration for reactive flow execution.

    Attributes:
        max_rounds: Maximum event processing rounds (prevents infinite loops)
        max_concurrent_agents: How many agents can run in parallel
        event_timeout: Timeout for waiting on events (seconds)
        enable_history: Whether to record all events for debugging
        stop_on_idle: Stop when no more events to process
        stop_events: Event types that signal flow completion
        flow_id: Optional fixed flow ID (auto-generated if None)
        checkpoint_every: Checkpoint after every N rounds (0 = disabled)
    """

    max_rounds: int = 100
    max_concurrent_agents: int = 10
    event_timeout: float = 30.0
    enable_history: bool = True
    stop_on_idle: bool = True
    stop_events: frozenset[str] = frozenset({"flow.completed", "flow.failed"})
    flow_id: str | None = None
    checkpoint_every: int = 0  # 0 = disabled, 1 = every round, etc.



# Backward compatibility alias
EventFlowConfig = ReactiveFlowConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class ReactiveFlowResult(FlowResult):
    """
    Result of event-driven flow execution.

    Extends FlowResult with reactive-specific metrics.
    """

    events_processed: int = 0
    """Total number of events processed."""

    reactions: list[Reaction] = field(default_factory=list)
    """All agent reactions that occurred."""

    event_history: list[Event] = field(default_factory=list)
    """Full event history if enabled."""

    final_event: Event | None = None
    """The event that terminated the flow."""

    checkpoint_id: str | None = None
    """Last checkpoint ID if checkpointing was enabled."""

    flow_id: str | None = None
    """Flow ID for this execution."""



# Backward compatibility alias
EventFlowResult = ReactiveFlowResult


class ReactiveFlow(BaseFlow):
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
            triggers=[react_to("task.created").when(needs_research).build()]
        )

        writer = Agent(
            name="writer",
            model=model,
            system_prompt="Write engaging content.",
        )
        writer_triggers = AgentTriggerConfig(
            triggers=[react_to("researcher.completed").build()]
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
        config: ReactiveFlowConfig | None = None,
        event_bus: Any | None = None,
        observer: Observer | None = None,
        thread_id_resolver: Callable[[CoreEvent, dict[str, Any]], str | None] | None = None,
        checkpointer: Checkpointer | None = None,
    ) -> None:
        """
        Initialize the reactive flow.

        Args:
            config: Flow configuration
            event_bus: Shared event bus (creates new one if not provided)
            observer: Optional observer for monitoring and tracing
            checkpointer: Optional checkpointer for persistent state
        """
        # Initialize base class
        # BaseFlow bus remains observability-only.
        super().__init__(event_bus=event_bus, observer=observer)
        
        self.config = config or ReactiveFlowConfig()
        # Dedicated core bus for orchestration events.
        self.events = CoreEventBus()
        self._agents_registry: dict[str, tuple[Agent, AgentTriggerConfig]] = {}
        self._pending_events: asyncio.Queue[CoreEvent] = asyncio.Queue()
        self._stop_event: Event | None = None
        self._thread_id_resolver = thread_id_resolver
        self._checkpointer = checkpointer

        # Container-like helpers (optional): shared memory and background tasks.
        self._shared_memory: Any | None = None
        self._spawned: set[asyncio.Task[Any]] = set()

        # Skills registry: event-triggered behavioral specializations
        self._skills_registry: dict[str, Skill] = {}

        # Current flow execution state (for checkpointing)
        self._flow_id: str | None = None
        self._last_checkpoint_id: str | None = None


    @property
    def agents(self) -> list[str]:
        """Names of registered agents (BaseFlow interface)."""
        return list(self._agents_registry.keys())

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

        # Connect agent to the flow's event bus for shared observability
        # This ensures tool events from agents are visible to the observer
        agent.event_bus = self._bus

        # Optional: if the flow is acting as a container, inject shared memory
        # into agents that don't already have one.
        if self._shared_memory is not None and getattr(agent, "memory_manager", None) is None:
            setup = getattr(agent, "_setup_memory", None)
            if callable(setup):
                setup(memory=self._shared_memory)

        self._agents_registry[agent.name] = (agent, trigger_config)

    def register_skill(
        self,
        skill: Skill | SkillBuilder,
    ) -> None:
        """
        Register a skill with the flow.

        Skills are event-triggered behavioral specializations. When an event
        matches a skill's trigger, the skill's prompt and tools are injected
        into any agent that also triggers on that event.

        Args:
            skill: The skill to register (or SkillBuilder)

        Example:
            ```python
            flow.register_skill(Skill(
                name="python_expert",
                trigger=Trigger(on="code.write"),
                prompt="You are a Python expert...",
                tools=[run_python],
            ))
            ```
        """
        if isinstance(skill, SkillBuilder):
            skill = skill.build()
        self._skills_registry[skill.name] = skill

    def unregister_skill(self, skill_name: str) -> None:
        """Remove a skill from the flow."""
        self._skills_registry.pop(skill_name, None)

    @property
    def skills(self) -> list[str]:
        """Names of registered skills."""
        return list(self._skills_registry.keys())

    def _get_matching_skills(self, event: CoreEvent) -> list[Skill]:
        """Get all skills that match an event, sorted by priority."""
        matching = [s for s in self._skills_registry.values() if s.matches(event)]
        return sorted(matching, key=lambda s: -s.priority)

    def with_memory(self, memory: Any | None = None) -> "ReactiveFlow":
        """Configure shared memory for agents registered to this flow.

        If set, agents without an existing `memory_manager` will receive this
        memory backend at registration time.
        """
        from agenticflow.memory import Memory

        self._shared_memory = memory or Memory()
        return self

    def spawn(self, coro: Any) -> asyncio.Task[Any]:
        """Spawn a background task and track it for cleanup."""
        task = asyncio.create_task(coro)
        self._spawned.add(task)

        def _done(_t: asyncio.Task[Any]) -> None:
            self._spawned.discard(_t)

        task.add_done_callback(_done)
        return task

    async def cancel_spawned(self) -> None:
        """Cancel any still-running spawned tasks."""
        if not self._spawned:
            return

        for task in list(self._spawned):
            if not task.done():
                task.cancel()

        await asyncio.gather(*self._spawned, return_exceptions=True)
        self._spawned.clear()

    def thread_by_data(self, key: str, *, prefix: str | None = None) -> "ReactiveFlow":
        """Derive `thread_id` from `event.data[key]`.

        This is a mid-level UX helper for per-entity memory without lambdas.

        Example:
            flow.thread_by_data("job_id")
        """
        from agenticflow.reactive.threading import thread_id_from_data

        self._thread_id_resolver = thread_id_from_data(key, prefix=prefix)
        return self

    def unregister(self, agent_name: str) -> None:
        """Remove an agent from the flow."""
        self._agents_registry.pop(agent_name, None)

    @property
    def memory(self) -> Any | None:
        """Shared memory configured via `with_memory()` (if any)."""
        return self._shared_memory

    @property
    def flow_id(self) -> str | None:
        """Current flow ID (set during run)."""
        return self._flow_id

    @property
    def last_checkpoint_id(self) -> str | None:
        """Most recent checkpoint ID (if checkpointing enabled)."""
        return self._last_checkpoint_id

    async def resume(
        self,
        state: FlowState,
        *,
        context: dict[str, Any] | None = None,
    ) -> ReactiveFlowResult:
        """
        Resume a flow from a checkpoint.

        Args:
            state: The flow state to resume from
            context: Optional context override (uses checkpoint context if not provided)

        Returns:
            ReactiveFlowResult with output and execution details

        Example:
            ```python
            # After crash, resume from last checkpoint
            state = await checkpointer.load_latest("my-flow-id")
            if state:
                result = await flow.resume(state)
            ```
        """
        import time

        from agenticflow.reactive.checkpointer import generate_checkpoint_id

        start_time = time.perf_counter()

        # Restore flow ID
        self._flow_id = state.flow_id
        self._running = True
        self._stop_event = None

        # Restore state
        task = state.task
        rounds = state.round
        events_processed = state.events_processed
        last_output = state.last_output
        context = context if context is not None else dict(state.context)

        # Restore reactions (convert from dicts)
        reactions: list[Reaction] = []
        # Note: We don't restore full Reaction objects as they contain
        # non-serializable Trigger objects. Start fresh from checkpoint.

        # Restore pending events
        for event_dict in state.pending_events:
            event = CoreEvent(
                id=event_dict.get("id", ""),
                name=event_dict.get("name", ""),
                data=event_dict.get("data", {}),
            )
            await self._pending_events.put(event)

        self._observe(
            TraceType.REACTIVE_FLOW_STARTED,
            {
                "task": task[:200],
                "resumed_from": state.checkpoint_id,
                "agents": list(self._agents_registry.keys()),
                "round": rounds,
            },
        )

        error: Exception | None = None

        try:
            while self._running and rounds < self.config.max_rounds:
                rounds += 1

                self._observe(
                    TraceType.REACTIVE_ROUND_STARTED,
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
                event_name = event.name

                self._observe(
                    TraceType.REACTIVE_EVENT_PROCESSED,
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

                # Checkpoint if enabled
                if self._checkpointer and self.config.checkpoint_every > 0:
                    if rounds % self.config.checkpoint_every == 0:
                        await self._save_checkpoint(
                            task=task,
                            rounds=rounds,
                            events_processed=events_processed,
                            last_output=last_output,
                            context=context,
                            reactions=reactions,
                        )

                self._observe(
                    TraceType.REACTIVE_ROUND_COMPLETED,
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
                TraceType.REACTIVE_FLOW_FAILED,
                {"error": str(e), "rounds": rounds, "events_processed": events_processed},
            )
            raise

        finally:
            self._running = False
            await self.cancel_spawned()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if not error:
            self._observe(
                TraceType.REACTIVE_FLOW_COMPLETED,
                {
                    "output_length": len(last_output),
                    "events_processed": events_processed,
                    "reactions": len(reactions),
                    "rounds": rounds,
                    "execution_time_ms": elapsed_ms,
                },
            )

        return EventFlowResult(
            output=last_output,
            events_processed=events_processed,
            reactions=reactions,
            event_history=self._bus._event_history.copy() if self.config.enable_history else [],
            final_event=self._stop_event,
            execution_time_ms=elapsed_ms,
            checkpoint_id=self._last_checkpoint_id,
            flow_id=self._flow_id,
        )

    async def _save_checkpoint(
        self,
        *,
        task: str,
        rounds: int,
        events_processed: int,
        last_output: str,
        context: dict[str, Any],
        reactions: list[Reaction],
    ) -> None:
        """Save current flow state to checkpoint."""
        if not self._checkpointer or not self._flow_id:
            return

        from agenticflow.reactive.checkpointer import FlowState, generate_checkpoint_id

        checkpoint_id = generate_checkpoint_id()
        self._last_checkpoint_id = checkpoint_id

        # Drain pending events to list (and re-queue)
        pending_list: list[dict[str, Any]] = []
        temp_events: list[CoreEvent] = []
        while not self._pending_events.empty():
            try:
                ev = self._pending_events.get_nowait()
                temp_events.append(ev)
                pending_list.append({"id": ev.id, "name": ev.name, "data": ev.data})
            except asyncio.QueueEmpty:
                break
        # Re-queue
        for ev in temp_events:
            await self._pending_events.put(ev)

        state = FlowState(
            flow_id=self._flow_id,
            checkpoint_id=checkpoint_id,
            task=task,
            events_processed=events_processed,
            pending_events=pending_list,
            context=context,
            reactions=[],  # Don't serialize reactions (contain non-serializable Triggers)
            last_output=last_output,
            round=rounds,
        )

        await self._checkpointer.save(state)

        self._observe(
            TraceType.REACTIVE_ROUND_STARTED,  # Reuse existing trace type
            {"checkpoint_saved": checkpoint_id, "round": rounds},
        )

    async def run(
        self,
        task: str,
        *,
        initial_event: str = "task.created",
        initial_data: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ReactiveFlowResult:
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

        from agenticflow.reactive.checkpointer import generate_flow_id

        start_time = time.perf_counter()

        # Initialize flow ID
        self._flow_id = self.config.flow_id or generate_flow_id()
        self._last_checkpoint_id = None

        # Initialize state
        self._running = True
        self._stop_event = None
        reactions: list[Reaction] = []
        events_processed = 0
        last_output = ""
        context = context or {}


        # Observe: user input received
        self._observe(
            TraceType.USER_INPUT,
            {"content": task, "source": "reactive_flow"},
        )

        # Observe: flow started
        self._observe(
            TraceType.REACTIVE_FLOW_STARTED,
            {
                "task": task[:200],
                "initial_event": initial_event,
                "agents": list(self._agents_registry.keys()),
                "config": {
                    "max_rounds": self.config.max_rounds,
                    "max_concurrent": self.config.max_concurrent_agents,
                },
            },
        )

        # Emit initial event
        initial = CoreEvent(
            name=initial_event,
            data={"task": task, **(initial_data or {})},
        )
        await self._pending_events.put(initial)
        await self.events.publish(initial)
        # Mirror to observability bus for subscribers/debugging
        await self._bus.publish(initial_event, {"task": task, **(initial_data or {})})

        self._observe(
            TraceType.REACTIVE_EVENT_EMITTED,
            {"event_name": initial_event, "event_id": initial.id},
        )

        # Process events until done
        rounds = 0
        error: Exception | None = None

        try:
            while self._running and rounds < self.config.max_rounds:
                rounds += 1

                self._observe(
                    TraceType.REACTIVE_ROUND_STARTED,
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
                event_name = event.name

                self._observe(
                    TraceType.REACTIVE_EVENT_PROCESSED,
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

                # Checkpoint if enabled
                if self._checkpointer and self.config.checkpoint_every > 0:
                    if rounds % self.config.checkpoint_every == 0:
                        await self._save_checkpoint(
                            task=task,
                            rounds=rounds,
                            events_processed=events_processed,
                            last_output=last_output,
                            context=context,
                            reactions=reactions,
                        )

                self._observe(
                    TraceType.REACTIVE_ROUND_COMPLETED,
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
                TraceType.REACTIVE_FLOW_FAILED,
                {"error": str(e), "rounds": rounds, "events_processed": events_processed},
            )
            raise

        finally:
            self._running = False
            await self.cancel_spawned()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Observe: flow completed
        if not error:
            self._observe(
                TraceType.REACTIVE_FLOW_COMPLETED,
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
            event_history=self._bus._event_history.copy() if self.config.enable_history else [],
            final_event=self._stop_event,
            execution_time_ms=elapsed_ms,
            checkpoint_id=self._last_checkpoint_id,
            flow_id=self._flow_id,
        )


    async def _process_event(
        self,
        event: CoreEvent,
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
        for agent, trigger_config in self._agents_registry.values():
            for trigger in trigger_config.get_matching_triggers(event):
                matching.append((agent, trigger))

        if not matching:
            self._observe(
                TraceType.REACTIVE_NO_MATCH,
                {"event_name": event.name, "event_id": event.id},
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
        event: CoreEvent,
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
        event_name = event.name

        # Observe: agent triggered
        self._observe(
            TraceType.REACTIVE_AGENT_TRIGGERED,
            {
                "agent": agent.name,
                "trigger_event": event_name,
                "trigger_on": str(trigger.on),
                "event_id": event.id,
            },
        )

        try:
            thread_id: str | None = None
            if self._thread_id_resolver is not None:
                try:
                    thread_id = self._thread_id_resolver(event, context)
                except TypeError:
                    # Backward compatible: resolver(event) only
                    thread_id = self._thread_id_resolver(event)  # type: ignore[misc]

            # === Skill Injection ===
            # Find matching skills and prepare context/tools
            matching_skills = self._get_matching_skills(event)
            skill_context = dict(context)  # Copy to avoid mutation
            added_tools: list[str] = []
            
            for skill in matching_skills:
                # Observe: skill activated
                self._observe(
                    TraceType.SKILL_ACTIVATED,
                    {
                        "skill": skill.name,
                        "agent": agent.name,
                        "trigger_event": event_name,
                    },
                )
                
                # Apply context enrichment if configured
                skill_context = skill.enrich_context(event, skill_context)
                
                # Temporarily add skill tools to agent
                for tool_fn in skill.tools:
                    tool_name = getattr(tool_fn, "name", None) or tool_fn.__name__
                    try:
                        agent.add_tool(tool_fn)
                        added_tools.append(tool_name)
                    except Exception:
                        pass  # Tool may already exist or agent doesn't support add_tool

            # Prefer a first-class reactive API when the agent provides it.
            # Important: don't treat arbitrary objects (e.g. MagicMock) as reactive.
            react_fn = getattr(agent, "react", None)
            if react_fn is not None and inspect.iscoroutinefunction(react_fn):
                try:
                    result = await agent.react(event, task=task, context=skill_context, thread_id=thread_id)
                except TypeError:
                    result = await agent.react(event, task=task, context=skill_context)
            else:
                # Backward compatible path: build prompt and call run()
                prompt = self._build_prompt(event, task, skill_context, matching_skills)
                result = await agent.run(prompt, context=skill_context, thread_id=thread_id)
            
            # === Skill Cleanup ===
            # Remove temporarily added skill tools
            for tool_name in added_tools:
                try:
                    agent.remove_tool(tool_name)
                except Exception:
                    pass  # Agent may not support remove_tool
            
            for skill in matching_skills:
                self._observe(
                    TraceType.SKILL_DEACTIVATED,
                    {"skill": skill.name, "agent": agent.name},
                )
            
            # Handle both string return and object with .output attribute
            output = result.output if hasattr(result, "output") else str(result)

            # Observe: agent completed
            self._observe(
                TraceType.REACTIVE_AGENT_COMPLETED,
                {
                    "agent": agent.name,
                    "output_length": len(output) if output else 0,
                    "trigger_event": event_name,
                },
            )

            # Emit completion event
            _, trigger_config = self._agents_registry[agent.name]
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
                TraceType.REACTIVE_AGENT_FAILED,
                {
                    "agent": agent.name,
                    "error": error,
                    "trigger_event": event_name,
                },
            )

            # Emit error event
            _, trigger_config = self._agents_registry[agent.name]
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
        event: CoreEvent,
        task: str,
        context: dict[str, Any],
        skills: list[Skill] | None = None,
    ) -> str:
        """Build prompt for agent from event context.
        
        Args:
            event: The triggering event.
            task: Original task.
            context: Shared context.
            skills: Optional list of active skills to inject.
        
        Returns:
            Constructed prompt string.
        """
        event_name = event.name
        parts = [f"Task: {task}", f"Event: {event_name}"]

        try:
            event_json = json.dumps(event.data or {}, default=str, ensure_ascii=False)
        except Exception:
            event_json = str(event.data or {})
        parts.append(f"Event Data: {event_json}")

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

        # === Inject Skill Prompts ===
        if skills:
            skill_prompts = []
            for skill in skills:
                skill_prompts.append(f"## Active Skill: {skill.name}\n{skill.prompt}")
            parts.append("\n" + "\n\n".join(skill_prompts))

        return "\n".join(parts)

    async def _emit_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Emit an orchestration event (core) and mirror it to observability."""
        core_event = CoreEvent(name=event_name, data=dict(data))
        await self._pending_events.put(core_event)
        await self.events.publish(core_event)
        await self._bus.publish(event_name, dict(data))

    async def emit(self, event_name: str, data: dict[str, Any] | None = None) -> None:
        """
        Manually emit an event into the flow.

        Useful for external triggers like webhooks, timers, or user input.

        Args:
            event_name: Name/type of the event
            data: Event data
        """
        await self._emit_event(event_name, data or {})


# Backward compatibility alias
EventFlow = ReactiveFlow
