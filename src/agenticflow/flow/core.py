"""Flow - Unified event-driven orchestration engine.

This is the core orchestration class that drives event-driven multi-agent
systems. Flow manages reactors (event handlers) and an event bus to enable
event-driven agent coordination.

The Flow replaces both the old Topology-based and Flow approaches
with a single, unified event-driven model.
"""

from __future__ import annotations

import asyncio
import fnmatch
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from agenticflow.events import Event, EventBus
from agenticflow.flow.config import FlowConfig, FlowResult, ReactorBinding
from agenticflow.flow.state import CoordinationManager
from agenticflow.observability.bus import TraceBus
from agenticflow.reactors.base import Reactor

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent
    from agenticflow.flow.checkpointer import Checkpointer, FlowState
    from agenticflow.flow.skills import Skill, SkillBuilder
    from agenticflow.flow.streaming import StreamChunk
    from agenticflow.middleware.base import Middleware
    from agenticflow.observability.observer import Observer


def _generate_id() -> str:
    """Generate a unique flow/reactor ID."""
    return uuid.uuid4().hex[:12]


def _matches_pattern(event_type: str, pattern: str) -> bool:
    """Check if event type matches a pattern (supports * wildcard)."""
    return fnmatch.fnmatch(event_type, pattern)


@dataclass
class _PendingReaction:
    """Tracks a pending reactor execution."""

    reactor_id: str
    event: Event
    binding: ReactorBinding
    task: asyncio.Task[Event | list[Event] | None] | None = None


class Flow:
    """Unified event-driven orchestration engine.

    Flow is the core class for building event-driven multi-agent systems.
    It manages reactors (event handlers) and coordinates their execution
    through an event bus.

    Key Concepts:
        - **Reactors**: Units that react to events (agents, functions, aggregators)
        - **Events**: Messages that trigger reactors
        - **Patterns**: Event type patterns with wildcard support
        - **Middleware**: Cross-cutting concerns (logging, retry, timeout)

    Example - Basic Flow:
        ```python
        from agenticflow import Agent, Flow

        # Create agents
        researcher = Agent(name="researcher", model=model)
        writer = Agent(name="writer", model=model)

        # Create flow and register reactors
        flow = Flow()
        flow.register(researcher, on="task.created", emits="research.done")
        flow.register(writer, on="research.done", emits="flow.done")

        # Run the flow
        result = await flow.run("Write about AI", initial_event="task.created")
        print(result.output)
        ```

    Example - With Patterns:
        ```python
        from agenticflow.flow import pipeline, supervisor

        # Pipeline pattern: sequential processing
        flow = pipeline([agent1, agent2, agent3])
        result = await flow.run("Process this")

        # Supervisor pattern: one coordinator, many workers
        flow = supervisor(
            coordinator=manager,
            workers=[analyst, writer, reviewer],
        )
        result = await flow.run("Complete this task")
        ```

    Example - With Aggregation:
        ```python
        from agenticflow.reactors import Aggregator

        flow = Flow()

        # Fan-out: multiple workers process in parallel
        flow.register(worker1, on="task.created")
        flow.register(worker2, on="task.created")
        flow.register(worker3, on="task.created")

        # Fan-in: aggregate results
        flow.register(
            Aggregator(collect=3, emit="all.done"),
            on="worker.*.done",
        )

        result = await flow.run("Parallel task")
        ```
    """

    def __init__(
        self,
        *,
        config: FlowConfig | None = None,
        event_bus: EventBus | None = None,
        observer: Observer | None = None,
        checkpointer: Checkpointer | None = None,
        thread_id_resolver: Callable[[Event, dict[str, object]], str | None]
        | None = None,
    ) -> None:
        """Initialize the Flow.

        Args:
            config: Flow configuration
            event_bus: Shared event bus (creates new one if not provided)
            observer: Optional observer for monitoring and tracing
            checkpointer: Optional checkpointer for persistent state
            thread_id_resolver: Optional function to resolve thread IDs from events
        """
        self.config = config or FlowConfig()
        self.events = event_bus or EventBus()
        self._observer = observer
        self._checkpointer = checkpointer
        self._thread_id_resolver = thread_id_resolver

        # Create TraceBus for observability events
        self._trace_bus = TraceBus()

        # Attach observer to trace bus if provided
        if self._observer:
            self._observer.attach(self._trace_bus)

        # Reactor registry: id -> reactor instance
        self._reactors: dict[str, Reactor] = {}

        # Bindings: how reactors connect to events
        self._bindings: list[ReactorBinding] = []

        # Middleware stack
        self._middleware: list[Middleware] = []

        # Execution state
        self._flow_id: str | None = None
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._stop_event: Event | None = None
        self._event_history: list[Event] = []
        self._last_checkpoint_id: str | None = None

        # Skills registry: name -> Skill
        self._skills_registry: dict[str, Skill] = {}

        # Source groups: name -> set of sources
        self._source_groups: dict[str, set[str]] = {}
        self._init_builtin_groups()

        # Coordination manager for stateful patterns
        self._coordination = CoordinationManager()

        # Container-like features
        self._shared_memory: object | None = None
        self._spawned: set[asyncio.Task[object]] = set()

    @property
    def reactors(self) -> list[str]:
        """List of registered reactor IDs."""
        return list(self._reactors.keys())

    @property
    def skills(self) -> list[str]:
        """Names of registered skills."""
        return list(self._skills_registry.keys())

    @property
    def memory(self) -> object | None:
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

    # -------------------------------------------------------------------------
    # Skills API
    # -------------------------------------------------------------------------

    def register_skill(self, skill: Skill | SkillBuilder) -> Self:
        """Register a skill with the flow.

        Skills are event-triggered behavioral specializations. When an event
        matches a skill's trigger, the skill's prompt and tools are injected
        into any agent that also triggers on that event.

        Args:
            skill: The skill to register (or SkillBuilder)

        Returns:
            Self for method chaining

        Example:
            ```python
            from agenticflow.flow.skills import skill

            python_skill = skill(
                "python_expert",
                on="code.write",
                prompt="You are a Python expert...",
                tools=[run_python],
            )

            flow.register_skill(python_skill)
            ```
        """
        from agenticflow.flow.skills import SkillBuilder

        if isinstance(skill, SkillBuilder):
            skill = skill.build()
        self._skills_registry[skill.name] = skill
        return self

    def _get_matching_skills(self, event: Event) -> list[Skill]:
        """Get all skills that match an event, sorted by priority."""

        matching = [s for s in self._skills_registry.values() if s.matches(event)]
        return sorted(matching, key=lambda s: -s.priority)

    # -------------------------------------------------------------------------
    # Source Groups API
    # -------------------------------------------------------------------------

    def _init_builtin_groups(self) -> None:
        """Initialize built-in source groups.

        Built-in groups:
            - :agents - Auto-populated with registered agent names
            - :system - System sources (flow, router, aggregator)
        """
        self._source_groups["agents"] = set()  # Auto-populated on agent registration
        self._source_groups["system"] = {"flow", "router", "aggregator"}

    def add_source_group(self, name: str, sources: list[str]) -> Self:
        """Define a named group of sources.

        Source groups allow cleaner multi-source filtering using :group syntax.
        Groups can be referenced in `after` parameter and pattern syntax.

        Args:
            name: Group name (without : prefix)
            sources: List of source names or patterns

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name starts with : or is empty

        Example:
            ```python
            # Define a group of analyst agents
            flow.add_source_group("analysts", ["agent1", "agent2", "agent3"])

            # Use in after parameter
            flow.register(aggregator, on="analysis.done", after=":analysts")

            # Use in pattern syntax
            flow.register(reviewer, on="*.done@:analysts")

            # Chain multiple groups
            flow.add_source_group("writers", ["writer1", "writer2"])
                .add_source_group("reviewers", ["reviewer1", "reviewer2"])
            ```
        """
        if not name or name.startswith(":"):
            raise ValueError("Group name cannot be empty or start with ':'")

        self._source_groups[name] = set(sources)
        return self

    def get_source_group(self, name: str) -> set[str]:
        """Get sources in a named group.

        Args:
            name: Group name (without : prefix)

        Returns:
            Set of source names, or empty set if group doesn't exist

        Example:
            ```python
            flow.add_source_group("analysts", ["agent1", "agent2"])
            sources = flow.get_source_group("analysts")
            # {'agent1', 'agent2'}

            # Nonexistent group returns empty set
            sources = flow.get_source_group("missing")
            # set()
            ```
        """
        return self._source_groups.get(name, set()).copy()

    def reset_coordination(self, binding_id: str) -> Self:
        """Manually reset a coordination point for a specific reactor binding.

        Use this to reset coordinations that have reset_after=False, allowing
        them to trigger again when all sources emit in the next cycle.

        Args:
            binding_id: Unique identifier for the reactor binding (from register())

        Returns:
            Self for method chaining

        Example:
            ```python
            # Register one-time coordination
            binding_id = flow.register(
                handler,
                on="task.done",
                when=all_sources(["a", "b", "c"], reset_after=False)
            )

            # Later, manually reset to allow triggering again
            flow.reset_coordination(binding_id)
            ```

        Note:
            For coordinations with reset_after=True (default), this method
            is not needed as they auto-reset after each completion.
        """
        self._coordination.reset_coordination(binding_id)
        return self

    # -------------------------------------------------------------------------
    # Memory API
    # -------------------------------------------------------------------------

    def with_memory(self, memory: object | None = None) -> Self:
        """Configure shared memory for agents registered to this flow.

        If set, agents without an existing `memory_manager` will receive this
        memory backend at registration time.

        Args:
            memory: Memory instance (creates new Memory if None)

        Returns:
            Self for method chaining

        Example:
            ```python
            from agenticflow.memory import Memory

            flow = Flow().with_memory(Memory())
            ```
        """
        from agenticflow.memory import Memory

        self._shared_memory = memory or Memory()
        return self

    # -------------------------------------------------------------------------
    # Background Tasks API
    # -------------------------------------------------------------------------

    def spawn(self, coro: Awaitable[object]) -> asyncio.Task[object]:
        """Spawn a background task and track it for cleanup.

        Args:
            coro: Coroutine to run in background

        Returns:
            The created asyncio.Task

        Example:
            ```python
            async def background_job():
                await asyncio.sleep(10)
                print("Background job complete")

            flow.spawn(background_job())
            ```
        """
        task = asyncio.create_task(coro)
        self._spawned.add(task)

        def _done(_t: asyncio.Task[object]) -> None:
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

    # -------------------------------------------------------------------------
    # Thread ID Resolution
    # -------------------------------------------------------------------------

    def thread_by_data(self, key: str, *, prefix: str | None = None) -> Self:
        """Derive `thread_id` from `event.data[key]`.

        This is a mid-level UX helper for per-entity memory without lambdas.

        Args:
            key: Key in event.data to use as thread ID
            prefix: Optional prefix to prepend to thread ID

        Returns:
            Self for method chaining

        Example:
            ```python
            flow.thread_by_data("job_id")
            flow.thread_by_data("user_id", prefix="user_")
            ```
        """

        def _thread_id_resolver(
            event: Event, _context: dict[str, object]
        ) -> str | None:
            data = getattr(event, "data", None) or {}
            value = data.get(key)
            if value is None:
                return None
            thread_id = str(value)
            return f"{prefix}{thread_id}" if prefix else thread_id

        self._thread_id_resolver = _thread_id_resolver
        return self

    def unregister(self, reactor_id: str) -> None:
        """Remove a reactor from the flow.

        Args:
            reactor_id: ID of the reactor to remove
        """
        self._reactors.pop(reactor_id, None)
        self._bindings = [b for b in self._bindings if b.reactor_id != reactor_id]

    def _normalize_patterns(
        self, on: str | list[str] | None
    ) -> tuple[frozenset[str], object | None]:
        """Convert on parameter to frozenset of patterns and extract source filter.

        Parses event@source syntax and extracts source filters.
        Supports :group references in pattern syntax.

        Args:
            on: Event pattern(s), optionally with source filter using @ separator

        Returns:
            Tuple of (event_patterns, source_filter)
            - event_patterns: frozenset of event patterns
            - source_filter: SourceFilter if any pattern had source, None otherwise

        Examples:
            >>> flow._normalize_patterns("agent.done@researcher")
            (frozenset({'agent.done'}), SourceFilter(...))

            >>> flow._normalize_patterns(["*.done@agent1", "*.done@agent2"])
            (frozenset({'*.done'}), SourceFilter(...))

            >>> flow._normalize_patterns("task.created")
            (frozenset({'task.created'}), None)

            >>> flow._normalize_patterns("*.done@:analysts")
            (frozenset({'*.done'}), SourceFilter(...))
        """
        from agenticflow.events.patterns import from_source
        from agenticflow.flow.parser import parse_pattern

        if on is None:
            return frozenset(), None

        patterns_list = [on] if isinstance(on, str) else on
        event_patterns = []
        source_filter = None

        for pattern in patterns_list:
            parsed = parse_pattern(pattern)
            event_patterns.append(parsed.event)

            # Combine source filters with OR logic if multiple patterns have sources
            if parsed.source:
                # Pass self for :group support
                new_filter = from_source(parsed.source, flow=self)
                if source_filter is None:
                    source_filter = new_filter
                else:
                    source_filter = source_filter | new_filter

        return frozenset(event_patterns), source_filter

    def register(
        self,
        reactor: Reactor | Agent | Callable[..., object],
        on: str | list[str] | None = None,
        *,
        name: str | None = None,
        priority: int = 0,
        when: Callable[[Event], bool] | None = None,
        after: str | list[str] | None = None,
        emits: str | None = None,
    ) -> str:
        """Register a reactor to respond to events.

        Args:
            reactor: A Reactor instance, Agent, or callable function
            on: Event type(s) to react to. Supports:
                - Simple patterns: "task.created"
                - Wildcards: "task.*"
                - Source filters: "task.created@api", "*.done@agent*"
                - Multiple separators: @, :, ->
                - Lists: ["event1@source1", "event2@source2"]
            name: Optional name/ID for the reactor (auto-generated if None)
            priority: Execution priority (higher values execute first)
            when: Optional condition function for filtering events
            after: Filter to events from specific source(s). Supports:
                - Exact match: after="researcher"
                - Multiple sources: after=["agent1", "agent2"]
                - Wildcard patterns: after="agent*"
                Cannot be used together with `when` parameter or `on` with @source.
            emits: Event type to emit after reactor completes

        Returns:
            The reactor ID

        Example:
            ```python
            # Register an agent
            flow.register(researcher, on="task.created")

            # Register with emit
            flow.register(processor, on="data.ready", emits="data.processed")

            # Register with condition
            flow.register(
                urgent_handler,
                on="task.*",
                when=lambda e: e.data.get("priority") == "high",
            )

            # Register with source filter (after parameter)
            flow.register(
                reviewer,
                on="agent.done",
                after="researcher"  # Only from researcher
            )

            # Register with source filter (pattern syntax)
            flow.register(
                reviewer,
                on="agent.done@researcher"  # Same as above
            )

            # Multiple patterns with sources
            flow.register(
                aggregator,
                on=["*.done@agent1", "*.done@agent2"]
            )

            # Wildcards in both event and source
            flow.register(
                monitor,
                on="*.error@agent*"
            )

            # Register a function
            flow.register(
                lambda event: event.data["value"] * 2,
                on="compute.request",
            )
            ```
        """
        # Track agents in :agents group
        from agenticflow.agent.base import Agent

        if isinstance(reactor, Agent):
            self._source_groups.setdefault("agents", set()).add(reactor.name)

        # Parse patterns to extract source filter from event@source syntax
        patterns, pattern_source_filter = self._normalize_patterns(on)

        # Validate conflicting parameters
        if after is not None and when is not None:
            raise ValueError(
                "Cannot specify both 'after' and 'when' parameters. "
                "Use 'after' for source filtering or 'when' for custom conditions."
            )

        if pattern_source_filter is not None and after is not None:
            raise ValueError(
                "Cannot specify both 'after' parameter and '@source' in pattern. "
                "Use one approach: either 'after=\"source\"' or 'on=\"event@source\"'."
            )

        if pattern_source_filter is not None and when is not None:
            raise ValueError(
                "Cannot specify both 'when' parameter and '@source' in pattern. "
                "Use 'when' for custom conditions or '@source' for source filtering, not both."
            )

        # Build final condition from available sources
        final_condition = when

        # Convert 'after' to condition (pass self for :group support)
        if after is not None:
            from agenticflow.events.patterns import from_source

            final_condition = from_source(after, flow=self)

        # Use pattern-extracted source filter if present
        if pattern_source_filter is not None:
            final_condition = pattern_source_filter

        # Wrap non-reactor types
        wrapped_reactor = self._wrap_reactor(reactor)

        # Generate ID if not provided
        reactor_id = name or getattr(reactor, "name", None) or _generate_id()

        # No initialization needed - StatefulSourceFilter is self-contained!
        # Filters now manage their own state internally.

        # Store reactor
        self._reactors[reactor_id] = wrapped_reactor

        # Create binding if patterns provided
        if patterns:
            binding = ReactorBinding(
                reactor_id=reactor_id,
                patterns=patterns,
                priority=priority,
                condition=final_condition,
                emits=emits,
            )
            self._bindings.append(binding)
            # Sort by priority (descending)
            self._bindings.sort(key=lambda b: b.priority, reverse=True)

        return reactor_id

    def _wrap_reactor(
        self, reactor: Reactor | Agent | Callable[..., object]
    ) -> Reactor:
        """Wrap non-Reactor types into Reactor instances."""
        # Already a Reactor (has handle method)
        if hasattr(reactor, "handle") and callable(reactor.handle):
            return reactor  # type: ignore

        # Agent -> AgentReactor (pass trace_bus for observability)
        if hasattr(reactor, "run") and hasattr(reactor, "name"):
            from agenticflow.agent.reactor import AgentReactor

            # Connect agent to Flow's trace bus for unified observability
            if hasattr(reactor, "trace_bus") and reactor.trace_bus is None:
                reactor.trace_bus = self._trace_bus

            return AgentReactor(reactor)  # type: ignore

        # Callable -> FunctionReactor
        if callable(reactor):
            from agenticflow.reactors.function import FunctionReactor

            return FunctionReactor(reactor)

        raise TypeError(
            f"Cannot wrap {type(reactor).__name__} as a Reactor. "
            "Expected Reactor, Agent, or callable."
        )

    def use(self, middleware: Middleware) -> Flow:
        """Add middleware to the flow.

        Middleware can intercept events, modify reactor execution,
        or add cross-cutting concerns like logging and retry.

        Args:
            middleware: Middleware instance to add

        Returns:
            Self for method chaining

        Example:
            ```python
            from agenticflow.middleware import LoggingMiddleware, RetryMiddleware

            flow = Flow()
            flow.use(LoggingMiddleware())
            flow.use(RetryMiddleware(max_retries=3))
            ```
        """
        self._middleware.append(middleware)
        return self

    def _find_matching_bindings(self, event: Event) -> list[ReactorBinding]:
        """Find all bindings that match an event type."""
        matches = []
        for binding in self._bindings:
            for pattern in binding.patterns:
                if _matches_pattern(event.name, pattern):
                    # Check condition if present
                    if binding.condition is None or binding.condition(event):
                        matches.append(binding)
                        break  # Don't add same binding twice
        return matches

    async def emit(
        self, event: Event | str, data: dict[str, object] | None = None
    ) -> None:
        """Emit an event into the flow.

        Args:
            event: Event instance or event type string
            data: Event data (if event is a string)

        Example:
            ```python
            # Emit an Event instance
            await flow.emit(Event(name="task.created", data={"task": "Do this"}))

            # Emit by name with data
            await flow.emit("task.created", {"task": "Do this"})
            ```
        """
        if isinstance(event, str):
            event = Event(name=event, data=data or {})

        await self._event_queue.put(event)

        if self.config.enable_history:
            self._event_history.append(event)

    async def run(
        self,
        task: str | None = None,
        *,
        initial_event: str | Event | None = "task.created",
        data: dict[str, object] | None = None,
    ) -> FlowResult:
        """Run the flow to completion.

        Args:
            task: The task/prompt to process
            initial_event: Starting event type or Event instance
            data: Additional data for the initial event

        Returns:
            FlowResult with execution details

        Example:
            ```python
            result = await flow.run(
                task="Write a blog post about AI",
                initial_event="task.created",
            )

            if result.success:
                print(result.output)
            else:
                print(f"Error: {result.error}")
            ```
        """
        # Initialize execution state
        self._flow_id = self.config.flow_id or _generate_id()
        self._event_queue = asyncio.Queue()
        self._event_history = []
        self._stop_event = None

        # Build initial event
        event_data = data or {}
        if task:
            event_data["task"] = task

        if isinstance(initial_event, str):
            initial = Event(
                name=initial_event,
                source="flow",
                data=event_data,
            )
        elif initial_event is not None:
            initial = initial_event
        else:
            return FlowResult(
                success=False,
                error="No initial event provided",
                flow_id=self._flow_id,
            )

        # Emit initial event
        await self.emit(initial)

        # Event processing loop
        rounds = 0
        events_processed = 0
        final_output: object | None = None

        try:
            while rounds < self.config.max_rounds:
                rounds += 1

                # Get next event (with timeout)
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=self.config.event_timeout,
                    )
                except TimeoutError:
                    if self.config.stop_on_idle:
                        break
                    continue

                events_processed += 1

                # Check for stop event
                if event.name in self.config.stop_events:
                    self._stop_event = event
                    final_output = event.data.get("output", event.data)
                    break

                # Find matching reactors
                bindings = self._find_matching_bindings(event)

                if not bindings and self.config.stop_on_idle:
                    # No reactors matched and queue is empty
                    if self._event_queue.empty():
                        break

                # Execute matching reactors
                for binding in bindings:
                    reactor = self._reactors[binding.reactor_id]

                    try:
                        # Execute reactor
                        result_events = await self._execute_reactor(
                            reactor, event, binding
                        )

                        # Emit result events
                        if result_events:
                            if isinstance(result_events, Event):
                                await self.emit(result_events)
                                final_output = result_events.data.get(
                                    "output", result_events.data
                                )
                            elif isinstance(result_events, list):
                                for e in result_events:
                                    await self.emit(e)
                                    final_output = e.data.get("output", e.data)

                        # Auto-emit if configured
                        if binding.emits and result_events:
                            out = (
                                result_events
                                if isinstance(result_events, Event)
                                else result_events[-1]
                            )
                            emit_event = Event(
                                name=binding.emits,
                                source=binding.reactor_id,
                                data=out.data
                                if isinstance(out, Event)
                                else {"result": out},
                                correlation_id=event.correlation_id,
                            )
                            await self.emit(emit_event)

                    except Exception as e:
                        if self.config.error_policy == "fail_fast":
                            return FlowResult(
                                success=False,
                                error=str(e),
                                events_processed=events_processed,
                                event_history=self._event_history
                                if self.config.enable_history
                                else [],
                                flow_id=self._flow_id,
                            )
                        # continue or retry handled by middleware
                        pass

            return FlowResult(
                success=True,
                output=final_output,
                events_processed=events_processed,
                event_history=self._event_history if self.config.enable_history else [],
                final_event=self._stop_event,
                flow_id=self._flow_id,
            )

        except Exception as e:
            return FlowResult(
                success=False,
                error=str(e),
                events_processed=events_processed,
                event_history=self._event_history if self.config.enable_history else [],
                flow_id=self._flow_id,
            )

    async def _execute_reactor(
        self,
        reactor: Reactor,
        event: Event,
        binding: ReactorBinding,
    ) -> Event | list[Event] | None:
        """Execute a reactor with middleware chain."""
        # Apply middleware (pre-processing)
        for mw in self._middleware:
            if hasattr(mw, "before"):
                event = await mw.before(event, reactor) or event

        # Execute reactor (handle takes event and context)
        from agenticflow.flow.context import FlowContext

        # Extract original task from first event's data
        original_task = None
        if self._event_history:
            first_event = self._event_history[0]
            original_task = first_event.data.get("task")

        ctx = FlowContext(
            flow_id=self._flow_id,
            event=event,
            history=list(self._event_history),
            original_task=original_task,
        )
        result = await reactor.handle(event, ctx)

        # Apply middleware (post-processing)
        for mw in reversed(self._middleware):
            if hasattr(mw, "after"):
                result = await mw.after(result, event, reactor) or result

        return result

    async def stream(
        self,
        task: str | None = None,
        *,
        initial_event: str | Event | None = "task.created",
        data: dict[str, object] | None = None,
    ) -> AsyncIterator[Event]:
        """Stream events as they occur during flow execution.

        Args:
            task: The task/prompt to process
            initial_event: Starting event type or Event instance
            data: Additional data for the initial event

        Yields:
            Events as they are processed

        Example:
            ```python
            async for event in flow.stream("Process this"):
                print(f"Event: {event.name}")
                if event.name == "agent.chunk":
                    print(event.data["content"], end="", flush=True)
            ```
        """
        # Initialize execution state
        self._flow_id = self.config.flow_id or _generate_id()
        self._event_queue = asyncio.Queue()
        self._event_history = []

        # Build initial event
        event_data = data or {}
        if task:
            event_data["task"] = task

        if isinstance(initial_event, str):
            initial = Event(name=initial_event, source="flow", data=event_data)
        elif initial_event:
            initial = initial_event
        else:
            return

        await self.emit(initial)
        yield initial

        rounds = 0
        while rounds < self.config.max_rounds:
            rounds += 1

            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=self.config.event_timeout,
                )
            except TimeoutError:
                if self.config.stop_on_idle:
                    break
                continue

            yield event

            if event.name in self.config.stop_events:
                break

            bindings = self._find_matching_bindings(event)

            for binding in bindings:
                reactor = self._reactors[binding.reactor_id]

                try:
                    result_events = await self._execute_reactor(reactor, event, binding)

                    if result_events:
                        if isinstance(result_events, Event):
                            await self.emit(result_events)
                        elif isinstance(result_events, list):
                            for e in result_events:
                                await self.emit(e)

                except Exception as e:
                    if self.config.error_policy == "fail_fast":
                        error_event = Event.error(source="flow", error=str(e))
                        yield error_event
                        return

    def clone(self) -> Flow:
        """Create a copy of this flow with the same configuration.

        Returns:
            New Flow instance with same config and bindings
        """
        new_flow = Flow(
            config=self.config,
            observer=self._observer,
            checkpointer=self._checkpointer,
            thread_id_resolver=self._thread_id_resolver,
        )
        new_flow._reactors = dict(self._reactors)
        new_flow._bindings = list(self._bindings)
        new_flow._middleware = list(self._middleware)
        new_flow._skills_registry = dict(self._skills_registry)
        new_flow._shared_memory = self._shared_memory
        return new_flow

    # -------------------------------------------------------------------------
    # Checkpointing API
    # -------------------------------------------------------------------------

    async def resume(
        self,
        state: FlowState,
        *,
        context: dict[str, object] | None = None,
    ) -> FlowResult:
        """Resume a flow from a checkpoint.

        Args:
            state: The flow state to resume from
            context: Optional context override (uses checkpoint context if not provided)

        Returns:
            FlowResult with output and execution details

        Example:
            ```python
            # After crash, resume from last checkpoint
            state = await checkpointer.load_latest("my-flow-id")
            if state:
                result = await flow.resume(state)
            ```
        """

        # Restore flow ID and state
        self._flow_id = state.flow_id
        self._last_checkpoint_id = state.checkpoint_id
        self._event_queue = asyncio.Queue()
        self._event_history = []
        self._stop_event = None

        # Restore pending events
        for event_dict in state.pending_events:
            event = Event(
                name=event_dict.get("name", ""),
                data=event_dict.get("data", {}),
            )
            await self._event_queue.put(event)

        # Use provided context or restore from state
        context if context is not None else dict(state.context)

        # Resume from last round
        rounds = state.round
        events_processed = state.events_processed
        final_output: object | None = state.last_output

        try:
            while rounds < self.config.max_rounds:
                rounds += 1

                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=self.config.event_timeout,
                    )
                except TimeoutError:
                    if self.config.stop_on_idle:
                        break
                    continue

                events_processed += 1

                if event.name in self.config.stop_events:
                    self._stop_event = event
                    final_output = event.data.get("output", event.data)
                    break

                bindings = self._find_matching_bindings(event)

                if not bindings and self.config.stop_on_idle:
                    if self._event_queue.empty():
                        break

                for binding in bindings:
                    reactor = self._reactors[binding.reactor_id]

                    try:
                        result_events = await self._execute_reactor(
                            reactor, event, binding
                        )

                        if result_events:
                            if isinstance(result_events, Event):
                                await self.emit(result_events)
                                final_output = result_events.data.get(
                                    "output", result_events.data
                                )
                            elif isinstance(result_events, list):
                                for e in result_events:
                                    await self.emit(e)
                                    final_output = e.data.get("output", e.data)

                        if binding.emits and result_events:
                            out = (
                                result_events
                                if isinstance(result_events, Event)
                                else result_events[-1]
                            )
                            emit_event = Event(
                                name=binding.emits,
                                source=binding.reactor_id,
                                data=out.data
                                if isinstance(out, Event)
                                else {"result": out},
                                correlation_id=event.correlation_id,
                            )
                            await self.emit(emit_event)

                    except Exception as e:
                        if self.config.error_policy == "fail_fast":
                            return FlowResult(
                                success=False,
                                error=str(e),
                                events_processed=events_processed,
                                event_history=self._event_history
                                if self.config.enable_history
                                else [],
                                flow_id=self._flow_id,
                            )

            return FlowResult(
                success=True,
                output=final_output,
                events_processed=events_processed,
                event_history=self._event_history if self.config.enable_history else [],
                final_event=self._stop_event,
                flow_id=self._flow_id,
            )

        except Exception as e:
            return FlowResult(
                success=False,
                error=str(e),
                events_processed=events_processed,
                event_history=self._event_history if self.config.enable_history else [],
                flow_id=self._flow_id,
            )
        finally:
            await self.cancel_spawned()

    async def run_streaming(
        self,
        task: str,
        *,
        initial_event: str = "task.created",
        initial_data: dict[str, object] | None = None,
        context: dict[str, object] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute the flow with streaming output.

        Similar to run(), but yields StreamChunk objects as agents
        process events, providing real-time token-by-token output.

        Args:
            task: The task/prompt to execute
            initial_event: Event type to emit at start
            initial_data: Additional data for initial event
            context: Shared context available to all agents

        Yields:
            StreamChunk: Streaming chunks from agent executions

        Example:
            ```python
            async for chunk in flow.run_streaming("Research quantum computing"):
                print(f"[{chunk.agent_name}] {chunk.content}", end="", flush=True)
                if chunk.is_final:
                    print()  # Newline after agent completes
            ```
        """
        from agenticflow.flow.streaming import StreamChunk

        # Initialize flow state
        self._flow_id = self.config.flow_id or _generate_id()
        self._last_checkpoint_id = None
        self._event_queue = asyncio.Queue()
        self._event_history = []
        self._stop_event = None

        run_context = context or {}

        # Build and emit initial event
        event_data = {"task": task, **(initial_data or {})}
        initial = Event(name=initial_event, source="flow", data=event_data)
        await self.emit(initial)

        rounds = 0

        try:
            while rounds < self.config.max_rounds:
                rounds += 1

                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=self.config.event_timeout,
                    )
                except TimeoutError:
                    if self.config.stop_on_idle:
                        break
                    continue

                if event.name in self.config.stop_events:
                    self._stop_event = event
                    break

                bindings = self._find_matching_bindings(event)

                for binding in bindings:
                    reactor = self._reactors[binding.reactor_id]

                    # Check if reactor supports streaming
                    if hasattr(reactor, "handle_streaming"):
                        async for chunk in reactor.handle_streaming(event, run_context):
                            yield chunk
                    else:
                        # Fall back to regular execution
                        try:
                            result_events = await self._execute_reactor(
                                reactor, event, binding
                            )

                            if result_events:
                                if isinstance(result_events, Event):
                                    await self.emit(result_events)
                                    # Emit a final chunk with the result
                                    yield StreamChunk(
                                        agent_name=binding.reactor_id,
                                        event_id=event.id,
                                        event_name=event.name,
                                        content=str(
                                            result_events.data.get("output", "")
                                        ),
                                        delta=str(result_events.data.get("output", "")),
                                        is_final=True,
                                        finish_reason="stop",
                                    )
                                elif isinstance(result_events, list):
                                    for e in result_events:
                                        await self.emit(e)

                            if binding.emits and result_events:
                                out = (
                                    result_events
                                    if isinstance(result_events, Event)
                                    else result_events[-1]
                                )
                                emit_event = Event(
                                    name=binding.emits,
                                    source=binding.reactor_id,
                                    data=out.data
                                    if isinstance(out, Event)
                                    else {"result": out},
                                    correlation_id=event.correlation_id,
                                )
                                await self.emit(emit_event)

                        except Exception as e:
                            yield StreamChunk(
                                agent_name=binding.reactor_id,
                                event_id=event.id,
                                event_name=event.name,
                                content=f"[Error: {e!s}]",
                                delta=f"[Error: {e!s}]",
                                is_final=True,
                                finish_reason="error",
                                metadata={"error": str(e)},
                            )

                if self._event_queue.empty() and self.config.stop_on_idle:
                    break

        finally:
            await self.cancel_spawned()

    def __repr__(self) -> str:
        return (
            f"Flow(reactors={len(self._reactors)}, "
            f"bindings={len(self._bindings)}, "
            f"skills={len(self._skills_registry)}, "
            f"config={self.config})"
        )
