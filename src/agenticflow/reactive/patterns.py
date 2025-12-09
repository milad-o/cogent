"""Reactive orchestration patterns.

Simple, clean API for common multi-agent coordination patterns.

Example:
    ```python
    from agenticflow.reactive import chain, fanout, route

    # Chain: sequential execution
    result = await chain(researcher, writer, editor).run("Write about AI")

    # Fan-out: parallel with optional aggregator
    result = await fanout(search1, search2, search3, then=aggregator).run("Find info")

    # Route: conditional dispatch
    result = await route(
        (lambda t: "code" in t, coder),
        (lambda t: "write" in t, writer),
        analyst,  # default
    ).run("Analyze the data")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from agenticflow.reactive.core import AgentTriggerConfig, TriggerCondition, react_to
from agenticflow.flow.reactive import ReactiveFlow, ReactiveFlowConfig, ReactiveFlowResult
from agenticflow.observability.event import Event
from agenticflow.observability.observer import Observer

# Backward compatibility aliases
EventFlow = ReactiveFlow
EventFlowConfig = ReactiveFlowConfig
EventFlowResult = ReactiveFlowResult

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent


# =============================================================================
# HIGH-LEVEL API (Functions that return runnable patterns)
# =============================================================================


def chain(
    *agents: Agent,
    start_event: str = "task.created",
    observer: Observer | None = None,
) -> Chain:
    """
    Create a sequential chain: A → B → C.

    Args:
        *agents: Agents to execute in order
        start_event: Event that starts the chain
        observer: Optional observer for monitoring

    Returns:
        Runnable Chain pattern

    Example:
        ```python
        result = await chain(researcher, writer, editor).run("Write about AI")

        # With observability
        observer = Observer.verbose()
        result = await chain(a, b, c, observer=observer).run("task")
        ```
    """
    return Chain(list(agents), start_event=start_event, observer=observer)


def fanout(
    *workers: Agent,
    then: Agent | None = None,
    trigger: str = "task.created",
    observer: Observer | None = None,
) -> FanOut:
    """
    Create a fan-out pattern: parallel workers with optional aggregator.

    Args:
        *workers: Agents to run in parallel
        then: Optional agent to aggregate results
        trigger: Event that triggers all workers
        observer: Optional observer for monitoring

    Returns:
        Runnable FanOut pattern

    Example:
        ```python
        result = await fanout(search1, search2, then=aggregator).run("Find info")
        ```
    """
    return FanOut(list(workers), then=then, trigger=trigger, observer=observer)


def route(
    *routes: Agent | tuple[Callable[[str], bool] | str | list[str], Agent],
    trigger: str = "task.created",
    observer: Observer | None = None,
) -> Router:
    """
    Create a router: dispatch to agent based on task content.

    Args:
        *routes: Routing rules in order of priority. Each can be:
                 - `agent` - Fallback (always matches)
                 - `(keywords, agent)` - Match if any keyword in task
                 - `(callable, agent)` - Custom condition function
                 
                 Keywords can be:
                 - String with `|` separator: `"code|math|calculate"`
                 - List of strings: `["code", "math", "calculate"]`
                 
        trigger: Event that triggers routing
        observer: Optional observer for monitoring

    Returns:
        Runnable Router pattern

    Example:
        ```python
        # Simple keyword routing (recommended)
        result = await route(
            ("analyze|data|chart", analyst),
            ("code|math|calculate", coder),
            ("notify|send|alert", communicator),
            general,  # fallback
        ).run("Analyze the sales data")
        
        # With list syntax
        result = await route(
            (["code", "python", "function"], coder),
            (["write", "draft", "compose"], writer),
        ).run("Write a Python function")
        
        # Custom condition (advanced)
        result = await route(
            (lambda t: len(t) > 100, deep_analyzer),
            (lambda t: "urgent" in t.lower(), fast_responder),
            general,
        ).run(task)
        ```
    """
    return Router(list(routes), trigger=trigger, observer=observer)


# =============================================================================
# PATTERN CLASSES (Mid-level API with .run())
# =============================================================================


@dataclass
class Chain:
    """
    Sequential chain pattern: A → B → C.

    Each agent waits for the previous to complete.
    """

    agents: list[Agent]
    start_event: str = "task.created"
    config: EventFlowConfig | None = None
    observer: Observer | None = None

    async def run(
        self,
        task: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> EventFlowResult:
        """Execute the chain."""
        flow = self._build()
        return await flow.run(task, initial_event=self.start_event, context=context)

    def _build(self) -> EventFlow:
        """Build the underlying EventFlow."""
        flow = EventFlow(config=self.config, observer=self.observer)

        for i, agent in enumerate(self.agents):
            if i == 0:
                trigger_event = self.start_event
            else:
                trigger_event = f"{self.agents[i - 1].name}.completed"

            flow.register(agent, [react_to(trigger_event)])

        return flow


@dataclass
class FanOut:
    """
    Fan-out pattern: parallel workers with optional aggregator.

    All workers run concurrently on the same trigger.
    """

    workers: list[Agent]
    then: Agent | None = None
    trigger: str = "task.created"
    config: EventFlowConfig | None = None
    observer: Observer | None = None

    async def run(
        self,
        task: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> EventFlowResult:
        """Execute the fan-out."""
        flow = self._build()
        return await flow.run(task, initial_event=self.trigger, context=context)

    def _build(self) -> EventFlow:
        """Build the underlying EventFlow."""
        flow = EventFlow(config=self.config, observer=self.observer)

        # All workers trigger on same event
        for worker in self.workers:
            flow.register(worker, [react_to(self.trigger)])

        # Aggregator triggers on any worker completion
        if self.then:
            worker_events = [f"{w.name}.completed" for w in self.workers]
            flow.register(self.then, [react_to(e) for e in worker_events])

        return flow


@dataclass
class FanIn:
    """
    Fan-in pattern: wait for ALL events before triggering.

    Agent only runs after receiving all required events.
    """

    wait_for: list[str]
    agent: Agent
    emit: str | None = None
    config: EventFlowConfig | None = None
    observer: Observer | None = None

    async def run(
        self,
        task: str,
        *,
        initial_event: str = "task.created",
        context: dict[str, Any] | None = None,
    ) -> EventFlowResult:
        """Execute the fan-in."""
        flow = self._build()
        return await flow.run(task, initial_event=initial_event, context=context)

    def _build(self) -> EventFlow:
        """Build the underlying EventFlow."""
        flow = EventFlow(config=self.config, observer=self.observer)

        received: set[str] = set()

        def make_condition(event_name: str) -> TriggerCondition:
            def condition(event: Event) -> bool:
                received.add(event_name)
                return received >= set(self.wait_for)
            return condition

        triggers = []
        for event_name in self.wait_for:
            trigger = react_to(event_name).when(make_condition(event_name))
            if self.emit:
                trigger = trigger.emits(self.emit)
            triggers.append(trigger)

        flow.register(self.agent, triggers)
        return flow


@dataclass
class Router:
    """
    Router pattern: dispatch to agent based on conditions.

    Routes are evaluated in order; first match wins.
    Supports keyword strings, lists, or custom callables.
    """

    routes: list[Agent | tuple[Callable[[str], bool] | str | list[str], Agent]]
    trigger: str = "task.created"
    config: EventFlowConfig | None = None
    observer: Observer | None = None

    async def run(
        self,
        task: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> EventFlowResult:
        """Execute the router."""
        flow = self._build(task)
        return await flow.run(task, initial_event=self.trigger, context=context)

    def _build(self, task: str = "") -> EventFlow:
        """Build the underlying EventFlow."""
        flow = EventFlow(config=self.config, observer=self.observer)

        for route_entry in self.routes:
            if isinstance(route_entry, tuple):
                condition, agent = route_entry
                condition_fn = self._normalize_condition(condition)
                
                # Convert task-based condition to event-based
                def make_event_condition(fn: Callable[[str], bool]) -> TriggerCondition:
                    def cond(event: Event) -> bool:
                        task_text = event.data.get("task", "")
                        return fn(task_text)
                    return cond
                flow.register(agent, [react_to(self.trigger).when(make_event_condition(condition_fn))])
            else:
                # Plain agent = fallback (always matches)
                flow.register(route_entry, [react_to(self.trigger)])

        return flow
    
    def _normalize_condition(
        self, 
        condition: Callable[[str], bool] | str | list[str],
    ) -> Callable[[str], bool]:
        """Convert keywords or callable to a condition function."""
        if callable(condition):
            return condition
        
        # Convert string or list to keyword matcher
        if isinstance(condition, str):
            keywords = [k.strip().lower() for k in condition.split("|")]
        else:
            keywords = [k.lower() for k in condition]
        
        def keyword_matcher(task: str) -> bool:
            task_lower = task.lower()
            return any(kw in task_lower for kw in keywords)
        
        return keyword_matcher


@dataclass
class Saga:
    """
    Saga pattern: steps with compensation on failure.

    If a step fails, all previous compensations run in reverse.
    """

    steps: list[tuple[Agent, Agent | None]]  # (forward, compensate)
    config: EventFlowConfig | None = None
    observer: Observer | None = None

    async def run(
        self,
        task: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> EventFlowResult:
        """Execute the saga."""
        flow = self._build()
        return await flow.run(task, initial_event="saga.started", context=context)

    def _build(self) -> EventFlow:
        """Build the underlying EventFlow."""
        flow = EventFlow(config=self.config, observer=self.observer)

        for i, (forward, compensate) in enumerate(self.steps):
            if i == 0:
                trigger_event = "saga.started"
            else:
                trigger_event = f"{self.steps[i - 1][0].name}.completed"

            flow.register(forward, [react_to(trigger_event)])

            if compensate:
                error_events = [
                    f"{self.steps[j][0].name}.error"
                    for j in range(i + 1, len(self.steps))
                ]
                if error_events:
                    flow.register(compensate, [react_to(e) for e in error_events])

        return flow


# =============================================================================
# LEGACY COMPATIBILITY (Deprecated - use new API)
# =============================================================================


@dataclass(kw_only=True)
class ChainPattern:
    """Deprecated: Use `chain()` or `Chain` instead."""

    agents: list[Agent]
    start_event: str = "task.created"
    config: EventFlowConfig | None = None

    def build(self) -> EventFlow:
        return Chain(self.agents, self.start_event, self.config)._build()


@dataclass(kw_only=True)
class FanOutPattern:
    """Deprecated: Use `fanout()` or `FanOut` instead."""

    trigger_event: str
    workers: list[Agent]
    collector: Agent | None = None
    collect_event: str = "fanout.all_completed"
    config: EventFlowConfig | None = None

    def build(self) -> EventFlow:
        return FanOut(self.workers, self.collector, self.trigger_event, self.config)._build()


@dataclass(kw_only=True)
class FanInPattern:
    """Deprecated: Use `FanIn` instead."""

    wait_for: list[str]
    agent: Agent
    emit_on_complete: str | None = None
    timeout: float = 60.0
    config: EventFlowConfig | None = None

    def build(self) -> EventFlow:
        return FanIn(self.wait_for, self.agent, self.emit_on_complete, self.config)._build()


@dataclass(kw_only=True)
class SagaPattern:
    """Deprecated: Use `Saga` instead."""

    steps: list[SagaStep]
    config: EventFlowConfig | None = None

    def build(self) -> EventFlow:
        converted = [(s.agent, s.compensate) for s in self.steps]
        return Saga(converted, self.config)._build()


@dataclass(frozen=True, slots=True, kw_only=True)
class SagaStep:
    """A step in a saga with optional compensation."""

    agent: Agent
    compensate: Agent | None = None


@dataclass(kw_only=True)
class RouterPattern:
    """Deprecated: Use `route()` or `Router` instead."""

    trigger_event: str
    routes: list[Route]
    config: EventFlowConfig | None = None

    def build(self) -> EventFlow:
        converted = []
        for r in self.routes:
            if r.condition:
                converted.append((lambda e: r.condition(e), r.agent))  # type: ignore
            else:
                converted.append(r.agent)
        return Router(converted, self.trigger_event, self.config)._build()


@dataclass(frozen=True, slots=True, kw_only=True)
class Route:
    """A route in a router pattern."""

    agent: Agent
    condition: TriggerCondition | None = None
    emit_on_complete: str | None = None
