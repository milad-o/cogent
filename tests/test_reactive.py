"""Tests for reactive flow orchestration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agenticflow.reactive import (
    # High-level
    chain,
    fanout,
    route,
    # Mid-level
    Chain,
    FanIn,
    FanOut,
    Router,
    Saga,
    # Low-level
    EventFlow,
    EventFlowConfig,
    EventFlowResult,
    Trigger,
    AgentTriggerConfig,
    react_to,
    when,
)
from agenticflow.observability.trace_record import TraceType
from agenticflow.events.event import Event


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent():
    """Create a mock agent."""

    def _create(name: str = "test_agent") -> MagicMock:
        agent = MagicMock()
        agent.name = name
        agent.run = AsyncMock(
            return_value=MagicMock(output=f"Output from {name}")
        )
        return agent

    return _create


@pytest.fixture
def researcher(mock_agent):
    return mock_agent("researcher")


@pytest.fixture
def writer(mock_agent):
    return mock_agent("writer")


@pytest.fixture
def editor(mock_agent):
    return mock_agent("editor")


# =============================================================================
# Tests: Trigger and on() builder
# =============================================================================


class TestTrigger:
    """Tests for Trigger matching."""

    def test_matches_event_type(self) -> None:
        """Trigger matches TraceType."""
        trigger = Trigger(on=TraceType.TASK_CREATED)
        event = Event(name="task.created", data={})

        assert trigger.matches(event)

    def test_matches_string_pattern(self) -> None:
        """Trigger matches string pattern."""
        trigger = Trigger(on="task.created")
        event = Event(name="task.created", data={})

        assert trigger.matches(event)

    def test_matches_wildcard_pattern(self) -> None:
        """Trigger matches wildcard pattern."""
        trigger = Trigger(on="task.*")
        event = Event(name="task.created", data={})

        assert trigger.matches(event)

    def test_condition_filters(self) -> None:
        """Trigger respects condition."""
        trigger = Trigger(
            on=TraceType.TASK_CREATED,
            condition=lambda e: e.data.get("priority") == "high",
        )

        high_priority = Event(name="task.created", data={"priority": "high"})
        low_priority = Event(name="task.created", data={"priority": "low"})

        assert trigger.matches(high_priority)
        assert not trigger.matches(low_priority)


class TestOnBuilder:
    """Tests for on() fluent builder."""

    def test_basic_on(self) -> None:
        """on() creates trigger builder."""
        builder = react_to("task.created")
        trigger = builder.build()

        assert isinstance(trigger, Trigger)

    def test_on_with_when(self) -> None:
        """on().when() adds condition."""
        trigger = react_to("task.created").when(lambda e: True).build()

        assert trigger.condition is not None

    def test_on_with_emits(self) -> None:
        """on().emits() sets emission."""
        trigger = react_to("task.created").emits("task.done").build()

        assert trigger.emits == "task.done"

    def test_on_with_priority(self) -> None:
        """on().with_priority() sets priority."""
        trigger = react_to("task.created").with_priority(10).build()

        assert trigger.priority == 10

    def test_chained_builder(self) -> None:
        """Builder methods can be chained."""
        trigger = (
            react_to("task.created")
            .when(lambda e: True)
            .emits("done")
            .with_priority(5)
            .build()
        )

        assert trigger.condition is not None
        assert trigger.emits == "done"
        assert trigger.priority == 5


# =============================================================================
# Tests: EventFlow
# =============================================================================


class TestEventFlow:
    """Tests for EventFlow orchestrator."""

    @pytest.mark.asyncio
    async def test_register_agent(self, mock_agent) -> None:
        """Can register agent with triggers."""
        agent = mock_agent("test")
        flow = EventFlow()

        flow.register(agent, [react_to("task.created")])

        assert "test" in flow.agents

    @pytest.mark.asyncio
    async def test_unregister_agent(self, mock_agent) -> None:
        """Can unregister agent."""
        agent = mock_agent("test")
        flow = EventFlow()
        flow.register(agent, [react_to("task.created")])

        flow.unregister("test")

        assert "test" not in flow.agents

    @pytest.mark.asyncio
    async def test_run_triggers_agent(self, mock_agent) -> None:
        """Run triggers matching agent."""
        agent = mock_agent("test")
        flow = EventFlow()
        flow.register(agent, [react_to("task.created")])

        result = await flow.run("Do something", initial_event="task.created")

        agent.run.assert_called_once()
        assert result.events_processed >= 1

    @pytest.mark.asyncio
    async def test_chain_triggers(self, researcher, writer) -> None:
        """Agents chain via completion events."""
        flow = EventFlow()
        flow.register(researcher, [react_to("task.created")])
        flow.register(writer, [react_to("researcher.completed")])

        result = await flow.run("Research and write", initial_event="task.created")

        researcher.run.assert_called_once()
        writer.run.assert_called_once()
        assert len(result.reactions) == 2

    @pytest.mark.asyncio
    async def test_parallel_triggers(self, mock_agent) -> None:
        """Multiple agents can trigger on same event."""
        agent1 = mock_agent("agent1")
        agent2 = mock_agent("agent2")

        flow = EventFlow()
        flow.register(agent1, [react_to("task.created")])
        flow.register(agent2, [react_to("task.created")])

        result = await flow.run("Parallel task", initial_event="task.created")

        agent1.run.assert_called_once()
        agent2.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_conditional_trigger(self, mock_agent) -> None:
        """Only matching conditions trigger."""
        agent = mock_agent("conditional")

        flow = EventFlow()
        flow.register(
            agent,
            [react_to("task.created").when(lambda e: "special" in e.data.get("task", ""))],
        )

        # Should not trigger
        result1 = await flow.run("Normal task", initial_event="task.created")
        assert len(result1.reactions) == 0

        # Should trigger
        result2 = await flow.run("This is special", initial_event="task.created")
        assert len(result2.reactions) == 1

    @pytest.mark.asyncio
    async def test_max_rounds_limit(self, mock_agent) -> None:
        """Flow respects max_rounds config."""
        agent = mock_agent("looper")
        # Make agent trigger itself
        config = EventFlowConfig(max_rounds=3)
        flow = EventFlow(config=config)
        flow.register(agent, [react_to("task.created"), react_to("looper.completed")])

        result = await flow.run("Loop task", initial_event="task.created")

        # Should stop after max_rounds
        assert len(result.reactions) <= 3

    @pytest.mark.asyncio
    async def test_result_contains_output(self, mock_agent) -> None:
        """Result contains agent output."""
        agent = mock_agent("test")
        flow = EventFlow()
        flow.register(agent, [react_to("task.created")])

        result = await flow.run("Task", initial_event="task.created")

        assert result.output == "Output from test"


# =============================================================================
# Tests: Chain Pattern
# =============================================================================


class TestChainPattern:
    """Tests for chain() and Chain."""

    @pytest.mark.asyncio
    async def test_chain_function(self, researcher, writer, editor) -> None:
        """chain() creates runnable chain."""
        result = await chain(researcher, writer, editor).run("Write something")

        researcher.run.assert_called_once()
        writer.run.assert_called_once()
        editor.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_chain_class(self, researcher, writer) -> None:
        """Chain class is runnable."""
        c = Chain([researcher, writer])
        result = await c.run("Task")

        researcher.run.assert_called_once()
        writer.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_chain_with_config(self, researcher, writer) -> None:
        """Chain accepts config."""
        config = EventFlowConfig(max_rounds=10)
        c = Chain([researcher, writer], config=config)

        result = await c.run("Task")

        assert result is not None

    @pytest.mark.asyncio
    async def test_chain_execution_order(self, mock_agent) -> None:
        """Chain executes in order via reactions."""
        # Create fresh agents for this test
        r = mock_agent("researcher")
        w = mock_agent("writer")
        e = mock_agent("editor")

        result = await chain(r, w, e).run("Task")

        # Verify execution order via reactions (which track agent calls)
        assert len(result.reactions) == 3
        assert result.reactions[0].agent_name == "researcher"
        assert result.reactions[1].agent_name == "writer"
        assert result.reactions[2].agent_name == "editor"


# =============================================================================
# Tests: FanOut Pattern
# =============================================================================


class TestFanOutPattern:
    """Tests for fanout() and FanOut."""

    @pytest.mark.asyncio
    async def test_fanout_function(self, mock_agent) -> None:
        """fanout() runs workers in parallel."""
        w1 = mock_agent("worker1")
        w2 = mock_agent("worker2")
        w3 = mock_agent("worker3")

        result = await fanout(w1, w2, w3).run("Parallel task")

        w1.run.assert_called_once()
        w2.run.assert_called_once()
        w3.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_fanout_with_aggregator(self, mock_agent) -> None:
        """fanout() with then= runs aggregator."""
        w1 = mock_agent("worker1")
        w2 = mock_agent("worker2")
        agg = mock_agent("aggregator")

        result = await fanout(w1, w2, then=agg).run("Task")

        w1.run.assert_called_once()
        w2.run.assert_called_once()
        agg.run.assert_called()  # Called multiple times (once per worker completion)

    @pytest.mark.asyncio
    async def test_fanout_class(self, mock_agent) -> None:
        """FanOut class is runnable."""
        w1 = mock_agent("w1")
        w2 = mock_agent("w2")

        f = FanOut([w1, w2])
        result = await f.run("Task")

        w1.run.assert_called_once()
        w2.run.assert_called_once()


# =============================================================================
# Tests: Router Pattern
# =============================================================================


class TestRouterPattern:
    """Tests for route() and Router."""

    @pytest.mark.asyncio
    async def test_route_function(self, mock_agent) -> None:
        """route() dispatches based on condition."""
        coder = mock_agent("coder")
        writer = mock_agent("writer")

        r = route(
            (lambda t: "code" in t, coder),
            (lambda t: "write" in t, writer),
        )

        await r.run("Write some code")

        coder.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_fallback(self, mock_agent) -> None:
        """route() uses fallback for no match."""
        coder = mock_agent("coder")
        fallback = mock_agent("fallback")

        r = route(
            (lambda t: "code" in t, coder),
            fallback,
        )

        await r.run("Something else")

        fallback.run.assert_called()

    @pytest.mark.asyncio
    async def test_router_class(self, mock_agent) -> None:
        """Router class is runnable."""
        agent = mock_agent("test")

        r = Router([agent])
        result = await r.run("Task")

        agent.run.assert_called_once()


# =============================================================================
# Tests: FanIn Pattern
# =============================================================================


class TestFanInPattern:
    """Tests for FanIn."""

    @pytest.mark.asyncio
    async def test_fanin_waits_for_all(self, mock_agent) -> None:
        """FanIn waits for all events."""
        collector = mock_agent("collector")

        f = FanIn(
            wait_for=["event1", "event2"],
            agent=collector,
        )

        # Build flow and check triggers
        flow = f._build()
        assert "collector" in flow.agents


# =============================================================================
# Tests: Saga Pattern
# =============================================================================


class TestSagaPattern:
    """Tests for Saga."""

    @pytest.mark.asyncio
    async def test_saga_runs_steps(self, mock_agent) -> None:
        """Saga runs forward steps."""
        step1 = mock_agent("step1")
        step2 = mock_agent("step2")

        s = Saga([(step1, None), (step2, None)])
        result = await s.run("Process order")

        step1.run.assert_called_once()
        step2.run.assert_called_once()

    def test_saga_registers_compensation(self, mock_agent) -> None:
        """Saga registers compensation triggers."""
        forward1 = mock_agent("forward1")
        compensate1 = mock_agent("compensate1")
        forward2 = mock_agent("forward2")

        s = Saga([(forward1, compensate1), (forward2, None)])
        flow = s._build()

        # Compensation should be registered
        assert "compensate1" in flow.agents


# =============================================================================
# Tests: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for reactive patterns."""

    @pytest.mark.asyncio
    async def test_pattern_produces_result(self, researcher, writer) -> None:
        """Patterns produce EventFlowResult."""
        result = await chain(researcher, writer).run("Task")

        assert isinstance(result, EventFlowResult)
        assert result.output is not None
        assert result.events_processed > 0
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_reactions_tracked(self, researcher, writer) -> None:
        """Reactions are tracked in result."""
        result = await chain(researcher, writer).run("Task")

        assert len(result.reactions) == 2
        assert result.reactions[0].agent_name == "researcher"
        assert result.reactions[1].agent_name == "writer"

    @pytest.mark.asyncio
    async def test_event_history_recorded(self, researcher) -> None:
        """Event history is recorded when enabled."""
        config = EventFlowConfig(enable_history=True)
        c = Chain([researcher], config=config)

        result = await c.run("Task")

        assert len(result.event_history) > 0


class TestObservability:
    """Tests for reactive observability integration."""

    @pytest.mark.asyncio
    async def test_observer_receives_events(self, researcher, writer) -> None:
        """Observer receives reactive flow events."""
        from agenticflow.reactive import Observer

        observer = Observer.off()  # Suppress output
        result = await chain(researcher, writer, observer=observer).run("Task")

        metrics = observer.metrics()
        assert metrics["total_events"] > 0
        # Should have reactive channel events
        assert "reactive" in metrics.get("by_channel", {})

    @pytest.mark.asyncio
    async def test_flow_started_event(self, researcher) -> None:
        """REACTIVE_FLOW_STARTED event is emitted."""
        from agenticflow.reactive import Observer

        observer = Observer.off()
        await chain(researcher, observer=observer).run("Task")

        # Check that flow started was recorded
        events = [e.event for e in observer._events]
        flow_started = [e for e in events if e.type == TraceType.REACTIVE_FLOW_STARTED]
        assert len(flow_started) == 1
        assert "task" in flow_started[0].data

    @pytest.mark.asyncio
    async def test_agent_triggered_event(self, researcher) -> None:
        """REACTIVE_AGENT_TRIGGERED event is emitted."""
        from agenticflow.reactive import Observer

        observer = Observer.off()
        await chain(researcher, observer=observer).run("Task")

        events = [e.event for e in observer._events]
        triggered = [e for e in events if e.type == TraceType.REACTIVE_AGENT_TRIGGERED]
        assert len(triggered) == 1
        assert triggered[0].data["agent"] == "researcher"

    @pytest.mark.asyncio
    async def test_flow_completed_event(self, researcher) -> None:
        """REACTIVE_FLOW_COMPLETED event is emitted."""
        from agenticflow.reactive import Observer

        observer = Observer.off()
        await chain(researcher, observer=observer).run("Task")

        events = [e.event for e in observer._events]
        completed = [e for e in events if e.type == TraceType.REACTIVE_FLOW_COMPLETED]
        assert len(completed) == 1
        assert "execution_time_ms" in completed[0].data
