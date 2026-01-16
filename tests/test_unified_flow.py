"""Tests for the unified Flow orchestration engine."""

import pytest
from datetime import datetime

from agenticflow.events import Event
from agenticflow.flow.core import Flow
from agenticflow.flow.config import FlowConfig, FlowResult


class TestEvent:
    """Tests for Event class."""

    def test_event_creation(self) -> None:
        """Test basic event creation."""
        event = Event(name="test.event", source="test", data={"key": "value"})

        assert event.name == "test.event"
        assert event.source == "test"
        assert event.data == {"key": "value"}
        assert event.id is not None
        assert event.timestamp is not None

    def test_event_done_factory(self) -> None:
        """Test Event.done() factory method."""
        event = Event.done(source="test", output="result")

        assert event.name == "agent.done"
        assert event.data["output"] == "result"
        assert event.source == "test"

    def test_event_error_factory(self) -> None:
        """Test Event.error() factory method."""
        event = Event.error(source="test", error="Something failed")

        assert event.name == "agent.error"
        assert event.data["error"] == "Something failed"
        assert event.source == "test"

    def test_event_task_created_factory(self) -> None:
        """Test Event.task_created() factory method."""
        event = Event.task_created("Do something", source="user")

        assert event.name == "task.created"
        assert event.data["task"] == "Do something"
        assert event.source == "user"

    def test_event_serialization(self) -> None:
        """Test event to_dict/from_dict."""
        original = Event(
            name="test.event",
            source="test",
            data={"key": "value"},
        )

        serialized = original.to_dict()
        assert serialized["name"] == "test.event"
        assert serialized["source"] == "test"
        assert serialized["data"] == {"key": "value"}

        restored = Event.from_dict(serialized)
        assert restored.name == original.name
        assert restored.source == original.source
        assert restored.data == original.data
        assert restored.id == original.id

    def test_event_with_correlation(self) -> None:
        """Test with_correlation() method."""
        original = Event(name="test", source="test")
        modified = original.with_correlation("corr-123")

        assert modified.correlation_id == "corr-123"
        assert original.correlation_id != "corr-123"
        assert modified.name == original.name

    def test_event_with_metadata(self) -> None:
        """Test with_metadata() method."""
        original = Event(name="test", source="test")
        modified = original.with_metadata(flow_id="flow-123", extra="data")

        assert modified.metadata["flow_id"] == "flow-123"
        assert modified.metadata["extra"] == "data"


class TestFlowConfig:
    """Tests for FlowConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FlowConfig()

        assert config.max_rounds == 100
        assert config.max_concurrent == 10
        assert config.event_timeout == 30.0
        assert config.enable_history is True
        assert config.stop_on_idle is True
        assert "flow.done" in config.stop_events

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = FlowConfig(
            max_rounds=50,
            max_concurrent=5,
            stop_events=frozenset({"done", "error"}),
        )

        assert config.max_rounds == 50
        assert config.max_concurrent == 5
        assert config.stop_events == frozenset({"done", "error"})

    def test_config_is_immutable(self) -> None:
        """FlowConfig is frozen."""
        config = FlowConfig()
        with pytest.raises(AttributeError):
            config.max_rounds = 999  # type: ignore


class TestFlowResult:
    """Tests for FlowResult."""

    def test_successful_result(self) -> None:
        """Test successful flow result."""
        result = FlowResult(success=True, output="done")

        assert result.success is True
        assert result.output == "done"
        assert bool(result) is True

    def test_failed_result(self) -> None:
        """Test failed flow result."""
        result = FlowResult(success=False, error="Something failed")

        assert result.success is False
        assert result.error == "Something failed"
        assert bool(result) is False

    def test_raise_for_error(self) -> None:
        """Test raise_for_error method."""
        result = FlowResult(success=False, error="Failed")

        with pytest.raises(RuntimeError, match="Failed"):
            result.raise_for_error()


class TestFlow:
    """Tests for the Flow orchestrator."""

    def test_flow_creation(self) -> None:
        """Flow can be created with optional config."""
        flow = Flow()
        assert flow is not None

        flow_with_config = Flow(config=FlowConfig(max_rounds=10))
        assert flow_with_config is not None

    def test_register_function(self) -> None:
        """Flow.register accepts a callable."""
        flow = Flow()

        def handler(event: Event) -> Event | None:
            return None

        flow.register(handler, on="test.event")
        # Should not raise

    def test_register_with_name(self) -> None:
        """Flow.register accepts a name parameter."""
        flow = Flow()

        def handler(event: Event) -> Event | None:
            return None

        flow.register(handler, on="test.event", name="my_handler")
        # Should not raise

    def test_register_with_priority(self) -> None:
        """Flow.register accepts a priority parameter."""
        flow = Flow()

        def handler(event: Event) -> Event | None:
            return None

        flow.register(handler, on="test.event", priority=10)
        # Should not raise

    def test_register_with_emits(self) -> None:
        """Flow.register accepts an emits parameter."""
        flow = Flow()

        def handler(event: Event) -> str:
            return "result"

        flow.register(handler, on="task.created", emits="task.done")
        # Should not raise

    @pytest.mark.asyncio
    async def test_simple_flow_run(self) -> None:
        """Flow.run executes a simple handler."""
        flow = Flow()
        results = []

        def handler(event: Event) -> Event:
            results.append(event.data)
            return Event(name="flow.done", source="handler", data={"output": "done"})

        flow.register(handler, on="task.created")

        result = await flow.run(task="test task")
        assert result is not None

    @pytest.mark.asyncio
    async def test_flow_with_chain(self) -> None:
        """Flow supports chaining reactors."""
        flow = Flow()
        calls = []

        def first(event: Event) -> Event:
            calls.append("first")
            return Event(name="step.two", source="first", data=event.data)

        def second(event: Event) -> Event:
            calls.append("second")
            return Event(name="flow.done", source="second", data={"output": "complete"})

        flow.register(first, on="task.created")
        flow.register(second, on="step.two")

        result = await flow.run(task="test")
        assert result is not None

    def test_flow_clone(self) -> None:
        """Flow can be cloned."""
        flow = Flow()
        flow.register(lambda e: None, on="test")

        cloned = flow.clone()
        assert cloned is not flow


class TestFlowPatternMatching:
    """Tests for event pattern matching."""

    def test_exact_pattern_registration(self) -> None:
        """Exact patterns can be registered."""
        flow = Flow()

        def handler(event: Event) -> None:
            pass

        flow.register(handler, on="task.created")
        # Should not raise

    def test_wildcard_pattern_registration(self) -> None:
        """Wildcard patterns can be registered."""
        flow = Flow()

        def handler(event: Event) -> None:
            pass

        flow.register(handler, on="task.*")
        # Should not raise

    def test_multi_wildcard_registration(self) -> None:
        """Multi-part wildcards can be registered."""
        flow = Flow()

        def handler(event: Event) -> None:
            pass

        flow.register(handler, on="agent.*.done")
        # Should not raise

    def test_multiple_handlers_same_pattern(self) -> None:
        """Multiple handlers can be registered for same pattern."""
        flow = Flow()

        def handler1(event: Event) -> None:
            pass

        def handler2(event: Event) -> None:
            pass

        flow.register(handler1, on="task.created")
        flow.register(handler2, on="task.created")
        # Should not raise
