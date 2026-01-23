"""Tests for Flow.register() with after parameter."""

import pytest

from agenticflow import Flow
from agenticflow.events import Event


class TestFlowAfterParameter:
    """Test Flow.register() after parameter."""

    @pytest.fixture
    def flow(self):
        """Create test flow."""
        return Flow()

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        class MockAgent:
            name = "test_agent"

            async def run(self, task, **kwargs):
                return f"Processed: {task}"

        return MockAgent()

    def test_after_with_single_source(self, flow, mock_agent):
        """after parameter with single source creates correct filter."""
        flow.register(mock_agent, on="agent.done", after="researcher")

        # Check binding was created with condition
        assert len(flow._bindings) == 1
        binding = flow._bindings[0]
        assert binding.condition is not None

        # Test filter behavior
        valid_event = Event(name="agent.done", source="researcher")
        invalid_event = Event(name="agent.done", source="other")

        assert binding.condition(valid_event) is True
        assert binding.condition(invalid_event) is False

    def test_after_with_multiple_sources(self, flow, mock_agent):
        """after parameter with list creates OR filter."""
        flow.register(
            mock_agent,
            on="agent.done",
            after=["agent1", "agent2", "agent3"]
        )

        binding = flow._bindings[0]

        # All listed sources should pass
        assert binding.condition(Event(name="test", source="agent1")) is True
        assert binding.condition(Event(name="test", source="agent2")) is True
        assert binding.condition(Event(name="test", source="agent3")) is True

        # Others should fail
        assert binding.condition(Event(name="test", source="agent4")) is False

    def test_after_with_wildcard_pattern(self, flow, mock_agent):
        """after parameter supports wildcard patterns."""
        flow.register(mock_agent, on="*.done", after="agent*")

        binding = flow._bindings[0]

        # Matching pattern
        assert binding.condition(Event(name="test", source="agent1")) is True
        assert binding.condition(Event(name="test", source="agentX")) is True

        # Not matching
        assert binding.condition(Event(name="test", source="other")) is False

    def test_after_and_when_conflict(self, flow, mock_agent):
        """Cannot specify both after and when."""
        with pytest.raises(ValueError, match="Cannot specify both 'after' and 'when'"):
            flow.register(
                mock_agent,
                on="event",
                after="source1",
                when=lambda e: True
            )

    def test_after_none_allowed(self, flow, mock_agent):
        """after=None is allowed (no filtering)."""
        flow.register(mock_agent, on="event", after=None)

        binding = flow._bindings[0]
        assert binding.condition is None

    def test_when_still_works(self, flow, mock_agent):
        """when parameter still works without after."""
        condition = lambda e: e.data.get("priority") == "high"
        flow.register(mock_agent, on="event", when=condition)

        binding = flow._bindings[0]
        assert binding.condition is condition


class TestFlowAfterIntegration:
    """Integration tests for after parameter with condition checking."""

    def test_after_creates_correct_condition_single_source(self):
        """Flow creates correct condition for single source."""
        flow = Flow()

        def handler(event, ctx):
            return None

        flow.register(handler, on="event", after="source1")

        # Check condition filters correctly
        binding = flow._bindings[0]
        assert binding.condition(Event(name="test", source="source1")) is True
        assert binding.condition(Event(name="test", source="source2")) is False

    def test_after_creates_correct_condition_multiple_sources(self):
        """Flow creates correct condition for multiple sources."""
        flow = Flow()

        def handler(event, ctx):
            return None

        flow.register(handler, on="event", after=["source1", "source2"])

        # Check all sources pass
        binding = flow._bindings[0]
        assert binding.condition(Event(name="test", source="source1")) is True
        assert binding.condition(Event(name="test", source="source2")) is True
        assert binding.condition(Event(name="test", source="source3")) is False

    def test_wildcard_after_creates_pattern_condition(self):
        """Wildcard patterns create correct matching condition."""
        flow = Flow()

        def handler(event, ctx):
            return None

        flow.register(handler, on="test", after="agent*")

        # Check wildcard matching
        binding = flow._bindings[0]
        assert binding.condition(Event(name="test", source="agent1")) is True
        assert binding.condition(Event(name="test", source="agentX")) is True
        assert binding.condition(Event(name="test", source="other")) is False

    def test_after_with_priority_ordering(self):
        """Bindings with after maintain priority ordering."""
        flow = Flow()

        def handler1(event, ctx):
            return None

        def handler2(event, ctx):
            return None

        # Register with different priorities
        flow.register(handler1, on="event", after="source1", priority=0)
        flow.register(handler2, on="event", after="source1", priority=10)

        # Check priority ordering is preserved
        bindings = [b for b in flow._bindings if "event" in b.patterns]
        assert len(bindings) == 2
        assert bindings[0].priority == 10  # High priority first
        assert bindings[1].priority == 0   # Low priority second


class TestBackwardCompatibility:
    """Ensure after parameter doesn't break existing functionality."""

    def test_register_without_after_still_works(self):
        """Registering without after parameter works as before."""
        flow = Flow()

        def handler(event, ctx):
            return None

        # Should not raise
        flow.register(handler, on="event")
        flow.register(handler, on="event", when=lambda e: True)
        flow.register(handler, on="event", priority=5)

    def test_existing_when_filters_work(self):
        """Existing when filters create correct conditions."""
        flow = Flow()

        condition = lambda e: e.data.get("value", 0) > 5

        def handler(event, ctx):
            return None

        flow.register(handler, on="event", when=condition)

        # Check condition was set correctly
        binding = flow._bindings[0]
        assert binding.condition is condition

        # Test condition behavior
        assert binding.condition(Event(name="event", data={"value": 10})) is True
        assert binding.condition(Event(name="event", data={"value": 3})) is False


class TestErrorMessages:
    """Test clear error messages."""

    def test_clear_error_when_both_after_and_when(self):
        """Error message is clear when using both after and when."""
        flow = Flow()

        with pytest.raises(ValueError) as exc_info:
            flow.register(
                lambda e, ctx: None,
                on="event",
                after="source",
                when=lambda e: True
            )

        assert "Cannot specify both 'after' and 'when'" in str(exc_info.value)
        assert "Use 'after' for source filtering" in str(exc_info.value)
