"""Tests for source groups functionality."""

import pytest

from agenticflow import Agent, Flow
from agenticflow.events import Event
from agenticflow.models import MockChatModel


class TestSourceGroupsBasics:
    """Test basic group operations."""

    def test_add_source_group(self):
        """Can add a source group."""
        flow = Flow()
        flow.add_source_group("analysts", ["agent1", "agent2", "agent3"])

        sources = flow.get_source_group("analysts")
        assert sources == {"agent1", "agent2", "agent3"}

    def test_add_source_group_chaining(self):
        """add_source_group returns self for chaining."""
        flow = Flow()
        result = (
            flow.add_source_group("group1", ["a"])
            .add_source_group("group2", ["b"])
            .add_source_group("group3", ["c"])
        )

        assert result is flow
        assert flow.get_source_group("group1") == {"a"}
        assert flow.get_source_group("group2") == {"b"}
        assert flow.get_source_group("group3") == {"c"}

    def test_get_nonexistent_group(self):
        """get_source_group returns empty set for missing groups."""
        flow = Flow()
        assert flow.get_source_group("missing") == set()

    def test_get_source_group_returns_copy(self):
        """get_source_group returns a copy, not the original set."""
        flow = Flow()
        flow.add_source_group("test", ["a", "b"])

        sources1 = flow.get_source_group("test")
        sources1.add("c")  # Modify the returned set

        sources2 = flow.get_source_group("test")
        assert sources2 == {"a", "b"}  # Original unchanged

    def test_empty_group_name_raises(self):
        """Cannot create group with empty name."""
        flow = Flow()
        with pytest.raises(ValueError, match="cannot be empty"):
            flow.add_source_group("", ["agent1"])

    def test_group_name_with_colon_raises(self):
        """Cannot create group name starting with :."""
        flow = Flow()
        with pytest.raises(ValueError, match="cannot.*start with ':'"):
            flow.add_source_group(":analysts", ["agent1"])

    def test_update_existing_group(self):
        """Can update an existing group by adding it again."""
        flow = Flow()
        flow.add_source_group("team", ["a", "b"])
        flow.add_source_group("team", ["c", "d"])  # Replace

        assert flow.get_source_group("team") == {"c", "d"}


class TestGroupReferencesInAfter:
    """Test :group syntax in after parameter."""

    def test_after_with_group_reference(self):
        """Can use :group in after parameter."""
        flow = Flow()
        flow.add_source_group("analysts", ["agent1", "agent2"])

        executed = []
        flow.register(
            lambda e: executed.append(e.source), on="task.done", after=":analysts"
        )

        binding = flow._bindings[0]
        assert binding.condition(Event(name="task.done", source="agent1")) is True
        assert binding.condition(Event(name="task.done", source="agent2")) is True
        assert binding.condition(Event(name="task.done", source="agent3")) is False

    def test_after_with_multiple_group_sources(self):
        """Group with multiple sources matches any of them."""
        flow = Flow()
        flow.add_source_group("workers", ["worker1", "worker2", "worker3"])

        flow.register(lambda e: None, on="task", after=":workers")

        binding = flow._bindings[0]
        assert binding.condition(Event(name="task", source="worker1")) is True
        assert binding.condition(Event(name="task", source="worker2")) is True
        assert binding.condition(Event(name="task", source="worker3")) is True
        assert binding.condition(Event(name="task", source="other")) is False

    def test_after_with_nonexistent_group(self):
        """Group reference to missing group matches nothing."""
        flow = Flow()

        flow.register(lambda e: None, on="task", after=":missing")

        binding = flow._bindings[0]
        assert binding.condition(Event(name="task", source="any")) is False

    def test_after_with_empty_group(self):
        """Group reference to empty group matches nothing."""
        flow = Flow()
        flow.add_source_group("empty", [])

        flow.register(lambda e: None, on="task", after=":empty")

        binding = flow._bindings[0]
        assert binding.condition(Event(name="task", source="any")) is False


class TestGroupReferencesInPatterns:
    """Test :group syntax in pattern syntax."""

    def test_pattern_with_group_reference(self):
        """Can use :group in pattern@source syntax."""
        flow = Flow()
        flow.add_source_group("workers", ["worker1", "worker2", "worker3"])

        flow.register(lambda e: None, on="task.done@:workers")

        binding = flow._bindings[0]
        assert binding.condition(Event(name="task.done", source="worker1")) is True
        assert binding.condition(Event(name="task.done", source="worker2")) is True
        assert binding.condition(Event(name="task.done", source="worker3")) is True
        assert binding.condition(Event(name="task.done", source="other")) is False

    def test_pattern_with_wildcard_and_group(self):
        """Can use wildcards in event with :group in source."""
        flow = Flow()
        flow.add_source_group("team", ["alice", "bob"])

        flow.register(lambda e: None, on="*.done@:team")

        # Check patterns extracted correctly
        binding = flow._bindings[0]
        assert "*.done" in binding.patterns

        # Check source filter in condition
        assert binding.condition(Event(name="task.done", source="alice")) is True
        assert binding.condition(Event(name="job.done", source="bob")) is True
        assert binding.condition(Event(name="task.done", source="charlie")) is False
        # Note: Pattern matching happens separately in Flow._find_matching_bindings()
        # Here we only test the source filter part

    def test_multiple_patterns_with_groups(self):
        """Multiple patterns can reference different groups."""
        flow = Flow()
        flow.add_source_group("writers", ["w1", "w2"])
        flow.add_source_group("reviewers", ["r1", "r2"])

        flow.register(lambda e: None, on=["task.done@:writers", "review.done@:reviewers"])

        binding = flow._bindings[0]

        # Check patterns extracted correctly
        assert "task.done" in binding.patterns
        assert "review.done" in binding.patterns

        # Check combined source filter (OR logic)
        # Filter should match writers OR reviewers
        assert binding.condition(Event(name="task.done", source="w1")) is True
        assert binding.condition(Event(name="task.done", source="w2")) is True
        assert binding.condition(Event(name="review.done", source="r1")) is True
        assert binding.condition(Event(name="review.done", source="r2")) is True
        # Not in any group
        assert binding.condition(Event(name="any", source="other")) is False


class TestBuiltinGroups:
    """Test built-in source groups."""

    def test_agents_group_exists(self):
        """:agents group is initialized on Flow creation."""
        flow = Flow()
        agents = flow.get_source_group("agents")
        assert agents == set()  # Empty initially

    def test_agents_group_auto_populated(self):
        """:agents group is auto-populated when agents registered."""
        flow = Flow()
        model = MockChatModel()

        agent1 = Agent(name="agent1", model=model)
        agent2 = Agent(name="agent2", model=model)

        flow.register(agent1, on="task")
        flow.register(agent2, on="task")

        agents = flow.get_source_group("agents")
        assert "agent1" in agents
        assert "agent2" in agents

    def test_agents_group_usable_in_after(self):
        """Can use :agents in after parameter."""
        flow = Flow()
        model = MockChatModel()

        agent1 = Agent(name="agent1", model=model)
        agent2 = Agent(name="agent2", model=model)

        flow.register(agent1, on="task")
        flow.register(agent2, on="task")

        # Register aggregator that waits for :agents
        executed = []
        flow.register(
            lambda e: executed.append(e.source), on="agent.done", after=":agents"
        )

        binding = flow._bindings[-1]
        assert binding.condition(Event(name="agent.done", source="agent1")) is True
        assert binding.condition(Event(name="agent.done", source="agent2")) is True
        assert binding.condition(Event(name="agent.done", source="other")) is False

    def test_system_group_exists(self):
        """:system group exists with default system sources."""
        flow = Flow()

        system = flow.get_source_group("system")
        assert "flow" in system
        assert "router" in system
        assert "aggregator" in system

    def test_system_group_usable_in_after(self):
        """Can use :system in after parameter."""
        flow = Flow()

        executed = []
        flow.register(lambda e: executed.append(e.source), on="event", after=":system")

        binding = flow._bindings[0]
        assert binding.condition(Event(name="event", source="flow")) is True
        assert binding.condition(Event(name="event", source="router")) is True
        assert binding.condition(Event(name="event", source="aggregator")) is True
        assert binding.condition(Event(name="event", source="agent1")) is False


class TestGroupEdgeCases:
    """Test edge cases and error handling."""

    def test_group_reference_without_colon_is_literal(self):
        """Source without : prefix is treated as literal source name."""
        flow = Flow()
        flow.add_source_group("analysts", ["agent1", "agent2"])

        # "analysts" without : is a literal source name
        flow.register(lambda e: None, on="task", after="analysts")

        binding = flow._bindings[0]
        # Matches only literal "analysts"
        assert binding.condition(Event(name="task", source="analysts")) is True
        assert binding.condition(Event(name="task", source="agent1")) is False
        assert binding.condition(Event(name="task", source="agent2")) is False

    def test_group_name_case_sensitive(self):
        """Group names are case-sensitive."""
        flow = Flow()
        flow.add_source_group("Team", ["a"])

        assert flow.get_source_group("Team") == {"a"}
        assert flow.get_source_group("team") == set()

    def test_from_source_raises_without_flow_parameter(self):
        """from_source with :group but no flow parameter raises ValueError."""
        from agenticflow.events.patterns import from_source

        with pytest.raises(ValueError, match="Flow instance required"):
            from_source(":analysts")

    def test_from_source_with_group_and_flow(self):
        """from_source with :group and flow parameter works."""
        from agenticflow.events.patterns import from_source

        flow = Flow()
        flow.add_source_group("team", ["alice", "bob"])

        filter_fn = from_source(":team", flow=flow)

        assert filter_fn(Event(name="task", source="alice")) is True
        assert filter_fn(Event(name="task", source="bob")) is True
        assert filter_fn(Event(name="task", source="charlie")) is False
