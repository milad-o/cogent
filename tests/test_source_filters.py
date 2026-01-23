"""Tests for source-based event filtering."""


from agenticflow.events import (
    Event,
    SourceFilter,
    any_source,
    from_source,
    matching_sources,
    not_from_source,
)


class TestSourceFilter:
    """Test SourceFilter class and composition."""

    def test_call_evaluates_predicate(self):
        """SourceFilter evaluates its predicate on call."""
        filter = SourceFilter(lambda e: e.source == "test")
        event = Event(name="test", source="test")
        assert filter(event) is True

        event2 = Event(name="test", source="other")
        assert filter(event2) is False

    def test_and_combines_filters(self):
        """& operator combines filters with AND logic."""
        filter1 = SourceFilter(lambda e: e.source.startswith("agent"))
        filter2 = SourceFilter(lambda e: "1" in e.source)
        combined = filter1 & filter2

        assert combined(Event(name="test", source="agent1")) is True
        assert combined(Event(name="test", source="agent2")) is False
        assert combined(Event(name="test", source="other1")) is False

    def test_or_combines_filters(self):
        """| operator combines filters with OR logic."""
        filter1 = from_source("agent1")
        filter2 = from_source("agent2")
        combined = filter1 | filter2

        assert combined(Event(name="test", source="agent1")) is True
        assert combined(Event(name="test", source="agent2")) is True
        assert combined(Event(name="test", source="agent3")) is False

    def test_invert_negates_filter(self):
        """~ operator negates filter."""
        filter = from_source("system")
        inverted = ~filter

        assert inverted(Event(name="test", source="system")) is False
        assert inverted(Event(name="test", source="user")) is True

    def test_complex_composition(self):
        """Complex boolean composition works correctly."""
        # (from_source("api") & high_priority) | from_source("admin")
        api_filter = from_source("api")
        high_priority = SourceFilter(lambda e: e.data.get("priority") == "high")
        admin_filter = from_source("admin")

        combined = (api_filter & high_priority) | admin_filter

        # API with high priority → True
        assert combined(Event(
            name="test",
            source="api",
            data={"priority": "high"}
        )) is True

        # API with low priority → False
        assert combined(Event(
            name="test",
            source="api",
            data={"priority": "low"}
        )) is False

        # Admin regardless of priority → True
        assert combined(Event(
            name="test",
            source="admin",
            data={"priority": "low"}
        )) is True


class TestFromSource:
    """Test from_source() function."""

    def test_exact_match_single_string(self):
        """from_source with single string matches exact source."""
        filter = from_source("researcher")

        assert filter(Event(name="test", source="researcher")) is True
        assert filter(Event(name="test", source="other")) is False
        assert filter(Event(name="test", source="researcher2")) is False

    def test_list_matches_any(self):
        """from_source with list matches any of the sources."""
        filter = from_source(["agent1", "agent2", "agent3"])

        assert filter(Event(name="test", source="agent1")) is True
        assert filter(Event(name="test", source="agent2")) is True
        assert filter(Event(name="test", source="agent3")) is True
        assert filter(Event(name="test", source="agent4")) is False
        assert filter(Event(name="test", source="other")) is False

    def test_wildcard_star(self):
        """from_source supports * wildcard."""
        filter = from_source("agent*")

        assert filter(Event(name="test", source="agent1")) is True
        assert filter(Event(name="test", source="agent2")) is True
        assert filter(Event(name="test", source="agentX")) is True
        assert filter(Event(name="test", source="other")) is False

    def test_wildcard_question_mark(self):
        """from_source supports ? wildcard."""
        filter = from_source("agent?")

        assert filter(Event(name="test", source="agent1")) is True
        assert filter(Event(name="test", source="agent2")) is True
        assert filter(Event(name="test", source="agentX")) is True
        assert filter(Event(name="test", source="agent12")) is False

    def test_wildcard_at_start(self):
        """Wildcard at start of pattern works."""
        filter = from_source("*_dev")

        assert filter(Event(name="test", source="api_dev")) is True
        assert filter(Event(name="test", source="test_dev")) is True
        assert filter(Event(name="test", source="dev")) is False
        assert filter(Event(name="test", source="api_prod")) is False

    def test_wildcard_in_middle(self):
        """Wildcard in middle of pattern works."""
        filter = from_source("api_*_prod")

        assert filter(Event(name="test", source="api_rest_prod")) is True
        assert filter(Event(name="test", source="api_webhook_prod")) is True
        assert filter(Event(name="test", source="api_prod")) is False

    def test_empty_list_matches_nothing(self):
        """from_source with empty list matches nothing."""
        filter = from_source([])

        assert filter(Event(name="test", source="anything")) is False


class TestNotFromSource:
    """Test not_from_source() function."""

    def test_excludes_exact_source(self):
        """not_from_source excludes exact source."""
        filter = not_from_source("system")

        assert filter(Event(name="test", source="system")) is False
        assert filter(Event(name="test", source="user")) is True
        assert filter(Event(name="test", source="api")) is True

    def test_excludes_list_of_sources(self):
        """not_from_source with list excludes all listed sources."""
        filter = not_from_source(["test", "debug", "dev"])

        assert filter(Event(name="test", source="test")) is False
        assert filter(Event(name="test", source="debug")) is False
        assert filter(Event(name="test", source="dev")) is False
        assert filter(Event(name="test", source="prod")) is True

    def test_excludes_wildcard_pattern(self):
        """not_from_source supports wildcard patterns."""
        filter = not_from_source("*_internal")

        assert filter(Event(name="test", source="api_internal")) is False
        assert filter(Event(name="test", source="db_internal")) is False
        assert filter(Event(name="test", source="api_public")) is True


class TestAnySource:
    """Test any_source() convenience function."""

    def test_matches_any_listed_source(self):
        """any_source matches any of the listed sources."""
        filter = any_source(["a", "b", "c"])

        assert filter(Event(name="test", source="a")) is True
        assert filter(Event(name="test", source="b")) is True
        assert filter(Event(name="test", source="c")) is True
        assert filter(Event(name="test", source="d")) is False

    def test_equivalent_to_from_source_with_list(self):
        """any_source is equivalent to from_source with list."""
        sources = ["agent1", "agent2"]
        filter1 = any_source(sources)
        filter2 = from_source(sources)

        event = Event(name="test", source="agent1")
        assert filter1(event) == filter2(event)


class TestMatchingSources:
    """Test matching_sources() convenience function."""

    def test_matches_pattern(self):
        """matching_sources matches wildcard pattern."""
        filter = matching_sources("agent*")

        assert filter(Event(name="test", source="agent1")) is True
        assert filter(Event(name="test", source="agentX")) is True
        assert filter(Event(name="test", source="other")) is False

    def test_equivalent_to_from_source_with_pattern(self):
        """matching_sources is equivalent to from_source with pattern."""
        pattern = "api_*"
        filter1 = matching_sources(pattern)
        filter2 = from_source(pattern)

        event = Event(name="test", source="api_rest")
        assert filter1(event) == filter2(event)


class TestRealWorldScenarios:
    """Test realistic use cases."""

    def test_review_after_research(self):
        """Reviewer only processes researcher output."""
        filter = from_source("researcher")

        research_event = Event(
            name="agent.done",
            source="researcher",
            data={"output": "Research complete"}
        )
        other_event = Event(
            name="agent.done",
            source="writer",
            data={"output": "Article written"}
        )

        assert filter(research_event) is True
        assert filter(other_event) is False

    def test_aggregate_from_multiple_analysts(self):
        """Aggregator collects from specific analysts only."""
        filter = from_source(["analyst1", "analyst2", "analyst3"])

        valid_events = [
            Event(name="analysis.done", source="analyst1"),
            Event(name="analysis.done", source="analyst2"),
            Event(name="analysis.done", source="analyst3"),
        ]
        invalid_events = [
            Event(name="analysis.done", source="analyst4"),
            Event(name="analysis.done", source="other"),
        ]

        for event in valid_events:
            assert filter(event) is True

        for event in invalid_events:
            assert filter(event) is False

    def test_high_priority_from_api_only(self):
        """Complex: High priority tasks from API source only."""
        api_filter = from_source("api")
        high_priority_filter = SourceFilter(
            lambda e: e.data.get("priority") == "high"
        )
        combined = api_filter & high_priority_filter

        # API + high priority → True
        assert combined(Event(
            name="task.created",
            source="api",
            data={"priority": "high"}
        )) is True

        # API + low priority → False
        assert combined(Event(
            name="task.created",
            source="api",
            data={"priority": "low"}
        )) is False

        # Non-API + high priority → False
        assert combined(Event(
            name="task.created",
            source="user",
            data={"priority": "high"}
        )) is False

    def test_exclude_internal_sources(self):
        """Filter out internal/system events."""
        filter = not_from_source(["system", "internal", "debug"])

        # User events pass through
        assert filter(Event(name="event", source="user")) is True
        assert filter(Event(name="event", source="api")) is True

        # Internal events blocked
        assert filter(Event(name="event", source="system")) is False
        assert filter(Event(name="event", source="internal")) is False
        assert filter(Event(name="event", source="debug")) is False

    def test_any_agent_matching_pattern(self):
        """Match any agent with specific naming convention."""
        filter = matching_sources("agent_*_prod")

        # Production agents
        assert filter(Event(name="event", source="agent_api_prod")) is True
        assert filter(Event(name="event", source="agent_worker_prod")) is True

        # Non-production agents
        assert filter(Event(name="event", source="agent_api_dev")) is False
        assert filter(Event(name="event", source="api_prod")) is False
