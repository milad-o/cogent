"""Tests for reactive skills."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agenticflow.reactive import (
    skill,
    Skill,
    SkillBuilder,
    Trigger,
    react_to,
)
from agenticflow.observability.event import Event, EventType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_event():
    """Create a mock event for testing."""
    def _create(name: str = "code.write", data: dict | None = None):
        from agenticflow.events.event import Event as CoreEvent
        return CoreEvent(name=name, data=data or {})
    return _create


@pytest.fixture
def sample_tool():
    """Create a sample tool function."""
    def my_tool(x: str) -> str:
        """A sample tool."""
        return f"processed: {x}"
    return my_tool


# =============================================================================
# Tests: skill() factory function
# =============================================================================


class TestSkillFactory:
    """Tests for the skill() factory function."""

    def test_skill_creates_instance(self) -> None:
        """skill() returns a Skill instance."""
        s = skill(
            "test_skill",
            on="test.event",
            prompt="You are a test expert.",
        )
        assert isinstance(s, Skill)
        assert s.name == "test_skill"
        assert s.prompt == "You are a test expert."

    def test_skill_with_condition(self, mock_event) -> None:
        """skill() with when= creates matching trigger."""
        s = skill(
            "python_expert",
            on="code.write",
            when=lambda e: e.data.get("language") == "python",
            prompt="Write Python code.",
        )

        # Should match
        py_event = mock_event("code.write", {"language": "python"})
        assert s.matches(py_event)

        # Should not match
        js_event = mock_event("code.write", {"language": "javascript"})
        assert not s.matches(js_event)

    def test_skill_with_tools(self, sample_tool) -> None:
        """skill() with tools= stores them as tuple."""
        s = skill(
            "tooled_skill",
            on="task.created",
            prompt="Use tools.",
            tools=[sample_tool],
        )
        assert len(s.tools) == 1
        assert s.tools[0] is sample_tool

    def test_skill_with_priority(self) -> None:
        """skill() with priority= sets priority."""
        s = skill(
            "high_priority",
            on="urgent.*",
            prompt="Handle urgently.",
            priority=100,
        )
        assert s.priority == 100

    def test_skill_with_context_enricher(self, mock_event) -> None:
        """skill() with context_enricher= applies enrichment."""
        def enrich(event, ctx):
            return {**ctx, "enriched": True}

        s = skill(
            "enricher_skill",
            on="test.event",
            prompt="Test.",
            context_enricher=enrich,
        )

        event = mock_event("test.event")
        result = s.enrich_context(event, {"original": True})
        assert result == {"original": True, "enriched": True}


# =============================================================================
# Tests: Skill dataclass
# =============================================================================


class TestSkill:
    """Tests for Skill dataclass."""

    def test_skill_matches_exact_event(self, mock_event) -> None:
        """Skill matches exact event name."""
        s = Skill(
            name="test",
            trigger=Trigger(on="task.completed"),
            prompt="Test prompt.",
        )
        event = mock_event("task.completed")
        assert s.matches(event)

    def test_skill_matches_wildcard(self, mock_event) -> None:
        """Skill matches wildcard pattern."""
        s = Skill(
            name="error_handler",
            trigger=Trigger(on="error.*"),
            prompt="Handle errors.",
        )
        assert s.matches(mock_event("error.network"))
        assert s.matches(mock_event("error.timeout"))
        assert not s.matches(mock_event("warning.low_memory"))

    def test_skill_default_context_enricher(self, mock_event) -> None:
        """Skill returns same context when no enricher."""
        s = Skill(
            name="simple",
            trigger=Trigger(on="test"),
            prompt="Test.",
        )
        ctx = {"key": "value"}
        result = s.enrich_context(mock_event("test"), ctx)
        assert result is ctx


# =============================================================================
# Tests: SkillBuilder (legacy API)
# =============================================================================


class TestSkillBuilder:
    """Tests for SkillBuilder (backward compatibility)."""

    def test_builder_creates_skill(self) -> None:
        """SkillBuilder.build() creates Skill."""
        builder = SkillBuilder("test", "test.event")
        builder.with_prompt("Test prompt.")
        s = builder.build()

        assert isinstance(s, Skill)
        assert s.name == "test"
        assert s.prompt == "Test prompt."

    def test_builder_requires_prompt(self) -> None:
        """SkillBuilder raises if no prompt."""
        builder = SkillBuilder("test", "test.event")
        with pytest.raises(ValueError, match="requires a prompt"):
            builder.build()

    def test_builder_chaining(self, sample_tool) -> None:
        """SkillBuilder methods chain correctly."""
        s = (
            SkillBuilder("chained", "event.*")
            .when(lambda e: True)
            .with_prompt("Chained prompt.")
            .with_tools([sample_tool])
            .with_priority(5)
            .build()
        )

        assert s.name == "chained"
        assert s.prompt == "Chained prompt."
        assert len(s.tools) == 1
        assert s.priority == 5

    def test_builder_iterable(self) -> None:
        """SkillBuilder can be used in list unpacking."""
        builder = SkillBuilder("iterable", "test")
        builder.with_prompt("Test.")

        skills = list(builder)
        assert len(skills) == 1
        assert isinstance(skills[0], Skill)


# =============================================================================
# Tests: Skill priority ordering
# =============================================================================


class TestSkillPriority:
    """Tests for skill priority ordering."""

    def test_skills_sorted_by_priority(self) -> None:
        """Higher priority skills should come first."""
        low = skill("low", on="task", prompt="Low.", priority=1)
        high = skill("high", on="task", prompt="High.", priority=100)
        medium = skill("medium", on="task", prompt="Medium.", priority=50)

        skills = [low, high, medium]
        sorted_skills = sorted(skills, key=lambda s: -s.priority)

        assert sorted_skills[0].name == "high"
        assert sorted_skills[1].name == "medium"
        assert sorted_skills[2].name == "low"
