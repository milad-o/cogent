"""
Tests for core enums and utilities.
"""

import pytest
from datetime import datetime, timezone

from agenticflow.core.enums import (
    TaskStatus,
    AgentStatus,
    EventType,
    Priority,
    AgentRole,
)
from agenticflow.core.utils import generate_id, now_utc, truncate_string


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_is_terminal_completed(self) -> None:
        assert TaskStatus.COMPLETED.is_terminal() is True

    def test_is_terminal_failed(self) -> None:
        assert TaskStatus.FAILED.is_terminal() is True

    def test_is_terminal_cancelled(self) -> None:
        assert TaskStatus.CANCELLED.is_terminal() is True

    def test_is_terminal_pending(self) -> None:
        assert TaskStatus.PENDING.is_terminal() is False

    def test_is_terminal_running(self) -> None:
        assert TaskStatus.RUNNING.is_terminal() is False

    def test_is_active_running(self) -> None:
        assert TaskStatus.RUNNING.is_active() is True

    def test_is_active_spawning(self) -> None:
        assert TaskStatus.SPAWNING.is_active() is True

    def test_is_active_pending(self) -> None:
        assert TaskStatus.PENDING.is_active() is False


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_is_available_idle(self) -> None:
        assert AgentStatus.IDLE.is_available() is True

    def test_is_available_thinking(self) -> None:
        assert AgentStatus.THINKING.is_available() is False

    def test_is_working_thinking(self) -> None:
        assert AgentStatus.THINKING.is_working() is True

    def test_is_working_acting(self) -> None:
        assert AgentStatus.ACTING.is_working() is True

    def test_is_working_idle(self) -> None:
        assert AgentStatus.IDLE.is_working() is False


class TestEventType:
    """Tests for EventType enum."""

    def test_category_task(self) -> None:
        assert EventType.TASK_CREATED.category == "task"

    def test_category_agent(self) -> None:
        assert EventType.AGENT_INVOKED.category == "agent"

    def test_category_system(self) -> None:
        assert EventType.SYSTEM_STARTED.category == "system"


class TestPriority:
    """Tests for Priority enum."""

    def test_comparison_less_than(self) -> None:
        assert Priority.LOW < Priority.NORMAL
        assert Priority.NORMAL < Priority.HIGH
        assert Priority.HIGH < Priority.CRITICAL

    def test_comparison_greater_than(self) -> None:
        assert Priority.CRITICAL > Priority.HIGH
        assert Priority.HIGH > Priority.NORMAL

    def test_comparison_equal(self) -> None:
        assert Priority.NORMAL <= Priority.NORMAL
        assert Priority.NORMAL >= Priority.NORMAL


class TestAgentRole:
    """Tests for AgentRole enum with the clean 4-role system."""

    def test_can_delegate_supervisor(self) -> None:
        assert AgentRole.SUPERVISOR.can_delegate() is True

    def test_can_delegate_worker(self) -> None:
        assert AgentRole.WORKER.can_delegate() is False

    def test_can_delegate_autonomous(self) -> None:
        assert AgentRole.AUTONOMOUS.can_delegate() is False

    def test_can_delegate_reviewer(self) -> None:
        assert AgentRole.REVIEWER.can_delegate() is False

    def test_can_finish_supervisor(self) -> None:
        assert AgentRole.SUPERVISOR.can_finish() is True

    def test_can_finish_worker(self) -> None:
        assert AgentRole.WORKER.can_finish() is False

    def test_can_finish_autonomous(self) -> None:
        assert AgentRole.AUTONOMOUS.can_finish() is True

    def test_can_finish_reviewer(self) -> None:
        assert AgentRole.REVIEWER.can_finish() is True

    def test_can_use_tools_supervisor(self) -> None:
        assert AgentRole.SUPERVISOR.can_use_tools() is False

    def test_can_use_tools_worker(self) -> None:
        assert AgentRole.WORKER.can_use_tools() is True

    def test_can_use_tools_autonomous(self) -> None:
        assert AgentRole.AUTONOMOUS.can_use_tools() is True

    def test_can_use_tools_reviewer(self) -> None:
        assert AgentRole.REVIEWER.can_use_tools() is False

    def test_only_four_roles_exist(self) -> None:
        """Ensure only the clean 4-role system exists."""
        roles = list(AgentRole)
        assert len(roles) == 4
        assert set(roles) == {
            AgentRole.WORKER,
            AgentRole.SUPERVISOR,
            AgentRole.AUTONOMOUS,
            AgentRole.REVIEWER,
        }


class TestGenerateId:
    """Tests for generate_id utility."""

    def test_returns_string(self) -> None:
        result = generate_id()
        assert isinstance(result, str)

    def test_correct_length(self) -> None:
        result = generate_id()
        assert len(result) == 8

    def test_unique_ids(self) -> None:
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100  # All unique


class TestNowUtc:
    """Tests for now_utc utility."""

    def test_returns_datetime(self) -> None:
        result = now_utc()
        assert isinstance(result, datetime)

    def test_has_timezone(self) -> None:
        result = now_utc()
        assert result.tzinfo is not None

    def test_is_utc(self) -> None:
        result = now_utc()
        assert result.tzinfo == timezone.utc


class TestTruncateString:
    """Tests for truncate_string utility."""

    def test_short_string_unchanged(self) -> None:
        result = truncate_string("hello", max_length=10)
        assert result == "hello"

    def test_long_string_truncated(self) -> None:
        result = truncate_string("hello world", max_length=8)
        assert result == "hello..."

    def test_exact_length_unchanged(self) -> None:
        result = truncate_string("hello", max_length=5)
        assert result == "hello"

    def test_empty_string(self) -> None:
        result = truncate_string("", max_length=10)
        assert result == ""
