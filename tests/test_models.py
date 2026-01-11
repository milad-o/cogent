"""
Tests for model classes (Event, Message, Task).
"""

import json
import pytest
from datetime import datetime, timezone

from agenticflow.core.enums import Priority, TaskStatus
from agenticflow.observability.trace_record import Trace, TraceType
from agenticflow.core.message import Message, MessageType
from agenticflow.tasks.task import Task


class TestEvent:
    """Tests for Event model."""

    def test_create_event(self) -> None:
        event = Event(type=TraceType.TASK_CREATED, data={"name": "test"})
        assert event.type == TraceType.TASK_CREATED
        assert event.data == {"name": "test"}
        assert event.id is not None
        assert event.timestamp is not None

    def test_event_to_dict(self) -> None:
        event = Event(
            type=TraceType.TASK_STARTED,
            data={"task_id": "123"},
            source="test",
        )
        result = event.to_dict()
        assert result["type"] == "task.started"
        assert result["data"] == {"task_id": "123"}
        assert result["source"] == "test"

    def test_event_to_json(self) -> None:
        event = Event(type=TraceType.SYSTEM_STARTED)
        result = event.to_json()
        parsed = json.loads(result)
        assert parsed["type"] == "system.started"

    def test_event_from_dict(self) -> None:
        data = {
            "id": "abc123",
            "type": "task.completed",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "data": {"result": "success"},
            "source": "test",
        }
        event = Event.from_dict(data)
        assert event.id == "abc123"
        assert event.type == TraceType.TASK_COMPLETED
        assert event.data == {"result": "success"}

    def test_event_category(self) -> None:
        event = Event(type=TraceType.AGENT_THINKING)
        assert event.category == "agent"

    def test_child_event(self) -> None:
        parent = Event(
            type=TraceType.TASK_STARTED,
            correlation_id="corr-123",
        )
        child = parent.child_event(
            TraceType.TOOL_CALLED,
            data={"tool": "test"},
        )
        assert child.parent_event_id == parent.id
        assert child.correlation_id == "corr-123"
        assert child.type == TraceType.TOOL_CALLED


class TestMessage:
    """Tests for Message model."""

    def test_create_message(self) -> None:
        msg = Message(
            content="Hello",
            sender_id="agent-1",
            receiver_id="agent-2",
        )
        assert msg.content == "Hello"
        assert msg.sender_id == "agent-1"
        assert msg.receiver_id == "agent-2"
        assert msg.message_type == MessageType.TEXT

    def test_message_to_dict(self) -> None:
        msg = Message(content="Test", sender_id="a1")
        result = msg.to_dict()
        assert result["content"] == "Test"
        assert result["sender_id"] == "a1"
        assert "timestamp" in result

    def test_message_reply(self) -> None:
        original = Message(content="Question?", sender_id="user")
        reply = original.reply(content="Answer!", sender_id="agent")
        assert reply.reply_to == original.id
        assert reply.receiver_id == original.sender_id
        assert reply.content == "Answer!"

    def test_message_is_broadcast(self) -> None:
        broadcast = Message(content="Announcement", sender_id="system")
        direct = Message(content="Hello", sender_id="a1", receiver_id="a2")
        assert broadcast.is_broadcast is True
        assert direct.is_broadcast is False


class TestTask:
    """Tests for Task model."""

    def test_create_task(self) -> None:
        task = Task(
            name="Test task",
            description="A test",
            tool="test_tool",
            args={"param": "value"},
        )
        assert task.name == "Test task"
        assert task.tool == "test_tool"
        assert task.status == TaskStatus.PENDING

    def test_task_to_dict(self) -> None:
        task = Task(name="Test", tool="tool")
        result = task.to_dict()
        assert result["name"] == "Test"
        assert result["tool"] == "tool"
        assert result["status"] == "pending"

    def test_task_from_dict(self) -> None:
        data = {
            "id": "task-123",
            "name": "Imported task",
            "status": "completed",
            "priority": 3,
        }
        task = Task.from_dict(data)
        assert task.id == "task-123"
        assert task.name == "Imported task"
        assert task.status == TaskStatus.COMPLETED
        assert task.priority == Priority.HIGH

    def test_task_is_complete(self) -> None:
        task = Task(name="Test")
        assert task.is_complete() is False
        task.status = TaskStatus.COMPLETED
        assert task.is_complete() is True

    def test_task_can_retry(self) -> None:
        task = Task(name="Test", max_retries=3)
        assert task.can_retry() is False  # Not failed

        task.status = TaskStatus.FAILED
        assert task.can_retry() is True

        task.retry_count = 3
        assert task.can_retry() is False  # Max retries reached

    def test_task_mark_scheduled(self) -> None:
        task = Task(name="Test")
        task.mark_scheduled()
        assert task.status == TaskStatus.SCHEDULED
        assert task.scheduled_at is not None

    def test_task_mark_started(self) -> None:
        task = Task(name="Test")
        task.mark_started()
        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None

    def test_task_mark_completed(self) -> None:
        task = Task(name="Test")
        task.mark_started()
        task.mark_completed(result="success")
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "success"
        assert task.completed_at is not None

    def test_task_mark_failed(self) -> None:
        task = Task(name="Test")
        task.mark_started()
        task.mark_failed(error="Something went wrong")
        assert task.status == TaskStatus.FAILED
        assert task.error == "Something went wrong"

    def test_task_increment_retry(self) -> None:
        task = Task(name="Test", max_retries=2)
        task.status = TaskStatus.FAILED

        assert task.increment_retry() is True
        assert task.retry_count == 1
        assert task.status == TaskStatus.PENDING

        task.status = TaskStatus.FAILED
        assert task.increment_retry() is True
        assert task.retry_count == 2

        task.status = TaskStatus.FAILED
        assert task.increment_retry() is False  # Max reached

    def test_task_create_subtask(self) -> None:
        parent = Task(name="Parent")
        child = parent.create_subtask(name="Child", tool="child_tool")
        assert child.parent_id == parent.id
        assert child.id in parent.subtask_ids

    def test_task_has_dependencies(self) -> None:
        task = Task(name="Test")
        assert task.has_dependencies() is False

        task.depends_on = ["other-task"]
        assert task.has_dependencies() is True
