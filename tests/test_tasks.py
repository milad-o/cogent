"""
Tests for TaskManager.
"""

import pytest
from unittest.mock import AsyncMock

from agenticflow.core.enums import Priority, TaskStatus
from agenticflow.observability.trace_record import TraceType
from agenticflow.observability.bus import TraceBus
from agenticflow.tasks.manager import TaskManager


class TestTaskManager:
    """Tests for TaskManager."""

    @pytest.fixture
    def event_bus(self) -> TraceBus:
        return TraceBus()

    @pytest.fixture
    def task_manager(self, event_bus: TraceBus) -> TaskManager:
        return TaskManager(event_bus)

    async def test_create_task(self, task_manager: TaskManager) -> None:
        task = await task_manager.create_task(
            name="Test task",
            tool="test_tool",
            args={"param": "value"},
        )

        assert task.name == "Test task"
        assert task.tool == "test_tool"
        assert task.id in task_manager.tasks

    async def test_create_task_emits_event(
        self, task_manager: TaskManager, event_bus: TraceBus
    ) -> None:
        handler = AsyncMock()
        event_bus.subscribe(TraceType.TASK_CREATED, handler)

        await task_manager.create_task(name="Test")

        handler.assert_awaited_once()
        event = handler.call_args[0][0]
        assert event.type == TraceType.TASK_CREATED

    async def test_create_subtask(self, task_manager: TaskManager) -> None:
        parent = await task_manager.create_task(name="Parent")
        child = await task_manager.create_subtask(
            parent_id=parent.id,
            name="Child",
            tool="child_tool",
        )

        assert child.parent_id == parent.id
        assert child.id in parent.subtask_ids

    async def test_create_subtask_emits_spawned_event(
        self, task_manager: TaskManager, event_bus: TraceBus
    ) -> None:
        handler = AsyncMock()
        event_bus.subscribe(TraceType.SUBTASK_SPAWNED, handler)

        parent = await task_manager.create_task(name="Parent")
        await task_manager.create_subtask(parent_id=parent.id, name="Child")

        handler.assert_awaited_once()

    async def test_update_status_to_running(
        self, task_manager: TaskManager
    ) -> None:
        task = await task_manager.create_task(name="Test")

        await task_manager.update_status(task.id, TaskStatus.RUNNING)

        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None

    async def test_update_status_to_completed(
        self, task_manager: TaskManager
    ) -> None:
        task = await task_manager.create_task(name="Test")
        await task_manager.update_status(task.id, TaskStatus.RUNNING)

        await task_manager.update_status(
            task.id,
            TaskStatus.COMPLETED,
            result="success",
        )

        assert task.status == TaskStatus.COMPLETED
        assert task.result == "success"
        assert task.completed_at is not None

    async def test_update_status_to_failed(
        self, task_manager: TaskManager
    ) -> None:
        task = await task_manager.create_task(name="Test")
        await task_manager.update_status(task.id, TaskStatus.RUNNING)

        await task_manager.update_status(
            task.id,
            TaskStatus.FAILED,
            error="Something went wrong",
        )

        assert task.status == TaskStatus.FAILED
        assert task.error == "Something went wrong"

    async def test_get_ready_tasks_no_dependencies(
        self, task_manager: TaskManager
    ) -> None:
        await task_manager.create_task(name="Task 1")
        await task_manager.create_task(name="Task 2")

        ready = await task_manager.get_ready_tasks()

        assert len(ready) == 2

    async def test_get_ready_tasks_with_dependencies(
        self, task_manager: TaskManager
    ) -> None:
        task1 = await task_manager.create_task(name="Task 1")
        task2 = await task_manager.create_task(
            name="Task 2",
            depends_on=[task1.id],
        )

        ready = await task_manager.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == task1.id

        # Complete task1
        await task_manager.update_status(task1.id, TaskStatus.COMPLETED)

        ready = await task_manager.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == task2.id

    async def test_get_ready_tasks_sorted_by_priority(
        self, task_manager: TaskManager
    ) -> None:
        await task_manager.create_task(name="Low", priority=Priority.LOW)
        await task_manager.create_task(name="High", priority=Priority.HIGH)
        await task_manager.create_task(name="Normal", priority=Priority.NORMAL)

        ready = await task_manager.get_ready_tasks()

        assert ready[0].name == "High"
        assert ready[1].name == "Normal"
        assert ready[2].name == "Low"

    async def test_retry_task(self, task_manager: TaskManager) -> None:
        task = await task_manager.create_task(name="Test", max_retries=2)
        await task_manager.update_status(
            task.id,
            TaskStatus.FAILED,
            error="First failure",
        )

        result = await task_manager.retry_task(task.id)

        assert result is True
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 1

    async def test_retry_task_max_reached(
        self, task_manager: TaskManager
    ) -> None:
        task = await task_manager.create_task(name="Test", max_retries=1)
        task.retry_count = 1
        task.status = TaskStatus.FAILED

        result = await task_manager.retry_task(task.id)

        assert result is False

    async def test_cancel_task(self, task_manager: TaskManager) -> None:
        task = await task_manager.create_task(name="Test")

        await task_manager.cancel_task(task.id)

        assert task.status == TaskStatus.CANCELLED

    async def test_aggregate_subtasks(
        self, task_manager: TaskManager
    ) -> None:
        parent = await task_manager.create_task(name="Parent")
        child1 = await task_manager.create_subtask(
            parent_id=parent.id, name="Child 1"
        )
        child2 = await task_manager.create_subtask(
            parent_id=parent.id, name="Child 2"
        )

        await task_manager.update_status(
            child1.id, TaskStatus.COMPLETED, result="Result 1"
        )
        await task_manager.update_status(
            child2.id, TaskStatus.COMPLETED, result="Result 2"
        )

        results = await task_manager.aggregate_subtasks(parent.id)

        assert results["Child 1"] == "Result 1"
        assert results["Child 2"] == "Result 2"

    async def test_check_subtasks_complete(
        self, task_manager: TaskManager
    ) -> None:
        parent = await task_manager.create_task(name="Parent")
        child = await task_manager.create_subtask(
            parent_id=parent.id, name="Child"
        )

        assert await task_manager.check_subtasks_complete(parent.id) is False

        await task_manager.update_status(child.id, TaskStatus.COMPLETED)

        assert await task_manager.check_subtasks_complete(parent.id) is True

    def test_get_task_tree(self, task_manager: TaskManager) -> None:
        # Synchronously add tasks for tree test
        from agenticflow.tasks.task import Task

        parent = Task(name="Parent")
        child1 = Task(name="Child 1", parent_id=parent.id)
        child2 = Task(name="Child 2", parent_id=parent.id)

        parent.subtask_ids = [child1.id, child2.id]

        task_manager.tasks[parent.id] = parent
        task_manager.tasks[child1.id] = child1
        task_manager.tasks[child2.id] = child2

        tree = task_manager.get_task_tree(parent.id)

        assert tree["name"] == "Parent"
        assert len(tree["subtasks"]) == 2

    async def test_get_stats(self, task_manager: TaskManager) -> None:
        await task_manager.create_task(name="Task 1")
        task2 = await task_manager.create_task(name="Task 2")
        await task_manager.update_status(task2.id, TaskStatus.COMPLETED)

        stats = task_manager.get_stats()

        assert stats["total"] == 2
        assert stats["by_status"]["pending"] == 1
        assert stats["by_status"]["completed"] == 1

    async def test_clear(self, task_manager: TaskManager) -> None:
        await task_manager.create_task(name="Task 1")
        await task_manager.create_task(name="Task 2")

        await task_manager.clear()

        assert len(task_manager.tasks) == 0
