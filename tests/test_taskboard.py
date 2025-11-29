"""
Tests for TaskBoard - task tracking and working memory for agents.

Tests the working memory system including:
- Task management
- Notes and observations
- Error tracking with learning
- Reflections (Reflexion-style memory)
"""

import pytest

from agenticflow.agent.taskboard import (
    TaskBoard,
    TaskBoardConfig,
    Task,
    TaskStatus,
    Note,
    ErrorRecord,
    LearnedPattern,
    Reflection,
)


class TestTask:
    """Tests for Task dataclass."""

    def test_create_task(self) -> None:
        """Test creating a basic task."""
        task = Task(id="task_1", description="Test task")
        assert task.id == "task_1"
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None

    def test_complete_task(self) -> None:
        """Test completing a task."""
        task = Task(id="task_1", description="Test task")
        task.complete("Task done successfully")
        
        assert task.status == TaskStatus.DONE
        assert task.result == "Task done successfully"
        assert task.completed_at is not None

    def test_fail_task(self) -> None:
        """Test failing a task."""
        task = Task(id="task_1", description="Test task")
        task.fail("Something went wrong")
        
        assert task.status == TaskStatus.FAILED
        assert task.error == "Something went wrong"
        assert task.completed_at is not None

    def test_start_task(self) -> None:
        """Test starting a task."""
        task = Task(id="task_1", description="Test task")
        task.start()
        assert task.status == TaskStatus.IN_PROGRESS

    def test_skip_task(self) -> None:
        """Test skipping a task."""
        task = Task(id="task_1", description="Test task")
        task.skip()
        assert task.status == TaskStatus.SKIPPED

    def test_to_dict(self) -> None:
        """Test serializing task to dict."""
        task = Task(id="task_1", description="Test task")
        data = task.to_dict()
        
        assert data["id"] == "task_1"
        assert data["description"] == "Test task"
        assert data["status"] == "pending"

    def test_str_representation(self) -> None:
        """Test string representation of tasks."""
        pending = Task(id="t1", description="Pending task")
        done = Task(id="t2", description="Done task")
        done.complete()
        
        assert "○" in str(pending)
        assert "✓" in str(done)


class TestTaskBoard:
    """Tests for TaskBoard."""

    def test_create_taskboard(self) -> None:
        """Test creating an empty taskboard."""
        board = TaskBoard()
        assert board.get_all_tasks() == []
        assert board.get_notes() == []
        assert board.get_goal() is None

    def test_add_task(self) -> None:
        """Test adding a single task."""
        board = TaskBoard()
        task_id = board.add_task("Search for information")
        
        assert task_id.startswith("task_")
        assert len(board.get_all_tasks()) == 1
        assert board.get_pending()[0].description == "Search for information"

    def test_add_multiple_tasks(self) -> None:
        """Test adding multiple tasks at once."""
        board = TaskBoard()
        ids = board.add_tasks(["Task 1", "Task 2", "Task 3"])
        
        assert len(ids) == 3
        assert len(board.get_all_tasks()) == 3

    def test_complete_task_by_description(self) -> None:
        """Test completing task by partial description match."""
        board = TaskBoard()
        board.add_task("Search for Python tutorials")
        
        result = board.complete_task("Python tutorials", "Found 5 articles")
        
        assert result is True
        tasks = board.get_done()
        assert len(tasks) == 1
        assert tasks[0].result == "Found 5 articles"

    def test_fail_task(self) -> None:
        """Test failing a task."""
        board = TaskBoard()
        board.add_task("Connect to API")
        
        result = board.fail_task("Connect", "Connection refused")
        
        assert result is True
        failed = board.get_failed()
        assert len(failed) == 1
        assert failed[0].error == "Connection refused"

    def test_start_task(self) -> None:
        """Test starting a task."""
        board = TaskBoard()
        board.add_task("Do something")
        
        result = board.start_task("Do something")
        
        assert result is True
        pending = board.get_pending()
        assert len(pending) == 1
        assert pending[0].status == TaskStatus.IN_PROGRESS

    def test_find_task_by_id(self) -> None:
        """Test finding task by ID."""
        board = TaskBoard()
        task_id = board.add_task("My task")
        
        found = board.find_task(task_id)
        assert found is not None
        assert found.description == "My task"

    def test_find_task_by_partial_description(self) -> None:
        """Test finding task by partial description."""
        board = TaskBoard()
        board.add_task("Search for Python tutorials")
        
        found = board.find_task("Python")
        assert found is not None
        assert "Python" in found.description

    def test_is_complete(self) -> None:
        """Test checking if all tasks are complete."""
        board = TaskBoard()
        board.add_tasks(["Task 1", "Task 2"])
        
        assert board.is_complete() is False
        
        board.complete_task("Task 1")
        assert board.is_complete() is False
        
        board.complete_task("Task 2")
        assert board.is_complete() is True


class TestTaskBoardNotes:
    """Tests for taskboard notes."""

    def test_add_note(self) -> None:
        """Test adding a note."""
        board = TaskBoard()
        note_id = board.add_note("Python is great for data science")
        
        assert note_id.startswith("note_")
        notes = board.get_notes()
        assert len(notes) == 1
        assert notes[0].content == "Python is great for data science"

    def test_note_categories(self) -> None:
        """Test different note categories."""
        board = TaskBoard()
        board.add_note("Observation", category="observation")
        board.add_note("Insight", category="insight")
        board.add_note("Question", category="question")
        
        observations = board.get_notes(category="observation")
        insights = board.get_notes(category="insight")
        questions = board.get_notes(category="question")
        
        assert len(observations) == 1
        assert len(insights) == 1
        assert len(questions) == 1

    def test_note_limit(self) -> None:
        """Test that notes are trimmed when over limit."""
        config = TaskBoardConfig(max_notes=5)
        board = TaskBoard(config)
        
        for i in range(10):
            board.add_note(f"Note {i}")
        
        notes = board.get_notes()
        assert len(notes) == 5
        assert notes[0].content == "Note 5"  # Oldest kept

    def test_get_notes_with_limit(self) -> None:
        """Test getting limited notes."""
        board = TaskBoard()
        for i in range(10):
            board.add_note(f"Note {i}")
        
        recent = board.get_notes(limit=3)
        assert len(recent) == 3
        assert recent[-1].content == "Note 9"


class TestTaskBoardErrors:
    """Tests for error tracking."""

    def test_record_error(self) -> None:
        """Test recording an error."""
        board = TaskBoard()
        board.record_error("search", ValueError("Not found"), {"query": "test"})
        
        # Error should be recorded internally
        assert len(board._errors) == 1
        assert board._errors[0].tool_name == "search"
        assert board._errors[0].error_type == "ValueError"

    def test_learn_pattern_from_error(self) -> None:
        """Test that patterns are learned from errors."""
        board = TaskBoard()
        board.record_error("api_call", TimeoutError("Connection timeout"))
        
        patterns = list(board._patterns.values())
        assert len(patterns) == 1
        assert patterns[0].tool_name == "api_call"

    def test_get_known_fix(self) -> None:
        """Test retrieving known fix."""
        board = TaskBoard()
        board.record_error("api_call", TimeoutError("Rate limit exceeded"))
        board.record_fix("api_call", "Rate limit exceeded", "Add exponential backoff")
        
        fix = board.get_known_fix("api_call", "Rate limit exceeded")
        assert fix == "Add exponential backoff"

    def test_get_known_fix_none(self) -> None:
        """Test getting known fix when none exists."""
        board = TaskBoard()
        fix = board.get_known_fix("unknown_tool", "Unknown error")
        assert fix is None


class TestTaskBoardReflections:
    """Tests for reflections (Reflexion-style memory)."""

    def test_add_reflection(self) -> None:
        """Test adding a reflection."""
        board = TaskBoard()
        board.set_goal("Complete data analysis")
        board.add_reflection("Breaking down queries works better", "strategy")
        
        reflections = board.get_reflections()
        assert len(reflections) == 1
        assert reflections[0].content == "Breaking down queries works better"
        assert reflections[0].category == "strategy"

    def test_reflection_categories(self) -> None:
        """Test filtering reflections by category."""
        board = TaskBoard()
        board.add_reflection("Query succeeded", "success")
        board.add_reflection("Query failed", "failure")
        board.add_reflection("Try smaller batches", "insight")
        
        successes = board.get_reflections(category="success")
        failures = board.get_reflections(category="failure")
        
        assert len(successes) == 1
        assert len(failures) == 1

    def test_reflections_persist_after_clear(self) -> None:
        """Test that reflections survive regular clear."""
        board = TaskBoard()
        board.add_task("Task 1")
        board.add_note("Note 1")
        board.add_reflection("Insight", "insight")
        
        board.clear()
        
        # Tasks and notes gone
        assert len(board.get_all_tasks()) == 0
        assert len(board.get_notes()) == 0
        
        # But reflections persist
        assert len(board.get_reflections()) == 1

    def test_clear_all(self) -> None:
        """Test clearing everything including reflections."""
        board = TaskBoard()
        board.record_error("tool", ValueError("Error"))
        board.add_reflection("Insight", "insight")
        
        board.clear_all()
        
        assert len(board._patterns) == 0
        assert len(board.get_reflections()) == 0


class TestTaskBoardSummary:
    """Tests for summary and context generation."""

    def test_summary_empty(self) -> None:
        """Test summary of empty board."""
        board = TaskBoard()
        summary = board.summary()
        assert "Empty" in summary or "empty" in summary.lower()

    def test_summary_with_goal(self) -> None:
        """Test summary includes goal."""
        board = TaskBoard()
        board.set_goal("Complete the analysis")
        
        summary = board.summary()
        assert "Complete the analysis" in summary

    def test_summary_with_tasks(self) -> None:
        """Test summary includes tasks."""
        board = TaskBoard()
        board.add_tasks(["Task 1", "Task 2", "Task 3"])
        board.complete_task("Task 1")
        
        summary = board.summary()
        assert "1/3" in summary
        assert "Task 1" in summary

    def test_get_context(self) -> None:
        """Test context generation for LLM."""
        board = TaskBoard()
        board.set_goal("Research topic")
        board.add_tasks(["Search", "Analyze"])
        board.complete_task("Search", "Found info")
        board.add_note("Important finding")
        
        context = board.get_context()
        
        assert "Goal" in context
        assert "Progress" in context

    def test_verify_completion(self) -> None:
        """Test completion verification."""
        board = TaskBoard()
        board.add_tasks(["Task 1", "Task 2"])
        
        result = board.verify_completion()
        assert result["complete"] is False
        assert result["pending_count"] == 2
        
        board.complete_task("Task 1", "Done")
        board.complete_task("Task 2", "Done")
        
        result = board.verify_completion()
        assert result["complete"] is True


class TestTaskBoardSerialization:
    """Tests for serialization."""

    def test_to_dict(self) -> None:
        """Test serializing to dict."""
        board = TaskBoard()
        board.set_goal("Test goal")
        board.add_task("Task 1")
        board.add_note("Note 1")
        board.add_reflection("Insight", "insight")
        
        data = board.to_dict()
        
        assert data["goal"] == "Test goal"
        assert len(data["tasks"]) == 1
        assert len(data["notes"]) == 1
        assert len(data["reflections"]) == 1


class TestTaskBoardConfig:
    """Tests for TaskBoardConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = TaskBoardConfig()
        assert config.auto_verify is True
        assert config.max_tasks == 50
        assert config.max_notes == 30
        assert config.track_errors is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TaskBoardConfig(
            auto_verify=False,
            max_tasks=10,
            max_notes=5,
        )
        assert config.auto_verify is False
        assert config.max_tasks == 10
        assert config.max_notes == 5

    def test_config_applied(self) -> None:
        """Test that config is applied to board."""
        config = TaskBoardConfig(max_notes=3)
        board = TaskBoard(config)
        
        for i in range(10):
            board.add_note(f"Note {i}")
        
        assert len(board.get_notes()) == 3
