"""
Tests for Scratchpad with Reflexion-style memory.

Tests the working memory system including:
- Todo list management
- Notes and observations
- Error tracking with self-correction
- Failure pattern learning (Reflexion)
- Episodic reflections
"""

import pytest
from datetime import datetime

from agenticflow.agent.scratchpad import (
    Scratchpad,
    TodoItem,
    TodoStatus,
    Note,
    ErrorRecord,
    FailurePattern,
    Reflection,
)


class TestTodoItem:
    """Tests for TodoItem dataclass."""

    def test_create_todo(self) -> None:
        """Test creating a basic todo item."""
        todo = TodoItem(id="todo_1", description="Test task")
        assert todo.id == "todo_1"
        assert todo.description == "Test task"
        assert todo.status == TodoStatus.PENDING
        assert todo.result is None
        assert todo.error is None

    def test_mark_done(self) -> None:
        """Test marking todo as done."""
        todo = TodoItem(id="todo_1", description="Test task")
        todo.mark_done("Task completed successfully")
        
        assert todo.status == TodoStatus.DONE
        assert todo.result == "Task completed successfully"
        assert todo.completed_at is not None

    def test_mark_failed(self) -> None:
        """Test marking todo as failed."""
        todo = TodoItem(id="todo_1", description="Test task")
        todo.mark_failed("Something went wrong")
        
        assert todo.status == TodoStatus.FAILED
        assert todo.error == "Something went wrong"
        assert todo.completed_at is not None

    def test_mark_in_progress(self) -> None:
        """Test marking todo as in progress."""
        todo = TodoItem(id="todo_1", description="Test task")
        todo.mark_in_progress()
        assert todo.status == TodoStatus.IN_PROGRESS

    def test_to_dict(self) -> None:
        """Test serializing todo to dict."""
        todo = TodoItem(id="todo_1", description="Test task")
        data = todo.to_dict()
        
        assert data["id"] == "todo_1"
        assert data["description"] == "Test task"
        assert data["status"] == "pending"

    def test_str_representation(self) -> None:
        """Test string representation of todos."""
        todo_pending = TodoItem(id="t1", description="Pending task")
        todo_done = TodoItem(id="t2", description="Done task")
        todo_done.mark_done()
        
        assert "○" in str(todo_pending)
        assert "●" in str(todo_done)


class TestScratchpad:
    """Tests for Scratchpad working memory."""

    def test_create_scratchpad(self) -> None:
        """Test creating an empty scratchpad."""
        pad = Scratchpad()
        assert pad.get_all_todos() == []
        assert pad.get_notes() == []
        assert pad.get_goal() is None
        assert pad.get_plan() is None

    def test_add_todo(self) -> None:
        """Test adding a single todo."""
        pad = Scratchpad()
        todo_id = pad.add_todo("Search for information")
        
        assert todo_id.startswith("todo_")
        assert len(pad.get_all_todos()) == 1
        assert pad.get_pending()[0].description == "Search for information"

    def test_add_multiple_todos(self) -> None:
        """Test adding multiple todos at once."""
        pad = Scratchpad()
        ids = pad.add_todos(["Task 1", "Task 2", "Task 3"])
        
        assert len(ids) == 3
        assert len(pad.get_all_todos()) == 3

    def test_mark_todo_done_by_description(self) -> None:
        """Test marking todo done by partial description match."""
        pad = Scratchpad()
        pad.add_todo("Search for Python tutorials")
        
        result = pad.mark_done("Python tutorials", "Found 5 articles")
        
        assert result is True
        todos = pad.get_done()
        assert len(todos) == 1
        assert todos[0].result == "Found 5 articles"

    def test_mark_todo_failed(self) -> None:
        """Test marking todo as failed."""
        pad = Scratchpad()
        pad.add_todo("Connect to API")
        
        result = pad.mark_failed("Connect", "Connection refused")
        
        assert result is True
        failed = pad.get_failed()
        assert len(failed) == 1
        assert failed[0].error == "Connection refused"

    def test_has_pending(self) -> None:
        """Test checking for pending todos."""
        pad = Scratchpad()
        assert pad.has_pending() is False
        
        pad.add_todo("Task 1")
        assert pad.has_pending() is True
        
        pad.mark_done("Task 1")
        assert pad.has_pending() is False

    def test_all_done(self) -> None:
        """Test checking if all todos are done."""
        pad = Scratchpad()
        pad.add_todos(["Task 1", "Task 2"])
        
        assert pad.all_done() is False
        
        pad.mark_done("Task 1")
        assert pad.all_done() is False
        
        pad.mark_done("Task 2")
        assert pad.all_done() is True

    def test_todo_summary(self) -> None:
        """Test todo summary generation."""
        pad = Scratchpad()
        pad.add_todos(["Task 1", "Task 2", "Task 3"])
        pad.mark_done("Task 1")
        pad.mark_failed("Task 2", "Error")
        
        summary = pad.todo_summary()
        assert "1/3 done" in summary
        assert "1 failed" in summary


class TestScratchpadNotes:
    """Tests for scratchpad notes functionality."""

    def test_add_note(self) -> None:
        """Test adding a note."""
        pad = Scratchpad()
        note_id = pad.note("Python is great for data science")
        
        assert note_id.startswith("note_")
        notes = pad.get_notes()
        assert len(notes) == 1
        assert notes[0].content == "Python is great for data science"

    def test_note_categories(self) -> None:
        """Test different note categories."""
        pad = Scratchpad()
        pad.observe("Observation note")
        pad.insight("Insight note")
        pad.question("Question note")
        
        observations = pad.get_notes(category="observation")
        insights = pad.get_notes(category="insight")
        questions = pad.get_notes(category="question")
        
        assert len(observations) == 1
        assert len(insights) == 1
        assert len(questions) == 1

    def test_note_limit(self) -> None:
        """Test that notes are trimmed when over limit."""
        pad = Scratchpad(max_notes=5)
        
        for i in range(10):
            pad.note(f"Note {i}")
        
        notes = pad.get_notes()
        assert len(notes) == 5
        assert notes[0].content == "Note 5"  # Oldest kept

    def test_get_notes_with_limit(self) -> None:
        """Test getting limited notes."""
        pad = Scratchpad()
        for i in range(10):
            pad.note(f"Note {i}")
        
        recent = pad.get_notes(limit=3)
        assert len(recent) == 3
        assert recent[-1].content == "Note 9"  # Most recent


class TestScratchpadErrors:
    """Tests for scratchpad error tracking."""

    def test_record_error(self) -> None:
        """Test recording an error."""
        pad = Scratchpad()
        pad.record_error("search", {"query": "test"}, ValueError("Not found"))
        
        errors = pad.get_recent_errors()
        assert len(errors) == 1
        assert errors[0].tool_name == "search"
        assert errors[0].error_type == "ValueError"

    def test_error_limit(self) -> None:
        """Test that errors are trimmed when over limit."""
        pad = Scratchpad(max_errors=3)
        
        for i in range(5):
            pad.record_error("tool", {}, Exception(f"Error {i}"))
        
        errors = pad.get_recent_errors(limit=10)
        assert len(errors) == 3

    def test_mark_error_corrected(self) -> None:
        """Test marking an error as corrected."""
        pad = Scratchpad()
        pad.record_error("search", {"query": "test"}, ValueError("Bad query"))
        
        pad.mark_error_corrected("search", success=True, strategy="Fixed query format")
        
        errors = pad.get_recent_errors()
        assert errors[0].correction_attempted is True
        assert errors[0].correction_successful is True
        assert errors[0].correction_strategy == "Fixed query format"

    def test_error_context(self) -> None:
        """Test error context generation."""
        pad = Scratchpad()
        pad.record_error("search", {"q": "test"}, ValueError("Invalid"))
        
        context = pad.get_error_context()
        assert "search" in context
        assert "ValueError" in context


class TestReflexionMemory:
    """Tests for Reflexion-style learning features."""

    def test_learn_failure_pattern(self) -> None:
        """Test learning a failure pattern."""
        pad = Scratchpad()
        pad.learn_failure_pattern("api_call", "Connection timeout", fix="Add retry")
        
        patterns = pad.get_failure_patterns()
        assert len(patterns) == 1
        assert patterns[0].tool_name == "api_call"
        assert patterns[0].successful_fix == "Add retry"

    def test_failure_pattern_counting(self) -> None:
        """Test that repeated failures increment counter."""
        pad = Scratchpad()
        pad.learn_failure_pattern("api_call", "Connection timeout")
        pad.learn_failure_pattern("api_call", "Connection timeout")
        pad.learn_failure_pattern("api_call", "Connection timeout")
        
        patterns = pad.get_failure_patterns("api_call")
        assert len(patterns) == 1
        assert patterns[0].failure_count == 3

    def test_error_pattern_generalization(self) -> None:
        """Test that error messages are generalized to patterns."""
        pad = Scratchpad()
        
        # Record similar errors with different specific values
        pad.learn_failure_pattern("file_read", "File '/path/to/file1.txt' not found")
        pad.learn_failure_pattern("file_read", "File '/path/to/file2.txt' not found")
        
        # Should be grouped as one pattern due to generalization
        patterns = pad.get_failure_patterns("file_read")
        assert len(patterns) == 1  # Both generalized to same pattern
        assert patterns[0].failure_count == 2

    def test_get_known_fix(self) -> None:
        """Test retrieving known fix for an error."""
        pad = Scratchpad()
        pad.learn_failure_pattern("api_call", "Rate limit exceeded", fix="Add exponential backoff")
        
        fix = pad.get_known_fix("api_call", "Rate limit exceeded")
        assert fix == "Add exponential backoff"

    def test_get_known_fix_none(self) -> None:
        """Test getting known fix when none exists."""
        pad = Scratchpad()
        fix = pad.get_known_fix("unknown_tool", "Unknown error")
        assert fix is None

    def test_add_reflection(self) -> None:
        """Test adding a reflection."""
        pad = Scratchpad()
        pad.set_goal("Complete data analysis")
        pad.add_reflection("Breaking down complex queries works better", "strategy")
        
        reflections = pad.get_reflections()
        assert len(reflections) == 1
        assert reflections[0].content == "Breaking down complex queries works better"
        assert reflections[0].reflection_type == "strategy"
        assert reflections[0].task_context == "Complete data analysis"

    def test_reflection_types(self) -> None:
        """Test filtering reflections by type."""
        pad = Scratchpad()
        pad.add_reflection("Query succeeded", "success")
        pad.add_reflection("Query failed", "failure")
        pad.add_reflection("Try smaller batches", "insight")
        
        successes = pad.get_reflections(reflection_type="success")
        failures = pad.get_reflections(reflection_type="failure")
        
        assert len(successes) == 1
        assert len(failures) == 1

    def test_reflexion_context(self) -> None:
        """Test generating Reflexion context for LLM."""
        pad = Scratchpad()
        
        # Add patterns with fixes
        pad.learn_failure_pattern("api", "Timeout error", fix="Use smaller batch size")
        
        # Add common pitfall (no fix)
        pad.learn_failure_pattern("parse", "Invalid JSON")
        pad.learn_failure_pattern("parse", "Invalid JSON")
        
        # Add reflection
        pad.add_reflection("Batching improved success rate", "insight")
        
        context = pad.get_reflexion_context()
        
        assert "Learned Fixes" in context
        assert "api" in context
        assert "smaller batch" in context
        assert "Reflections" in context

    def test_failure_patterns_persist_after_clear(self) -> None:
        """Test that failure patterns survive regular clear."""
        pad = Scratchpad()
        pad.add_todo("Task 1")
        pad.note("Note 1")
        pad.learn_failure_pattern("tool", "Error", fix="Fix it")
        pad.add_reflection("Insight", "insight")
        
        pad.clear()  # Clear regular state
        
        # Todos and notes should be gone
        assert len(pad.get_all_todos()) == 0
        assert len(pad.get_notes()) == 0
        
        # But Reflexion memory should persist
        assert len(pad.get_failure_patterns()) == 1
        assert len(pad.get_reflections()) == 1

    def test_clear_all(self) -> None:
        """Test clearing everything including Reflexion memory."""
        pad = Scratchpad()
        pad.learn_failure_pattern("tool", "Error")
        pad.add_reflection("Insight", "insight")
        
        pad.clear_all()
        
        assert len(pad.get_failure_patterns()) == 0
        assert len(pad.get_reflections()) == 0


class TestScratchpadContext:
    """Tests for scratchpad context generation."""

    def test_get_context_empty(self) -> None:
        """Test getting context from empty scratchpad."""
        pad = Scratchpad()
        context = pad.get_context()
        assert context == ""

    def test_get_context_with_goal(self) -> None:
        """Test context includes goal."""
        pad = Scratchpad()
        pad.set_goal("Complete the analysis")
        
        context = pad.get_context()
        assert "Goal" in context
        assert "Complete the analysis" in context

    def test_get_context_with_plan(self) -> None:
        """Test context includes plan."""
        pad = Scratchpad()
        pad.set_plan("Step 1: Search, Step 2: Analyze")
        
        context = pad.get_context()
        assert "Plan" in context

    def test_get_context_full(self) -> None:
        """Test full context generation."""
        pad = Scratchpad()
        pad.set_goal("Research topic")
        pad.add_todos(["Search", "Analyze"])
        pad.mark_done("Search", "Found info")
        pad.note("Important finding")
        pad.record_error("api", {}, ValueError("Error"))
        pad.learn_failure_pattern("api", "Error", fix="Retry")
        
        context = pad.get_context()
        
        assert "Goal" in context
        assert "Progress" in context
        assert "Notes" in context
        assert "errors" in context.lower()

    def test_completion_checklist(self) -> None:
        """Test completion checklist."""
        pad = Scratchpad()
        pad.add_todos(["Task 1", "Task 2"])
        
        checklist = pad.get_completion_checklist()
        assert checklist["all_todos_done"] is False
        assert checklist["pending_count"] == 2
        assert checklist["ready_to_complete"] is False
        
        pad.mark_done("Task 1")
        pad.mark_done("Task 2")
        
        checklist = pad.get_completion_checklist()
        assert checklist["all_todos_done"] is True
        assert checklist["ready_to_complete"] is True


class TestScratchpadSerialization:
    """Tests for scratchpad serialization."""

    def test_to_dict(self) -> None:
        """Test serializing scratchpad to dict."""
        pad = Scratchpad()
        pad.set_goal("Test goal")
        pad.add_todo("Task 1")
        pad.note("Note 1")
        pad.record_error("tool", {}, ValueError("Error"))
        pad.learn_failure_pattern("tool", "Error pattern")
        pad.add_reflection("Insight", "insight")
        
        data = pad.to_dict()
        
        assert data["goal"] == "Test goal"
        assert len(data["todos"]) == 1
        assert len(data["notes"]) == 1
        assert len(data["errors"]) == 1
        assert len(data["failure_patterns"]) == 1
        assert len(data["reflections"]) == 1
