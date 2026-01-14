"""Tests for checkpointing functionality."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agenticflow.reactive import (
    EventFlow,
    EventFlowConfig,
    react_to,
)
from agenticflow.reactive.checkpointer import (
    FlowState,
    MemoryCheckpointer,
    FileCheckpointer,
    generate_checkpoint_id,
    generate_flow_id,
)


# =============================================================================
# FlowState Tests
# =============================================================================


class TestFlowState:
    """Tests for FlowState serialization."""

    def test_to_dict(self) -> None:
        """FlowState serializes to dict."""
        state = FlowState(
            flow_id="test-flow",
            checkpoint_id="cp-123",
            task="Test task",
            events_processed=5,
            last_output="Hello",
            round=3,
        )
        
        data = state.to_dict()
        
        assert data["flow_id"] == "test-flow"
        assert data["checkpoint_id"] == "cp-123"
        assert data["task"] == "Test task"
        assert data["events_processed"] == 5
        assert data["last_output"] == "Hello"
        assert data["round"] == 3

    def test_from_dict(self) -> None:
        """FlowState deserializes from dict."""
        data = {
            "flow_id": "test-flow",
            "checkpoint_id": "cp-456",
            "task": "Another task",
            "events_processed": 10,
            "pending_events": [{"id": "e1", "name": "test.event", "data": {}}],
            "context": {"key": "value"},
            "last_output": "World",
            "round": 5,
            "timestamp": "2024-01-01T00:00:00+00:00",
        }
        
        state = FlowState.from_dict(data)
        
        assert state.flow_id == "test-flow"
        assert state.checkpoint_id == "cp-456"
        assert state.task == "Another task"
        assert state.events_processed == 10
        assert len(state.pending_events) == 1
        assert state.context == {"key": "value"}
        assert state.last_output == "World"
        assert state.round == 5

    def test_round_trip(self) -> None:
        """FlowState survives round-trip serialization."""
        original = FlowState(
            flow_id="flow-1",
            checkpoint_id="cp-1",
            task="Round trip test",
            events_processed=42,
            pending_events=[{"id": "x", "name": "foo", "data": {"a": 1}}],
            context={"user": "test"},
            last_output="Result",
            round=7,
        )
        
        restored = FlowState.from_dict(original.to_dict())
        
        assert restored.flow_id == original.flow_id
        assert restored.checkpoint_id == original.checkpoint_id
        assert restored.task == original.task
        assert restored.events_processed == original.events_processed
        assert restored.pending_events == original.pending_events
        assert restored.context == original.context
        assert restored.last_output == original.last_output
        assert restored.round == original.round


# =============================================================================
# MemoryCheckpointer Tests
# =============================================================================


class TestMemoryCheckpointer:
    """Tests for in-memory checkpointer."""

    @pytest.mark.asyncio
    async def test_save_and_load(self) -> None:
        """Can save and load a checkpoint."""
        checkpointer = MemoryCheckpointer()
        state = FlowState(
            flow_id="f1",
            checkpoint_id="cp1",
            task="Test",
        )
        
        await checkpointer.save(state)
        loaded = await checkpointer.load("cp1")
        
        assert loaded is not None
        assert loaded.flow_id == "f1"
        assert loaded.checkpoint_id == "cp1"

    @pytest.mark.asyncio
    async def test_load_latest(self) -> None:
        """Can load the latest checkpoint for a flow."""
        checkpointer = MemoryCheckpointer()
        
        await checkpointer.save(FlowState(flow_id="f1", checkpoint_id="cp1", task="T", round=1))
        await checkpointer.save(FlowState(flow_id="f1", checkpoint_id="cp2", task="T", round=2))
        await checkpointer.save(FlowState(flow_id="f1", checkpoint_id="cp3", task="T", round=3))
        
        latest = await checkpointer.load_latest("f1")
        
        assert latest is not None
        assert latest.checkpoint_id == "cp3"
        assert latest.round == 3

    @pytest.mark.asyncio
    async def test_list_checkpoints(self) -> None:
        """Can list checkpoints for a flow."""
        checkpointer = MemoryCheckpointer()
        
        await checkpointer.save(FlowState(flow_id="f1", checkpoint_id="cp1", task="T"))
        await checkpointer.save(FlowState(flow_id="f1", checkpoint_id="cp2", task="T"))
        await checkpointer.save(FlowState(flow_id="f2", checkpoint_id="cp3", task="T"))
        
        f1_checkpoints = await checkpointer.list_checkpoints("f1")
        
        assert len(f1_checkpoints) == 2
        assert "cp2" in f1_checkpoints
        assert "cp1" in f1_checkpoints

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """Can delete a checkpoint."""
        checkpointer = MemoryCheckpointer()
        state = FlowState(flow_id="f1", checkpoint_id="cp1", task="T")
        
        await checkpointer.save(state)
        deleted = await checkpointer.delete("cp1")
        loaded = await checkpointer.load("cp1")
        
        assert deleted is True
        assert loaded is None

    @pytest.mark.asyncio
    async def test_max_checkpoints_pruning(self) -> None:
        """Old checkpoints are pruned when limit reached."""
        checkpointer = MemoryCheckpointer(max_checkpoints_per_flow=3)
        
        for i in range(5):
            await checkpointer.save(FlowState(
                flow_id="f1",
                checkpoint_id=f"cp{i}",
                task="T",
            ))
        
        checkpoints = await checkpointer.list_checkpoints("f1")
        
        assert len(checkpoints) == 3
        # Oldest (cp0, cp1) should be pruned
        assert "cp0" not in checkpoints
        assert "cp1" not in checkpoints
        assert "cp4" in checkpoints


# =============================================================================
# FileCheckpointer Tests
# =============================================================================


class TestFileCheckpointer:
    """Tests for file-based checkpointer."""

    @pytest.mark.asyncio
    async def test_save_and_load(self) -> None:
        """Can save and load from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = FileCheckpointer(Path(tmpdir))
            state = FlowState(
                flow_id="f1",
                checkpoint_id="cp1",
                task="Test",
                last_output="Result",
            )
            
            await checkpointer.save(state)
            loaded = await checkpointer.load("cp1")
            
            assert loaded is not None
            assert loaded.flow_id == "f1"
            assert loaded.last_output == "Result"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self) -> None:
        """Loading nonexistent checkpoint returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = FileCheckpointer(Path(tmpdir))
            
            loaded = await checkpointer.load("does-not-exist")
            
            assert loaded is None

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """Can delete a file checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = FileCheckpointer(Path(tmpdir))
            state = FlowState(flow_id="f1", checkpoint_id="cp1", task="T")
            
            await checkpointer.save(state)
            deleted = await checkpointer.delete("cp1")
            loaded = await checkpointer.load("cp1")
            
            assert deleted is True
            assert loaded is None


# =============================================================================
# ID Generation Tests
# =============================================================================


class TestIdGeneration:
    """Tests for ID generation."""

    def test_generate_checkpoint_id(self) -> None:
        """Checkpoint IDs are unique and prefixed."""
        id1 = generate_checkpoint_id()
        id2 = generate_checkpoint_id()
        
        assert id1.startswith("cp_")
        assert id2.startswith("cp_")
        assert id1 != id2

    def test_generate_flow_id(self) -> None:
        """Flow IDs are unique and prefixed."""
        id1 = generate_flow_id()
        id2 = generate_flow_id()
        
        assert id1.startswith("flow_")
        assert id2.startswith("flow_")
        assert id1 != id2


# =============================================================================
# ReactiveFlow Checkpointing Integration Tests
# =============================================================================


class TestReactiveFlowCheckpointing:
    """Tests for checkpointing in ReactiveFlow."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        def _create(name: str = "test_agent") -> MagicMock:
            agent = MagicMock()
            agent.name = name
            agent.run = AsyncMock(
                return_value=MagicMock(output=f"Output from {name}")
            )
            return agent
        return _create

    @pytest.mark.asyncio
    async def test_flow_has_flow_id(self, mock_agent) -> None:
        """Flow generates a flow_id."""
        agent = mock_agent("test")
        flow = EventFlow()
        flow.register(agent, [react_to("task.created")])
        
        result = await flow.run("Test task", initial_event="task.created")
        
        assert result.flow_id is not None
        assert result.flow_id.startswith("flow_")

    @pytest.mark.asyncio
    async def test_flow_with_custom_flow_id(self, mock_agent) -> None:
        """Flow uses custom flow_id from config."""
        agent = mock_agent("test")
        config = EventFlowConfig(flow_id="custom-flow-123")
        flow = EventFlow(config=config)
        flow.register(agent, [react_to("task.created")])
        
        result = await flow.run("Test task", initial_event="task.created")
        
        assert result.flow_id == "custom-flow-123"

    @pytest.mark.asyncio
    async def test_checkpointing_saves_state(self, mock_agent) -> None:
        """Flow saves checkpoints when checkpointer is configured."""
        agent = mock_agent("test")
        checkpointer = MemoryCheckpointer()
        config = EventFlowConfig(checkpoint_every=1)  # Checkpoint every round
        flow = EventFlow(config=config, checkpointer=checkpointer)
        flow.register(agent, [react_to("task.created")])
        
        result = await flow.run("Test task", initial_event="task.created")
        
        # Should have saved at least one checkpoint
        assert result.checkpoint_id is not None
        checkpoints = await checkpointer.list_checkpoints(result.flow_id)
        assert len(checkpoints) >= 1

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, mock_agent) -> None:
        """Can resume flow from checkpoint."""
        agent = mock_agent("test")
        checkpointer = MemoryCheckpointer()
        config = EventFlowConfig(checkpoint_every=1)
        
        # Create a checkpoint manually
        state = FlowState(
            flow_id="resume-test-flow",
            checkpoint_id="cp-resume",
            task="Resume task",
            events_processed=1,
            pending_events=[{"id": "e1", "name": "test.continue", "data": {"task": "Resume task"}}],
            context={},
            last_output="Previous output",
            round=1,
        )
        await checkpointer.save(state)
        
        # Create flow and register agent for the pending event
        flow = EventFlow(config=config, checkpointer=checkpointer)
        flow.register(agent, [react_to("test.continue")])
        
        # Resume from checkpoint
        result = await flow.resume(state)
        
        assert result.flow_id == "resume-test-flow"
        assert agent.run.called  # Agent was triggered by pending event
