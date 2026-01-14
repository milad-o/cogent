"""Tests for topology context management strategies."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agenticflow.topologies import (
    Mesh,
    AgentConfig,
    SlidingWindowStrategy,
    SummarizationStrategy,
    StructuredHandoffStrategy,
    StructuredHandoff,
    BlackboardStrategy,
    CompositeStrategy,
)


# ==================== Fixtures ====================


@pytest.fixture
def sample_history():
    """Sample round history for testing."""
    return [
        {"agent_a": "Round 1 from A", "agent_b": "Round 1 from B"},
        {"agent_a": "Round 2 from A", "agent_b": "Round 2 from B"},
        {"agent_a": "Round 3 from A", "agent_b": "Round 3 from B"},
        {"agent_a": "Round 4 from A", "agent_b": "Round 4 from B"},
    ]


@pytest.fixture
def mock_model():
    """Create a mock chat model."""
    model = MagicMock()
    response = MagicMock()
    response.content = "Summary of the discussion"
    model.ainvoke = AsyncMock(return_value=response)
    return model


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    def _create(name: str):
        agent = MagicMock()
        agent.name = name
        agent.run = AsyncMock(return_value=f"Output from {name}")
        return agent
    return _create


# ==================== SlidingWindowStrategy Tests ====================


class TestSlidingWindowStrategy:
    """Tests for SlidingWindowStrategy."""

    @pytest.mark.asyncio
    async def test_keeps_only_n_rounds(self, sample_history):
        """Test that only last N rounds are kept."""
        strategy = SlidingWindowStrategy(max_rounds=2)
        
        context = await strategy.build_context(sample_history, 5, "test task")
        
        # Should not contain older rounds
        assert "Round 1" not in context
        assert "Round 2" not in context
        # Should contain recent rounds
        assert "Round 3" in context
        assert "Round 4" in context

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_history(self):
        """Test empty history returns empty string."""
        strategy = SlidingWindowStrategy(max_rounds=3)
        
        context = await strategy.build_context([], 1, "test task")
        
        assert context == ""

    @pytest.mark.asyncio
    async def test_handles_fewer_rounds_than_max(self, sample_history):
        """Test when history has fewer rounds than max."""
        strategy = SlidingWindowStrategy(max_rounds=10)
        
        context = await strategy.build_context(sample_history[:2], 3, "test task")
        
        # Should contain all available rounds
        assert "Round 1" in context
        assert "Round 2" in context

    def test_default_max_rounds(self):
        """Test default max_rounds is 3."""
        strategy = SlidingWindowStrategy()
        assert strategy.max_rounds == 3


# ==================== SummarizationStrategy Tests ====================


class TestSummarizationStrategy:
    """Tests for SummarizationStrategy."""

    @pytest.mark.asyncio
    async def test_keeps_recent_rounds_full(self, sample_history, mock_model):
        """Test that recent rounds are kept in full."""
        strategy = SummarizationStrategy(model=mock_model, keep_full_rounds=2)
        
        context = await strategy.build_context(sample_history, 5, "test task")
        
        # Recent rounds should be in full
        assert "Round 3" in context or "Round 4" in context

    @pytest.mark.asyncio
    async def test_summarizes_older_rounds(self, sample_history, mock_model):
        """Test that older rounds are summarized."""
        strategy = SummarizationStrategy(model=mock_model, keep_full_rounds=1)
        
        context = await strategy.build_context(sample_history, 5, "test task")
        
        # Model should have been called for summarization
        assert mock_model.ainvoke.called

    @pytest.mark.asyncio
    async def test_caches_summary(self, sample_history, mock_model):
        """Test that summary is cached across calls."""
        strategy = SummarizationStrategy(model=mock_model, keep_full_rounds=1)
        
        # First call
        await strategy.build_context(sample_history[:3], 4, "test task")
        call_count_1 = mock_model.ainvoke.call_count
        
        # Second call with same history - should use cache
        await strategy.build_context(sample_history[:3], 4, "test task")
        call_count_2 = mock_model.ainvoke.call_count
        
        # Should not have called model again
        assert call_count_2 == call_count_1


# ==================== StructuredHandoffStrategy Tests ====================


class TestStructuredHandoffStrategy:
    """Tests for StructuredHandoffStrategy."""

    @pytest.mark.asyncio
    async def test_extracts_structured_data(self, mock_model):
        """Test extraction of structured data."""
        mock_model.ainvoke.return_value.content = """{
            "decisions": ["Use Python"],
            "key_findings": ["Performance is good"],
            "open_questions": ["Which database?"],
            "action_items": ["Research options"]
        }"""
        
        strategy = StructuredHandoffStrategy(model=mock_model)
        history = [{"agent_a": "I think we should use Python for this project"}]
        
        context = await strategy.build_context(history, 2, "test task")
        
        assert "DECISIONS" in context
        assert "Python" in context

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self, mock_model):
        """Test handling of invalid JSON response."""
        mock_model.ainvoke.return_value.content = "Not valid JSON"
        
        strategy = StructuredHandoffStrategy(model=mock_model)
        history = [{"agent_a": "Some output"}]
        
        # Should not raise
        context = await strategy.build_context(history, 2, "test task")
        # Should return empty or minimal context
        assert isinstance(context, str)


class TestStructuredHandoff:
    """Tests for StructuredHandoff dataclass."""

    def test_to_context_formats_correctly(self):
        """Test context formatting."""
        handoff = StructuredHandoff(
            decisions=["Decision 1"],
            key_findings=["Finding 1"],
            open_questions=["Question 1"],
            action_items=["Action 1"],
        )
        
        context = handoff.to_context()
        
        assert "DECISIONS" in context
        assert "Decision 1" in context
        assert "KEY FINDINGS" in context
        assert "OPEN QUESTIONS" in context

    def test_empty_handoff(self):
        """Test empty handoff returns empty string."""
        handoff = StructuredHandoff()
        assert handoff.to_context() == ""


# ==================== BlackboardStrategy Tests ====================


class TestBlackboardStrategy:
    """Tests for BlackboardStrategy."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory."""
        memory = MagicMock()
        memory.recall = AsyncMock(side_effect=lambda k: f"Value for {k}" if k == "decisions" else None)
        memory.remember = AsyncMock()
        return memory

    @pytest.mark.asyncio
    async def test_reads_specific_keys(self, mock_memory, sample_history):
        """Test that only specified keys are read."""
        strategy = BlackboardStrategy(memory=mock_memory, keys=["decisions", "findings"])
        
        context = await strategy.build_context(sample_history, 5, "test task")
        
        # Should have called recall for each key
        assert mock_memory.recall.call_count == 2
        # Should include value that was found
        assert "decisions" in context.lower()

    @pytest.mark.asyncio
    async def test_helper_methods(self, mock_memory):
        """Test write and read helper methods."""
        strategy = BlackboardStrategy(memory=mock_memory)
        
        await strategy.write("test_key", "test_value")
        mock_memory.remember.assert_called_with("test_key", "test_value")
        
        await strategy.read("test_key")
        mock_memory.recall.assert_called_with("test_key")


# ==================== CompositeStrategy Tests ====================


class TestCompositeStrategy:
    """Tests for CompositeStrategy."""

    @pytest.mark.asyncio
    async def test_combines_strategies(self, sample_history):
        """Test that multiple strategies are combined."""
        strategy1 = SlidingWindowStrategy(max_rounds=1)
        strategy2 = SlidingWindowStrategy(max_rounds=2)
        
        composite = CompositeStrategy(strategies=[strategy1, strategy2])
        
        context = await composite.build_context(sample_history, 5, "test task")
        
        # Should contain output from both strategies
        assert "PREVIOUS ROUNDS" in context  # Multiple instances due to separator


# ==================== Mesh Integration Tests ====================


class TestMeshWithContextStrategy:
    """Tests for Mesh topology with context strategies."""

    def test_mesh_has_default_strategy(self, mock_agent):
        """Test that Mesh defaults to SlidingWindowStrategy."""
        mesh = Mesh(
            agents=[AgentConfig(agent=mock_agent("a"))],
            max_rounds=2,
        )
        
        assert mesh.context_strategy is not None
        assert isinstance(mesh.context_strategy, SlidingWindowStrategy)

    def test_mesh_accepts_custom_strategy(self, mock_agent):
        """Test that Mesh accepts custom strategy."""
        custom_strategy = SlidingWindowStrategy(max_rounds=5)
        
        mesh = Mesh(
            agents=[AgentConfig(agent=mock_agent("a"))],
            context_strategy=custom_strategy,
        )
        
        assert mesh.context_strategy is custom_strategy
        assert mesh.context_strategy.max_rounds == 5

    @pytest.mark.asyncio
    async def test_mesh_uses_strategy(self, mock_agent):
        """Test that Mesh actually uses the context strategy."""
        # Create a mock strategy to verify it's called
        mock_strategy = MagicMock()
        mock_strategy.build_context = AsyncMock(return_value="Custom context")
        
        mesh = Mesh(
            agents=[
                AgentConfig(agent=mock_agent("a")),
                AgentConfig(agent=mock_agent("b")),
            ],
            max_rounds=2,
            context_strategy=mock_strategy,
        )
        
        await mesh.run("Test task")
        
        # Strategy should have been called (once per round after first)
        assert mock_strategy.build_context.called
