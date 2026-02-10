"""Test Agent Cognitive Compressor (ACC) implementation."""

import pytest

from cogent.memory.acc import (
    AgentCognitiveCompressor,
    BoundedMemoryState,
    MemoryItem,
    SemanticForgetGate,
)


class TestMemoryItem:
    """Test MemoryItem dataclass."""

    def test_create_memory_item(self):
        """Test creating a memory item."""
        item = MemoryItem(
            content="User wants to book a flight to Paris",
            type="constraint",
            relevance=1.0,
        )

        assert item.content == "User wants to book a flight to Paris"
        assert item.type == "constraint"
        assert item.relevance == 1.0
        assert not item.verified

    def test_access_updates_timestamp(self):
        """Test that access() updates last_accessed."""
        item = MemoryItem(content="Test", type="entity")
        original_time = item.last_accessed

        # Access the item
        item.access()

        assert item.last_accessed >= original_time

    def test_serialization(self):
        """Test to_dict/from_dict serialization."""
        item = MemoryItem(
            content="Test memory",
            type="entity",
            relevance=0.8,
            verified=True,
        )

        # Serialize
        data = item.to_dict()
        assert data["content"] == "Test memory"
        assert data["type"] == "entity"
        assert data["relevance"] == 0.8
        assert data["verified"] is True

        # Deserialize
        restored = MemoryItem.from_dict(data)
        assert restored.content == item.content
        assert restored.type == item.type
        assert restored.relevance == item.relevance
        assert restored.verified == item.verified


class TestBoundedMemoryState:
    """Test BoundedMemoryState."""

    def test_create_empty_state(self):
        """Test creating empty bounded state."""
        state = BoundedMemoryState()

        assert state.total_items == 0
        assert not state.is_full
        assert state.utilization == 0.0

    def test_add_items(self):
        """Test adding items to bounded state."""
        state = BoundedMemoryState()

        # Add constraints
        state.constraints.append(
            MemoryItem(content="Book flight to Paris", type="constraint")
        )
        state.constraints.append(MemoryItem(content="Under $500", type="constraint"))

        # Add entities
        state.entities.append(MemoryItem(content="Paris", type="entity"))

        assert state.total_items == 3
        assert len(state.constraints) == 2
        assert len(state.entities) == 1

    def test_utilization(self):
        """Test memory utilization calculation."""
        state = BoundedMemoryState(
            max_constraints=10,
            max_entities=10,
            max_actions=10,
            max_context=10,
        )

        # Add 20 items (50% utilization)
        for i in range(20):
            state.entities.append(MemoryItem(content=f"Entity {i}", type="entity"))

        assert state.total_items == 20
        assert state.utilization == 0.5  # 20/40 = 0.5

    def test_serialization(self):
        """Test state serialization."""
        state = BoundedMemoryState()
        state.constraints.append(
            MemoryItem(content="Test constraint", type="constraint")
        )
        state.entities.append(MemoryItem(content="Test entity", type="entity"))

        # Serialize
        data = state.to_dict()
        assert len(data["constraints"]) == 1
        assert len(data["entities"]) == 1

        # Deserialize
        restored = BoundedMemoryState.from_dict(data)
        assert len(restored.constraints) == 1
        assert len(restored.entities) == 1
        assert restored.constraints[0].content == "Test constraint"


class TestSemanticForgetGate:
    """Test SemanticForgetGate."""

    def test_create_forget_gate(self):
        """Test creating forget gate."""
        gate = SemanticForgetGate(decay_rate=0.1)
        assert gate.decay_rate == 0.1

    def test_simple_similarity(self):
        """Test simple keyword-based similarity."""
        gate = SemanticForgetGate()

        score = gate._simple_similarity("book flight to paris", "paris travel flight")
        assert score > 0  # Should have some overlap

        score = gate._simple_similarity("hello world", "goodbye moon")
        assert score == 0  # No overlap

    def test_compute_relevance(self):
        """Test relevance scoring."""
        gate = SemanticForgetGate()

        item = MemoryItem(
            content="Book flight to Paris",
            type="constraint",  # High priority
            relevance=1.0,
        )

        score = gate.compute_relevance(item, "Find flights to Paris")
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be relevant

    def test_should_forget(self):
        """Test forget decision."""
        gate = SemanticForgetGate()

        # High relevance, verified - don't forget
        item1 = MemoryItem(
            content="Important constraint",
            type="constraint",
            relevance=0.9,
            verified=True,
        )
        assert not gate.should_forget(item1, threshold=0.3)

        # Low relevance, not verified - forget
        item2 = MemoryItem(
            content="Old context",
            type="context",
            relevance=0.2,
            verified=False,
        )
        assert gate.should_forget(item2, threshold=0.3)

    def test_prune_memory(self):
        """Test pruning to max size."""
        gate = SemanticForgetGate()

        # Create 10 items
        items = [
            MemoryItem(content=f"Item {i}", type="entity", relevance=i / 10)
            for i in range(10)
        ]

        # Prune to keep only 5
        kept = gate.prune_memory(items, max_size=5, current_task="test task")

        assert len(kept) == 5
        # Should keep highest relevance items
        assert all(item.relevance >= 0.5 for item in kept[:5])


class TestAgentCognitiveCompressor:
    """Test AgentCognitiveCompressor."""

    def test_create_acc(self):
        """Test creating ACC."""
        state = BoundedMemoryState()
        gate = SemanticForgetGate()
        acc = AgentCognitiveCompressor(state=state, forget_gate=gate)

        assert acc.state == state
        assert acc.forget_gate == gate

    @pytest.mark.asyncio
    async def test_extract_artifacts(self):
        """Test artifact extraction from conversation turn."""
        state = BoundedMemoryState()
        acc = AgentCognitiveCompressor(state=state)

        artifacts = await acc._extract_artifacts(
            user_message="I need to book a flight to Paris under $500",
            assistant_message="I'll help you find affordable flights to Paris.",
            tool_calls=[{"name": "search_flights", "args": {"destination": "Paris"}}],
        )

        assert len(artifacts) > 0
        # Should extract constraint (need, requirement)
        constraints = [a for a in artifacts if a.type == "constraint"]
        assert len(constraints) > 0

    @pytest.mark.asyncio
    async def test_update_from_turn(self):
        """Test updating ACC from conversation turn."""
        state = BoundedMemoryState()
        acc = AgentCognitiveCompressor(state=state)

        await acc.update_from_turn(
            user_message="Book a flight to Paris",
            assistant_message="I'll search for flights.",
            tool_calls=[{"name": "search_flights", "args": {}}],
            current_task="Help with travel booking",
        )

        # Should have extracted some memories
        assert state.total_items > 0

    @pytest.mark.asyncio
    async def test_prune_maintains_bounds(self):
        """Test that pruning maintains bounded size."""
        state = BoundedMemoryState(max_entities=5)
        acc = AgentCognitiveCompressor(state=state)

        # Add many entities
        for i in range(20):
            state.entities.append(MemoryItem(content=f"Entity {i}", type="entity"))

        # Prune
        await acc._prune_state("test task")

        # Should be within bounds
        assert len(state.entities) <= state.max_entities

    def test_format_for_prompt(self):
        """Test formatting memory for LLM context."""
        state = BoundedMemoryState()
        state.constraints.append(
            MemoryItem(content="Book flight under $500", type="constraint")
        )
        state.entities.append(MemoryItem(content="Paris", type="entity"))
        state.actions.append(MemoryItem(content="Called search_flights", type="action"))

        acc = AgentCognitiveCompressor(state=state)
        formatted = acc.format_for_prompt("Find flights")

        # Should have sections
        assert "Task Requirements" in formatted
        assert "Known Facts" in formatted
        assert "Previous Actions" in formatted
        assert "Book flight under $500" in formatted

    def test_get_stats(self):
        """Test memory statistics."""
        state = BoundedMemoryState()
        state.constraints.append(MemoryItem(content="Test", type="constraint"))
        state.entities.append(MemoryItem(content="Test", type="entity"))

        acc = AgentCognitiveCompressor(state=state)
        stats = acc.get_stats()

        assert stats["total_items"] == 2
        assert stats["constraints"] == 1
        assert stats["entities"] == 1
        assert 0.0 <= stats["utilization"] <= 1.0
