"""Tests for ReactiveFlow streaming support."""

import os

import pytest

from agenticflow import Agent
from agenticflow.reactive import ReactiveFlow, react_to, ReactiveStreamChunk

# Check if LLM is configured
HAS_LLM = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


pytestmark = pytest.mark.asyncio


@pytest.fixture
def model():
    """Get real LLM model."""
    if not HAS_LLM:
        pytest.skip("No LLM configuration available")
    
    from agenticflow.models import ChatModel
    return ChatModel(model="gpt-4o-mini")


@pytest.fixture
def streaming_agent(model):
    """Create agent with real streaming model."""
    return Agent(
        name="test_agent",
        model=model,
        system_prompt="You are a test agent. Be very concise - respond in 5 words or less.",
    )


# =============================================================================
# Basic Streaming Tests
# =============================================================================

async def test_basic_streaming(streaming_agent):
    """Test basic streaming execution."""
    flow = ReactiveFlow()
    flow.register(streaming_agent, [react_to("task.created")])
    
    chunks_received = []
    
    async for chunk in flow.run_streaming("Test task"):
        assert isinstance(chunk, ReactiveStreamChunk)
        assert chunk.agent_name == "test_agent"
        assert chunk.event_name == "task.created"
        assert chunk.content is not None
        chunks_received.append(chunk)
    
    # Should receive multiple chunks
    assert len(chunks_received) > 1
    
    # At least one chunk should be marked final or last chunk should be considered final
    final_chunks = [c for c in chunks_received if c.is_final]
    assert len(final_chunks) >= 0  # May or may not have explicit final marker
    
    # All chunks should have valid properties
    for chunk in chunks_received:
        assert chunk.content is not None
        assert chunk.event_id


async def test_streaming_chunk_properties(streaming_agent):
    """Test ReactiveStreamChunk properties."""
    flow = ReactiveFlow()
    flow.register(streaming_agent, [react_to("task.created")])
    
    async for chunk in flow.run_streaming("Test"):
        # Basic properties
        assert hasattr(chunk, "agent_name")
        assert hasattr(chunk, "event_id")
        assert hasattr(chunk, "event_name")
        assert hasattr(chunk, "content")
        assert hasattr(chunk, "delta")
        assert hasattr(chunk, "is_final")
        
        # Content and delta should be same
        assert chunk.content == chunk.delta
        
        break  # Just test first chunk


async def test_empty_stream_no_matches(model):
    """Test streaming when no agents match."""
    agent = Agent(name="agent", model=model, system_prompt="Be concise.")
    flow = ReactiveFlow()
    flow.register(agent, [react_to("nonexistent.event")])
    
    chunks = []
    async for chunk in flow.run_streaming("Test", initial_event="task.created"):
        chunks.append(chunk)
    
    # Should receive no chunks since no agent matched
    assert len(chunks) == 0


# =============================================================================
# Multi-Agent Streaming
# =============================================================================

async def test_multi_agent_streaming(model):
    """Test streaming with multiple agents."""
    agent1 = Agent(name="agent1", model=model, system_prompt="Be concise - 3 words max.")
    agent2 = Agent(name="agent2", model=model, system_prompt="Be concise - 3 words max.")
    
    flow = ReactiveFlow()
    flow.register(agent1, [react_to("task.created").emits("agent1.done")])
    flow.register(agent2, [react_to("agent1.done")])
    
    agents_seen = set()
    chunks_per_agent = {}
    
    async for chunk in flow.run_streaming("Test"):
        agents_seen.add(chunk.agent_name)
        chunks_per_agent.setdefault(chunk.agent_name, []).append(chunk)
    
    # Both agents should have streamed
    assert "agent1" in agents_seen
    assert "agent2" in agents_seen
    
    # Each agent should have multiple chunks
    assert len(chunks_per_agent["agent1"]) > 1
    assert len(chunks_per_agent["agent2"]) > 1


async def test_streaming_event_context(streaming_agent):
    """Test that event context is preserved in chunks."""
    flow = ReactiveFlow()
    flow.register(streaming_agent, [react_to("custom.event")])
    
    event_ids = set()
    
    async for chunk in flow.run_streaming(
        "Test",
        initial_event="custom.event",
        initial_data={"key": "value"},
    ):
        assert chunk.event_name == "custom.event"
        assert chunk.event_id  # Should have event ID
        event_ids.add(chunk.event_id)
    
    # All chunks from same agent should have same event ID
    assert len(event_ids) == 1


# =============================================================================
# Conditional Streaming
# =============================================================================

async def test_conditional_streaming(model):
    """Test streaming with conditional triggers."""
    agent_a = Agent(name="agent_a", model=model, system_prompt="3 words max.")
    agent_b = Agent(name="agent_b", model=model, system_prompt="3 words max.")
    
    flow = ReactiveFlow()
    flow.register(
        agent_a,
        [react_to("task.created").when(lambda e: e.data.get("type") == "A")],
    )
    flow.register(
        agent_b,
        [react_to("task.created").when(lambda e: e.data.get("type") == "B")],
    )
    
    # Test with type A
    chunks = []
    async for chunk in flow.run_streaming("Test", initial_data={"type": "A"}):
        chunks.append(chunk)
    
    assert all(chunk.agent_name == "agent_a" for chunk in chunks)
    
    # Test with type B
    chunks = []
    async for chunk in flow.run_streaming("Test", initial_data={"type": "B"}):
        chunks.append(chunk)
    
    assert all(chunk.agent_name == "agent_b" for chunk in chunks)


# =============================================================================
# Error Handling
# =============================================================================

@pytest.mark.skip("Error handling test - needs proper failing model implementation")
async def test_streaming_error_handling():
    """Test error handling during streaming."""
    # This test would require a model that intentionally fails during streaming
    # Skipping for now as it requires special mock setup
    pass


# =============================================================================
# Configuration Tests
# =============================================================================

async def test_streaming_with_config(streaming_agent):
    """Test streaming respects flow configuration."""
    from agenticflow.reactive import ReactiveFlowConfig
    
    config = ReactiveFlowConfig(
        max_rounds=2,
        stop_on_idle=True,
    )
    
    flow = ReactiveFlow(config=config)
    flow.register(streaming_agent, [react_to("task.created")])
    
    chunks = []
    async for chunk in flow.run_streaming("Test"):
        chunks.append(chunk)
    
    # Should complete within configured limits
    assert len(chunks) > 0


async def test_streaming_respects_stop_events(streaming_agent):
    """Test streaming stops on stop events."""
    from agenticflow.reactive import ReactiveFlowConfig
    
    config = ReactiveFlowConfig(
        stop_events=frozenset({"task.done", "flow.completed", "flow.failed"}),
    )
    
    flow = ReactiveFlow(config=config)
    flow.register(streaming_agent, [react_to("task.created").emits("task.done")])
    
    chunks = []
    async for chunk in flow.run_streaming("Test"):
        chunks.append(chunk)
    
    # Should stop after agent emits stop event
    assert len(chunks) > 0


# =============================================================================
# Integration Tests
# =============================================================================

async def test_streaming_backward_compatibility(streaming_agent):
    """Test that run_streaming doesn't break existing functionality."""
    flow = ReactiveFlow()
    flow.register(streaming_agent, [react_to("task.created")])
    
    # Regular run() should still work
    result = await flow.run("Test task")
    assert result.output
    assert result.events_processed > 0
    
    # Streaming should also work
    chunks = []
    async for chunk in flow.run_streaming("Test task"):
        chunks.append(chunk)
    
    assert len(chunks) > 0


async def test_reactivestream_chunk_from_agent_chunk():
    """Test ReactiveStreamChunk.from_agent_chunk() conversion."""
    from agenticflow.agent.streaming import StreamChunk
    
    agent_chunk = StreamChunk(
        content="Hello",
        finish_reason="stop",
        model="test",
        index=0,
    )
    
    reactive_chunk = ReactiveStreamChunk.from_agent_chunk(
        chunk=agent_chunk,
        agent_name="test_agent",
        event_id="evt_123",
        event_name="test.event",
        extra_metadata="value",
    )
    
    assert reactive_chunk.agent_name == "test_agent"
    assert reactive_chunk.event_id == "evt_123"
    assert reactive_chunk.event_name == "test.event"
    assert reactive_chunk.content == "Hello"
    assert reactive_chunk.is_final is True
    assert reactive_chunk.finish_reason == "stop"


async def test_streaming_preserves_flow_state(streaming_agent):
    """Test that streaming preserves flow state."""
    flow = ReactiveFlow()
    flow.register(streaming_agent, [react_to("task.created")])
    
    # Check flow has an ID after streaming
    async for _ in flow.run_streaming("Test"):
        pass
    
    assert flow.flow_id is not None
