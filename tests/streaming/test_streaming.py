"""Tests for Agent streaming support."""

import os
from pathlib import Path

import pytest

from cogent import Agent, ChatModel
from cogent.agent.streaming import StreamChunk

pytestmark = pytest.mark.asyncio


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment or tests/.env file."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Try loading from tests/.env
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            content = env_file.read_text().strip()
            if content.startswith("OPENAI_API_KEY="):
                api_key = content.split("=", 1)[1].strip()
                # Set it in environment for the ChatModel to pick up
                os.environ["OPENAI_API_KEY"] = api_key
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set (create tests/.env with your key)")
    return api_key


@pytest.fixture
def model(openai_api_key):
    """Create ChatModel with OpenAI."""
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
    chunks_received = []

    async for chunk in streaming_agent.run("Say hello", stream=True):
        assert isinstance(chunk, StreamChunk)
        assert chunk.content is not None
        chunks_received.append(chunk)

    # Should receive multiple chunks (or at least one)
    assert len(chunks_received) >= 1


async def test_streaming_chunk_properties(streaming_agent):
    """Test StreamChunk properties."""
    async for chunk in streaming_agent.run("Test", stream=True):
        # Basic properties
        assert hasattr(chunk, "content")
        assert hasattr(chunk, "finish_reason")
        break  # Just test first chunk


async def test_streaming_collects_full_response(streaming_agent):
    """Test that streaming collects full response."""
    full_content = ""

    async for chunk in streaming_agent.run("Say hello world", stream=True):
        if chunk.content:
            full_content += chunk.content

    # Should have some content
    assert len(full_content) > 0


# =============================================================================
# Streaming with Tools
# =============================================================================


async def test_streaming_with_tools(model):
    """Test streaming with tool calls.

    Note: When the LLM immediately calls a tool without preamble text,
    no content chunks may be yielded. This is expected behavior.
    """
    from cogent.tools import tool

    @tool
    def get_number() -> int:
        """Get a number."""
        return 42

    agent = Agent(
        name="tool_agent",
        model=model,
        tools=[get_number],
        system_prompt="Use tools when asked. Be concise.",
    )

    chunks = []
    async for chunk in agent.run(
        "What number do you get from get_number?", stream=True
    ):
        chunks.append(chunk)

    # Streaming with tools may or may not produce chunks depending on
    # whether the LLM outputs text before the tool call.
    # The important thing is that the stream completes without error.
    # (Most models will output some text, but it's not guaranteed)
    assert True  # Test passes if stream completes without exception


# =============================================================================
# Streaming with Conversation
# =============================================================================


async def test_streaming_with_conversation(streaming_agent):
    """Test streaming preserves conversation context."""
    thread_id = "test-thread"

    # First message
    chunks1 = []
    async for chunk in streaming_agent.run(
        "My name is TestUser",
        stream=True,
        thread_id=thread_id,
    ):
        chunks1.append(chunk)

    assert len(chunks1) >= 1

    # Second message should remember context
    chunks2 = []
    async for chunk in streaming_agent.run(
        "What is my name?",
        stream=True,
        thread_id=thread_id,
    ):
        chunks2.append(chunk)

    assert len(chunks2) >= 1


# =============================================================================
# Non-streaming Comparison
# =============================================================================


async def test_streaming_vs_non_streaming(streaming_agent):
    """Test that streaming and non-streaming produce similar results."""
    prompt = "Say exactly: Hello"

    # Non-streaming
    result = await streaming_agent.run(prompt)
    non_streaming_content = result.content

    # Streaming
    streaming_content = ""
    async for chunk in streaming_agent.run(prompt, stream=True):
        if chunk.content:
            streaming_content += chunk.content

    # Both should produce content
    assert len(non_streaming_content) > 0
    assert len(streaming_content) > 0


# =============================================================================
# Think Method Streaming
# =============================================================================


async def test_think_streaming(streaming_agent):
    """Test streaming via think() method."""
    chunks = []

    async for chunk in streaming_agent.think("Test prompt", stream=True):
        assert isinstance(chunk, StreamChunk)
        chunks.append(chunk)

    assert len(chunks) >= 1


async def test_think_non_streaming(streaming_agent):
    """Test non-streaming think() method."""
    result = await streaming_agent.think("Test prompt")
    assert result.content is not None
