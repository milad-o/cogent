"""Tests for Agent-to-Agent (A2A) communication in ReactiveFlow."""

import asyncio
from pathlib import Path

import pytest

from agenticflow import Agent, ChatModel
from agenticflow.reactive import (
    AgentRequest,
    AgentResponse,
    ReactiveContext,
    ReactiveFlow,
    create_request,
    create_response,
    react_to,
)
from agenticflow.events import Event


pytestmark = pytest.mark.asyncio


# Fixtures
@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment or tests/.env file."""
    import os

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


# =============================================================================
# AgentRequest Tests
# =============================================================================


def test_agent_request_creation():
    """Test creating an agent request."""
    request = AgentRequest(
        from_agent="coordinator",
        to_agent="specialist",
        task="analyze data",
        data={"file": "data.csv"},
    )

    assert request.from_agent == "coordinator"
    assert request.to_agent == "specialist"
    assert request.task == "analyze data"
    assert request.data == {"file": "data.csv"}
    assert request.correlation_id  # Auto-generated
    assert request.reply_to == "agent.response"


def test_agent_request_to_event():
    """Test converting request to event."""
    request = AgentRequest(
        from_agent="coordinator",
        to_agent="specialist",
        task="process task",
        correlation_id="test123",
    )

    event = request.to_event()

    assert event.name == "agent.request"
    assert event.data["from_agent"] == "coordinator"
    assert event.data["to_agent"] == "specialist"
    assert event.data["task"] == "process task"
    assert event.correlation_id == "test123"


def test_create_request_helper():
    """Test create_request helper function."""
    request = create_request(
        from_agent="agent1",
        to_agent="agent2",
        task="do something",
        data={"key": "value"},
        timeout_ms=5000,
    )

    assert isinstance(request, AgentRequest)
    assert request.from_agent == "agent1"
    assert request.to_agent == "agent2"
    assert request.task == "do something"
    assert request.data == {"key": "value"}
    assert request.timeout_ms == 5000


# =============================================================================
# AgentResponse Tests
# =============================================================================


def test_agent_response_creation():
    """Test creating an agent response."""
    response = AgentResponse(
        from_agent="specialist",
        to_agent="coordinator",
        correlation_id="test123",
        result={"analysis": "complete"},
        success=True,
    )

    assert response.from_agent == "specialist"
    assert response.to_agent == "coordinator"
    assert response.correlation_id == "test123"
    assert response.result == {"analysis": "complete"}
    assert response.success is True
    assert response.error is None


def test_agent_response_to_event():
    """Test converting response to event."""
    response = AgentResponse(
        from_agent="specialist",
        to_agent="coordinator",
        correlation_id="test123",
        result="done",
    )

    event = response.to_event()

    assert event.name == "agent.response"
    assert event.data["from_agent"] == "specialist"
    assert event.data["to_agent"] == "coordinator"
    assert event.data["result"] == "done"
    assert event.correlation_id == "test123"


def test_create_response_helper():
    """Test create_response helper function."""
    response = create_response(
        from_agent="agent2",
        to_agent="agent1",
        correlation_id="test456",
        result={"status": "ok"},
        success=True,
    )

    assert isinstance(response, AgentResponse)
    assert response.from_agent == "agent2"
    assert response.to_agent == "agent1"
    assert response.correlation_id == "test456"
    assert response.result == {"status": "ok"}
    assert response.success is True


def test_response_with_error():
    """Test response with error."""
    response = create_response(
        from_agent="agent2",
        to_agent="agent1",
        correlation_id="test789",
        result=None,
        success=False,
        error="Something went wrong",
    )

    assert response.success is False
    assert response.error == "Something went wrong"


# =============================================================================
# ReactiveContext Tests
# =============================================================================


def test_reactive_context_creation():
    """Test creating a reactive context."""
    event = Event(name="test.event", data={"key": "value"})
    context = ReactiveContext(
        current_agent="agent1",
        event=event,
        task="test task",
        data={"shared": "data"},
    )

    assert context.current_agent == "agent1"
    assert context.event == event
    assert context.task == "test task"
    assert context.data == {"shared": "data"}


def test_reactive_context_dict_access():
    """Test dict-like access to context data."""
    context = ReactiveContext(
        current_agent="agent1",
        event=Event(name="test", data={}),
        task="task",
        data={"key1": "value1", "key2": "value2"},
    )

    assert context["key1"] == "value1"
    assert context.get("key2") == "value2"
    assert context.get("nonexistent", "default") == "default"
    assert "key1" in context
    assert "nonexistent" not in context

    context["key3"] = "value3"
    assert context["key3"] == "value3"


async def test_reactive_context_delegate_to_without_wait():
    """Test delegating without waiting for response."""
    event = Event(name="test.event", data={})
    event_queue = asyncio.Queue()
    
    context = ReactiveContext(
        current_agent="coordinator",
        event=event,
        task="task",
        event_queue=event_queue,
    )

    result = await context.delegate_to(
        agent_name="specialist",
        task="process data",
        data={"file": "data.csv"},
        wait=False,
    )

    assert result is None  # No wait, no result
    assert not event_queue.empty()
    
    delegated_event = await event_queue.get()
    assert delegated_event.name == "agent.request"
    assert delegated_event.data["to_agent"] == "specialist"
    assert delegated_event.data["task"] == "process data"


def test_reactive_context_reply():
    """Test creating a reply from context."""
    request_event = Event(
        name="agent.request",
        data={
            "from_agent": "coordinator",
            "to_agent": "specialist",
            "task": "analyze",
            "correlation_id": "test123",
        },
        correlation_id="test123",
    )

    context = ReactiveContext(
        current_agent="specialist",
        event=request_event,
        task="task",
    )

    response = context.reply(result={"status": "done"}, success=True)

    assert response.from_agent == "specialist"
    assert response.to_agent == "coordinator"
    assert response.correlation_id == "test123"
    assert response.result == {"status": "done"}
    assert response.success is True


# =============================================================================
# Integration Tests
# =============================================================================


async def test_basic_delegation_flow(model):
    """Test basic delegation between two agents."""
    coordinator_output = []
    specialist_output = []

    # Define coordinator that delegates
    coordinator = Agent(
        name="coordinator",
        model=model,
        system_prompt="You coordinate tasks. When asked to analyze something, respond with exactly: 'DELEGATING'",
    )

    # Define specialist that processes delegated tasks
    specialist = Agent(
        name="specialist",
        model=model,
        system_prompt="You are a specialist. Respond with exactly: 'PROCESSED'",
    )

    flow = ReactiveFlow()
    flow.register(coordinator, [react_to("task.created")])
    flow.register(specialist, [react_to("agent.request")])

    result = await flow.run(
        "Analyze data",
        initial_event="task.created",
    )

    assert result.events_processed > 0
    assert len(result.reactions) >= 1


async def test_request_response_correlation(model):
    """Test correlation IDs link requests and responses."""
    # Specialist that replies to requests
    specialist = Agent(
        name="specialist",
        model=model,
        system_prompt="You process requests and reply. Say 'DONE' briefly.",
    )

    flow = ReactiveFlow()
    flow.register(specialist, [react_to("agent.request")])

    # Create and emit a request event manually
    request = create_request(
        from_agent="test",
        to_agent="specialist",
        task="do something",
    )

    result = await flow.run(
        "Process this request",
        initial_event="agent.request",
        initial_data=request.to_event().data,
    )

    # Verify specialist was triggered
    assert len(result.reactions) >= 1
    assert result.reactions[0].agent_name == "specialist"


# Skip error tests for now - would require mock setup
async def test_delegation_timeout():
    """Test delegation with timeout."""
    pytest.skip("Timeout test requires special mock setup")


async def test_delegation_error_handling():
    """Test error handling in delegation."""
    pytest.skip("Error handling test requires special mock setup")
