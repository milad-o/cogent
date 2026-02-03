"""Tests for executor subagent integration."""

import pytest
from unittest.mock import MagicMock

from cogent.agent.base import Agent
from cogent.core.context import RunContext
from cogent.core.messages import ToolMessage
from cogent.executors.native import NativeExecutor
from cogent.models import BaseChatModel


def create_mock_model():
    """Create a mock model for testing."""
    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.model = "mock-model"
    mock_model.bind_tools = MagicMock(return_value=mock_model)
    return mock_model


@pytest.mark.asyncio
async def test_executor_recognizes_subagent():
    """Test that executor can identify subagent tool calls."""
    model = create_mock_model()
    
    # Create mock subagent that returns a simple response
    class MockSubagent:
        def __init__(self, name):
            self.name = name
            self.config = type('obj', (object,), {'description': f'{name} specialist'})()
        
        async def run(self, task, context=None):
            from cogent.core.response import Response, ResponseMetadata, TokenUsage
            return Response(
                content=f"Result from {self.name}: {task}",
                metadata=ResponseMetadata(
                    agent=self.name,
                    model="gpt-4o-mini",
                    tokens=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                    duration=0.5,
                ),
            )
    
    analyst = MockSubagent("analyst")
    
    # Create coordinator with subagent
    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst},
    )
    
    # Create executor
    executor = NativeExecutor(coordinator)
    
    # Verify subagent is recognized
    assert coordinator._subagent_registry.has_subagent("analyst")


@pytest.mark.asyncio
async def test_executor_runs_subagent():
    """Test that executor can execute subagent and return ToolMessage."""
    model = create_mock_model()
    
    # Create mock subagent
    class MockSubagent:
        def __init__(self, name):
            self.name = name
            self.config = type('obj', (object,), {'description': f'{name} specialist'})()
        
        async def run(self, task, context=None):
            from cogent.core.response import Response, ResponseMetadata, TokenUsage
            return Response(
                content=f"Analyzed: {task}",
                metadata=ResponseMetadata(
                    agent=self.name,
                    model="gpt-4o-mini",
                    tokens=TokenUsage(prompt_tokens=100, completion_tokens=150, total_tokens=250),
                    duration=1.2,
                ),
            )
    
    analyst = MockSubagent("analyst")
    
    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst},
    )
    
    executor = NativeExecutor(coordinator)
    
    # Simulate tool call from LLM
    tool_call = {
        "name": "analyst",
        "args": {"task": "Analyze Q4 sales data"},
        "id": "call_123",
    }
    
    # Execute via executor
    result = await executor._run_single_tool(tool_call, run_context=None)
    
    # Verify result is ToolMessage
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call_123"
    assert "Analyzed: Analyze Q4 sales data" in result.content


@pytest.mark.asyncio
async def test_subagent_response_cached():
    """Test that subagent Response objects are cached in registry."""
    model = create_mock_model()
    
    class MockSubagent:
        def __init__(self, name):
            self.name = name
            self.config = type('obj', (object,), {'description': f'{name} specialist'})()
        
        async def run(self, task, context=None):
            from cogent.core.response import Response, ResponseMetadata, TokenUsage
            return Response(
                content=f"Done: {task}",
                metadata=ResponseMetadata(
                    agent=self.name,
                    model="gpt-4o-mini",
                    tokens=TokenUsage(prompt_tokens=50, completion_tokens=75, total_tokens=125),
                    duration=0.8,
                ),
            )
    
    analyst = MockSubagent("analyst")
    
    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst},
    )
    
    executor = NativeExecutor(coordinator)
    
    # Before execution, no cached responses
    assert len(coordinator._subagent_registry.get_responses()) == 0
    
    # Execute subagent
    tool_call = {
        "name": "analyst",
        "args": {"task": "Test task"},
        "id": "call_456",
    }
    
    await executor._run_single_tool(tool_call)
    
    # After execution, response should be cached
    responses = coordinator._subagent_registry.get_responses()
    assert len(responses) == 1
    assert responses[0].content == "Done: Test task"
    assert responses[0].metadata.tokens.total_tokens == 125


@pytest.mark.asyncio
async def test_subagent_with_context_propagation():
    """Test that RunContext propagates to subagent."""
    model = create_mock_model()
    
    received_context = None
    
    class MockSubagent:
        def __init__(self, name):
            self.name = name
            self.config = type('obj', (object,), {'description': f'{name} specialist'})()
        
        async def run(self, task, context=None):
            nonlocal received_context
            received_context = context
            
            from cogent.core.response import Response, ResponseMetadata
            return Response(
                content="Result",
                metadata=ResponseMetadata(agent=self.name, model="gpt-4o-mini"),
            )
    
    analyst = MockSubagent("analyst")
    
    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst},
    )
    
    executor = NativeExecutor(coordinator)
    
    # Create context
    context = RunContext(query="original query", metadata={"user_id": "123"})
    
    # Execute subagent with context
    tool_call = {
        "name": "analyst",
        "args": {"task": "Test"},
        "id": "call_789",
    }
    
    await executor._run_single_tool(tool_call, run_context=context)
    
    # Verify context was propagated
    assert received_context is context
    assert received_context.query == "original query"
    assert received_context.metadata["user_id"] == "123"


@pytest.mark.asyncio
async def test_subagent_missing_task_parameter():
    """Test error handling when task parameter is missing."""
    model = create_mock_model()
    
    class MockSubagent:
        def __init__(self, name):
            self.name = name
            self.config = type('obj', (object,), {'description': f'{name} specialist'})()
    
    analyst = MockSubagent("analyst")
    
    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst},
    )
    
    executor = NativeExecutor(coordinator)
    
    # Call without task parameter
    tool_call = {
        "name": "analyst",
        "args": {},  # Missing task!
        "id": "call_error",
    }
    
    result = await executor._run_single_tool(tool_call)
    
    # Should return error message
    assert isinstance(result, ToolMessage)
    assert "Error" in result.content
    assert "task" in result.content.lower()
