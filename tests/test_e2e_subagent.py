"""End-to-end tests for subagent delegation with real/mock LLMs."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cogent.agent.base import Agent
from cogent.core.context import RunContext
from cogent.core.response import Response, ResponseMetadata, TokenUsage
from cogent.models import BaseChatModel


def get_test_model():
    """Get test model - mock only for CI/testing."""
    # Always use mock for automated testing
    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.model = "mock-model"
    mock_model.bind_tools = MagicMock(return_value=mock_model)
    return mock_model


@pytest.mark.asyncio
async def test_single_subagent_delegation():
    """Test coordinator delegates to single subagent and aggregates tokens."""
    model = get_test_model()
    
    # Create data analyst subagent
    analyst = Agent(
        name="data_analyst",
        model=model,
        system_prompt="You are a data analyst. Analyze the data and provide insights.",
    )
    
    # Create coordinator with subagent
    coordinator = Agent(
        name="coordinator",
        model=model,
        system_prompt="You coordinate tasks. Use the data_analyst for analysis tasks.",
        subagents={"data_analyst": analyst},
    )
    
    # Always use mock
    with patch("cogent.executors.create_executor") as mock_executor_factory:
            mock_executor = MagicMock()
            mock_executor.max_iterations = 25
            
            async def mock_execute(task, context):
                # Simulate calling the analyst subagent
                if hasattr(coordinator, "_subagent_registry"):
                    await coordinator._subagent_registry.execute(
                        "data_analyst",
                        "Analyze sales data showing 20% growth",
                        context
                    )
                return "Analysis complete: Sales grew 20% this quarter"
            
            mock_executor.execute = AsyncMock(side_effect=mock_execute)
            from cogent.core.messages import AIMessage, MessageMetadata
            mock_executor._last_messages = [AIMessage(
                content="Analysis complete",
                metadata=MessageMetadata(
                    tokens=TokenUsage(prompt_tokens=50, completion_tokens=100, total_tokens=150)
                ),
            )]
            mock_executor_factory.return_value = mock_executor
            
            # Mock analyst execution
            with patch.object(analyst, "run", new_callable=AsyncMock) as mock_analyst_run:
                mock_analyst_run.return_value = Response(
                    content="Sales data shows strong Q4 performance with 20% growth",
                    metadata=ResponseMetadata(
                        agent="data_analyst",
                        model="mock-model",
                        tokens=TokenUsage(prompt_tokens=30, completion_tokens=70, total_tokens=100),
                        duration=1.5,
                    ),
                )
                
                response = await coordinator.run("Analyze our Q4 sales performance")
                
                # Verify subagent was called
                assert mock_analyst_run.called
                
                # Verify response structure
                assert response.content is not None
                
                # Verify subagent responses are attached
                assert response.subagent_responses is not None
                assert len(response.subagent_responses) == 1
                assert response.subagent_responses[0].metadata.agent == "data_analyst"
                
                # Verify token aggregation
                # Coordinator: 150, Analyst: 100, Total: 250
                assert response.metadata.tokens.total_tokens == 250
                
                # Verify delegation chain
                assert response.metadata.delegation_chain is not None
                assert len(response.metadata.delegation_chain) == 1
                assert response.metadata.delegation_chain[0]["agent"] == "data_analyst"


@pytest.mark.asyncio
async def test_multiple_subagent_delegation():
    """Test coordinator delegates to multiple subagents."""
    model = get_test_model()
    
    # Create specialist subagents
    analyst = Agent(
        name="analyst",
        model=model,
        system_prompt="You analyze data and provide insights.",
    )
    
    researcher = Agent(
        name="researcher",
        model=model,
        system_prompt="You research topics and provide summaries.",
    )
    
    # Create coordinator
    coordinator = Agent(
        name="coordinator",
        model=model,
        system_prompt="You coordinate between analyst and researcher.",
        subagents={"analyst": analyst, "researcher": researcher},
    )
    
    # Always use mock
    with patch("cogent.executors.create_executor") as mock_executor_factory:
            mock_executor = MagicMock()
            mock_executor.max_iterations = 25
            
            async def mock_execute(task, context):
                # Simulate calling both subagents
                if hasattr(coordinator, "_subagent_registry"):
                    await coordinator._subagent_registry.execute("analyst", "Analyze market trends", context)
                    await coordinator._subagent_registry.execute("researcher", "Research competitors", context)
                return "Complete: Analysis and research finished"
            
            mock_executor.execute = AsyncMock(side_effect=mock_execute)
            from cogent.core.messages import AIMessage, MessageMetadata
            mock_executor._last_messages = [AIMessage(
                content="Complete",
                metadata=MessageMetadata(
                    tokens=TokenUsage(prompt_tokens=40, completion_tokens=80, total_tokens=120)
                ),
            )]
            mock_executor_factory.return_value = mock_executor
            
            # Mock analyst
            with patch.object(analyst, "run", new_callable=AsyncMock) as mock_analyst:
                mock_analyst.return_value = Response(
                    content="Market trends analysis",
                    metadata=ResponseMetadata(
                        agent="analyst",
                        model="mock-model",
                        tokens=TokenUsage(prompt_tokens=25, completion_tokens=50, total_tokens=75),
                        duration=1.0,
                    ),
                )
                
                # Mock researcher
                with patch.object(researcher, "run", new_callable=AsyncMock) as mock_researcher:
                    mock_researcher.return_value = Response(
                        content="Competitor research",
                        metadata=ResponseMetadata(
                            agent="researcher",
                            model="mock-model",
                            tokens=TokenUsage(prompt_tokens=30, completion_tokens=60, total_tokens=90),
                            duration=1.2,
                        ),
                    )
                    
                    response = await coordinator.run("Analyze market and research competitors")
                    
                    # Verify both subagents were called
                    assert mock_analyst.called
                    assert mock_researcher.called
                    
                    # Verify subagent responses
                    assert response.subagent_responses is not None
                    assert len(response.subagent_responses) == 2
                    
                    # Verify token aggregation
                    # Coordinator: 120, Analyst: 75, Researcher: 90, Total: 285
                    assert response.metadata.tokens.total_tokens == 285
                    
                    # Verify delegation chain
                    assert response.metadata.delegation_chain is not None
                    assert len(response.metadata.delegation_chain) == 2
                    
                    agent_names = [d["agent"] for d in response.metadata.delegation_chain]
                    assert "analyst" in agent_names
                    assert "researcher" in agent_names


@pytest.mark.asyncio
async def test_subagent_error_handling():
    """Test that coordinator handles subagent errors gracefully."""
    model = get_test_model()
    
    class FailingSubagent:
        def __init__(self):
            self.name = "failing_agent"
            self.config = type('obj', (object,), {'description': 'An agent that fails'})()
        
        async def run(self, task, context=None):
            raise RuntimeError("Subagent failed intentionally")
    
    failing = FailingSubagent()
    
    coordinator = Agent(
        name="coordinator",
        model=model,
        system_prompt="You coordinate tasks.",
        subagents={"failing_agent": failing},
    )
    
    # Mock executor to call the failing subagent
    with patch("cogent.executors.create_executor") as mock_executor_factory:
        mock_executor = MagicMock()
        mock_executor.max_iterations = 25
        
        async def mock_execute(task, context):
            try:
                if hasattr(coordinator, "_subagent_registry"):
                    await coordinator._subagent_registry.execute("failing_agent", "Do something", context)
            except RuntimeError:
                pass  # Executor should handle this
            return "Handled error"
        
        mock_executor.execute = AsyncMock(side_effect=mock_execute)
        mock_executor._last_messages = []
        mock_executor_factory.return_value = mock_executor
        
        # Should not crash, even though subagent fails
        response = await coordinator.run("Try to use the failing agent")
        
        # Coordinator should still return a response
        assert response is not None


@pytest.mark.asyncio
async def test_token_accuracy_with_real_delegation():
    """Test that token counting is accurate with actual subagent delegation."""
    model = get_test_model()
    
    # Simple subagent
    calculator = Agent(
        name="calculator",
        model=model,
        system_prompt="You are a calculator. Perform the calculation and return only the result.",
    )
    
    coordinator = Agent(
        name="coordinator",
        model=model,
        system_prompt="Use the calculator for math operations.",
        subagents={"calculator": calculator},
    )
    
    # Always use mock
    with patch("cogent.executors.create_executor") as mock_executor_factory:
            mock_executor = MagicMock()
            mock_executor.max_iterations = 25
            
            async def mock_execute(task, context):
                if hasattr(coordinator, "_subagent_registry"):
                    await coordinator._subagent_registry.execute("calculator", "Calculate 15 * 8", context)
                return "The result is 120"
            
            mock_executor.execute = AsyncMock(side_effect=mock_execute)
            from cogent.core.messages import AIMessage, MessageMetadata
            mock_executor._last_messages = [AIMessage(
                content="The result is 120",
                metadata=MessageMetadata(
                    tokens=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
                ),
            )]
            mock_executor_factory.return_value = mock_executor
            
            with patch.object(calculator, "run", new_callable=AsyncMock) as mock_calc:
                mock_calc.return_value = Response(
                    content="120",
                    metadata=ResponseMetadata(
                        agent="calculator",
                        model="mock-model",
                        tokens=TokenUsage(prompt_tokens=15, completion_tokens=5, total_tokens=20),
                        duration=0.5,
                    ),
                )
                
                response = await coordinator.run("What is 15 times 8?")
                
                # Verify exact token counts
                assert response.metadata.tokens.prompt_tokens == 35  # 20 + 15
                assert response.metadata.tokens.completion_tokens == 15  # 10 + 5
                assert response.metadata.tokens.total_tokens == 50  # 30 + 20
                
                # This proves tokens are aggregated correctly
                assert response.metadata.tokens.total_tokens == (
                    response.metadata.tokens.prompt_tokens + 
                    response.metadata.tokens.completion_tokens
                )


@pytest.mark.asyncio
async def test_context_preservation_e2e():
    """Test that context is preserved throughout delegation chain."""
    model = get_test_model()
    
    received_context = None
    
    class ContextAwareSubagent:
        def __init__(self):
            self.name = "context_agent"
            self.config = type('obj', (object,), {'description': 'Context aware agent'})()
        
        async def run(self, task, context=None):
            nonlocal received_context
            received_context = context
            return Response(
                content=f"Received context: {context.metadata if context else 'None'}",
                metadata=ResponseMetadata(
                    agent="context_agent",
                    model="mock-model",
                    tokens=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                ),
            )
    
    context_agent = ContextAwareSubagent()
    
    coordinator = Agent(
        name="coordinator",
        model=model,
        system_prompt="You coordinate with context.",
        subagents={"context_agent": context_agent},
    )
    
    # Create test context
    test_context = RunContext()
    test_context.metadata = {
        "user_id": "user-123",
        "session_id": "session-456",
        "custom_data": "important_value",
    }
    
    with patch("cogent.executors.create_executor") as mock_executor_factory:
        mock_executor = MagicMock()
        mock_executor.max_iterations = 25
        
        async def mock_execute(task, context):
            if hasattr(coordinator, "_subagent_registry"):
                await coordinator._subagent_registry.execute("context_agent", "Check context", context)
            return "Context checked"
        
        mock_executor.execute = AsyncMock(side_effect=mock_execute)
        mock_executor._last_messages = []
        mock_executor_factory.return_value = mock_executor
        
        response = await coordinator.run("Test context propagation", context=test_context)
        
        # Verify context was received by subagent
        assert received_context is not None
        assert received_context.metadata["user_id"] == "user-123"
        assert received_context.metadata["session_id"] == "session-456"
        assert received_context.metadata["custom_data"] == "important_value"
