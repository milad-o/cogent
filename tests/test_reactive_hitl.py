"""Tests for Human-in-the-Loop (HITL) support in reactive flows."""

import asyncio
import pytest

from agenticflow import Agent
from agenticflow.agent.hitl import (
    InterruptReason,
    PendingAction,
)
from agenticflow.reactive.core import ReactionType, Trigger
from agenticflow.reactive.flow import EventFlow, EventFlowConfig


class MockHITLHandler:
    """Mock HITL handler for testing."""
    
    def __init__(self, auto_approve: bool = True, delay: float = 0.0):
        self.auto_approve = auto_approve
        self.delay = delay
        self.requests = []
    
    async def request_approval(self, request) -> bool:
        """Mock approval request."""
        self.requests.append(request)
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return self.auto_approve


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    
    class MockModel:
        def __init__(self):
            self.calls = []
        
        async def generate(self, messages, **kwargs):
            self.calls.append({"messages": messages, "kwargs": kwargs})
            return type('obj', (object,), {
                'message': type('obj', (object,), {
                    'content': "I'll help with that task."
                })(),
                'stop_reason': 'end_turn'
            })()
    
    return MockModel()


@pytest.mark.asyncio
async def test_hitl_approval_required(mock_model):
    """Test that AWAIT_HUMAN reaction type triggers approval."""
    handler = MockHITLHandler(auto_approve=True)
    
    agent = Agent(
        name="test_agent",
        model=mock_model,
        system_prompt="Test agent",
    )
    
    flow = EventFlow(
        config=EventFlowConfig(max_rounds=5),
        hitl_handler=handler,
    )
    
    # Register agent with AWAIT_HUMAN trigger
    trigger = Trigger(
        on="task.created",
        reaction=ReactionType.AWAIT_HUMAN,
    )
    flow.register(agent, [trigger])
    
    # Run flow
    result = await flow.run("Test task")
    
    # Verify approval was requested
    assert len(handler.requests) == 1
    assert handler.requests[0].agent_name == "test_agent"
    
    # Verify agent executed after approval
    assert len(mock_model.calls) > 0


@pytest.mark.asyncio
async def test_hitl_rejection_stops_execution(mock_model):
    """Test that rejecting approval prevents agent execution."""
    handler = MockHITLHandler(auto_approve=False)
    
    agent = Agent(
        name="test_agent",
        model=mock_model,
        system_prompt="Test agent",
    )
    
    flow = EventFlow(
        config=EventFlowConfig(max_rounds=5),
        hitl_handler=handler,
    )
    
    trigger = Trigger(
        on="task.created",
        reaction=ReactionType.AWAIT_HUMAN,
    )
    flow.register(agent, [trigger])
    
    result = await flow.run("Test task")
    
    # Verify approval was requested
    assert len(handler.requests) == 1
    
    # Verify agent did NOT execute (rejection prevented it)
    assert len(mock_model.calls) == 0
    
    # Verify reaction shows rejection
    assert len(result.reactions) > 0
    rejected_reaction = result.reactions[0]
    assert rejected_reaction.error == "human_rejected"


@pytest.mark.asyncio
async def test_hitl_with_regular_triggers(mock_model):
    """Test that HITL works alongside regular triggers."""
    handler = MockHITLHandler(auto_approve=True)
    
    # Agent 1: Requires approval
    agent1 = Agent(
        name="supervised_agent",
        model=mock_model,
        system_prompt="Agent requiring approval",
    )
    
    # Agent 2: No approval needed
    agent2 = Agent(
        name="autonomous_agent",
        model=mock_model,
        system_prompt="Autonomous agent",
    )
    
    flow = EventFlow(
        config=EventFlowConfig(max_rounds=10),
        hitl_handler=handler,
    )
    
    # Register with different reaction types
    flow.register(agent1, [Trigger(on="task.created", reaction=ReactionType.AWAIT_HUMAN)])
    flow.register(agent2, [Trigger(on="supervised_agent.completed", reaction=ReactionType.RUN)])
    
    result = await flow.run("Test task")
    
    # Only agent1 should require approval
    assert len(handler.requests) == 1
    assert handler.requests[0].agent_name == "supervised_agent"
    
    # Both agents should execute
    assert len(result.reactions) >= 2


@pytest.mark.asyncio
async def test_hitl_flow_events_emitted(mock_model):
    """Test that flow.paused and flow.resumed events are emitted."""
    handler = MockHITLHandler(auto_approve=True)
    
    agent = Agent(
        name="test_agent",
        model=mock_model,
        system_prompt="Test agent",
    )
    
    flow = EventFlow(
        config=EventFlowConfig(max_rounds=5, enable_history=True),
        hitl_handler=handler,
    )
    
    trigger = Trigger(
        on="task.created",
        reaction=ReactionType.AWAIT_HUMAN,
    )
    flow.register(agent, [trigger])
    
    result = await flow.run("Test task")
    
    # Check for flow.paused and flow.resumed in event history
    event_names = [e.name for e in result.event_history if hasattr(e, 'name')]
    
    # Should have paused and resumed events
    assert "flow.paused" in event_names
    assert "flow.resumed" in event_names


@pytest.mark.asyncio
async def test_hitl_without_handler_uses_default(mock_model):
    """Test that AWAIT_HUMAN without handler falls back gracefully."""
    agent = Agent(
        name="test_agent",
        model=mock_model,
        system_prompt="Test agent",
    )
    
    flow = EventFlow(
        config=EventFlowConfig(max_rounds=5),
        hitl_handler=None,  # No handler
    )
    
    trigger = Trigger(
        on="task.created",
        reaction=ReactionType.AWAIT_HUMAN,
    )
    flow.register(agent, [trigger])
    
    # Should still run - just skips HITL if no handler
    result = await flow.run("Test task")
    
    # Agent should execute normally
    assert len(mock_model.calls) > 0


@pytest.mark.asyncio
async def test_hitl_with_breakpoint_metadata(mock_model):
    """Test that breakpoint metadata is passed to handler."""
    handler = MockHITLHandler(auto_approve=True)
    
    agent = Agent(
        name="test_agent",
        model=mock_model,
        system_prompt="Test agent",
    )
    
    flow = EventFlow(
        config=EventFlowConfig(max_rounds=5),
        hitl_handler=handler,
    )
    
    # Create trigger with breakpoint
    from dataclasses import dataclass
    
    @dataclass
    class TestBreakpoint:
        name: str
        prompt: str
    
    breakpoint = TestBreakpoint(
        name="approval_checkpoint",
        prompt="Do you approve this action?",
    )
    
    trigger = Trigger(
        on="task.created",
        reaction=ReactionType.AWAIT_HUMAN,
        breakpoint=breakpoint,
    )
    flow.register(agent, [trigger])
    
    result = await flow.run("Test task")
    
    # Verify breakpoint was passed to handler
    assert len(handler.requests) == 1
    assert handler.requests[0].breakpoint == breakpoint


@pytest.mark.asyncio
async def test_hitl_concurrent_approval_requests(mock_model):
    """Test multiple agents requiring approval in parallel."""
    handler = MockHITLHandler(auto_approve=True, delay=0.1)
    
    # Create multiple agents
    agents = [
        Agent(name=f"agent_{i}", model=mock_model, system_prompt=f"Agent {i}")
        for i in range(3)
    ]
    
    flow = EventFlow(
        config=EventFlowConfig(max_rounds=10, max_concurrent_agents=3),
        hitl_handler=handler,
    )
    
    # Register all with AWAIT_HUMAN
    for agent in agents:
        trigger = Trigger(on="task.created", reaction=ReactionType.AWAIT_HUMAN)
        flow.register(agent, [trigger])
    
    result = await flow.run("Test task")
    
    # All agents should have requested approval
    assert len(handler.requests) == 3
    
    # All should have been approved and executed
    assert len([r for r in result.reactions if not r.error]) == 3


@pytest.mark.asyncio  
async def test_hitl_chained_approval(mock_model):
    """Test approval in a chain of agents."""
    handler = MockHITLHandler(auto_approve=True)
    
    agent1 = Agent(name="agent1", model=mock_model, system_prompt="First agent")
    agent2 = Agent(name="agent2", model=mock_model, system_prompt="Second agent")
    
    flow = EventFlow(
        config=EventFlowConfig(max_rounds=10),
        hitl_handler=handler,
    )
    
    # Agent1 requires approval, then emits event for agent2
    flow.register(
        agent1,
        [Trigger(on="task.created", reaction=ReactionType.AWAIT_HUMAN, emits="agent1.done")]
    )
    # Agent2 also requires approval when triggered by agent1
    flow.register(
        agent2,
        [Trigger(on="agent1.done", reaction=ReactionType.AWAIT_HUMAN)]
    )
    
    result = await flow.run("Test task")
    
    # Both agents should have requested approval
    assert len(handler.requests) == 2
    assert handler.requests[0].agent_name == "agent1"
    assert handler.requests[1].agent_name == "agent2"
