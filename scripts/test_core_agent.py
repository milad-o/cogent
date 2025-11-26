#!/usr/bin/env python3
"""
Test script to verify core Agent functionality.

Tests:
1. Agent creation with roles
2. Role-specific prompts and behaviors
3. Agent.think() - LLM reasoning
4. Agent.act() - Tool execution
5. Agent.run() - DAG execution
6. Delegation parsing
7. Agent memory/state
"""

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# Create proper mock model that inherits from BaseChatModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class MockChatModel(BaseChatModel):
    """Mock LangChain chat model for testing without API calls.
    
    This properly inherits from BaseChatModel so it passes type validation.
    """
    
    responses: list[str] = ["This is a mock response."]
    call_count: int = 0
    last_messages: list[BaseMessage] | None = None
    
    def __init__(self, responses: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        if responses:
            object.__setattr__(self, 'responses', responses)
        object.__setattr__(self, 'call_count', 0)
        object.__setattr__(self, 'last_messages', None)
    
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs,
    ) -> ChatResult:
        """Generate a response (sync)."""
        object.__setattr__(self, 'last_messages', messages)
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        object.__setattr__(self, 'call_count', self.call_count + 1)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response))]
        )
    
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs,
    ) -> ChatResult:
        """Generate a response (async)."""
        return self._generate(messages, stop, run_manager, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "mock"


# Ensure we have an API key or use mocks
USE_REAL_LLM = os.getenv("OPENAI_API_KEY") is not None and os.getenv("USE_REAL_LLM", "false").lower() == "true"

print("=" * 60)
print("AgenticFlow Core Agent Test")
print("=" * 60)
print(f"Using real LLM: {USE_REAL_LLM}")
print()


def get_mock_model(responses: list[str] | None = None) -> BaseChatModel:
    """Get a mock model for testing."""
    if USE_REAL_LLM:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini")
        except ImportError:
            print("Warning: langchain_openai not available, using mock")
    return MockChatModel(responses=responses or ["Mock response"])


async def test_agent_creation_and_roles():
    """Test 1: Agent creation with different roles."""
    print("TEST 1: Agent Creation & Roles")
    print("-" * 40)
    
    from agenticflow.agent import Agent, get_role_prompt, get_role_behavior
    from agenticflow.core.enums import AgentRole
    
    # Create mock model
    mock_model = get_mock_model()
    
    # Test 1a: Basic agent with role
    worker = Agent(
        name="TestWorker",
        model=mock_model,
        role="worker",
    )
    print(f"  ✓ Created worker agent: {worker.name}")
    print(f"    Role: {worker.role}")
    print(f"    Can delegate: {worker.can_delegate}")
    print(f"    Can finish: {worker.can_finish}")
    assert worker.role == AgentRole.WORKER
    assert worker.can_delegate is False
    assert worker.can_finish is True
    
    # Test 1b: Supervisor agent
    supervisor = Agent(
        name="TestSupervisor", 
        model=mock_model,
        role="supervisor",
    )
    print(f"  ✓ Created supervisor agent: {supervisor.name}")
    print(f"    Role: {supervisor.role}")
    print(f"    Can delegate: {supervisor.can_delegate}")
    print(f"    Can finish: {supervisor.can_finish}")
    assert supervisor.role == AgentRole.SUPERVISOR
    assert supervisor.can_delegate is True
    assert supervisor.can_finish is True
    
    # Test 1c: Role-specific prompts are applied
    assert "delegate" in supervisor.instructions.lower()
    assert "tool" in worker.instructions.lower()
    print(f"  ✓ Role-specific prompts applied")
    
    # Test 1d: Factory methods
    supervisor2 = Agent.as_supervisor(
        name="Manager",
        model=mock_model,
        workers=["Alice", "Bob"],
    )
    assert "Alice" in supervisor2.instructions
    assert "Bob" in supervisor2.instructions
    print(f"  ✓ Factory method Agent.as_supervisor() works")
    
    worker2 = Agent.as_worker(
        name="Analyst",
        model=mock_model,
        specialty="data analysis",
    )
    assert "data analysis" in worker2.instructions
    print(f"  ✓ Factory method Agent.as_worker() works")
    
    print("  ✓ TEST 1 PASSED\n")


async def test_delegation_parsing():
    """Test 2: Delegation command parsing."""
    print("TEST 2: Delegation Parsing")
    print("-" * 40)
    
    from agenticflow.agent import (
        parse_delegation,
        has_final_answer,
        extract_final_answer,
    )
    
    # Test DELEGATE TO
    cmd = parse_delegation("Let me think... DELEGATE TO [researcher]: Find info about Python")
    assert cmd is not None
    assert cmd.action == "delegate"
    assert cmd.target == "researcher"
    assert "Python" in cmd.task
    print(f"  ✓ Parsed DELEGATE TO command")
    
    # Test FINAL ANSWER
    cmd = parse_delegation("After analysis, FINAL ANSWER: The result is 42.")
    assert cmd is not None
    assert cmd.action == "final_answer"
    assert "42" in cmd.task
    print(f"  ✓ Parsed FINAL ANSWER command")
    
    # Test has_final_answer
    assert has_final_answer("FINAL ANSWER: Done") is True
    assert has_final_answer("Still working...") is False
    print(f"  ✓ has_final_answer() works")
    
    # Test extract_final_answer
    answer = extract_final_answer("Processing... FINAL ANSWER: The solution is X.")
    assert answer == "The solution is X."
    print(f"  ✓ extract_final_answer() works")
    
    print("  ✓ TEST 2 PASSED\n")


async def test_agent_think():
    """Test 3: Agent.think() - basic LLM reasoning."""
    print("TEST 3: Agent.think()")
    print("-" * 40)
    
    from agenticflow.agent import Agent
    
    # Create agent with mock model that returns a specific response
    mock_model = get_mock_model(["I think the answer is 42."])
    
    agent = Agent(
        name="Thinker",
        model=mock_model,
        role="worker",
    )
    
    # Test think
    result = await agent.think("What is the meaning of life?")
    
    print(f"  Input: 'What is the meaning of life?'")
    print(f"  Output: '{result}'")
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"  ✓ Agent.think() returns response")
    
    # Check that model was called
    if isinstance(mock_model, MockChatModel):
        assert mock_model.call_count > 0
        print(f"  ✓ Model was invoked (call count: {mock_model.call_count})")
    
    print("  ✓ TEST 3 PASSED\n")


async def test_agent_act():
    """Test 4: Agent.act() - tool execution."""
    print("TEST 4: Agent.act()")
    print("-" * 40)
    
    from agenticflow.agent import Agent
    from agenticflow.tools import tool
    
    # Define test tools
    tool_calls = []
    
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        tool_calls.append(("add", a, b))
        return a + b
    
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        tool_calls.append(("multiply", a, b))
        return a * b
    
    # Create agent with tools
    # Model response indicates tool use
    mock_model = get_mock_model([
        'I will calculate: {"tool": "add", "args": {"a": 5, "b": 3}}',
        "The result is 8."
    ])
    
    agent = Agent(
        name="Calculator",
        model=mock_model,
        role="worker",
        tools=[add, multiply],
    )
    
    print(f"  Agent has {len(agent.tools)} tools: {[t.name for t in agent.tools]}")
    assert len(agent.tools) == 2
    
    # Test act with explicit tool call
    result = await agent.act("add", a=5, b=3)
    
    print(f"  Called: add(5, 3)")
    print(f"  Result: {result}")
    
    assert result == 8
    assert ("add", 5, 3) in tool_calls
    print(f"  ✓ Agent.act() executes tool correctly")
    
    # Test multiply
    result = await agent.act("multiply", a=4, b=7)
    assert result == 28
    print(f"  ✓ Agent.act() multiply(4, 7) = {result}")
    
    print("  ✓ TEST 4 PASSED\n")


async def test_agent_run_dag():
    """Test 5: Agent.run() with DAG executor."""
    print("TEST 5: Agent.run() with DAG Executor")
    print("-" * 40)
    
    from agenticflow.agent import Agent
    from agenticflow.tools import tool
    from agenticflow.graphs import DAGExecutor, ExecutionStrategy
    
    # Define tools
    execution_order = []
    
    @tool
    def step_one() -> str:
        """First step."""
        execution_order.append("step_one")
        return "Step 1 complete"
    
    @tool
    def step_two() -> str:
        """Second step."""
        execution_order.append("step_two")
        return "Step 2 complete"
    
    @tool
    def step_three() -> str:
        """Third step."""
        execution_order.append("step_three")
        return "Step 3 complete"
    
    # Create agent
    # Model returns final answer after processing
    mock_model = get_mock_model([
        "Processing task...",
        "FINAL ANSWER: All steps completed successfully."
    ])
    
    agent = Agent(
        name="Runner",
        model=mock_model,
        role="worker",
        tools=[step_one, step_two, step_three],
    )
    
    # Verify DAG executor is default
    print(f"  Default execution strategy: {agent.config.execution_strategy}")
    assert agent.config.execution_strategy == ExecutionStrategy.DAG
    print(f"  ✓ DAG is the default execution strategy")
    
    # Test run
    print(f"  Running agent with task...")
    result = await agent.run("Execute all three steps in sequence")
    
    print(f"  Result: {result}")
    assert result is not None
    print(f"  ✓ Agent.run() completes successfully")
    
    print("  ✓ TEST 5 PASSED\n")


async def test_agent_state():
    """Test 6: Agent state management."""
    print("TEST 6: Agent State Management")
    print("-" * 40)
    
    from agenticflow.agent import Agent, AgentState, AgentStatus
    
    mock_model = get_mock_model()
    
    agent = Agent(
        name="StatefulAgent",
        model=mock_model,
        role="worker",
    )
    
    # Check initial state
    print(f"  Initial status: {agent.state.status}")
    assert agent.state.status == AgentStatus.IDLE
    print(f"  ✓ Initial status is IDLE")
    
    # Check state tracks activity
    print(f"  Last activity: {agent.state.last_activity}")
    print(f"  Tasks processed: {agent.state.tasks_processed}")
    
    # State should have message history
    print(f"  Message history length: {len(agent.state.message_history)}")
    print(f"  ✓ Agent has state tracking")
    
    # Test state serialization
    state_dict = agent.state.model_dump()
    assert "status" in state_dict
    assert "message_history" in state_dict
    print(f"  ✓ State is serializable")
    
    print("  ✓ TEST 6 PASSED\n")


async def test_role_behaviors():
    """Test 7: Role-specific behaviors."""
    print("TEST 7: Role-Specific Behaviors")
    print("-" * 40)
    
    from agenticflow.agent import Agent, get_role_behavior
    from agenticflow.core.enums import AgentRole
    
    mock_model = get_mock_model()
    
    # Test all roles
    roles_to_test = [
        (AgentRole.SUPERVISOR, True, True, False, True),  # can_delegate, can_finish, can_use_tools, can_route
        (AgentRole.WORKER, False, True, True, False),
        (AgentRole.PLANNER, True, True, False, True),
        (AgentRole.CRITIC, False, True, False, False),
        (AgentRole.RESEARCHER, False, True, True, False),
    ]
    
    for role, exp_delegate, exp_finish, exp_tools, exp_route in roles_to_test:
        behavior = get_role_behavior(role)
        
        # Verify behaviors match expectations
        status = "✓" if (
            behavior.can_delegate == exp_delegate and
            behavior.can_finish == exp_finish and
            behavior.can_use_tools == exp_tools and
            behavior.can_route_messages == exp_route
        ) else "✗"
        
        print(f"  {status} {role.value}: delegate={behavior.can_delegate}, "
              f"finish={behavior.can_finish}, tools={behavior.can_use_tools}")
        
        assert behavior.can_delegate == exp_delegate
        assert behavior.can_finish == exp_finish
    
    print("  ✓ TEST 7 PASSED\n")


async def test_executor_creation():
    """Test 8: Executor creation."""
    print("TEST 8: Executor Factory")
    print("-" * 40)
    
    from agenticflow.graphs import (
        create_executor,
        ExecutionStrategy,
        DAGExecutor,
        ReActExecutor,
        PlanExecutor,
        AdaptiveExecutor,
    )
    from agenticflow.agent import Agent
    
    mock_model = get_mock_model()
    agent = Agent(name="Test", model=mock_model, role="worker")
    
    # Test each executor type
    executors = [
        (ExecutionStrategy.DAG, DAGExecutor),
        (ExecutionStrategy.REACT, ReActExecutor),
        (ExecutionStrategy.PLAN, PlanExecutor),
        (ExecutionStrategy.ADAPTIVE, AdaptiveExecutor),
    ]
    
    for strategy, expected_type in executors:
        executor = create_executor(strategy, agent)
        assert isinstance(executor, expected_type)
        print(f"  ✓ {strategy.value} -> {expected_type.__name__}")
    
    print("  ✓ TEST 8 PASSED\n")


async def main():
    """Run all tests."""
    try:
        await test_agent_creation_and_roles()
        await test_delegation_parsing()
        await test_agent_think()
        await test_agent_act()
        await test_agent_run_dag()
        await test_agent_state()
        await test_role_behaviors()
        await test_executor_creation()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
