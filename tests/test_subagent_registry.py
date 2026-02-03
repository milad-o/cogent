"""Tests for SubagentRegistry."""

import pytest

from cogent.agent.base import Agent
from cogent.agent.config import AgentConfig
from cogent.agent.subagent import SubagentRegistry
from cogent.core.context import RunContext
from cogent.core.response import Response, ResponseMetadata
from cogent.models.openai import OpenAIChat


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.config = AgentConfig(
            name=name,
            model=OpenAIChat(model="gpt-4o-mini"),
            description=description,
        )
        self._run_called_with = None
    
    async def run(self, task: str, context: RunContext | None = None) -> Response:
        """Mock run method."""
        self._run_called_with = {"task": task, "context": context}
        
        # Return mock Response
        return Response(
            content=f"Result from {self.name}: {task}",
            metadata=ResponseMetadata(
                agent=self.name,
                model="gpt-4o-mini",
                duration=1.0,
            ),
        )


# ============================================================================
# Registration Tests
# ============================================================================

def test_register_single_agent():
    """Test registering a single subagent."""
    registry = SubagentRegistry()
    agent = MockAgent("analyst")
    
    registry.register("analyst", agent)
    
    assert registry.has_subagent("analyst")
    assert registry.count == 1
    assert "analyst" in registry.agent_names


def test_register_multiple_agents():
    """Test registering multiple subagents."""
    registry = SubagentRegistry()
    
    registry.register("analyst", MockAgent("analyst"))
    registry.register("writer", MockAgent("writer"))
    registry.register("researcher", MockAgent("researcher"))
    
    assert registry.has_subagent("analyst")
    assert registry.has_subagent("writer")
    assert registry.has_subagent("researcher")
    assert registry.count == 3
    assert set(registry.agent_names) == {"analyst", "writer", "researcher"}


def test_has_subagent_false_for_unregistered():
    """Test has_subagent returns False for unregistered names."""
    registry = SubagentRegistry()
    registry.register("analyst", MockAgent("analyst"))
    
    assert not registry.has_subagent("writer")
    assert not registry.has_subagent("unknown")
    assert not registry.has_subagent("")


def test_register_overwrites_existing():
    """Test re-registering overwrites existing agent."""
    registry = SubagentRegistry()
    
    agent1 = MockAgent("analyst", "First agent")
    agent2 = MockAgent("analyst", "Second agent")
    
    registry.register("analyst", agent1)
    registry.register("analyst", agent2)
    
    assert registry.count == 1
    # Should be the second agent
    assert registry._agents["analyst"] is agent2


# ============================================================================
# Execution Tests
# ============================================================================

@pytest.mark.asyncio
async def test_execute_basic():
    """Test basic subagent execution."""
    registry = SubagentRegistry()
    agent = MockAgent("analyst")
    registry.register("analyst", agent)
    
    response = await registry.execute("analyst", "Analyze data")
    
    assert response.content == "Result from analyst: Analyze data"
    assert agent._run_called_with["task"] == "Analyze data"


@pytest.mark.asyncio
async def test_execute_with_context():
    """Test subagent execution with context propagation."""
    registry = SubagentRegistry()
    agent = MockAgent("analyst")
    registry.register("analyst", agent)
    
    context = RunContext(query="original query", metadata={"user_id": "user123"})
    response = await registry.execute("analyst", "Analyze data", context=context)
    
    assert agent._run_called_with["context"] is context
    assert agent._run_called_with["context"].query == "original query"
    assert agent._run_called_with["context"].metadata["user_id"] == "user123"


@pytest.mark.asyncio
async def test_execute_unregistered_raises_error():
    """Test executing unregistered subagent raises KeyError."""
    registry = SubagentRegistry()
    
    with pytest.raises(KeyError, match="Subagent 'unknown' not registered"):
        await registry.execute("unknown", "Do something")


@pytest.mark.asyncio
async def test_execute_caches_response():
    """Test that execute caches Response objects."""
    registry = SubagentRegistry()
    registry.register("analyst", MockAgent("analyst"))
    registry.register("writer", MockAgent("writer"))
    
    # Execute both subagents
    response1 = await registry.execute("analyst", "Task 1")
    response2 = await registry.execute("writer", "Task 2")
    
    # Check responses are cached
    cached = registry.get_responses()
    assert len(cached) == 2
    assert cached[0] is response1
    assert cached[1] is response2


# ============================================================================
# Response Caching Tests
# ============================================================================

def test_get_responses_empty():
    """Test get_responses returns empty list initially."""
    registry = SubagentRegistry()
    assert registry.get_responses() == []


@pytest.mark.asyncio
async def test_get_responses_returns_copy():
    """Test get_responses returns a copy, not original list."""
    registry = SubagentRegistry()
    registry.register("analyst", MockAgent("analyst"))
    
    await registry.execute("analyst", "Task")
    
    responses1 = registry.get_responses()
    responses2 = registry.get_responses()
    
    # Should be different list objects
    assert responses1 is not responses2
    # But with same contents
    assert responses1 == responses2


@pytest.mark.asyncio
async def test_clear_removes_cached_responses():
    """Test clear() removes all cached responses."""
    registry = SubagentRegistry()
    registry.register("analyst", MockAgent("analyst"))
    
    await registry.execute("analyst", "Task 1")
    await registry.execute("analyst", "Task 2")
    
    assert len(registry.get_responses()) == 2
    
    registry.clear()
    
    assert len(registry.get_responses()) == 0


@pytest.mark.asyncio
async def test_clear_does_not_remove_registered_agents():
    """Test clear() only removes responses, not registered agents."""
    registry = SubagentRegistry()
    registry.register("analyst", MockAgent("analyst"))
    registry.register("writer", MockAgent("writer"))
    
    await registry.execute("analyst", "Task")
    
    registry.clear()
    
    # Agents still registered
    assert registry.has_subagent("analyst")
    assert registry.has_subagent("writer")
    assert registry.count == 2
    
    # But responses cleared
    assert len(registry.get_responses()) == 0


# ============================================================================
# Documentation Generation Tests
# ============================================================================

def test_generate_documentation_empty():
    """Test documentation generation with no subagents."""
    registry = SubagentRegistry()
    docs = registry.generate_documentation()
    assert docs == ""


def test_generate_documentation_single_agent():
    """Test documentation generation with one subagent."""
    registry = SubagentRegistry()
    registry.register("analyst", MockAgent("analyst", "Data analysis specialist"))
    
    docs = registry.generate_documentation()
    
    assert "# Specialist Agents" in docs
    assert "analyst" in docs
    assert "Data analysis specialist" in docs
    assert "call them like a tool" in docs


def test_generate_documentation_multiple_agents():
    """Test documentation generation with multiple subagents."""
    registry = SubagentRegistry()
    registry.register("analyst", MockAgent("analyst", "Data specialist"))
    registry.register("writer", MockAgent("writer", "Writing specialist"))
    
    docs = registry.generate_documentation()
    
    assert "analyst" in docs
    assert "Data specialist" in docs
    assert "writer" in docs
    assert "Writing specialist" in docs


def test_generate_documentation_no_description():
    """Test documentation with agent that has no description."""
    registry = SubagentRegistry()
    registry.register("analyst", MockAgent("analyst", ""))
    
    docs = registry.generate_documentation()
    
    assert "analyst" in docs
    assert "No description available" in docs


# ============================================================================
# Property Tests
# ============================================================================

def test_agent_names_property():
    """Test agent_names property."""
    registry = SubagentRegistry()
    
    assert registry.agent_names == []
    
    registry.register("analyst", MockAgent("analyst"))
    registry.register("writer", MockAgent("writer"))
    
    names = registry.agent_names
    assert len(names) == 2
    assert "analyst" in names
    assert "writer" in names


def test_count_property():
    """Test count property."""
    registry = SubagentRegistry()
    
    assert registry.count == 0
    
    registry.register("analyst", MockAgent("analyst"))
    assert registry.count == 1
    
    registry.register("writer", MockAgent("writer"))
    assert registry.count == 2


def test_repr():
    """Test string representation."""
    registry = SubagentRegistry()
    
    # Empty registry
    repr_str = repr(registry)
    assert "SubagentRegistry" in repr_str
    assert "none" in repr_str
    assert "cached_responses=0" in repr_str
    
    # With agents
    registry.register("analyst", MockAgent("analyst"))
    registry.register("writer", MockAgent("writer"))
    
    repr_str = repr(registry)
    assert "analyst" in repr_str
    assert "writer" in repr_str
