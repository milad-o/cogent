"""Tests for agent spawning functionality."""

import asyncio

import pytest

from agenticflow.agent.spawning import (
    AgentSpec,
    SpawningConfig,
    SpawnedAgentInfo,
    SpawnManager,
    create_spawn_tool,
)
from agenticflow.tools.base import tool


class TestAgentSpec:
    """Tests for AgentSpec dataclass."""
    
    def test_create_basic_spec(self):
        """Test creating a basic agent spec."""
        spec = AgentSpec(role="researcher")
        assert spec.role == "researcher"
        assert spec.system_prompt is None
        assert spec.tools is None
        assert spec.model is None
        assert spec.description == "Specialist agent for researcher tasks"
    
    def test_create_full_spec(self):
        """Test creating a fully specified agent spec."""
        @tool
        def my_tool(x: str) -> str:
            return x
        
        spec = AgentSpec(
            role="coder",
            system_prompt="You write code",
            tools=[my_tool],
            description="Writes clean Python code",
        )
        assert spec.role == "coder"
        assert spec.system_prompt == "You write code"
        assert len(spec.tools) == 1
        assert spec.description == "Writes clean Python code"
    
    def test_auto_description(self):
        """Test automatic description generation."""
        spec = AgentSpec(role="analyst")
        assert "analyst" in spec.description


class TestSpawningConfig:
    """Tests for SpawningConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SpawningConfig()
        assert config.max_concurrent == 10
        assert config.max_depth == 3
        assert config.max_total_spawns == 50
        assert config.ephemeral is True
        assert config.inherit_tools is False
        assert config.inherit_model is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SpawningConfig(
            max_concurrent=5,
            max_depth=2,
            max_total_spawns=20,
            ephemeral=False,
        )
        assert config.max_concurrent == 5
        assert config.max_depth == 2
        assert config.max_total_spawns == 20
        assert config.ephemeral is False
    
    def test_config_with_specs(self):
        """Test configuration with predefined specs."""
        config = SpawningConfig(
            specs={
                "researcher": AgentSpec(role="researcher"),
                "writer": AgentSpec(role="writer"),
            }
        )
        assert config.has_predefined_roles()
        assert set(config.get_available_roles()) == {"researcher", "writer"}
    
    def test_get_spec_existing(self):
        """Test getting an existing spec."""
        spec = AgentSpec(role="coder", system_prompt="Code writer")
        config = SpawningConfig(specs={"coder": spec})
        
        retrieved = config.get_spec("coder")
        assert retrieved is spec
    
    def test_get_spec_nonexistent(self):
        """Test getting a non-existent spec."""
        config = SpawningConfig()
        assert config.get_spec("unknown") is None
    
    def test_config_with_available_tools(self):
        """Test configuration with available tools pool."""
        @tool
        def search(q: str) -> str:
            return q
        
        @tool
        def write(text: str) -> str:
            return text
        
        config = SpawningConfig(available_tools=[search, write])
        assert len(config.available_tools) == 2


class TestSpawnedAgentInfo:
    """Tests for SpawnedAgentInfo tracking."""
    
    def test_create_info(self):
        """Test creating spawn info."""
        info = SpawnedAgentInfo(
            id="spawn_123",
            role="researcher",
            task="Find information",
            parent_id="parent_456",
            depth=1,
        )
        assert info.id == "spawn_123"
        assert info.role == "researcher"
        assert info.task == "Find information"
        assert info.parent_id == "parent_456"
        assert info.depth == 1
        assert info.status == "running"
        assert info.result is None
        assert info.error is None


class TestSpawnManager:
    """Tests for SpawnManager."""
    
    def test_can_spawn_within_limits(self):
        """Test spawn check within limits."""
        from unittest.mock import MagicMock
        
        parent = MagicMock()
        parent.id = "parent_123"
        
        config = SpawningConfig(max_concurrent=10, max_depth=3, max_total_spawns=50)
        manager = SpawnManager(parent, config, current_depth=0)
        
        can, reason = manager.can_spawn()
        assert can is True
        assert reason == "OK"
    
    def test_cannot_spawn_max_depth(self):
        """Test spawn blocked at max depth."""
        from unittest.mock import MagicMock
        
        parent = MagicMock()
        parent.id = "parent_123"
        
        config = SpawningConfig(max_depth=2)
        manager = SpawnManager(parent, config, current_depth=2)  # At max depth
        
        can, reason = manager.can_spawn()
        assert can is False
        assert "depth" in reason.lower()
    
    def test_cannot_spawn_max_total(self):
        """Test spawn blocked at max total."""
        from unittest.mock import MagicMock
        
        parent = MagicMock()
        parent.id = "parent_123"
        
        config = SpawningConfig(max_total_spawns=5)
        manager = SpawnManager(parent, config)
        manager._total_spawns = 5  # Already at limit
        
        can, reason = manager.can_spawn()
        assert can is False
        assert "total" in reason.lower()
    
    def test_active_count_tracking(self):
        """Test active spawn count tracking."""
        from unittest.mock import MagicMock
        
        parent = MagicMock()
        parent.id = "parent_123"
        
        config = SpawningConfig()
        manager = SpawnManager(parent, config)
        
        assert manager.active_count == 0
        assert manager.total_spawns == 0
    
    def test_get_summary(self):
        """Test getting manager summary."""
        from unittest.mock import MagicMock
        
        parent = MagicMock()
        parent.id = "parent_123"
        
        config = SpawningConfig(max_depth=3)
        manager = SpawnManager(parent, config, current_depth=1)
        
        summary = manager.get_summary()
        assert summary["active_count"] == 0
        assert summary["total_spawns"] == 0
        assert summary["current_depth"] == 1
        assert summary["max_depth"] == 3


class TestCreateSpawnTool:
    """Tests for spawn tool creation."""
    
    def test_create_tool_basic(self):
        """Test creating basic spawn tool."""
        from unittest.mock import MagicMock
        
        parent = MagicMock()
        parent.id = "parent_123"
        
        config = SpawningConfig()
        manager = SpawnManager(parent, config)
        
        spawn_tool = create_spawn_tool(manager, config)
        
        assert spawn_tool.name == "spawn_agent"
        assert "spawn" in spawn_tool.description.lower()
    
    def test_create_tool_with_roles(self):
        """Test spawn tool shows predefined roles."""
        from unittest.mock import MagicMock
        
        parent = MagicMock()
        parent.id = "parent_123"
        
        config = SpawningConfig(
            specs={
                "researcher": AgentSpec(role="researcher", description="Finds info"),
                "coder": AgentSpec(role="coder", description="Writes code"),
            }
        )
        manager = SpawnManager(parent, config)
        
        spawn_tool = create_spawn_tool(manager, config)
        
        assert "researcher" in spawn_tool.description
        assert "coder" in spawn_tool.description
    
    def test_create_tool_with_available_tools(self):
        """Test spawn tool shows available tools."""
        from unittest.mock import MagicMock
        
        @tool
        def web_search(q: str) -> str:
            return q
        
        parent = MagicMock()
        parent.id = "parent_123"
        
        config = SpawningConfig(available_tools=[web_search])
        manager = SpawnManager(parent, config)
        
        spawn_tool = create_spawn_tool(manager, config)
        
        assert "web_search" in spawn_tool.description


class TestAgentSpawningIntegration:
    """Integration tests for agent spawning."""
    
    def test_agent_with_spawning_config(self):
        """Test agent initialization with spawning config."""
        from agenticflow import Agent
        from agenticflow.models import MockChatModel
        
        model = MockChatModel(responses=["test"])
        config = SpawningConfig(max_concurrent=5)
        
        agent = Agent(
            name="Supervisor",
            model=model,
            spawning=config,
        )
        
        assert agent.can_spawn is True
        assert agent.spawn_manager is not None
        # Spawn tool should be added
        tool_names = [t.name for t in agent.all_tools]
        assert "spawn_agent" in tool_names
    
    def test_agent_without_spawning(self):
        """Test agent without spawning capability."""
        from agenticflow import Agent
        from agenticflow.models import MockChatModel
        
        model = MockChatModel(responses=["test"])
        
        agent = Agent(
            name="Worker",
            model=model,
        )
        
        assert agent.can_spawn is False
        assert agent.spawn_manager is None
    
    @pytest.mark.asyncio
    async def test_spawn_raises_without_config(self):
        """Test that spawn raises error when not configured."""
        from agenticflow import Agent
        from agenticflow.models import MockChatModel
        
        model = MockChatModel(responses=["test"])
        agent = Agent(name="Worker", model=model)
        
        with pytest.raises(RuntimeError, match="Spawning not enabled"):
            await agent.spawn("researcher", "Do research")
    
    @pytest.mark.asyncio
    async def test_parallel_map_raises_without_config(self):
        """Test that parallel_map raises error when not configured."""
        from agenticflow import Agent
        from agenticflow.models import MockChatModel
        
        model = MockChatModel(responses=["test"])
        agent = Agent(name="Worker", model=model)
        
        with pytest.raises(RuntimeError, match="Spawning not enabled"):
            await agent.parallel_map(["a", "b"], "Process {item}")
