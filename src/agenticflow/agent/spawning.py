"""
Agent Spawning - Dynamic creation of ephemeral specialist agents.

This module enables supervisor agents to spawn child agents on-demand
for parallel task execution. Spawned agents are ephemeral by default
and automatically cleaned up after their task completes.

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.agent.spawning import SpawningConfig, AgentSpec
    
    # With predefined specialists
    supervisor = Agent(
        name="Supervisor",
        model=model,
        spawning=SpawningConfig(
            max_concurrent=10,
            specs={
                "researcher": AgentSpec(
                    system_prompt="You research topics thoroughly",
                    tools=[web_search],
                ),
            },
        ),
    )
    
    # Dynamic spawning (LLM decides)
    supervisor = Agent(
        name="Supervisor",
        model=model, 
        spawning=SpawningConfig(
            max_concurrent=5,
            available_tools=[web_search, filesystem],
        ),
    )
    ```
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from agenticflow.observability.trace_record import TraceType
from agenticflow.core.utils import generate_id, now_utc
from agenticflow.tools.base import BaseTool, tool

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent


@dataclass
class AgentSpec:
    """
    Specification for a spawnable agent role.
    
    Defines the configuration for a specialist agent that can be
    spawned by a supervisor. If not provided, the LLM dynamically
    creates the role configuration.
    
    Attributes:
        role: Name/identifier for this specialist role.
        system_prompt: System prompt for the specialist.
        tools: List of tools available to this specialist.
        model: Model to use (None = inherit from parent).
        description: Human-readable description of the role.
        
    Example:
        ```python
        researcher = AgentSpec(
            role="researcher",
            system_prompt="You are a thorough researcher who finds accurate information.",
            tools=[web_search, fetch_page],
            description="Researches topics and gathers information",
        )
        ```
    """
    
    role: str
    system_prompt: str | None = None
    tools: list[BaseTool | Callable] | None = None
    model: Any | None = None  # ChatModel or None to inherit
    description: str | None = None
    
    def __post_init__(self):
        if not self.description:
            self.description = f"Specialist agent for {self.role} tasks"


@dataclass
class SpawningConfig:
    """
    Configuration for agent spawning capabilities.
    
    Controls how a supervisor agent can spawn child agents.
    
    Attributes:
        max_concurrent: Maximum number of concurrent spawned agents.
        max_depth: Maximum recursive spawning depth (spawns spawning spawns).
        max_total_spawns: Maximum total spawns per task execution.
        ephemeral: Whether spawned agents auto-cleanup after task.
        inherit_tools: Whether spawns inherit parent's tools by default.
        inherit_model: Whether spawns use parent's model by default.
        specs: Predefined specialist specifications by role name.
        available_tools: Pool of tools available for dynamic spawning.
        
    Example:
        ```python
        config = SpawningConfig(
            max_concurrent=10,
            max_depth=2,
            specs={
                "researcher": AgentSpec(role="researcher", ...),
                "coder": AgentSpec(role="coder", ...),
            },
        )
        ```
    """
    
    max_concurrent: int = 10
    max_depth: int = 3
    max_total_spawns: int = 50
    ephemeral: bool = True
    inherit_tools: bool = False
    inherit_model: bool = True
    specs: dict[str, AgentSpec] = field(default_factory=dict)
    available_tools: list[BaseTool | Callable] = field(default_factory=list)
    
    def get_spec(self, role: str) -> AgentSpec | None:
        """Get predefined spec for a role, or None for dynamic creation."""
        return self.specs.get(role)
    
    def has_predefined_roles(self) -> bool:
        """Check if any roles are predefined."""
        return len(self.specs) > 0
    
    def get_available_roles(self) -> list[str]:
        """Get list of predefined role names."""
        return list(self.specs.keys())


@dataclass
class SpawnedAgentInfo:
    """
    Tracking information for a spawned agent.
    
    Used internally to track spawned agents and their lifecycle.
    """
    
    id: str
    role: str
    task: str
    parent_id: str
    depth: int
    created_at: Any = field(default_factory=now_utc)
    completed_at: Any | None = None
    result: str | None = None
    error: str | None = None
    status: str = "running"  # running, completed, failed, cancelled


class SpawnManager:
    """
    Manages spawned agents for a parent agent.
    
    Handles:
    - Tracking active spawns
    - Enforcing limits (concurrent, total, depth)
    - Cleanup of ephemeral agents
    - Parallel execution coordination
    """
    
    __slots__ = (
        "_parent",
        "_config",
        "_active_spawns",
        "_spawn_history",
        "_total_spawns",
        "_semaphore",
        "_current_depth",
    )
    
    def __init__(
        self,
        parent: "Agent",
        config: SpawningConfig,
        current_depth: int = 0,
    ):
        self._parent = parent
        self._config = config
        self._active_spawns: dict[str, SpawnedAgentInfo] = {}
        self._spawn_history: list[SpawnedAgentInfo] = []
        self._total_spawns = 0
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._current_depth = current_depth
    
    @property
    def active_count(self) -> int:
        """Number of currently active spawned agents."""
        return len(self._active_spawns)
    
    @property
    def total_spawns(self) -> int:
        """Total number of agents spawned (including completed)."""
        return self._total_spawns
    
    def can_spawn(self) -> tuple[bool, str]:
        """Check if spawning is allowed. Returns (allowed, reason)."""
        if self._current_depth >= self._config.max_depth:
            return False, f"Max spawn depth ({self._config.max_depth}) reached"
        if self._total_spawns >= self._config.max_total_spawns:
            return False, f"Max total spawns ({self._config.max_total_spawns}) reached"
        if self.active_count >= self._config.max_concurrent:
            return False, f"Max concurrent spawns ({self._config.max_concurrent}) reached"
        return True, "OK"
    
    async def spawn(
        self,
        role: str,
        task: str,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
    ) -> str:
        """
        Spawn a new agent and execute the task.
        
        Args:
            role: Role/type of the specialist.
            task: Task for the spawned agent to execute.
            system_prompt: Custom system prompt (overrides spec).
            tools: Tool names to enable (from available_tools).
            
        Returns:
            Result from the spawned agent.
            
        Raises:
            RuntimeError: If spawning is not allowed.
        """
        from agenticflow.agent.base import Agent
        
        can, reason = self.can_spawn()
        if not can:
            raise RuntimeError(f"Cannot spawn: {reason}")
        
        spawn_id = generate_id("spawn")
        self._total_spawns += 1
        
        # Get or create spec
        spec = self._config.get_spec(role)
        
        # Determine system prompt
        if system_prompt:
            effective_prompt = system_prompt
        elif spec and spec.system_prompt:
            effective_prompt = spec.system_prompt
        else:
            # Dynamic: create prompt for the role
            effective_prompt = f"You are a specialist {role}. Complete the assigned task efficiently and thoroughly."
        
        # Determine model
        if spec and spec.model:
            model = spec.model
        elif self._config.inherit_model:
            model = self._parent.model
        else:
            model = self._parent.model  # Fallback
        
        # Determine tools
        spawn_tools: list[BaseTool | Callable] = []
        if spec and spec.tools:
            spawn_tools = list(spec.tools)
        elif tools:
            # Select from available tools by name
            tool_map = {
                (t.name if isinstance(t, BaseTool) else t.__name__): t 
                for t in self._config.available_tools
            }
            spawn_tools = [tool_map[name] for name in tools if name in tool_map]
        elif self._config.inherit_tools:
            spawn_tools = list(self._parent.config.tools)
        
        # Create spawn info
        info = SpawnedAgentInfo(
            id=spawn_id,
            role=role,
            task=task,
            parent_id=self._parent.id,
            depth=self._current_depth + 1,
        )
        self._active_spawns[spawn_id] = info
        
        # Emit spawn event
        await self._emit_spawn_event(TraceType.AGENT_SPAWNED, info)
        
        # Create the spawned agent
        spawned = Agent(
            name=f"{self._parent.name}_{role}_{spawn_id[-8:]}",
            model=model,
            system_prompt=effective_prompt,
            tools=spawn_tools if spawn_tools else None,
        )
        
        # Inherit infrastructure
        spawned.event_bus = self._parent.event_bus
        spawned.tool_registry = self._parent.tool_registry
        
        # If spawned agent can also spawn (depth allows), set it up
        if self._current_depth + 1 < self._config.max_depth:
            child_config = SpawningConfig(
                max_concurrent=self._config.max_concurrent,
                max_depth=self._config.max_depth,
                max_total_spawns=self._config.max_total_spawns - self._total_spawns,
                ephemeral=self._config.ephemeral,
                inherit_tools=self._config.inherit_tools,
                inherit_model=self._config.inherit_model,
                specs=self._config.specs,
                available_tools=self._config.available_tools,
            )
            spawned._spawn_manager = SpawnManager(
                spawned, child_config, self._current_depth + 1
            )
        
        # Execute with semaphore for concurrency control
        async with self._semaphore:
            try:
                result = await spawned.run(task)
                info.result = str(result)
                info.status = "completed"
                info.completed_at = now_utc()
                await self._emit_spawn_event(TraceType.AGENT_SPAWN_COMPLETED, info)
                return str(result)
            except Exception as e:
                info.error = str(e)
                info.status = "failed"
                info.completed_at = now_utc()
                await self._emit_spawn_event(TraceType.AGENT_SPAWN_FAILED, info)
                raise
            finally:
                # Cleanup
                del self._active_spawns[spawn_id]
                self._spawn_history.append(info)
                if self._config.ephemeral:
                    await self._emit_spawn_event(TraceType.AGENT_DESPAWNED, info)
    
    async def spawn_many(
        self,
        tasks: list[tuple[str, str]],  # [(role, task), ...]
    ) -> list[str]:
        """
        Spawn multiple agents in parallel.
        
        Args:
            tasks: List of (role, task) tuples.
            
        Returns:
            List of results in same order as tasks.
        """
        async def _spawn_one(role: str, task: str) -> str:
            try:
                return await self.spawn(role, task)
            except Exception as e:
                return f"Error: {e}"
        
        return await asyncio.gather(*[
            _spawn_one(role, task) for role, task in tasks
        ])
    
    async def parallel_map(
        self,
        items: list[Any],
        task_template: str,
        role: str = "worker",
    ) -> list[str]:
        """
        Map a task template over items in parallel.
        
        Args:
            items: Items to process.
            task_template: Template with {item} placeholder.
            role: Role for spawned workers.
            
        Returns:
            List of results.
            
        Example:
            ```python
            results = await manager.parallel_map(
                items=["Apple", "Google", "Microsoft"],
                task_template="Research {item} and summarize",
                role="researcher",
            )
            ```
        """
        tasks = [
            (role, task_template.format(item=item))
            for item in items
        ]
        return await self.spawn_many(tasks)
    
    async def _emit_spawn_event(
        self,
        event_type: TraceType,
        info: SpawnedAgentInfo,
    ) -> None:
        """Emit spawning-related event."""
        event_bus = getattr(self._parent, "event_bus", None)
        if event_bus:
            await event_bus.publish(event_type.value, {
                "parent_agent": self._parent.name,
                "parent_id": self._parent.id,
                "spawn_id": info.id,
                "role": info.role,
                "task": info.task[:200] if info.task else "",
                "depth": info.depth,
                "status": info.status,
                "result_preview": (info.result or "")[:200] if info.result else None,
                "error": info.error,
                "active_spawns": self.active_count,
                "total_spawns": self.total_spawns,
            })
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of spawning activity."""
        return {
            "active_count": self.active_count,
            "total_spawns": self.total_spawns,
            "completed": sum(1 for s in self._spawn_history if s.status == "completed"),
            "failed": sum(1 for s in self._spawn_history if s.status == "failed"),
            "current_depth": self._current_depth,
            "max_depth": self._config.max_depth,
        }


def create_spawn_tool(manager: SpawnManager, config: SpawningConfig) -> BaseTool:
    """
    Create the spawn_agent tool for an agent.
    
    This tool allows the LLM to spawn specialist agents.
    """
    
    # Build role descriptions for the tool
    if config.has_predefined_roles():
        roles_desc = "Available roles:\n" + "\n".join(
            f"- {role}: {spec.description or 'Specialist'}"
            for role, spec in config.specs.items()
        )
    else:
        roles_desc = "You can define any role. The agent will be created dynamically."
    
    # Build tools description
    if config.available_tools:
        tool_names = [
            t.name if isinstance(t, BaseTool) else t.__name__
            for t in config.available_tools
        ]
        tools_desc = f"Available tools for spawns: {', '.join(tool_names)}"
    else:
        tools_desc = "No specific tools available for spawns."
    
    @tool(
        name="spawn_agent",
        description=f"""Spawn a specialist agent to handle a specific subtask.
        
Use this to parallelize work or delegate to specialists.

{roles_desc}

{tools_desc}

The spawned agent will execute the task and return the result.
You can spawn multiple agents for parallel execution.
""",
    )
    async def spawn_agent(
        role: str,
        task: str,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
    ) -> str:
        """
        Spawn a specialist agent to execute a task.
        
        Args:
            role: The specialist role (e.g., "researcher", "analyst", "coder").
            task: The specific task for the spawned agent.
            system_prompt: Optional custom instructions for the agent.
            tools: Optional list of tool names to enable.
            
        Returns:
            The result from the spawned agent.
        """
        try:
            result = await manager.spawn(
                role=role,
                task=task,
                system_prompt=system_prompt,
                tools=tools,
            )
            return f"[{role}] completed: {result}"
        except RuntimeError as e:
            return f"[spawn failed] {e}"
        except Exception as e:
            return f"[{role}] error: {e}"
    
    return spawn_agent


__all__ = [
    "AgentSpec",
    "SpawningConfig", 
    "SpawnedAgentInfo",
    "SpawnManager",
    "create_spawn_tool",
]
