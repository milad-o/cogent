"""
AgentConfig - configuration for an Agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agenticflow.core.enums import AgentRole


@dataclass
class AgentConfig:
    """
    Configuration for an Agent.
    
    Defines the agent's identity, capabilities, and behavior parameters.
    Configuration is immutable after creation to ensure consistent behavior.
    
    Attributes:
        name: Human-readable agent name
        role: Agent's role in the system
        description: Detailed description of agent's purpose
        model_name: LLM model to use (e.g., "gpt-4o", "claude-3-opus")
        temperature: LLM temperature parameter (0.0-2.0)
        max_tokens: Maximum tokens in LLM response
        system_prompt: System prompt defining agent behavior
        tools: List of tool names this agent can use
        max_concurrent_tasks: Maximum parallel tasks
        timeout_seconds: Task timeout in seconds
        retry_on_error: Whether to auto-retry on errors
        max_retries: Maximum retry attempts
        
    Example:
        ```python
        config = AgentConfig(
            name="DataAnalyst",
            role=AgentRole.SPECIALIST,
            description="Analyzes data and generates insights",
            model_name="gpt-4o",
            temperature=0.3,
            system_prompt="You are an expert data analyst...",
            tools=["analyze_data", "create_visualization"],
            max_concurrent_tasks=3,
        )
        ```
    """

    name: str
    role: AgentRole = AgentRole.WORKER
    description: str = ""

    # LLM Configuration
    model_name: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    system_prompt: str | None = None

    # Capabilities
    tools: list[str] = field(default_factory=list)

    # Execution Parameters
    max_concurrent_tasks: int = 5
    timeout_seconds: float = 300.0
    retry_on_error: bool = True
    max_retries: int = 3

    # Metadata
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Agent name cannot be empty")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def with_tools(self, tools: list[str]) -> AgentConfig:
        """
        Create a new config with additional tools.
        
        Args:
            tools: Tool names to add
            
        Returns:
            New AgentConfig with the tools
        """
        return AgentConfig(
            name=self.name,
            role=self.role,
            description=self.description,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt,
            tools=list(set(self.tools + tools)),
            max_concurrent_tasks=self.max_concurrent_tasks,
            timeout_seconds=self.timeout_seconds,
            retry_on_error=self.retry_on_error,
            max_retries=self.max_retries,
            metadata=self.metadata.copy(),
        )

    def with_system_prompt(self, prompt: str) -> AgentConfig:
        """
        Create a new config with a different system prompt.
        
        Args:
            prompt: New system prompt
            
        Returns:
            New AgentConfig with the prompt
        """
        return AgentConfig(
            name=self.name,
            role=self.role,
            description=self.description,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=prompt,
            tools=self.tools.copy(),
            max_concurrent_tasks=self.max_concurrent_tasks,
            timeout_seconds=self.timeout_seconds,
            retry_on_error=self.retry_on_error,
            max_retries=self.max_retries,
            metadata=self.metadata.copy(),
        )

    def can_use_tool(self, tool_name: str) -> bool:
        """
        Check if agent is configured to use a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if tool is in agent's tool list, or if no tools are restricted
        """
        # Empty tools list means all tools are allowed
        if not self.tools:
            return True
        return tool_name in self.tools

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "role": self.role.value,
            "description": self.description,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "timeout_seconds": self.timeout_seconds,
            "retry_on_error": self.retry_on_error,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AgentConfig:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            role=AgentRole(data.get("role", "worker")),
            description=data.get("description", ""),
            model_name=data.get("model_name"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens"),
            system_prompt=data.get("system_prompt"),
            tools=data.get("tools", []),
            max_concurrent_tasks=data.get("max_concurrent_tasks", 5),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            retry_on_error=data.get("retry_on_error", True),
            max_retries=data.get("max_retries", 3),
            metadata=data.get("metadata", {}),
        )
