"""
AgentConfig - configuration for an Agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from agenticflow.core.enums import AgentRole

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from agenticflow.agents.resilience import ResilienceConfig
    from agenticflow.providers import ModelSpec


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
        model: LLM model - can be:
            - String: "openai/gpt-4o", "anthropic/claude-3-5-sonnet-latest"
            - String (legacy): "openai:gpt-4o", "azure:gpt-4"
            - String (auto-detect): "gpt-4o" -> infers OpenAI
            - ModelSpec: Explicit specification object
            - LangChain model object: ChatOpenAI, AzureChatOpenAI, etc.
            - Dict: {"provider": "openai", "model": "gpt-4o", ...}
        model_name: Deprecated, use 'model' instead
        temperature: LLM temperature parameter (0.0-2.0)
        max_tokens: Maximum tokens in LLM response
        system_prompt: System prompt defining agent behavior
        tools: List of tool names this agent can use
        max_concurrent_tasks: Maximum parallel tasks
        timeout_seconds: Task timeout in seconds (deprecated, use resilience_config)
        retry_on_error: Whether to auto-retry on errors (deprecated, use resilience_config)
        max_retries: Maximum retry attempts (deprecated, use resilience_config)
        model_kwargs: Additional kwargs passed to model creation
        resilience_config: Advanced resilience configuration (retry, circuit breaker, fallback)
        fallback_tools: Mapping of tool -> fallback tools for graceful degradation
        
    Example:
        ```python
        # Simple - model name string (auto-detects provider)
        config = AgentConfig(
            name="DataAnalyst",
            role=AgentRole.SPECIALIST,
            model="gpt-4o",  # Auto-detects OpenAI
        )
        
        # Provider/model format (preferred)
        config = AgentConfig(
            name="Writer",
            model="anthropic/claude-3-5-sonnet-latest",
        )
        
        # Using ModelSpec for full control
        from agenticflow.providers import ModelSpec
        config = AgentConfig(
            name="PreciseAgent",
            model=ModelSpec(
                provider="openai",
                model="gpt-4o",
                temperature=0.3,
                max_tokens=2000,
            ),
        )
        
        # Azure OpenAI with Managed Identity
        from agenticflow.providers import AzureOpenAIProvider, AzureAuthMethod
        provider = AzureOpenAIProvider(
            endpoint="https://my-resource.openai.azure.com",
            auth_method=AzureAuthMethod.MANAGED_IDENTITY,
        )
        config = AgentConfig(
            name="AzureAgent",
            model=provider.create_chat_model("gpt-4o-deployment"),
        )
        
        # Direct LangChain model object
        from langchain_openai import ChatOpenAI
        config = AgentConfig(
            name="CustomAgent",
            model=ChatOpenAI(model="gpt-4o", temperature=0.5),
        )
        ```
    """

    name: str
    role: AgentRole = AgentRole.WORKER
    description: str = ""

    # LLM Configuration - flexible model specification
    model: str | ModelSpec | BaseChatModel | dict[str, Any] | None = None
    model_name: str | None = None  # Deprecated, for backwards compatibility
    temperature: float = 0.7
    max_tokens: int | None = None
    system_prompt: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    # Capabilities
    tools: list[str] = field(default_factory=list)

    # Execution Parameters
    max_concurrent_tasks: int = 5
    timeout_seconds: float = 300.0  # Deprecated: use resilience_config
    retry_on_error: bool = True  # Deprecated: use resilience_config
    max_retries: int = 3  # Deprecated: use resilience_config
    
    # Resilience Configuration (intelligent retry, circuit breaker, fallback)
    resilience_config: ResilienceConfig | None = None
    fallback_tools: dict[str, list[str]] = field(default_factory=dict)  # tool -> [fallbacks]

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
        
        # Handle backwards compatibility: model_name -> model
        if self.model_name and not self.model:
            object.__setattr__(self, 'model', self.model_name)

    @property
    def effective_model(self) -> str | Any | None:
        """Get the effective model (handles backwards compatibility)."""
        return self.model or self.model_name

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
            model=self.model,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt,
            model_kwargs=self.model_kwargs.copy(),
            tools=list(set(self.tools + tools)),
            max_concurrent_tasks=self.max_concurrent_tasks,
            timeout_seconds=self.timeout_seconds,
            retry_on_error=self.retry_on_error,
            max_retries=self.max_retries,
            resilience_config=self.resilience_config,
            fallback_tools=self.fallback_tools.copy(),
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
            model=self.model,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=prompt,
            model_kwargs=self.model_kwargs.copy(),
            tools=self.tools.copy(),
            max_concurrent_tasks=self.max_concurrent_tasks,
            timeout_seconds=self.timeout_seconds,
            retry_on_error=self.retry_on_error,
            max_retries=self.max_retries,
            resilience_config=self.resilience_config,
            fallback_tools=self.fallback_tools.copy(),
            metadata=self.metadata.copy(),
        )
    
    def with_resilience(self, config: ResilienceConfig) -> AgentConfig:
        """
        Create a new config with resilience configuration.
        
        Args:
            config: ResilienceConfig for retry, circuit breaker, fallback
            
        Returns:
            New AgentConfig with resilience
            
        Example:
            ```python
            from agenticflow.agents.resilience import ResilienceConfig
            
            config = AgentConfig(
                name="ResilientAgent",
                model="gpt-4o",
            ).with_resilience(ResilienceConfig.aggressive())
            ```
        """
        return AgentConfig(
            name=self.name,
            role=self.role,
            description=self.description,
            model=self.model,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt,
            model_kwargs=self.model_kwargs.copy(),
            tools=self.tools.copy(),
            max_concurrent_tasks=self.max_concurrent_tasks,
            timeout_seconds=self.timeout_seconds,
            retry_on_error=self.retry_on_error,
            max_retries=self.max_retries,
            resilience_config=config,
            fallback_tools=self.fallback_tools.copy(),
            metadata=self.metadata.copy(),
        )
    
    def with_fallbacks(self, fallback_tools: dict[str, list[str]]) -> AgentConfig:
        """
        Create a new config with fallback tool mappings.
        
        Args:
            fallback_tools: Mapping of primary tool -> list of fallback tools
            
        Returns:
            New AgentConfig with fallbacks
            
        Example:
            ```python
            config = AgentConfig(
                name="ResilientAgent",
                model="gpt-4o",
                tools=["web_search", "cached_search", "local_search"],
            ).with_fallbacks({
                "web_search": ["cached_search", "local_search"],
            })
            ```
        """
        return AgentConfig(
            name=self.name,
            role=self.role,
            description=self.description,
            model=self.model,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt,
            model_kwargs=self.model_kwargs.copy(),
            tools=self.tools.copy(),
            max_concurrent_tasks=self.max_concurrent_tasks,
            timeout_seconds=self.timeout_seconds,
            retry_on_error=self.retry_on_error,
            max_retries=self.max_retries,
            resilience_config=self.resilience_config,
            fallback_tools={**self.fallback_tools, **fallback_tools},
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
        # Only serialize model if it's a string (not an object)
        model_value = self.model if isinstance(self.model, str) else None
        return {
            "name": self.name,
            "role": self.role.value,
            "description": self.description,
            "model": model_value,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "model_kwargs": self.model_kwargs,
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
            model=data.get("model"),
            model_name=data.get("model_name"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens"),
            system_prompt=data.get("system_prompt"),
            model_kwargs=data.get("model_kwargs", {}),
            tools=data.get("tools", []),
            max_concurrent_tasks=data.get("max_concurrent_tasks", 5),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            retry_on_error=data.get("retry_on_error", True),
            max_retries=data.get("max_retries", 3),
            metadata=data.get("metadata", {}),
        )
