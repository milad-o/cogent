"""
AgentConfig - configuration for an Agent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agenticflow.core.enums import AgentRole
from agenticflow.executors.base import ExecutionStrategy

if TYPE_CHECKING:
    from agenticflow.agent.resilience import ResilienceConfig
    from agenticflow.models.anthropic import AnthropicChat
    from agenticflow.models.azure import AzureOpenAIChat
    from agenticflow.models.base import BaseChatModel
    from agenticflow.models.openai import OpenAIChat

# Type for interrupt rules - bool or callable that takes (tool_name, args) -> bool
InterruptRule = bool | Callable[[str, dict[str, Any]], bool]


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
        model: LLM model - must be a native AgenticFlow chat model instance.
            Use native models:
            - ChatModel(model="gpt-4o")
            - AzureOpenAIChat(...)
            - AnthropicChat(model="claude-3-5-sonnet-latest")
            - GeminiChat(model="gemini-2.0-flash")
            - OllamaChat(model="llama3.2")
            Or use create_chat() factory function.
        temperature: LLM temperature parameter (0.0-2.0) - only used if model is None
        max_tokens: Maximum tokens in LLM response - only used if model is None
        system_prompt: System prompt defining agent behavior
        tools: List of tool names this agent can use
        max_concurrent_tasks: Maximum parallel tasks
        resilience_config: Advanced resilience configuration (retry, circuit breaker, fallback)
        fallback_tools: Mapping of tool -> fallback tools for graceful degradation

    Example:
        ```python
        from agenticflow.models import ChatModel, create_chat

        # OpenAI
        config = AgentConfig(
            name="DataAnalyst",
            role=AgentRole.SPECIALIST,
            model=ChatModel(model="gpt-4o"),
        )

        # Anthropic
        config = AgentConfig(
            name="Writer",
            model=AnthropicChat(model="claude-3-5-sonnet-latest"),
        )

        # Azure OpenAI with Managed Identity
        from agenticflow.models.azure import AzureOpenAIChat
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )
        config = AgentConfig(
            name="AzureAgent",
            model=AzureOpenAIChat(
                azure_deployment="gpt-4o",
                azure_endpoint="https://my-resource.openai.azure.com",
                azure_ad_token_provider=token_provider,
            ),
        )

        # Native models - RECOMMENDED
        from agenticflow.models.openai import OpenAIChat
        from agenticflow.models.azure import AzureOpenAIChat
        from agenticflow.models.anthropic import AnthropicChat

        # OpenAI native
        config = AgentConfig(
            name="FastAgent",
            model=OpenAIChat(model="gpt-4o"),
        )

        # Azure native with Managed Identity (Entra ID)
        from agenticflow.models.azure import AzureEntraAuth
        config = AgentConfig(
            name="AzureNative",
            model=AzureOpenAIChat(
                deployment="gpt-4o",
                azure_endpoint="https://my-resource.openai.azure.com",
                entra=AzureEntraAuth(method="managed_identity"),
            ),
        )

        # Anthropic native
        config = AgentConfig(
            name="ClaudeAgent",
            model=AnthropicChat(model="claude-sonnet-4-20250514"),
        )
        ```
    """

    name: str
    role: AgentRole = AgentRole.WORKER
    description: str = ""

    # LLM Configuration - native BaseChatModel
    # Native models: from agenticflow.models.openai, azure, anthropic, groq, gemini, etc.
    model: BaseChatModel | OpenAIChat | AzureOpenAIChat | AnthropicChat | None = None
    temperature: float = 0.7  # Used only if model is None (for lazy creation)
    max_tokens: int | None = None  # Used only if model is None
    system_prompt: str | None = None
    model_kwargs: dict[str, Any] = field(
        default_factory=dict
    )  # For lazy model creation

    # Streaming Configuration
    stream: bool = False  # Enable token-by-token streaming by default

    # Capabilities
    tools: list[str] = field(default_factory=list)

    # Execution Parameters
    max_concurrent_tasks: int = 5
    timeout_seconds: float = 300.0  # Deprecated: use resilience_config
    retry_on_error: bool = True  # Deprecated: use resilience_config
    max_retries: int = 3  # Deprecated: use resilience_config

    # Execution Strategy - how the agent processes tasks with tools
    # - NATIVE: High-performance parallel execution (DEFAULT)
    # - SEQUENTIAL: Sequential tool execution for ordered tasks
    # - TREE_SEARCH: LATS-style MCTS with backtracking (BEST ACCURACY)
    execution_strategy: ExecutionStrategy = ExecutionStrategy.NATIVE

    # Resilience Configuration (intelligent retry, circuit breaker, fallback)
    resilience_config: ResilienceConfig | None = None
    fallback_tools: dict[str, list[str]] = field(
        default_factory=dict
    )  # tool -> [fallbacks]

    # Human-in-the-Loop Configuration
    # Map tool names to interrupt rules:
    # - True: Always require human approval
    # - False: Never require approval (auto-approve)
    # - Callable[[str, dict], bool]: Dynamic decision based on tool name and args
    # - "*": Wildcard rule for unlisted tools
    interrupt_on: dict[str, InterruptRule] = field(default_factory=dict)

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
        
        # Convert string execution_strategy to enum
        if isinstance(self.execution_strategy, str):
            strategy_map = {
                "native": ExecutionStrategy.NATIVE,
                "sequential": ExecutionStrategy.SEQUENTIAL,
                "tree_search": ExecutionStrategy.TREE_SEARCH,
            }
            strategy = strategy_map.get(self.execution_strategy.lower())
            if strategy is None:
                valid = ", ".join(strategy_map.keys())
                raise ValueError(f"Invalid execution_strategy: {self.execution_strategy}. Valid: {valid}")
            object.__setattr__(self, "execution_strategy", strategy)

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
            interrupt_on=self.interrupt_on.copy(),
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
            interrupt_on=self.interrupt_on.copy(),
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
            from agenticflow.agent.resilience import ResilienceConfig

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
            interrupt_on=self.interrupt_on.copy(),
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
            interrupt_on=self.interrupt_on.copy(),
            metadata=self.metadata.copy(),
        )

    def with_interrupt_on(self, interrupt_on: dict[str, InterruptRule]) -> AgentConfig:
        """
        Create a new config with human-in-the-loop interrupt rules.

        Args:
            interrupt_on: Mapping of tool names to interrupt rules.
                - True: Always require human approval
                - False: Auto-approve (never interrupt)
                - Callable[[str, dict], bool]: Dynamic decision based on args
                - "*": Wildcard rule for unlisted tools

        Returns:
            New AgentConfig with interrupt rules

        Example:
            ```python
            config = AgentConfig(
                name="CautiousAgent",
                model="gpt-4o",
                tools=["read_file", "write_file", "delete_file"],
            ).with_interrupt_on({
                "delete_file": True,  # Always require approval
                "write_file": lambda name, args: "/important/" in args.get("path", ""),
                "read_file": False,  # Auto-approve
            })
            ```
        """
        return AgentConfig(
            name=self.name,
            role=self.role,
            description=self.description,
            model=self.model,
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
            fallback_tools=self.fallback_tools.copy(),
            interrupt_on={**self.interrupt_on, **interrupt_on},
            metadata=self.metadata.copy(),
        )

    @property
    def effective_model(self) -> BaseChatModel | None:
        """Get the effective model for this config.

        Returns the model if set, otherwise None.
        Lazy model creation from model_name is no longer supported -
        users should pass native model instances directly.
        """
        return self.model

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
        """Convert to dictionary (model and callables are not serialized)."""
        # Filter out callable interrupt rules (can't serialize functions)
        serializable_interrupt_on = {
            k: v for k, v in self.interrupt_on.items() if isinstance(v, bool)
        }
        return {
            "name": self.name,
            "role": self.role.value,
            "description": self.description,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "model_kwargs": self.model_kwargs,
            "tools": self.tools,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "timeout_seconds": self.timeout_seconds,
            "retry_on_error": self.retry_on_error,
            "max_retries": self.max_retries,
            "interrupt_on": serializable_interrupt_on,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict, model: BaseChatModel | None = None) -> AgentConfig:
        """Create from dictionary.

        Args:
            data: Configuration dictionary.
            model: Chat model instance to use.
        """
        return cls(
            name=data["name"],
            role=AgentRole(data.get("role", "worker")),
            description=data.get("description", ""),
            model=model,
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens"),
            system_prompt=data.get("system_prompt"),
            model_kwargs=data.get("model_kwargs", {}),
            tools=data.get("tools", []),
            max_concurrent_tasks=data.get("max_concurrent_tasks", 5),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            retry_on_error=data.get("retry_on_error", True),
            max_retries=data.get("max_retries", 3),
            interrupt_on=data.get("interrupt_on", {}),
            metadata=data.get("metadata", {}),
        )
