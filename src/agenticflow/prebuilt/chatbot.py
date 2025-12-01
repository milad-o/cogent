"""
Prebuilt Chatbot with conversation memory.

A simple, ready-to-use chatbot that maintains conversation history
across sessions using memory checkpointing. Inherits from Agent for
full access to all agent capabilities.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from agenticflow import Agent

if TYPE_CHECKING:
    from agenticflow.agent.memory import AgentMemory, MemorySaver, MemoryStore
    from agenticflow.agent.resilience import ResilienceConfig
    from agenticflow.models.base import BaseChatModel
    from agenticflow.tools.base import BaseTool


class Chatbot(Agent):
    """
    A prebuilt chatbot with conversation memory.
    
    Inherits from Agent, so you get full access to all agent capabilities:
    - tools, capabilities, streaming, reasoning, structured output
    - memory, store, interceptors, resilience, observability
    - run(), chat(), think(), and all other Agent methods
    
    Features:
    - Default personality for conversational interactions
    - Memory enabled by default (pass memory=False to disable)
    - All Agent capabilities available
    
    Example:
        ```python
        from agenticflow.models import ChatModel
        from agenticflow.prebuilt import Chatbot
        
        bot = Chatbot(
            model=ChatModel(model="gpt-4o-mini"),
            personality="You are a friendly coding assistant.",
        )
        
        # Chat with memory
        response1 = await bot.chat("Hi, I'm learning Python", thread_id="user-1")
        response2 = await bot.chat("What should I learn first?", thread_id="user-1")
        # Bot remembers you're learning Python!
        
        # Use any Agent method
        result = await bot.run("Help me write a function", strategy="react")
        ```
    
    Example with tools:
        ```python
        from agenticflow.tools import tool
        
        @tool
        def search_docs(query: str) -> str:
            '''Search documentation.'''
            return f"Results for: {query}"
        
        bot = Chatbot(
            model=model,
            personality="You are a helpful docs assistant.",
            tools=[search_docs],  # Full tools support!
        )
        
        # Bot can now use tools
        result = await bot.run("Find info about async/await")
        ```
    """
    
    DEFAULT_PERSONALITY = """You are a helpful, friendly assistant.

You engage in natural conversation, remember context from earlier in the chat,
and provide clear, concise responses. If you don't know something, say so honestly."""
    
    def __init__(
        self,
        model: BaseChatModel,
        *,
        # Chatbot-specific
        personality: str | None = None,
        # All Agent parameters
        name: str = "Chatbot",
        tools: Sequence[BaseTool | str] | None = None,
        capabilities: Sequence[Any] | None = None,
        memory: bool | MemorySaver | AgentMemory | None = True,  # Default: enabled
        store: MemoryStore | None = None,
        intercept: Sequence[Any] | None = None,
        stream: bool = False,
        reasoning: bool | Any = False,
        output: type | dict | None = None,
        verbose: bool | str = False,
        resilience: ResilienceConfig | None = None,
        interrupt_on: dict[str, Any] | None = None,
        observer: Any | None = None,
        taskboard: bool | Any = None,
    ) -> None:
        """
        Create a chatbot with conversation memory.
        
        Args:
            model: Chat model (e.g., ChatModel, AzureChat).
            personality: Optional personality/system prompt.
                If not provided, uses a default helpful assistant prompt.
            
            **All Agent parameters are supported:**
            name: Name for the chatbot.
            tools: List of tools the chatbot can use.
            capabilities: Capabilities to attach.
            memory: Enable conversation memory (default: True).
                - True: In-memory persistence
                - MemorySaver: Custom checkpointer
                - AgentMemory: Full memory manager
                - None/False: Disable memory
            store: Long-term memory store for cross-thread state.
            intercept: Interceptors (gates, guards, prompt adapters).
            stream: Enable streaming responses.
            reasoning: Enable extended thinking mode.
            output: Structured output schema (Pydantic model, dataclass, etc.).
            verbose: Observability level (False, True, "verbose", "debug", "trace").
            resilience: Retry and fallback configuration.
            interrupt_on: HITL tool approval rules.
            observer: Custom observer for rich observability.
            taskboard: Enable task tracking.
        """
        instructions = personality or self.DEFAULT_PERSONALITY
        
        super().__init__(
            name=name,
            model=model,
            instructions=instructions,
            tools=tools or [],
            capabilities=capabilities,
            memory=memory,
            store=store,
            intercept=intercept,
            stream=stream,
            reasoning=reasoning,
            output=output,
            verbose=verbose,
            resilience=resilience,
            interrupt_on=interrupt_on,
            observer=observer,
            taskboard=taskboard,
        )
    
    @property
    def personality(self) -> str:
        """Get the chatbot's personality (system prompt)."""
        return self._config.instructions or self.DEFAULT_PERSONALITY
    
    def set_personality(self, personality: str) -> None:
        """Update the chatbot's personality.
        
        Args:
            personality: New personality/system prompt.
        """
        self._config = self._config.model_copy(update={"instructions": personality})


def create_chatbot(
    model: BaseChatModel,
    *,
    personality: str | None = None,
    name: str = "Chatbot",
    tools: Sequence[BaseTool | str] | None = None,
    capabilities: Sequence[Any] | None = None,
    memory: bool | MemorySaver | AgentMemory | None = True,
    store: MemoryStore | None = None,
    intercept: Sequence[Any] | None = None,
    stream: bool = False,
    reasoning: bool | Any = False,
    output: type | dict | None = None,
    verbose: bool | str = False,
    resilience: ResilienceConfig | None = None,
    interrupt_on: dict[str, Any] | None = None,
    observer: Any | None = None,
    taskboard: bool | Any = None,
) -> Chatbot:
    """
    Create a chatbot with conversation memory.
    
    This is a convenience function that creates a Chatbot instance.
    
    Args:
        model: Chat model.
        personality: Optional system prompt / personality.
        name: Chatbot name.
        tools: Optional tools for the chatbot.
        capabilities: Capabilities to attach.
        memory: Enable conversation memory (default: True).
        store: Long-term memory store.
        intercept: Interceptors for execution hooks.
        stream: Enable streaming responses.
        reasoning: Enable extended thinking mode.
        output: Structured output schema.
        verbose: Observability level.
        resilience: Retry and fallback configuration.
        interrupt_on: HITL tool approval rules.
        observer: Custom observer.
        taskboard: Enable task tracking.
        
    Returns:
        A configured Chatbot instance (which is an Agent).
        
    Example:
        ```python
        from agenticflow.models import ChatModel
        from agenticflow.prebuilt import create_chatbot
        
        bot = create_chatbot(
            model=ChatModel(model="gpt-4o-mini"),
            personality="You are a pirate who speaks in pirate slang.",
        )
        
        response = await bot.chat("Hello!", thread_id="user-123")
        # "Ahoy there, matey! What be ye needin' today?"
        ```
    """
    return Chatbot(
        model=model,
        personality=personality,
        name=name,
        tools=tools,
        capabilities=capabilities,
        memory=memory,
        store=store,
        intercept=intercept,
        stream=stream,
        reasoning=reasoning,
        output=output,
        verbose=verbose,
        resilience=resilience,
        interrupt_on=interrupt_on,
        observer=observer,
        taskboard=taskboard,
    )
