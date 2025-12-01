"""
Prebuilt Chatbot with conversation memory.

A simple, ready-to-use chatbot that maintains conversation history
across sessions using memory checkpointing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from agenticflow import Agent

if TYPE_CHECKING:
    from agenticflow.agent.resilience import ResilienceConfig
    from agenticflow.context import RunContext
    from agenticflow.models.base import BaseChatModel
    from agenticflow.tools.base import BaseTool


class Chatbot:
    """
    A prebuilt chatbot with conversation memory.
    
    Features:
    - Maintains conversation history per thread
    - Optional personality customization
    - Optional tools for enhanced capabilities
    - Built-in memory persistence
    
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
        ```
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        *,
        name: str = "Chatbot",
        personality: str | None = None,
        tools: list[BaseTool] | None = None,
        memory: bool = True,
        # Agent configuration
        intercept: Sequence[Any] | None = None,
        stream: bool = False,
        reasoning: bool | Any = False,
        output: type | dict | None = None,
        verbose: bool | str = False,
        resilience: ResilienceConfig | None = None,
    ) -> None:
        """
        Create a chatbot.
        
        Args:
            model: Native chat model (e.g., ChatModel, AzureChat).
            name: Name for the chatbot.
            personality: Optional personality/system prompt.
                If not provided, uses a default helpful assistant prompt.
            tools: Optional list of tools the chatbot can use.
            memory: Whether to enable conversation memory (default: True).
            intercept: Interceptors for execution hooks (gates, guards, prompt adapters).
            stream: Enable streaming responses by default.
            reasoning: Enable extended thinking mode.
            output: Structured output schema (Pydantic model, dataclass, etc.).
            verbose: Observability level (False, True, "verbose", "debug", "trace").
            resilience: Retry and fallback configuration.
        """
        default_personality = """You are a helpful, friendly assistant.
        
You engage in natural conversation, remember context from earlier in the chat,
and provide clear, concise responses. If you don't know something, say so honestly."""
        
        instructions = personality or default_personality
        
        self._agent = Agent(
            name=name,
            model=model,
            instructions=instructions,
            tools=tools or [],
            memory=memory,
            intercept=intercept,
            stream=stream,
            reasoning=reasoning,
            output=output,
            verbose=verbose,
            resilience=resilience,
        )
    
    @property
    def agent(self) -> Agent:
        """Access the underlying Agent if needed."""
        return self._agent
    
    @property
    def name(self) -> str:
        """Chatbot name."""
        return self._agent.name
    
    async def chat(
        self,
        message: str,
        thread_id: str | None = None,
    ) -> str:
        """
        Send a message and get a response.
        
        Args:
            message: Your message to the chatbot.
            thread_id: Conversation thread ID. Messages with the same
                thread_id share conversation history.
                
        Returns:
            The chatbot's response.
        """
        return await self._agent.chat(message, thread_id=thread_id)
    
    async def chat_with_tools(
        self,
        message: str,
        thread_id: str | None = None,
        *,
        context: dict[str, Any] | RunContext | None = None,
        strategy: str = "dag",
        verbose: bool = False,
        max_iterations: int = 10,
    ) -> str:
        """
        Chat with tool execution enabled.
        
        Use this when you want the chatbot to use its tools
        to answer your question.
        
        Args:
            message: Your message.
            thread_id: Conversation thread ID.
            context: Optional context dict or RunContext for tools/interceptors.
            strategy: Execution strategy ("dag", "react", "plan").
            verbose: Show detailed progress with tool calls.
            max_iterations: Maximum LLM iterations (default: 10).
            
        Returns:
            The chatbot's response (may include tool results).
        """
        if verbose:
            from agenticflow.observability import OutputConfig, ProgressTracker
            tracker = ProgressTracker(OutputConfig.verbose())
            return await self._agent.run(
                message,
                context=context,
                strategy=strategy,
                tracker=tracker,
                max_iterations=max_iterations,
            )
        
        return await self._agent.run(
            message,
            context=context,
            strategy=strategy,
            max_iterations=max_iterations,
        )
    
    def clear_history(self, thread_id: str) -> None:
        """
        Clear conversation history for a thread.
        
        Args:
            thread_id: The thread to clear.
        """
        # Access memory manager to clear
        if hasattr(self._agent, "_memory"):
            # Memory clearing would go here
            pass


def create_chatbot(
    model: BaseChatModel,
    *,
    name: str = "Chatbot",
    personality: str | None = None,
    tools: list[BaseTool] | None = None,
    memory: bool = True,
    # Agent configuration
    intercept: Sequence[Any] | None = None,
    stream: bool = False,
    reasoning: bool | Any = False,
    output: type | dict | None = None,
    verbose: bool | str = False,
    resilience: ResilienceConfig | None = None,
) -> Chatbot:
    """
    Create a chatbot with conversation memory.
    
    This is a convenience function that creates a Chatbot instance.
    
    Args:
        model: Native chat model.
        name: Chatbot name.
        personality: Optional system prompt / personality.
        tools: Optional tools for the chatbot.
        memory: Enable conversation memory (default: True).
        intercept: Interceptors for execution hooks.
        stream: Enable streaming responses.
        reasoning: Enable extended thinking mode.
        output: Structured output schema.
        verbose: Observability level.
        resilience: Retry and fallback configuration.
        
    Returns:
        A configured Chatbot instance.
        
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
        name=name,
        personality=personality,
        tools=tools,
        memory=memory,
        intercept=intercept,
        stream=stream,
        reasoning=reasoning,
        output=output,
        verbose=verbose,
        resilience=resilience,
    )
