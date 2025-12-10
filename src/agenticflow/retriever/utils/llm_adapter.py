"""LLM adapter utilities for retrievers.

Provides automatic adaptation of chat models to the simple .generate() interface
expected by LLM-powered retrievers (SummaryIndex, KeywordTableIndex, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agenticflow.models import BaseChatModel


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM interfaces expected by retrievers."""
    
    async def generate(self, prompt: str) -> str:
        """Generate a response from a prompt string."""
        ...


class ChatModelAdapter:
    """Adapter to make chat models compatible with retriever LLM interface.
    
    AgenticFlow chat models use `.ainvoke(messages)` with role-based messages,
    while LLM-powered retrievers expect a simple `.generate(prompt)` interface
    for model-agnostic operation.
    
    This adapter bridges the two interfaces automatically.
    
    Example:
        ```python
        from agenticflow.models import ChatModel
        from agenticflow.retriever import SummaryIndex
        
        # No adapter needed - SummaryIndex auto-wraps chat models
        llm = ChatModel(model="gpt-4o-mini")
        index = SummaryIndex(llm=llm, vectorstore=vs)
        
        # The adapter is created internally, transparent to the user
        ```
    """
    
    def __init__(self, model: BaseChatModel) -> None:
        """Create adapter for a chat model.
        
        Args:
            model: Chat model instance to adapt.
        """
        self._model = model
    
    async def generate(self, prompt: str) -> str:
        """Generate a response from a prompt string.
        
        Args:
            prompt: Text prompt to generate from.
            
        Returns:
            Generated text response.
        """
        messages = [{"role": "user", "content": prompt}]
        response = await self._model.ainvoke(messages)
        return response.content


def adapt_llm(llm: Any) -> LLMProtocol:
    """Automatically adapt an LLM to the retriever interface.
    
    If the LLM already has a `.generate()` method, returns it as-is.
    If it's a chat model with `.ainvoke()`, wraps it in ChatModelAdapter.
    
    Args:
        llm: LLM instance (chat model or any object with .generate()).
        
    Returns:
        LLM with .generate() interface.
        
    Raises:
        TypeError: If the LLM doesn't support either interface.
    
    Example:
        ```python
        from agenticflow.models import ChatModel
        from agenticflow.retriever.utils import adapt_llm
        
        chat_model = ChatModel()
        llm = adapt_llm(chat_model)  # Returns ChatModelAdapter
        
        response = await llm.generate("Summarize this document...")
        ```
    """
    # Already has the right interface
    if isinstance(llm, LLMProtocol):
        return llm
    
    # Chat model - wrap it
    if hasattr(llm, "ainvoke"):
        return ChatModelAdapter(llm)
    
    # Unknown interface
    raise TypeError(
        f"LLM must have either .generate(prompt) or .ainvoke(messages) method. "
        f"Got: {type(llm).__name__}"
    )
