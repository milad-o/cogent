"""
Context management interceptors.

Interceptors for managing conversation context size, including
summarization and compression when approaching token limits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agenticflow.interceptors.base import (
    Interceptor,
    InterceptContext,
    InterceptResult,
    Phase,
)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars ≈ 1 token)."""
    return len(text) // 4


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    """Convert messages to text for token counting."""
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # Handle content arrays (e.g., with images)
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
        # Include tool calls in estimate
        if "tool_calls" in msg:
            parts.append(json.dumps(msg["tool_calls"]))
    return " ".join(parts)


@dataclass
class ContextCompressor(Interceptor):
    """Compresses conversation context when approaching token limits.
    
    When the message history exceeds `threshold_tokens`, this interceptor
    uses the agent's model to summarize older messages, keeping the
    conversation manageable.
    
    The summarization preserves:
    - System message (always kept)
    - Recent messages (last `keep_recent` messages)
    - Key information from older messages (summarized)
    
    Attributes:
        threshold_tokens: Trigger compression when exceeding this limit.
        keep_recent: Number of recent messages to preserve unchanged.
        summary_prompt: Prompt template for summarization.
        
    Example:
        ```python
        from agenticflow import Agent
        from agenticflow.interceptors import ContextCompressor
        
        agent = Agent(
            name="assistant",
            model=model,
            intercept=[
                ContextCompressor(
                    threshold_tokens=8000,
                    keep_recent=6,
                ),
            ],
        )
        ```
    """
    
    threshold_tokens: int = 8000
    keep_recent: int = 4
    summary_prompt: str = (
        "Summarize this conversation history concisely. "
        "Preserve key facts, decisions, and context. "
        "Keep tool results and important details. "
        "Be brief but complete:\n\n{history}"
    )
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.threshold_tokens < 1000:
            raise ValueError("threshold_tokens must be at least 1000")
        if self.keep_recent < 1:
            raise ValueError("keep_recent must be at least 1")
    
    async def pre_think(self, ctx: InterceptContext) -> InterceptResult:
        """Check and compress context before model call if needed."""
        messages = ctx.messages
        
        # Estimate current token count
        text = _messages_to_text(messages)
        tokens = _estimate_tokens(text)
        
        # Track compression in state
        ctx.state.setdefault("context_compressor", {
            "compressions": 0,
            "original_tokens": tokens,
        })
        
        if tokens <= self.threshold_tokens:
            return InterceptResult.ok()
        
        # Need to compress
        compressed = await self._compress_messages(ctx, messages)
        
        # Update stats
        new_tokens = _estimate_tokens(_messages_to_text(compressed))
        ctx.state["context_compressor"]["compressions"] += 1
        ctx.state["context_compressor"]["compressed_tokens"] = new_tokens
        ctx.state["context_compressor"]["saved_tokens"] = tokens - new_tokens
        
        return InterceptResult.modify_messages(compressed)
    
    async def _compress_messages(
        self,
        ctx: InterceptContext,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compress messages by summarizing older ones."""
        # Separate system message if present
        system_msg = None
        working_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg
            else:
                working_messages.append(msg)
        
        # If too few messages, can't compress
        if len(working_messages) <= self.keep_recent:
            return messages
        
        # Split into old (to summarize) and recent (to keep)
        old_messages = working_messages[:-self.keep_recent]
        recent_messages = working_messages[-self.keep_recent:]
        
        # Format old messages for summarization
        history_text = self._format_for_summary(old_messages)
        
        # Get summary from model
        summary = await self._get_summary(ctx, history_text)
        
        # Build new message list
        result = []
        
        # Keep system message if present
        if system_msg:
            result.append(system_msg)
        
        # Add summary as a system note
        result.append({
            "role": "system",
            "content": f"[Previous conversation summary]\n{summary}",
        })
        
        # Add recent messages
        result.extend(recent_messages)
        
        return result
    
    def _format_for_summary(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for summarization prompt."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if isinstance(content, str):
                parts.append(f"{role.upper()}: {content}")
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                if text_parts:
                    parts.append(f"{role.upper()}: {' '.join(text_parts)}")
            
            # Include tool calls
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    parts.append(f"TOOL CALL: {name}({args})")
            
            # Include tool results
            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                parts.append(f"TOOL RESULT [{tool_id}]: {content}")
        
        return "\n".join(parts)
    
    async def _get_summary(
        self,
        ctx: InterceptContext,
        history: str,
    ) -> str:
        """Get summary from the agent's model."""
        prompt = self.summary_prompt.format(history=history)
        
        # Use the agent's model directly
        try:
            response = await ctx.agent.model.ainvoke([
                {"role": "user", "content": prompt}
            ])
            
            # Extract content from response
            if hasattr(response, "content"):
                return response.content
            elif isinstance(response, dict):
                return response.get("content", str(response))
            else:
                return str(response)
                
        except Exception as e:
            # Fallback: just truncate if model fails
            return f"[Summary unavailable: {e}]\n{history[:500]}..."
    
    def stats(self, ctx: InterceptContext) -> dict[str, Any]:
        """Get compression statistics from context state."""
        return ctx.state.get("context_compressor", {
            "compressions": 0,
            "original_tokens": 0,
        })


@dataclass
class TokenLimiter(Interceptor):
    """Simple token limiter that stops execution when limit is reached.
    
    Unlike ContextCompressor which summarizes, this interceptor simply
    stops execution with a message when the token limit is exceeded.
    Useful for hard limits where summarization isn't appropriate.
    
    Attributes:
        max_tokens: Maximum allowed tokens in context.
        message: Message to return when limit exceeded.
        
    Example:
        ```python
        agent = Agent(
            name="assistant",
            model=model,
            intercept=[
                TokenLimiter(max_tokens=16000),
            ],
        )
        ```
    """
    
    max_tokens: int = 16000
    message: str = "Context limit reached. Please start a new conversation."
    
    async def pre_think(self, ctx: InterceptContext) -> InterceptResult:
        """Check token count before model call."""
        text = _messages_to_text(ctx.messages)
        tokens = _estimate_tokens(text)
        
        if tokens > self.max_tokens:
            return InterceptResult.stop(self.message)
        
        return InterceptResult.ok()


__all__ = [
    "ContextCompressor",
    "TokenLimiter",
]
