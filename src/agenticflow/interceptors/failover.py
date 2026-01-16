"""
Failover - Model fallback interceptor.

Automatically switch to backup models when the primary model fails
due to rate limits, timeouts, or other errors.

Example:
    from agenticflow import Agent
    from agenticflow.interceptors import Failover
    from agenticflow.models import ChatModel

    agent = Agent(
        name="assistant",
        model=ChatModel(model="gpt-4o"),
        intercept=[
            Failover(
                fallbacks=["gpt-4o-mini", "claude-sonnet-4-20250514"],
                on=["rate_limit", "timeout", "error"],
            ),
        ],
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agenticflow.interceptors.base import (
    InterceptContext,
    Interceptor,
    InterceptResult,
)

if TYPE_CHECKING:
    pass


class FailoverTrigger(Enum):
    """Triggers that activate model fallback."""
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    ERROR = "error"
    CONTEXT_LENGTH = "context_length"


@dataclass
class FailoverState:
    """Tracks failover state across the execution."""
    current_model_index: int = 0
    errors: list[tuple[str, Exception]] = field(default_factory=list)
    triggered: bool = False


class Failover(Interceptor):
    """Automatic model fallback on errors.

    Switches to backup models when the primary model fails.
    Tracks state across the execution and rotates through fallbacks.

    Args:
        fallbacks: List of fallback models (names or instances).
        on: List of triggers - "rate_limit", "timeout", "error", "context_length".
        max_retries_per_model: Max retries before switching models.
        on_fallback: Callback when fallback is triggered.

    Example:
        from agenticflow.models.openai import OpenAIChat
        from agenticflow.models.anthropic import AnthropicChat

        # With model names (will be created automatically)
        failover = Failover(
            fallbacks=["gpt-4o-mini", "claude-sonnet-4-20250514"],
        )

        # With model instances
        failover = Failover(
            fallbacks=[
                OpenAIChat(model="gpt-4o-mini"),
                AnthropicChat(model="claude-sonnet-4-20250514"),
            ],
        )
    """

    STATE_KEY = "_failover_state"

    def __init__(
        self,
        fallbacks: list[str | Any],
        on: list[str] | None = None,
        max_retries_per_model: int = 2,
        on_fallback: Callable[[str, str, Exception], None] | None = None,
    ) -> None:
        """Initialize Failover.

        Args:
            fallbacks: List of fallback model names or instances.
            on: Triggers - defaults to ["rate_limit", "timeout", "error"].
            max_retries_per_model: Retries before switching (default: 2).
            on_fallback: Callback(from_model, to_model, error) when switching.
        """
        self.fallbacks = fallbacks
        self.triggers = {FailoverTrigger(t) for t in (on or ["rate_limit", "timeout", "error"])}
        self.max_retries_per_model = max_retries_per_model
        self.on_fallback = on_fallback
        self._resolved_models: list[Any] | None = None

    def _get_state(self, ctx: InterceptContext) -> FailoverState:
        """Get or create failover state."""
        if self.STATE_KEY not in ctx.state:
            ctx.state[self.STATE_KEY] = FailoverState()
        return ctx.state[self.STATE_KEY]

    def _resolve_models(self) -> list[Any]:
        """Resolve model names to model instances."""
        if self._resolved_models is not None:
            return self._resolved_models

        resolved = []
        for fb in self.fallbacks:
            if isinstance(fb, str):
                # Create model from name
                from agenticflow.models import create_chat
                try:
                    resolved.append(create_chat(fb))
                except Exception:
                    # If creation fails, try OpenAI directly
                    from agenticflow.models.openai import OpenAIChat
                    resolved.append(OpenAIChat(model=fb))
            else:
                resolved.append(fb)

        self._resolved_models = resolved
        return resolved

    def _should_trigger(self, error: Exception) -> bool:
        """Check if error should trigger fallback."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        for trigger in self.triggers:
            if trigger == FailoverTrigger.RATE_LIMIT:
                if "rate" in error_str or "429" in error_str or "ratelimit" in error_type:
                    return True
            elif trigger == FailoverTrigger.TIMEOUT:
                if "timeout" in error_str or "timeout" in error_type:
                    return True
            elif trigger == FailoverTrigger.CONTEXT_LENGTH:
                if "context" in error_str or "token" in error_str or "length" in error_str:
                    return True
            elif trigger == FailoverTrigger.ERROR:
                # Generic error trigger
                return True

        return False

    async def on_error(self, ctx: InterceptContext) -> InterceptResult:
        """Handle errors and potentially switch models."""
        if ctx.error is None:
            return InterceptResult.ok()

        state = self._get_state(ctx)

        # Check if we should trigger failover
        if not self._should_trigger(ctx.error):
            return InterceptResult.ok()

        # Track the error
        current_model_name = getattr(ctx.agent.model, "model", "unknown")
        state.errors.append((current_model_name, ctx.error))

        # Check if we've exceeded retries for this model
        model_errors = sum(1 for m, _ in state.errors if m == current_model_name)
        if model_errors < self.max_retries_per_model:
            # Let normal retry handle it
            return InterceptResult.ok()

        # Switch to next model
        models = self._resolve_models()
        if state.current_model_index >= len(models):
            # All fallbacks exhausted
            return InterceptResult.ok()

        # Get next model
        next_model = models[state.current_model_index]
        state.current_model_index += 1
        state.triggered = True

        # Callback
        if self.on_fallback:
            next_model_name = getattr(next_model, "model", "unknown")
            self.on_fallback(current_model_name, next_model_name, ctx.error)

        return InterceptResult.use_model(next_model)

    async def pre_think(self, ctx: InterceptContext) -> InterceptResult:
        """Check if we need to use a fallback model."""
        state = self._get_state(ctx)

        # If fallback was triggered, ensure we're using the right model
        if state.triggered and state.current_model_index > 0:
            models = self._resolve_models()
            idx = min(state.current_model_index - 1, len(models) - 1)
            return InterceptResult.use_model(models[idx])

        return InterceptResult.ok()


__all__ = [
    "Failover",
    "FailoverTrigger",
    "FailoverState",
]
