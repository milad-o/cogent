"""
PromptAdapter - Dynamic system prompt modification.

Modify the system prompt based on context, conversation stage, or custom logic.

Example:
    from agenticflow import Agent
    from agenticflow.interceptors import PromptAdapter

    class PersonalizedPrompt(PromptAdapter):
        async def adapt(self, prompt: str, ctx: InterceptContext) -> str:
            user = ctx.run_context.user_name if ctx.run_context else "User"
            return f"{prompt}\\n\\nAddress the user as {user}."

    agent = Agent(
        name="assistant",
        model=model,
        system_prompt="You are a helpful assistant.",
        intercept=[PersonalizedPrompt()],
    )
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

from agenticflow.interceptors.base import (
    InterceptContext,
    Interceptor,
    InterceptResult,
)


class PromptAdapter(Interceptor):
    """Base class for dynamic prompt modification.

    PromptAdapter modifies the system prompt before the first model call (PRE_RUN).
    This allows personalization, context injection, or behavior changes.

    Override the `adapt` method to implement your modification logic.

    Example:
        class PersonalizedPrompt(PromptAdapter):
            async def adapt(self, prompt: str, ctx: InterceptContext) -> str:
                # Access run context
                if ctx.run_context and hasattr(ctx.run_context, "user_name"):
                    return f"{prompt}\\n\\nUser's name: {ctx.run_context.user_name}"
                return prompt
    """

    @abstractmethod
    async def adapt(
        self,
        prompt: str,
        ctx: InterceptContext,
    ) -> str:
        """Adapt the system prompt.

        Args:
            prompt: Current system prompt (may be empty).
            ctx: Intercept context with run_context, state, etc.

        Returns:
            Modified system prompt.
        """
        ...

    async def pre_run(self, ctx: InterceptContext) -> InterceptResult:
        """Adapt prompt before execution starts."""
        # Get current prompt from agent config
        current_prompt = ""
        if hasattr(ctx.agent, "config") and hasattr(ctx.agent.config, "system_prompt"):
            current_prompt = ctx.agent.config.system_prompt or ""

        adapted = await self.adapt(current_prompt, ctx)
        return InterceptResult.modify_prompt(adapted)


class ContextPrompt(PromptAdapter):
    """Inject context data into the system prompt.

    Appends information from RunContext to the system prompt.
    Useful for passing user-specific data to the agent.

    Args:
        template: Format string for context injection.
                  Use {key} placeholders for RunContext attributes.
        separator: Separator between original prompt and context.

    Example:
        # Template with placeholders
        adapter = ContextPrompt(
            template="User ID: {user_id}\\nRole: {role}\\nSession: {session_id}"
        )

        # Use with RunContext
        @dataclass
        class AppContext(RunContext):
            user_id: str
            role: str
            session_id: str

        await agent.run(
            "Help me",
            context=AppContext(user_id="123", role="admin", session_id="abc"),
        )
    """

    def __init__(
        self,
        template: str,
        separator: str = "\n\n",
    ) -> None:
        """Initialize ContextPrompt.

        Args:
            template: Format string with {key} placeholders.
            separator: Separator between prompt and context.
        """
        self.template = template
        self.separator = separator

    async def adapt(
        self,
        prompt: str,
        ctx: InterceptContext,
    ) -> str:
        """Inject context into prompt."""
        if ctx.run_context is None:
            return prompt

        # Get attributes from RunContext
        context_data = {}
        for key in dir(ctx.run_context):
            if not key.startswith("_"):
                value = getattr(ctx.run_context, key, None)
                if not callable(value):
                    context_data[key] = value

        try:
            context_str = self.template.format(**context_data)
        except KeyError:
            # Missing keys - try with available data only
            context_str = self.template
            for key, value in context_data.items():
                context_str = context_str.replace(f"{{{key}}}", str(value))

        if prompt:
            return f"{prompt}{self.separator}{context_str}"
        return context_str


class ConversationPrompt(PromptAdapter):
    """Modify prompt based on conversation stage.

    Changes the system prompt as the conversation progresses,
    allowing different behaviors at different stages.

    Args:
        stages: Dict mapping message count thresholds to prompt additions.

    Example:
        adapter = ConversationPrompt({
            0: "Start with a warm greeting.",
            5: "The user is now engaged. Be more detailed.",
            10: "This is a long conversation. Be concise.",
        })
    """

    def __init__(
        self,
        stages: dict[int, str],
        separator: str = "\n\n",
    ) -> None:
        """Initialize ConversationPrompt.

        Args:
            stages: Dict mapping message count to prompt addition.
            separator: Separator between prompt and stage text.
        """
        self.stages = stages
        self.separator = separator

    async def adapt(
        self,
        prompt: str,
        ctx: InterceptContext,
    ) -> str:
        """Add stage-specific instructions."""
        message_count = len(ctx.messages) if ctx.messages else 0

        # Find the applicable stage
        additions = []
        for threshold in sorted(self.stages.keys()):
            if message_count >= threshold:
                additions.append(self.stages[threshold])

        if not additions:
            return prompt

        stage_text = " ".join(additions)
        if prompt:
            return f"{prompt}{self.separator}{stage_text}"
        return stage_text


class LambdaPrompt(PromptAdapter):
    """Simple lambda-based prompt adapter.

    For quick customization without subclassing.

    Args:
        adapter_fn: Function(prompt, ctx) -> new_prompt

    Example:
        adapter = LambdaPrompt(
            lambda prompt, ctx: f"{prompt}\\n\\nBe concise."
        )
    """

    def __init__(
        self,
        adapter_fn: Callable[[str, InterceptContext], str],
    ) -> None:
        """Initialize LambdaPrompt.

        Args:
            adapter_fn: Sync function to adapt prompt.
        """
        self.adapter_fn = adapter_fn

    async def adapt(
        self,
        prompt: str,
        ctx: InterceptContext,
    ) -> str:
        """Apply the lambda function."""
        return self.adapter_fn(prompt, ctx)


__all__ = [
    "PromptAdapter",
    "ContextPrompt",
    "ConversationPrompt",
    "LambdaPrompt",
]
