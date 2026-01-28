"""
Model registry for Cogent.

Provides model alias resolution and automatic provider detection.
Enables high-level API: Agent("Helper", model="gpt4")
"""

from __future__ import annotations

from typing import Any

# Model aliases - short names to full model identifiers
MODEL_ALIASES: dict[str, str] = {
    # OpenAI
    "gpt4": "gpt-4o",
    "gpt4o": "gpt-4o",
    "gpt4-mini": "gpt-4o-mini",
    "gpt4-turbo": "gpt-4-turbo",
    "gpt35": "gpt-3.5-turbo",
    "gpt-35": "gpt-3.5-turbo",
    "gpt5": "gpt-5.2",
    "gpt5-mini": "gpt-5-mini",
    "gpt5-nano": "gpt-5-nano",
    "o1": "o1",
    "o1-mini": "o1-mini",
    "o3": "o3-mini",
    "o3-mini": "o3-mini",
    "o4": "o4-mini",
    "o4-mini": "o4-mini",
    # Anthropic
    "claude": "claude-sonnet-4-20250514",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-opus": "claude-opus-4-20250514",
    "claude-haiku": "claude-haiku-4-20250323",
    # Google Gemini
    "gemini": "gemini-2.5-flash",
    "gemini-flash": "gemini-2.5-flash",
    "gemini-pro": "gemini-2.5-pro",
    "gemini-exp": "gemini-2.0-flash-thinking-exp-1219",
    "gemini3": "gemini-3-pro",
    "gemini-3": "gemini-3-pro",
    "gemini-3-flash": "gemini-3-flash",
    # Groq (fast inference)
    "llama": "llama-3.3-70b-versatile",
    "llama-70b": "llama-3.3-70b-versatile",
    "llama-8b": "llama-3.1-8b-instant",
    "mixtral": "mixtral-8x7b-32768",
    "qwen": "qwen-2.5-72b-versatile",
    # Ollama (local)
    "ollama": "llama3.2",
    "ollama-llama": "llama3.2",
    # Mistral
    "mistral": "mistral-large-latest",
    "mistral-large": "mistral-large-latest",
    "mistral-large-3": "mistral-large-3",
    "mistral-medium": "mistral-medium-3.1",
    "mistral-small": "mistral-small-3.2",
    "ministral": "ministral-3-14b",
    "ministral-3": "ministral-3-14b",
    "magistral": "magistral-medium-1.2",
    "codestral": "codestral",
    "devstral": "devstral-2",
    # Cohere
    "command": "command-a-03-2025",
    "command-a": "command-a-03-2025",
    "command-r": "command-r-plus",
    "command-r7b": "command-r7b-12-2024",
    "command-light": "command-r",
    # xAI (Grok)
    "grok": "grok-4-1-fast",  # Fast agentic model (2M context, best for tools)
    "grok-4": "grok-4-0709",  # Flagship reasoning model
    "grok-fast": "grok-4-1-fast",  # Fast agentic model
    "grok-fast-reasoning": "grok-4-1-fast-reasoning",  # With reasoning
    "grok-fast-non-reasoning": "grok-4-1-fast-non-reasoning",  # Without reasoning
    "grok-vision": "grok-2-vision-1212",  # Image understanding
    "grok-code": "grok-code-fast-1",  # Code-optimized
    # Legacy models
    "grok-3": "grok-3",
    "grok-3-mini": "grok-3-mini",
}

# Provider detection patterns - model prefix to provider mapping
MODEL_PROVIDERS: dict[str, str] = {
    # OpenAI patterns
    "gpt-": "openai",
    "o1-": "openai",
    "o1": "openai",
    "o3-": "openai",
    "o3": "openai",
    "o4-": "openai",
    "o4": "openai",
    "text-embedding-": "openai",
    "gpt-oss-": "openai",
    "gpt-audio": "openai",
    "gpt-realtime": "openai",
    "gpt-image-": "openai",
    "chatgpt-": "openai",
    "sora-": "openai",
    "dall-e-": "openai",
    "whisper-": "openai",
    "tts-": "openai",
    "davinci-": "openai",
    "babbage-": "openai",
    # Anthropic
    "claude-": "anthropic",
    # Google Gemini
    "gemini-": "gemini",
    "gemini-flash": "gemini",
    "gemini-pro": "gemini",
    "gemini-exp": "gemini",
    "models/gemini": "gemini",
    "nano-banana": "gemini",
    "text-embedding-": "gemini",  # Gemini also has text-embedding models
    # Groq
    "llama-": "groq",
    "llama3": "groq",
    "mixtral-": "groq",
    "qwen-": "groq",
    "qwen2": "groq",
    "deepseek-": "groq",
    "gemma-": "groq",
    "gemma2-": "groq",
    # Mistral
    "mistral-": "mistral",
    "codestral-": "mistral",
    "codestral": "mistral",
    "ministral-": "mistral",
    "open-mistral-": "mistral",
    "pixtral-": "mistral",
    "pixtral": "mistral",
    "magistral-": "mistral",
    "devstral-": "mistral",
    "voxtral-": "mistral",
    "ocr-": "mistral",  # Mistral OCR models
    # Cloudflare Workers AI
    "@cf/": "cloudflare",
    # Cohere
    "command-": "cohere",
    "command": "cohere",
    "c4ai-aya-": "cohere",
    "embed-english-": "cohere",
    "embed-multilingual-": "cohere",
    "embed-v": "cohere",  # embed-v4.0
    "rerank-": "cohere",
    # xAI (Grok)
    "grok-": "xai",
    "grok": "xai",
}


def resolve_model(model_str: str) -> tuple[str, str]:
    """Resolve model string to (provider, model_name).

    Supports:
    - Aliases: "gpt4" → ("openai", "gpt-4o")
    - Provider prefix: "anthropic:claude-sonnet-4" → ("anthropic", "claude-sonnet-4")
    - Full names: "gpt-4o" → ("openai", "gpt-4o")
    - Pattern matching: Auto-detects provider from model name

    Args:
        model_str: Model string (alias, provider:model, or full name)

    Returns:
        Tuple of (provider, model_name)

    Raises:
        ValueError: If model string cannot be resolved

    Examples:
        >>> resolve_model("gpt4")
        ('openai', 'gpt-4o')

        >>> resolve_model("anthropic:claude-sonnet-4")
        ('anthropic', 'claude-sonnet-4')

        >>> resolve_model("gemini-2.0-flash-exp")
        ('gemini', 'gemini-2.0-flash-exp')
    """
    if not model_str or not isinstance(model_str, str):
        raise ValueError(f"Invalid model string: {model_str!r}")

    model_str = model_str.strip()
    model_lower = model_str.lower()

    if model_lower in {"ollama", "ollama-llama"}:
        return "ollama", MODEL_ALIASES[model_lower]

    # Handle provider:model syntax
    if ":" in model_str:
        parts = model_str.split(":", 1)
        if len(parts) == 2:
            provider, model_name = parts
            provider = provider.strip().lower()
            model_name = model_name.strip()

            # Resolve model alias if present
            if model_name.lower() in MODEL_ALIASES:
                model_name = MODEL_ALIASES[model_name.lower()]

            return provider, model_name

    # Check if it's an alias
    model_lower = model_str.lower()
    if model_lower in MODEL_ALIASES:
        resolved_model = MODEL_ALIASES[model_lower]
        # Now detect provider for the resolved model
        model_str = resolved_model

    # Auto-detect provider from model name patterns
    for pattern, provider in MODEL_PROVIDERS.items():
        if model_str.startswith(pattern):
            return provider, model_str

    # Special case: if it's just "o1" or "o3" (exact match)
    if model_str in ["o1", "o1-mini", "o3", "o3-mini"]:
        return "openai", model_str

    # Cannot auto-detect provider - raise error
    raise ValueError(
        f"Cannot auto-detect provider for model: {model_str!r}. "
        f"Use explicit provider syntax: 'provider:model' (e.g., 'openai:{model_str}') "
        f"or create_chat('provider', '{model_str}')"
    )


def resolve_and_create_model(
    model_str: str,
    **kwargs: Any,
) -> Any:  # Returns BaseChatModel
    """Resolve model string and create model instance.

    Convenience function that combines resolve_model() and create_chat().

    Args:
        model_str: Model string to resolve
        **kwargs: Additional arguments passed to create_chat()

    Returns:
        Chat model instance

    Examples:
        >>> llm = resolve_and_create_model("gpt4")
        >>> llm = resolve_and_create_model("anthropic:claude-sonnet-4")
        >>> llm = resolve_and_create_model("gemini", temperature=0.7)
    """
    from cogent.models import create_chat

    provider, model_name = resolve_model(model_str)
    return create_chat(provider, model_name, **kwargs)


def list_model_aliases() -> dict[str, str]:
    """Get all available model aliases.

    Returns:
        Dictionary of alias -> full model name
    """
    return MODEL_ALIASES.copy()


def get_provider_for_model(model_name: str) -> str | None:
    """Get provider name for a model.

    Args:
        model_name: Full model name

    Returns:
        Provider name or None if not found
    """
    try:
        provider, _ = resolve_model(model_name)
        return provider
    except ValueError:
        return None
