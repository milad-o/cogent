"""
Model registry for AgenticFlow.

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
    "o1": "o1",
    "o1-mini": "o1-mini",
    "o3": "o3-mini",
    "o3-mini": "o3-mini",
    
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
    
    # Groq (fast inference)
    "llama": "llama-3.3-70b-versatile",
    "llama-70b": "llama-3.3-70b-versatile",
    "llama-8b": "llama-3.1-8b-instant",
    "mixtral": "mixtral-8x7b-32768",
    "qwen": "qwen-2.5-72b-versatile",
    
    # Ollama (local)
    "ollama": "llama3.2",
    "ollama-llama": "llama3.2",
    
    # Cohere
    "command": "command-r-plus",
    "command-r": "command-r-plus",
    "command-light": "command-r",
}

# Provider detection patterns - model prefix to provider mapping
MODEL_PROVIDERS: dict[str, str] = {
    # OpenAI patterns
    "gpt-": "openai",
    "o1-": "openai",
    "o1": "openai",
    "o3-": "openai",
    "o3": "openai",
    "text-embedding": "openai",
    
    # Anthropic
    "claude-": "anthropic",
    
    # Google Gemini
    "gemini-": "gemini",
    "models/gemini": "gemini",
    
    # Groq
    "llama-": "groq",
    "mixtral-": "groq",
    "qwen-": "groq",
    "deepseek-": "groq",
    "gemma-": "groq",

    # Mistral
    "mistral-": "mistral",
    "codestral-": "mistral",
    "ministral-": "mistral",
    "open-mistral-": "mistral",
    "pixtral-": "mistral",
    "magistral-": "mistral",
    "devstral-": "mistral",
    "labs-mistral-": "mistral",
    "labs-devstral-": "mistral",
    "voxtral-": "mistral",
    
    # Cloudflare Workers AI
    "@cf/": "cloudflare",
    
    # Cohere
    "command-": "cohere",
    "embed-": "cohere",
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
    
    # Default to OpenAI for unknown models (common default)
    return "openai", model_str


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
    from agenticflow.models import create_chat
    
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
