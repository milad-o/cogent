"""
Simple model helpers for AgenticFlow examples.

Configuration priority:
1. Function arguments (highest)
2. models.yaml file in examples directory
3. Environment variables (.env file)
4. Defaults (lowest)

Provides backward-compatible get_model() for legacy examples.
New examples should use the 3-tier API directly:
    
    Agent(name="...", model="gemini")  # Recommended!

Usage (legacy):
    from models import get_model
    
    model = get_model()  # Uses config from models.yaml or .env
    model = get_model("claude")  # Override
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache


@lru_cache
def _load_config() -> dict:
    """Load configuration from models.yaml if it exists."""
    config_file = Path(__file__).parent / "models.yaml"
    
    if not config_file.exists():
        return {}
    
    try:
        import yaml
        with open(config_file) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # PyYAML not installed, fall back to env vars
        return {}
    except Exception:
        # YAML parsing error, fall back to env vars
        return {}


def _get_config_value(key: str, default: str | None = None) -> str | None:
    """Get config value from models.yaml, then env vars, then default."""
    # Try models.yaml first
    config = _load_config()
    if key in config:
        return config[key]
    
    # Fall back to environment variable
    return os.getenv(key, default)


def get_model(provider: str | None = None):
    """Get a chat model for the specified provider.
    
    Thin wrapper around the 3-tier API for backward compatibility.
    
    Args:
        provider: Model provider name (gemini, openai, anthropic, etc.)
                 If None, reads from models.yaml or LLM_PROVIDER env var
    
    Returns:
        Configured chat model instance.
        
    Example:
        model = get_model()  # Uses config from models.yaml or .env
        model = get_model("claude")  # Override to Anthropic
    """
    from agenticflow.models import create_chat
    
    # Get provider from config priority chain
    if provider is None:
        provider = _get_config_value("llm_provider") or _get_config_value("LLM_PROVIDER", "gemini")
    
    # Map common aliases
    provider_map = {
        "gpt": "openai",
        "gpt4": "openai",
        "claude": "anthropic",
    }
    provider = provider_map.get(provider, provider)
    
    # Use the 3-tier API - it handles API key loading automatically
    return create_chat(provider)


def get_embeddings(provider: str | None = None):
    """Get embeddings model for the specified provider.
    
    Args:
        provider: Embeddings provider (openai, gemini, cohere, etc.)
                 If None, reads from models.yaml or EMBEDDING_PROVIDER env var
    
    Returns:
        Configured embeddings model instance.
    """
    from agenticflow.models import create_embeddings
    
    if provider is None:
        provider = _get_config_value("embedding_provider") or _get_config_value("EMBEDDING_PROVIDER", "openai")
    
    return create_embeddings(provider)
