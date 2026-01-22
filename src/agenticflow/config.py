"""
Configuration loader for AgenticFlow.

Loads API keys and settings from:
1. Project-level: ./agenticflow.toml or ./agenticflow.yaml
2. User-level: ~/.agenticflow/config.toml or ~/.agenticflow/config.yaml
3. Environment variables (highest priority when used)
4. .env files (automatically loaded from current directory)

Priority: Explicit params > Environment vars > Config file
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Auto-load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load from .env in current directory
except ImportError:
    pass  # dotenv not installed, rely on system env vars

# Try to import yaml, but make it optional
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def find_config_file() -> Path | None:
    """Find configuration file in standard locations.
    
    Checks in order:
    1. ./agenticflow.toml
    2. ./agenticflow.yaml (if pyyaml installed)
    3. ~/.agenticflow/config.toml
    4. ~/.agenticflow/config.yaml (if pyyaml installed)
    
    Returns:
        Path to config file or None if not found
    """
    # Project-level configs
    project_toml = Path.cwd() / "agenticflow.toml"
    if project_toml.exists():
        return project_toml
    
    if HAS_YAML:
        project_yaml = Path.cwd() / "agenticflow.yaml"
        if project_yaml.exists():
            return project_yaml
    
    # User-level configs
    user_config_dir = Path.home() / ".agenticflow"
    
    user_toml = user_config_dir / "config.toml"
    if user_toml.exists():
        return user_toml
    
    if HAS_YAML:
        user_yaml = user_config_dir / "config.yaml"
        if user_yaml.exists():
            return user_yaml
    
    return None


def load_toml(path: Path) -> dict[str, Any]:
    """Load TOML configuration file.
    
    Uses built-in tomllib (Python 3.11+).
    
    Args:
        path: Path to TOML file
        
    Returns:
        Configuration dictionary
    """
    try:
        import tomllib
    except ImportError:
        # Python < 3.11
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            raise ImportError(
                "TOML support requires Python 3.11+ or 'tomli' package. "
                "Install with: uv add tomli"
            )
    
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML configuration file.
    
    Requires pyyaml package.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Configuration dictionary
    """
    if not HAS_YAML:
        raise ImportError(
            "YAML support requires 'pyyaml' package. "
            "Install with: uv add pyyaml"
        )
    
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config() -> dict[str, Any]:
    """Load configuration from file.
    
    Returns:
        Configuration dictionary or empty dict if no config found
    """
    config_path = find_config_file()
    if not config_path:
        return {}
    
    try:
        if config_path.suffix == ".toml":
            return load_toml(config_path)
        elif config_path.suffix in (".yaml", ".yml"):
            return load_yaml(config_path)
        else:
            return {}
    except Exception as e:
        # Don't fail if config can't be loaded, just warn
        import warnings
        warnings.warn(f"Failed to load config from {config_path}: {e}")
        return {}


def get_api_key(provider: str, explicit_key: str | None = None) -> str | None:
    """Get API key for provider with priority handling.
    
    Priority:
    1. Explicit key parameter (highest)
    2. Environment variable
    3. Config file (lowest)
    
    Args:
        provider: Provider name (openai, anthropic, gemini, etc.)
        explicit_key: Explicitly provided API key
        
    Returns:
        API key or None
    """
    # 1. Explicit parameter (highest priority)
    if explicit_key:
        return explicit_key
    
    # 2. Environment variable
    env_vars = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "groq": ["GROQ_API_KEY"],
        "azure": ["AZURE_OPENAI_API_KEY"],
        "cohere": ["COHERE_API_KEY", "CO_API_KEY"],
        "cloudflare": ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY", "CF_API_KEY"],
        "mistral": ["MISTRAL_API_KEY"],
        "together": ["TOGETHER_API_KEY"],
    }
    
    provider_lower = provider.lower()
    if provider_lower in env_vars:
        for env_var in env_vars[provider_lower]:
            key = os.environ.get(env_var)
            if key:
                return key
    
    # 3. Config file (lowest priority)
    config = load_config()
    models_config = config.get("models", {})
    provider_config = models_config.get(provider_lower, {})
    
    return provider_config.get("api_key")


def get_provider_config(provider: str) -> dict[str, Any]:
    """Get full configuration for provider from config file.
    
    Args:
        provider: Provider name
        
    Returns:
        Provider configuration dictionary
    """
    config = load_config()
    models_config = config.get("models", {})
    return models_config.get(provider.lower(), {})


def get_model_override(provider: str, kind: str) -> str | None:
    """Get chat/embedding model override from env or config.

    Priority:
    1. Environment variables (e.g., OPENAI_CHAT_MODEL)
    2. Config file (models.<provider>.chat_model / embedding_model)

    Args:
        provider: Provider name (openai, gemini, etc.)
        kind: "chat" or "embedding"

    Returns:
        Model name override or None
    """
    provider_key = provider.lower()
    if provider_key == "google":
        provider_key = "gemini"

    env_map: dict[str, dict[str, list[str]]] = {
        "chat": {
            "openai": ["OPENAI_CHAT_MODEL"],
            "gemini": ["GEMINI_CHAT_MODEL"],
            "groq": ["GROQ_CHAT_MODEL"],
            "mistral": ["MISTRAL_CHAT_MODEL"],
            "cohere": ["COHERE_CHAT_MODEL"],
            "cloudflare": ["CLOUDFLARE_CHAT_MODEL"],
            "github": ["GITHUB_CHAT_MODEL"],
            "ollama": ["OLLAMA_CHAT_MODEL"],
        },
        "embedding": {
            "openai": ["OPENAI_EMBEDDING_MODEL"],
            "gemini": ["GEMINI_EMBEDDING_MODEL"],
            "mistral": ["MISTRAL_EMBEDDING_MODEL"],
            "cohere": ["COHERE_EMBEDDING_MODEL"],
            "cloudflare": ["CLOUDFLARE_EMBEDDING_MODEL"],
            "github": ["GITHUB_EMBEDDING_MODEL"],
            "ollama": ["OLLAMA_EMBEDDING_MODEL"],
        },
    }

    for env_var in env_map.get(kind, {}).get(provider_key, []):
        value = os.environ.get(env_var)
        if value:
            return value

    config = load_config()
    models_config = config.get("models", {})
    provider_config = models_config.get(provider_key, {})
    key_name = "chat_model" if kind == "chat" else "embedding_model"
    return provider_config.get(key_name)


def get_default_model() -> str | None:
    """Get default model from config file.
    
    Returns:
        Default model string or None
    """
    config = load_config()
    models_config = config.get("models", {})
    return models_config.get("default")
