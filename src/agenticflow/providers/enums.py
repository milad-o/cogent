"""Provider enums and constants."""

from enum import Enum


class Provider(str, Enum):
    """Supported LLM/embedding providers.

    Use these values when specifying providers programmatically.

    Example:
        >>> from agenticflow.providers import Provider, create_model
        >>> model = create_model(f"{Provider.OPENAI}/gpt-4o")
    """

    OPENAI = "openai"
    AZURE = "azure"
    AZURE_OPENAI = "azure_openai"  # Alias for azure
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GEMINI = "gemini"  # Alias for google
    OLLAMA = "ollama"
    BEDROCK = "bedrock"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GROQ = "groq"
    FIREWORKS = "fireworks"
    TOGETHER = "together"
    HUGGINGFACE = "huggingface"
    VOYAGE = "voyage"

    def __str__(self) -> str:
        """Return the string value for use in spec strings."""
        return self.value


# Default models per provider
DEFAULT_CHAT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "azure": "gpt-4o-mini",
    "azure_openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "google": "gemini-1.5-flash",
    "gemini": "gemini-1.5-flash",
    "ollama": "llama3.2",
    "mistral": "mistral-large-latest",
    "groq": "llama-3.3-70b-versatile",
    "cohere": "command-r-plus",
    "bedrock": "anthropic.claude-3-sonnet-20240229-v1:0",
    "fireworks": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    "together": "meta-llama/Llama-3-70b-chat-hf",
}

DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "azure": "text-embedding-3-small",
    "azure_openai": "text-embedding-3-small",
    "cohere": "embed-english-v3.0",
    "ollama": "nomic-embed-text",
    "google": "models/embedding-001",
    "gemini": "models/embedding-001",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "bedrock": "amazon.titan-embed-text-v1",
    "mistral": "mistral-embed",
    "voyage": "voyage-2",
}


__all__ = [
    "Provider",
    "DEFAULT_CHAT_MODELS",
    "DEFAULT_EMBEDDING_MODELS",
]
