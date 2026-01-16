"""Embedding providers for vector store.

This module re-exports embeddings from agenticflow.models for backward compatibility.
Prefer importing directly from agenticflow.models for new code:

    from agenticflow.models import create_embedding, MockEmbedding
    from agenticflow.models.openai import OpenAIEmbedding
    from agenticflow.models.ollama import OllamaEmbedding

Example:
    >>> from agenticflow.models import MockEmbedding
    >>> embeddings = MockEmbedding()
    >>> vectors = embeddings.embed(["Hello", "World"])
"""

from __future__ import annotations

# Re-export from models for backward compatibility
from agenticflow.models.mock import MockEmbedding
from agenticflow.models.ollama import OllamaEmbedding as OllamaEmbeddings
from agenticflow.models.openai import OpenAIEmbedding as OpenAIEmbeddings

# Backward compat alias
MockEmbeddings = MockEmbedding


# ============================================================
# Embedding Dimensions for Common Models (utility)
# ============================================================

EMBEDDING_DIMENSIONS: dict[str, int] = {
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Ollama
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}


def get_embedding_dimension(model: str) -> int:
    """Get the embedding dimension for a model.

    Args:
        model: Model name.

    Returns:
        Embedding dimension, or 1536 as default.
    """
    return EMBEDDING_DIMENSIONS.get(model, 1536)


__all__ = [
    # Re-exports from models (backward compat)
    "OpenAIEmbeddings",
    "OllamaEmbeddings",
    # Testing (re-exported from models)
    "MockEmbedding",
    "MockEmbeddings",  # backward compat alias
    # Utilities
    "EMBEDDING_DIMENSIONS",
    "get_embedding_dimension",
]
