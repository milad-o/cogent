"""Token-based text splitter."""

from __future__ import annotations

from typing import Any

from agenticflow.document.splitters.base import BaseSplitter
from agenticflow.document.splitters.character import RecursiveCharacterSplitter
from agenticflow.document.types import Document


class TokenSplitter(BaseSplitter):
    """Split text by token count.

    Uses tiktoken for accurate OpenAI token counting.

    Args:
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap in tokens.
        model_name: Model name for tiktoken encoding.

    Example:
        >>> splitter = TokenSplitter(chunk_size=500, model_name="gpt-4")
        >>> chunks = splitter.split_text(text)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        model_name: str = "gpt-4",
        **kwargs: Any,
    ):
        self._tokenizer = None
        self._model_name = model_name

        # Initialize parent with token length function
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_length,
            **kwargs,
        )

    @property
    def tokenizer(self) -> Any:
        """Lazy-load tiktoken encoder."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.encoding_for_model(self._model_name)
            except ImportError:
                raise ImportError(
                    "Token splitting requires 'tiktoken'. "
                    "Install with: uv add tiktoken"
                )
            except KeyError:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def _token_length(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def split_text(self, text: str) -> list[Document]:
        """Split text by token count."""
        # Use recursive character splitting with token length function
        splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
        )
        chunks = splitter.split_text(text)

        # Add token count metadata
        for chunk in chunks:
            chunk.metadata["token_count"] = self._token_length(chunk.text)

        return chunks


__all__ = ["TokenSplitter"]
