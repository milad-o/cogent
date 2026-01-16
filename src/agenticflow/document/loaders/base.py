"""Base class for document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agenticflow.document.types import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders.

    All file-type-specific loaders should inherit from this class
    and implement the `load` method.

    Attributes:
        encoding: Default text encoding to use.

    Example:
        >>> class CustomLoader(BaseLoader):
        ...     async def load(self, path: str | Path, **kwargs) -> list[Document]:
        ...         path = Path(path)
        ...         content = path.read_text()
        ...         return [Document(content=content, metadata={"source": str(path)})]
    """

    # File extensions this loader supports
    supported_extensions: list[str] = []

    def __init__(self, encoding: str = "utf-8") -> None:
        """Initialize the loader.

        Args:
            encoding: Default text encoding.
        """
        self.encoding = encoding

    @abstractmethod
    async def load(self, path: str | Path, **kwargs: Any) -> list[Document]:
        """Load documents from a file.

        Args:
            path: Path to the file to load (str or Path).
            **kwargs: Additional loader-specific options.

        Returns:
            List of Document objects (may be multiple for multi-page files).

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is invalid.
            ImportError: If required dependencies are missing.
        """
        ...

    def load_sync(self, path: str | Path, **kwargs: Any) -> list[Document]:
        """Synchronous version of load.

        Default implementation uses asyncio.run().
        Subclasses can override for true sync implementation.

        Args:
            path: Path to the file to load (str or Path).
            **kwargs: Additional loader-specific options.

        Returns:
            List of Document objects.
        """
        import asyncio
        return asyncio.run(self.load(path, **kwargs))

    def can_load(self, path: str | Path) -> bool:
        """Check if this loader can handle the given file.

        Args:
            path: Path to check.

        Returns:
            True if this loader supports the file type.
        """
        path = Path(path)
        ext = path.suffix.lower()
        return ext in self.supported_extensions

    def _create_document(
        self,
        content: str,
        path: Path,
        **extra_metadata: Any,
    ) -> Document:
        """Helper to create a document with standard metadata.

        Args:
            content: Document content.
            path: Source file path.
            **extra_metadata: Additional metadata fields.

        Returns:
            Document with populated metadata.
        """
        metadata = {
            "source": str(path),
            "filename": path.name,
            "file_type": path.suffix.lower(),
            **extra_metadata,
        }
        return Document(text=content, metadata=metadata)


__all__ = ["BaseLoader"]
