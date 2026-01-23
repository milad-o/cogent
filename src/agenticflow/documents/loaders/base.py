"""Base class for document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from agenticflow.core import Document, DocumentMetadata


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
    supported_extensions: ClassVar[list[str]] = []

    def __init__(self, encoding: str = "utf-8") -> None:
        """Initialize the loader.

        Args:
            encoding: Default text encoding.
        """
        self.encoding = encoding

    @abstractmethod
    async def load(self, path: str | Path, **kwargs: object) -> list[Document]:
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

    def load_sync(self, path: str | Path, **kwargs: object) -> list[Document]:
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
        page: int | None = None,
        **extra_metadata: object,
    ) -> Document:
        """Helper to create a document with standard metadata.

        Args:
            content: Document content.
            path: Source file path.
            page: Page number for multi-page documents.
            **extra_metadata: Additional metadata fields.

        Returns:
            Document with populated metadata.
        """
        # Determine source_type from file extension
        ext = path.suffix.lower().lstrip(".")
        source_type_map = {
            "txt": "text",
            "md": "markdown",
            "rst": "rst",
            "pdf": "pdf",
            "docx": "docx",
            "html": "html",
            "htm": "html",
            "csv": "csv",
            "json": "json",
            "jsonl": "jsonl",
            "xlsx": "xlsx",
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
        }
        source_type = source_type_map.get(ext, ext or "unknown")

        custom = dict(extra_metadata) if extra_metadata else {}
        custom.setdefault("file_type", path.suffix.lower())

        metadata = DocumentMetadata(
            source=str(path),
            source_type=source_type,
            page=page,
            loader=self.__class__.__name__,
            custom=custom,
        )
        return Document(text=content, metadata=metadata)


__all__ = ["BaseLoader"]
