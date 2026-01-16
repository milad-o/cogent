"""Plain text file loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenticflow.document.loaders.base import BaseLoader
from agenticflow.document.types import Document


class TextLoader(BaseLoader):
    """Loader for plain text files.

    Supports .txt, .text, .rst, and .log files.
    Automatically handles encoding detection for common encodings.

    Example:
        >>> loader = TextLoader()
        >>> docs = await loader.load(Path("readme.txt"))
    """

    supported_extensions = [".txt", ".text", ".rst", ".log"]

    # Fallback encodings to try if UTF-8 fails
    FALLBACK_ENCODINGS = ["latin-1", "cp1252", "iso-8859-1"]

    async def load(self, path: str | Path, **kwargs: Any) -> list[Document]:
        """Load a plain text file.

        Args:
            path: Path to the text file (str or Path).
            **kwargs: Optional 'encoding' to override default.

        Returns:
            List containing a single Document.
        """
        path = Path(path)
        encoding = kwargs.get("encoding", self.encoding)
        content = self._read_with_fallback(path, encoding)

        return [self._create_document(content, path)]

    def _read_with_fallback(self, path: Path, encoding: str) -> str:
        """Read file with encoding fallbacks.

        Args:
            path: File path.
            encoding: Primary encoding to try.

        Returns:
            File content as string.
        """
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            pass

        # Try fallback encodings
        for enc in self.FALLBACK_ENCODINGS:
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue

        # Last resort: read with error replacement
        return path.read_bytes().decode("utf-8", errors="replace")


__all__ = ["TextLoader"]
