"""HTML file loader."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from agenticflow.document.loaders.base import BaseLoader
from agenticflow.document.types import Document


class HTMLLoader(BaseLoader):
    """Loader for HTML files.

    Extracts text content from HTML, removing scripts and styles.
    Uses BeautifulSoup if available, falls back to regex.

    Example:
        >>> loader = HTMLLoader()
        >>> docs = await loader.load(Path("page.html"))
        >>> print(docs[0].metadata.get("title"))
    """

    supported_extensions = [".html", ".htm"]

    def __init__(
        self,
        encoding: str = "utf-8",
        remove_elements: list[str] | None = None,
    ) -> None:
        """Initialize the loader.

        Args:
            encoding: Text encoding.
            remove_elements: HTML elements to remove (default: script, style, nav, footer, header).
        """
        super().__init__(encoding)
        self.remove_elements = remove_elements or [
            "script", "style", "nav", "footer", "header"
        ]

    async def load(self, path: str | Path, **kwargs: Any) -> list[Document]:
        """Load an HTML file.

        Args:
            path: Path to the HTML file (str or Path).
            **kwargs: Optional 'encoding' to override default.

        Returns:
            List containing a single Document.
        """
        path = Path(path)
        encoding = kwargs.get("encoding", self.encoding)
        raw = path.read_text(encoding=encoding)

        # Try BeautifulSoup first
        try:
            text, title = self._extract_with_bs4(raw)
        except ImportError:
            text, title = self._extract_with_regex(raw)

        extra_meta = {}
        if title:
            extra_meta["title"] = title

        return [self._create_document(text, path, **extra_meta)]

    def _extract_with_bs4(self, raw: str) -> tuple[str, str | None]:
        """Extract text using BeautifulSoup.

        Args:
            raw: Raw HTML content.

        Returns:
            Tuple of (text content, title or None).
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")

        # Remove unwanted elements
        for element in soup(self.remove_elements):
            element.decompose()

        # Extract title
        title = soup.title.string if soup.title else None

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        return text, title

    def _extract_with_regex(self, raw: str) -> tuple[str, str | None]:
        """Extract text using regex (fallback).

        Args:
            raw: Raw HTML content.

        Returns:
            Tuple of (text content, title or None).
        """
        # Extract title
        title = None
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>",
            raw,
            re.IGNORECASE | re.DOTALL,
        )
        if title_match:
            title = html.unescape(title_match.group(1).strip())

        text = raw

        # Remove scripts and styles
        for element in self.remove_elements:
            pattern = rf"<{element}[^>]*>.*?</{element}>"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        # Remove all tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Decode entities
        text = html.unescape(text)

        # Clean whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text, title


__all__ = ["HTMLLoader"]
