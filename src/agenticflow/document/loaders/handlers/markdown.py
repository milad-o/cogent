"""Markdown file loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenticflow.document.loaders.base import BaseLoader
from agenticflow.document.types import Document


class MarkdownLoader(BaseLoader):
    """Loader for Markdown files.
    
    Supports .md and .markdown files.
    Optionally extracts YAML frontmatter into metadata.
    
    Example:
        >>> loader = MarkdownLoader()
        >>> docs = await loader.load(Path("README.md"))
        >>> print(docs[0].metadata.get("title"))  # From frontmatter
    """
    
    supported_extensions = [".md", ".markdown"]
    
    def __init__(
        self,
        encoding: str = "utf-8",
        extract_frontmatter: bool = True,
    ) -> None:
        """Initialize the loader.
        
        Args:
            encoding: Text encoding.
            extract_frontmatter: Whether to parse YAML frontmatter.
        """
        super().__init__(encoding)
        self.extract_frontmatter = extract_frontmatter
    
    async def load(self, path: Path, **kwargs: Any) -> list[Document]:
        """Load a Markdown file.
        
        Args:
            path: Path to the Markdown file.
            **kwargs: Optional 'encoding' to override default.
            
        Returns:
            List containing a single Document.
        """
        encoding = kwargs.get("encoding", self.encoding)
        content = path.read_text(encoding=encoding)
        metadata: dict[str, Any] = {}
        
        # Extract YAML frontmatter if present
        if self.extract_frontmatter and content.startswith("---"):
            content, metadata = self._extract_frontmatter(content)
        
        return [self._create_document(content, path, **metadata)]
    
    def _extract_frontmatter(self, content: str) -> tuple[str, dict[str, Any]]:
        """Extract YAML frontmatter from content.
        
        Args:
            content: Full document content starting with ---.
            
        Returns:
            Tuple of (content without frontmatter, frontmatter dict).
        """
        parts = content.split("---", 2)
        if len(parts) < 3:
            return content, {}
        
        frontmatter_text = parts[1].strip()
        remaining_content = parts[2].strip()
        
        # Parse simple key: value frontmatter
        metadata: dict[str, Any] = {}
        for line in frontmatter_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                
                # Handle quoted strings
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Handle lists (simple single-line format)
                if value.startswith("[") and value.endswith("]"):
                    value = [v.strip().strip("\"'") for v in value[1:-1].split(",")]
                
                metadata[key] = value
        
        return remaining_content, metadata


__all__ = ["MarkdownLoader"]
