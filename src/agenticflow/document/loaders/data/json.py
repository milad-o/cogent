"""JSON file loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenticflow.document.loaders.base import BaseLoader
from agenticflow.document.types import Document


class JSONLoader(BaseLoader):
    """Loader for JSON and JSONL files.

    For JSON arrays, each item becomes a separate Document.
    For JSON objects, the entire object becomes one Document.
    For JSONL files, each line becomes a separate Document.

    Example:
        >>> loader = JSONLoader()
        >>> docs = await loader.load(Path("data.json"))
        >>>
        >>> # Or with a content key
        >>> loader = JSONLoader(content_key="text")
        >>> docs = await loader.load(Path("articles.json"))
    """

    supported_extensions = [".json", ".jsonl", ".ndjson"]

    def __init__(
        self,
        encoding: str = "utf-8",
        content_key: str | None = None,
        metadata_keys: list[str] | None = None,
    ) -> None:
        """Initialize the loader.

        Args:
            encoding: Text encoding.
            content_key: JSON key to use as document content (None = full JSON).
            metadata_keys: JSON keys to extract as metadata.
        """
        super().__init__(encoding)
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []

    async def load(self, path: str | Path, **kwargs: Any) -> list[Document]:
        """Load a JSON or JSONL file.

        Args:
            path: Path to the JSON file (str or Path).
            **kwargs: Optional 'encoding' override.

        Returns:
            List of Documents.
        """
        path = Path(path)
        encoding = kwargs.get("encoding", self.encoding)
        ext = path.suffix.lower()

        if ext in {".jsonl", ".ndjson"}:
            return self._load_jsonl(path, encoding)
        else:
            return self._load_json(path, encoding)

    def _load_json(self, path: Path, encoding: str) -> list[Document]:
        """Load a regular JSON file.

        Args:
            path: File path.
            encoding: Text encoding.

        Returns:
            List of Documents.
        """
        content = path.read_text(encoding=encoding)
        data = json.loads(content)

        if isinstance(data, list):
            # Array: each item is a document
            documents = []
            for i, item in enumerate(data):
                doc = self._item_to_document(item, path, index=i, total_items=len(data))
                documents.append(doc)
            return documents
        else:
            # Object: one document
            return [self._item_to_document(data, path)]

    def _load_jsonl(self, path: Path, encoding: str) -> list[Document]:
        """Load a JSONL file.

        Args:
            path: File path.
            encoding: Text encoding.

        Returns:
            List of Documents.
        """
        content = path.read_text(encoding=encoding)
        lines = [line.strip() for line in content.split("\n") if line.strip()]

        documents = []
        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                doc = self._item_to_document(
                    data, path,
                    line=i + 1,
                    total_lines=len(lines),
                )
                documents.append(doc)
            except json.JSONDecodeError:
                continue

        return documents

    def _item_to_document(
        self,
        item: Any,
        path: Path,
        **extra_meta: Any,
    ) -> Document:
        """Convert a JSON item to a Document.

        Args:
            item: JSON item (dict or any value).
            path: Source file path.
            **extra_meta: Additional metadata.

        Returns:
            Document instance.
        """
        # Extract content
        if self.content_key and isinstance(item, dict):
            content = str(item.get(self.content_key, ""))
        else:
            content = json.dumps(item, indent=2, ensure_ascii=False)

        # Extract metadata
        metadata: dict[str, Any] = {}
        if self.metadata_keys and isinstance(item, dict):
            for key in self.metadata_keys:
                if key in item:
                    metadata[key] = item[key]

        return self._create_document(content, path, **metadata, **extra_meta)


__all__ = ["JSONLoader"]
