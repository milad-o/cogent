"""Utility functions for document loading."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agenticflow.document.types import Document


async def load_documents(
    source: str | Path | Sequence[str | Path],
    **kwargs: Any,
) -> list[Document]:
    """Load documents from file(s) or directory.

    Convenience function that auto-detects source type.

    Args:
        source: File path, directory path, or list of paths.
        **kwargs: Additional arguments for loaders.

    Returns:
        List of loaded documents.

    Example:
        >>> # Load single file
        >>> docs = await load_documents("report.pdf")
        >>>
        >>> # Load directory
        >>> docs = await load_documents("./documents")
        >>>
        >>> # Load multiple files
        >>> docs = await load_documents(["doc1.pdf", "doc2.md"])
    """
    from agenticflow.document.loaders.loader import DocumentLoader

    # Extract encoding for loader
    encoding = kwargs.pop("encoding", "utf-8")
    loader = DocumentLoader(encoding=encoding)

    if isinstance(source, (list, tuple)):
        return await loader.load_many(source, **kwargs)

    path = Path(source)
    if path.is_dir():
        return await loader.load_directory(path, **kwargs)
    else:
        return await loader.load(path, **kwargs)


def load_documents_sync(
    source: str | Path | Sequence[str | Path],
    **kwargs: Any,
) -> list[Document]:
    """Synchronous version of load_documents.

    Args:
        source: File path, directory path, or list of paths.
        **kwargs: Additional arguments for loaders.

    Returns:
        List of loaded documents.
    """
    return asyncio.run(load_documents(source, **kwargs))


__all__ = ["load_documents", "load_documents_sync"]
