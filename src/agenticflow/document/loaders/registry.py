"""Loader registry for file type to loader mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pathlib import Path
    
    from agenticflow.document.loaders.base import BaseLoader
    from agenticflow.document.types import Document


# Global registry of loaders
_LOADER_REGISTRY: dict[str, type[BaseLoader]] = {}


def _build_default_registry() -> dict[str, type[BaseLoader]]:
    """Build the default loader registry.
    
    Imports handlers lazily to avoid circular imports.
    
    Returns:
        Dictionary mapping extensions to loader classes.
    """
    from agenticflow.document.loaders.code import CodeLoader
    from agenticflow.document.loaders.data import CSVLoader, JSONLoader, XLSXLoader
    from agenticflow.document.loaders.markup import HTMLLoader, MarkdownLoader
    from agenticflow.document.loaders.pdf import PDFLoader
    from agenticflow.document.loaders.plaintext import TextLoader
    from agenticflow.document.loaders.word import WordLoader
    
    registry: dict[str, type[BaseLoader]] = {}
    
    # Register each loader for its supported extensions
    loaders = [
        TextLoader,
        MarkdownLoader,
        HTMLLoader,
        PDFLoader,
        WordLoader,
        CSVLoader,
        JSONLoader,
        XLSXLoader,
        CodeLoader,
    ]
    
    for loader_class in loaders:
        for ext in loader_class.supported_extensions:
            registry[ext] = loader_class
    
    return registry


def get_loader(extension: str) -> type[BaseLoader] | None:
    """Get the loader class for a file extension.
    
    Args:
        extension: File extension (with or without leading dot).
        
    Returns:
        Loader class or None if not found.
    """
    global _LOADER_REGISTRY
    
    if not _LOADER_REGISTRY:
        _LOADER_REGISTRY = _build_default_registry()
    
    if not extension.startswith("."):
        extension = f".{extension}"
    
    return _LOADER_REGISTRY.get(extension.lower())


def register_loader(
    extension: str,
    loader: type[BaseLoader] | Callable[[Path], Any],
) -> None:
    """Register a custom loader for a file extension.
    
    Args:
        extension: File extension (e.g., ".xyz").
        loader: Loader class or function.
        
    Example:
        >>> from agenticflow.document.loaders import register_loader, BaseLoader
        >>> 
        >>> class MyLoader(BaseLoader):
        ...     supported_extensions = [".xyz"]
        ...     async def load(self, path, **kwargs):
        ...         return [Document(content=path.read_text())]
        >>> 
        >>> register_loader(".xyz", MyLoader)
    """
    global _LOADER_REGISTRY
    
    if not _LOADER_REGISTRY:
        _LOADER_REGISTRY = _build_default_registry()
    
    if not extension.startswith("."):
        extension = f".{extension}"
    
    _LOADER_REGISTRY[extension.lower()] = loader  # type: ignore


def list_supported_extensions() -> list[str]:
    """Get list of all supported file extensions.
    
    Returns:
        Sorted list of supported extensions.
    """
    global _LOADER_REGISTRY
    
    if not _LOADER_REGISTRY:
        _LOADER_REGISTRY = _build_default_registry()
    
    return sorted(_LOADER_REGISTRY.keys())


# Expose the registry for direct access (backward compatibility)
def _get_loaders() -> dict[str, type[BaseLoader]]:
    """Get the full loader registry."""
    global _LOADER_REGISTRY
    
    if not _LOADER_REGISTRY:
        _LOADER_REGISTRY = _build_default_registry()
    
    return _LOADER_REGISTRY


# Create a lazy property-like access
class _LoadersProxy:
    """Proxy object for lazy loading the LOADERS dict."""
    
    def __getitem__(self, key: str) -> type[BaseLoader] | None:
        return get_loader(key)
    
    def __contains__(self, key: str) -> bool:
        return get_loader(key) is not None
    
    def keys(self) -> list[str]:
        return list_supported_extensions()
    
    def values(self):
        return _get_loaders().values()
    
    def items(self):
        return _get_loaders().items()
    
    def get(self, key: str, default=None):
        return get_loader(key) or default
    
    def __iter__(self):
        return iter(list_supported_extensions())


LOADERS = _LoadersProxy()


__all__ = [
    "LOADERS",
    "get_loader",
    "register_loader",
    "list_supported_extensions",
]
