"""Main DocumentLoader class."""

from __future__ import annotations

import asyncio
import mimetypes
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from agenticflow.document.loaders.base import BaseLoader
from agenticflow.document.loaders.registry import get_loader, register_loader
from agenticflow.document.types import Document

from typing import Any, Callable, Sequence


class DocumentLoader:
    """Universal document loader supporting multiple file types.
    
    Automatically detects file type and uses the appropriate loader.
    Supports text, PDF, DOCX, CSV, JSON, Excel, HTML, and code files.
    
    Args:
        encoding: Default text encoding (default: utf-8).
        
    Example:
        >>> loader = DocumentLoader()
        >>> 
        >>> # Load single file
        >>> docs = await loader.load("report.pdf")
        >>> 
        >>> # Load multiple files
        >>> docs = await loader.load_many(["doc1.pdf", "data.csv"])
        >>> 
        >>> # Load directory with glob pattern
        >>> docs = await loader.load_directory("./docs", glob="**/*.md")
    """
    
    def __init__(self, encoding: str = "utf-8") -> None:
        """Initialize the loader.
        
        Args:
            encoding: Default text encoding.
        """
        self.encoding = encoding
        self._loader_cache: dict[str, BaseLoader] = {}
    
    def register_loader(
        self,
        extension: str,
        loader: type[BaseLoader] | Callable[[Path], Any],
    ) -> None:
        """Register a custom loader for a file extension.
        
        Args:
            extension: File extension (e.g., ".xyz").
            loader: Loader class or async function.
        """
        register_loader(extension, loader)
        # Clear cache to pick up new loader
        self._loader_cache.clear()
    
    def _get_loader_instance(self, extension: str) -> BaseLoader | Callable | None:
        """Get or create a loader instance for an extension.
        
        Args:
            extension: File extension.
            
        Returns:
            Loader instance, callable, or None.
        """
        if extension not in self._loader_cache:
            loader_class = get_loader(extension)
            if loader_class:
                # Check if it's a class (BaseLoader subclass) or function
                if isinstance(loader_class, type) and issubclass(loader_class, BaseLoader):
                    self._loader_cache[extension] = loader_class(encoding=self.encoding)
                else:
                    # It's a callable (function), store as-is
                    self._loader_cache[extension] = loader_class
        
        return self._loader_cache.get(extension)
    
    async def load(
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> list[Document]:
        """Load a single file.
        
        Args:
            path: Path to file.
            **kwargs: Additional arguments passed to loader.
            
        Returns:
            List of documents (may be multiple for multi-page files).
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file type is not supported.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        
        ext = path.suffix.lower()
        loader = self._get_loader_instance(ext)
        
        if not loader:
            # Try mime type detection for text files
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type and mime_type.startswith("text/"):
                loader = self._get_loader_instance(".txt")
            
            if not loader:
                from agenticflow.document.loaders.registry import list_supported_extensions
                raise ValueError(
                    f"Unsupported file type: {ext}. "
                    f"Supported: {', '.join(list_supported_extensions())}"
                )
        
        # Handle both BaseLoader instances and callable functions
        if isinstance(loader, BaseLoader):
            return await loader.load(path, **kwargs)
        else:
            # It's a callable (custom function loader)
            import inspect
            if inspect.iscoroutinefunction(loader):
                return await loader(path, **kwargs)
            else:
                return loader(path, **kwargs)
    
    def load_sync(
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> list[Document]:
        """Synchronous version of load.
        
        Args:
            path: Path to file.
            **kwargs: Additional arguments passed to loader.
            
        Returns:
            List of documents.
        """
        return asyncio.run(self.load(path, **kwargs))
    
    async def load_many(
        self,
        paths: Sequence[str | Path],
        **kwargs: Any,
    ) -> list[Document]:
        """Load multiple files concurrently.
        
        Args:
            paths: List of file paths.
            **kwargs: Additional arguments passed to loaders.
            
        Returns:
            Combined list of all documents.
        """
        tasks = [self.load(p, **kwargs) for p in paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        documents = []
        for result in results:
            if isinstance(result, Exception):
                # Log error but continue with other files
                continue
            documents.extend(result)
        
        return documents
    
    async def load_directory(
        self,
        directory: str | Path,
        glob: str = "**/*",
        recursive: bool = True,
        exclude: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Load all files from a directory.
        
        Args:
            directory: Directory path.
            glob: Glob pattern for file matching (default: all files).
            recursive: Whether to search recursively (deprecated, use glob).
            exclude: Regex patterns to exclude.
            **kwargs: Additional arguments passed to loaders.
            
        Returns:
            List of all loaded documents.
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        exclude = exclude or []
        exclude_patterns = [re.compile(p) for p in exclude]
        
        # Collect matching files
        files = []
        for path in directory.glob(glob):
            if not path.is_file():
                continue
            
            # Check exclusions
            rel_path = str(path.relative_to(directory))
            if any(p.search(rel_path) for p in exclude_patterns):
                continue
            
            # Check if we have a loader
            if get_loader(path.suffix.lower()) is not None:
                files.append(path)
        
        return await self.load_many(files, **kwargs)


__all__ = ["DocumentLoader"]
