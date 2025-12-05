"""Source code file loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenticflow.document.loaders.base import BaseLoader
from agenticflow.document.types import Document


# Map extensions to language names
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
    ".vue": "vue",
    ".svelte": "svelte",
    ".r": "r",
    ".jl": "julia",
    ".lua": "lua",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".fs": "fsharp",
    ".clj": "clojure",
    ".lisp": "lisp",
    ".scm": "scheme",
}


class CodeLoader(BaseLoader):
    """Loader for source code files.
    
    Supports a wide variety of programming languages.
    Adds language metadata for downstream processing.
    
    Example:
        >>> loader = CodeLoader()
        >>> docs = await loader.load(Path("main.py"))
        >>> print(docs[0].metadata["language"])  # "python"
    """
    
    supported_extensions = list(EXTENSION_TO_LANGUAGE.keys())
    
    # Fallback encodings for code files
    FALLBACK_ENCODINGS = ["latin-1", "cp1252"]
    
    async def load(self, path: str | Path, **kwargs: Any) -> list[Document]:
        """Load a source code file.
        
        Args:
            path: Path to the code file (str or Path).
            **kwargs: Optional 'encoding' override.
            
        Returns:
            List containing a single Document.
        """
        path = Path(path)
        encoding = kwargs.get("encoding", self.encoding)
        content = self._read_with_fallback(path, encoding)
        
        # Detect language from extension
        ext = path.suffix.lower()
        language = EXTENSION_TO_LANGUAGE.get(ext, "text")
        
        return [
            self._create_document(
                content,
                path,
                language=language,
                line_count=content.count("\n") + 1,
            )
        ]
    
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
        
        for enc in self.FALLBACK_ENCODINGS:
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        
        return path.read_bytes().decode("utf-8", errors="replace")


__all__ = ["CodeLoader", "EXTENSION_TO_LANGUAGE"]
