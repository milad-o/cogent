"""
FileSystem capability - sandboxed file operations.

Provides secure file system access with configurable allowed paths,
enabling agents to read, write, search, and manage files safely.

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import FileSystem
    
    # Allow access only to specific directories
    fs = FileSystem(allowed_paths=["./data", "./output"])
    
    agent = Agent(
        name="Assistant",
        model=model,
        capabilities=[fs],
    )
    
    # Agent can now work with files in allowed directories
    await agent.run("Read the config.json file and summarize it")
    await agent.run("Create a report.txt with the analysis results")
    ```
"""

from __future__ import annotations

import fnmatch
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agenticflow.tools.base import BaseTool, tool

from agenticflow.capabilities.base import BaseCapability


@dataclass
class FileInfo:
    """Information about a file or directory."""
    
    path: str
    name: str
    is_dir: bool
    size: int
    modified: datetime
    created: datetime | None = None
    extension: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "is_dir": self.is_dir,
            "size": self.size,
            "modified": self.modified.isoformat(),
            "created": self.created.isoformat() if self.created else None,
            "extension": self.extension,
        }


class FileSystem(BaseCapability):
    """
    FileSystem capability for sandboxed file operations.
    
    Provides tools for reading, writing, listing, and searching files
    within specified allowed directories.
    
    Args:
        allowed_paths: List of paths the agent can access. Supports absolute
            paths and relative paths (resolved from cwd). If empty, allows
            current working directory only.
        deny_patterns: Glob patterns to deny (e.g., ["*.env", "*.key", ".git/*"])
        max_file_size: Maximum file size to read/write in bytes (default: 10MB)
        allow_write: Whether to allow write operations (default: True)
        allow_delete: Whether to allow delete operations (default: False)
        encoding: Default file encoding (default: "utf-8")
    
    Example:
        ```python
        # Read-only access to docs
        fs = FileSystem(
            allowed_paths=["./docs"],
            allow_write=False,
        )
        
        # Full access with security
        fs = FileSystem(
            allowed_paths=["./workspace"],
            deny_patterns=["*.env", "*.key", ".git/*", "__pycache__/*"],
            allow_delete=True,
        )
        ```
    """
    
    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        deny_patterns: list[str] | None = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        allow_write: bool = True,
        allow_delete: bool = False,
        encoding: str = "utf-8",
        name: str = "filesystem",
    ):
        self._name = name
        self._tools_cache: list[BaseTool] | None = None
        
        # Resolve and normalize allowed paths
        if allowed_paths:
            self._allowed_paths = [
                Path(p).resolve() for p in allowed_paths
            ]
        else:
            # Default to current working directory
            self._allowed_paths = [Path.cwd()]
        
        self._deny_patterns = deny_patterns or [
            "*.env",
            "*.key",
            "*.pem",
            "*.secret",
            ".git/*",
            ".env*",
            "**/node_modules/*",
            "**/__pycache__/*",
            "**/.venv/*",
        ]
        
        self._max_file_size = max_file_size
        self._allow_write = allow_write
        self._allow_delete = allow_delete
        self._encoding = encoding
    
    @property
    def name(self) -> str:
        """Capability name."""
        return self._name
    
    @property
    def description(self) -> str:
        """Capability description."""
        return "Sandboxed file system operations with security controls"
    
    @property
    def tools(self) -> list[BaseTool]:
        """Get file system tools."""
        if self._tools_cache is None:
            self._tools_cache = self._build_tools()
        return self._tools_cache
    
    @property
    def allowed_paths(self) -> list[Path]:
        """Get allowed paths."""
        return self._allowed_paths
    
    @property
    def allow_write(self) -> bool:
        """Check if write is allowed."""
        return self._allow_write
    
    @property
    def allow_delete(self) -> bool:
        """Check if delete is allowed."""
        return self._allow_delete
    
    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is within allowed directories."""
        resolved = path.resolve()
        
        for allowed in self._allowed_paths:
            try:
                resolved.relative_to(allowed)
                return True
            except ValueError:
                continue
        
        return False
    
    def _is_path_denied(self, path: Path) -> bool:
        """Check if path matches any deny pattern."""
        path_str = str(path)
        
        for pattern in self._deny_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
            if fnmatch.fnmatch(path.name, pattern):
                return True
        
        return False
    
    def _validate_path(self, path_str: str, must_exist: bool = True) -> Path:
        """
        Validate and resolve a path.
        
        Args:
            path_str: Path string to validate
            must_exist: Whether the path must exist
            
        Returns:
            Resolved Path object
            
        Raises:
            PermissionError: If path is not allowed or denied
            FileNotFoundError: If must_exist and path doesn't exist
        """
        path = Path(path_str).resolve()
        
        if not self._is_path_allowed(path):
            allowed = ", ".join(str(p) for p in self._allowed_paths)
            raise PermissionError(
                f"Path '{path}' is outside allowed directories: {allowed}"
            )
        
        if self._is_path_denied(path):
            raise PermissionError(
                f"Path '{path}' matches a denied pattern"
            )
        
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        return path
    
    def _get_file_info(self, path: Path) -> FileInfo:
        """Get file information."""
        stat = path.stat()
        
        return FileInfo(
            path=str(path),
            name=path.name,
            is_dir=path.is_dir(),
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            created=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            extension=path.suffix if path.is_file() else None,
        )
    
    # =========================================================================
    # Core Operations (Direct API)
    # =========================================================================
    
    def read(self, path: str, encoding: str | None = None) -> str:
        """
        Read file contents.
        
        Args:
            path: Path to file
            encoding: File encoding (default: utf-8)
            
        Returns:
            File contents as string
        """
        validated = self._validate_path(path)
        
        if validated.is_dir():
            raise IsADirectoryError(f"Cannot read directory: {validated}")
        
        if validated.stat().st_size > self._max_file_size:
            raise ValueError(
                f"File too large ({validated.stat().st_size} bytes). "
                f"Max: {self._max_file_size} bytes"
            )
        
        return validated.read_text(encoding=encoding or self._encoding)
    
    def read_bytes(self, path: str) -> bytes:
        """Read file as bytes."""
        validated = self._validate_path(path)
        
        if validated.is_dir():
            raise IsADirectoryError(f"Cannot read directory: {validated}")
        
        if validated.stat().st_size > self._max_file_size:
            raise ValueError(f"File too large. Max: {self._max_file_size} bytes")
        
        return validated.read_bytes()
    
    def write(
        self,
        path: str,
        content: str,
        encoding: str | None = None,
        append: bool = False,
    ) -> Path:
        """
        Write content to file.
        
        Args:
            path: Path to file
            content: Content to write
            encoding: File encoding
            append: Whether to append instead of overwrite
            
        Returns:
            Path to written file
        """
        if not self._allow_write:
            raise PermissionError("Write operations are disabled")
        
        validated = self._validate_path(path, must_exist=False)
        
        # Create parent directories if needed
        validated.parent.mkdir(parents=True, exist_ok=True)
        
        if append and validated.exists():
            existing = validated.read_text(encoding=encoding or self._encoding)
            content = existing + content
        
        validated.write_text(content, encoding=encoding or self._encoding)
        return validated
    
    def write_bytes(self, path: str, content: bytes) -> Path:
        """Write bytes to file."""
        if not self._allow_write:
            raise PermissionError("Write operations are disabled")
        
        validated = self._validate_path(path, must_exist=False)
        validated.parent.mkdir(parents=True, exist_ok=True)
        validated.write_bytes(content)
        return validated
    
    def list_dir(
        self,
        path: str = ".",
        pattern: str | None = None,
        recursive: bool = False,
    ) -> list[FileInfo]:
        """
        List directory contents.
        
        Args:
            path: Directory path
            pattern: Glob pattern to filter (e.g., "*.py")
            recursive: Whether to list recursively
            
        Returns:
            List of FileInfo objects
        """
        validated = self._validate_path(path)
        
        if not validated.is_dir():
            raise NotADirectoryError(f"Not a directory: {validated}")
        
        results = []
        
        if recursive:
            if pattern:
                items = validated.rglob(pattern)
            else:
                items = validated.rglob("*")
        else:
            if pattern:
                items = validated.glob(pattern)
            else:
                items = validated.iterdir()
        
        for item in items:
            # Skip denied paths
            if self._is_path_denied(item):
                continue
            
            try:
                results.append(self._get_file_info(item))
            except (PermissionError, OSError):
                # Skip files we can't access
                continue
        
        return sorted(results, key=lambda x: (not x.is_dir, x.name.lower()))
    
    def search(
        self,
        pattern: str,
        path: str = ".",
        content_pattern: str | None = None,
        max_results: int = 100,
    ) -> list[FileInfo]:
        """
        Search for files.
        
        Args:
            pattern: Glob pattern for filenames (e.g., "*.py", "**/*.md")
            path: Directory to search in
            content_pattern: Optional text to search within files
            max_results: Maximum number of results
            
        Returns:
            List of matching FileInfo objects
        """
        validated = self._validate_path(path)
        results = []
        
        for item in validated.rglob(pattern):
            if len(results) >= max_results:
                break
            
            if self._is_path_denied(item):
                continue
            
            if item.is_dir():
                continue
            
            # Content search
            if content_pattern:
                try:
                    content = item.read_text(encoding=self._encoding)
                    if content_pattern.lower() not in content.lower():
                        continue
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            try:
                results.append(self._get_file_info(item))
            except (PermissionError, OSError):
                continue
        
        return results
    
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        try:
            validated = self._validate_path(path, must_exist=False)
            return validated.exists()
        except PermissionError:
            return False
    
    def info(self, path: str) -> FileInfo:
        """Get file/directory information."""
        validated = self._validate_path(path)
        return self._get_file_info(validated)
    
    def mkdir(self, path: str, parents: bool = True) -> Path:
        """Create directory."""
        if not self._allow_write:
            raise PermissionError("Write operations are disabled")
        
        validated = self._validate_path(path, must_exist=False)
        validated.mkdir(parents=parents, exist_ok=True)
        return validated
    
    def copy(self, src: str, dst: str) -> Path:
        """Copy file or directory."""
        if not self._allow_write:
            raise PermissionError("Write operations are disabled")
        
        src_path = self._validate_path(src)
        dst_path = self._validate_path(dst, must_exist=False)
        
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path)
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
        
        return dst_path
    
    def move(self, src: str, dst: str) -> Path:
        """Move/rename file or directory."""
        if not self._allow_write:
            raise PermissionError("Write operations are disabled")
        
        src_path = self._validate_path(src)
        dst_path = self._validate_path(dst, must_exist=False)
        
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        
        return dst_path
    
    def delete(self, path: str, recursive: bool = False) -> bool:
        """
        Delete file or directory.
        
        Args:
            path: Path to delete
            recursive: If True, delete directories with contents
            
        Returns:
            True if deleted successfully
        """
        if not self._allow_delete:
            raise PermissionError("Delete operations are disabled")
        
        validated = self._validate_path(path)
        
        if validated.is_dir():
            if recursive:
                shutil.rmtree(validated)
            else:
                validated.rmdir()  # Only works if empty
        else:
            validated.unlink()
        
        return True
    
    # =========================================================================
    # Tool Generation
    # =========================================================================
    
    def _build_tools(self) -> list[BaseTool]:
        """Build file system tools for agent use."""
        tools = [
            self._read_file_tool(),
            self._list_directory_tool(),
            self._search_files_tool(),
            self._file_info_tool(),
        ]
        
        if self._allow_write:
            tools.extend([
                self._write_file_tool(),
                self._create_directory_tool(),
                self._copy_file_tool(),
                self._move_file_tool(),
            ])
        
        if self._allow_delete:
            tools.append(self._delete_file_tool())
        
        return tools
    
    def _read_file_tool(self) -> BaseTool:
        fs = self
        
        @tool
        def read_file(path: str) -> str:
            """
            Read the contents of a file.
            
            Args:
                path: Path to the file to read
            
            Returns:
                The file contents as text
            """
            try:
                content = fs.read(path)
                lines = content.count('\n') + 1
                return f"[File: {path} ({lines} lines)]\n\n{content}"
            except Exception as e:
                return f"Error reading file: {e}"
        
        return read_file
    
    def _write_file_tool(self) -> BaseTool:
        fs = self
        
        @tool
        def write_file(path: str, content: str) -> str:
            """
            Write content to a file. Creates the file if it doesn't exist.
            
            Args:
                path: Path to the file to write
                content: The content to write to the file
            
            Returns:
                Confirmation message
            """
            try:
                result = fs.write(path, content)
                return f"Successfully wrote {len(content)} characters to {result}"
            except Exception as e:
                return f"Error writing file: {e}"
        
        return write_file
    
    def _list_directory_tool(self) -> BaseTool:
        fs = self
        
        @tool
        def list_directory(path: str = ".", pattern: str = "", recursive: bool = False) -> str:
            """
            List contents of a directory.
            
            Args:
                path: Directory path to list (default: current directory)
                pattern: Optional glob pattern to filter (e.g., "*.py", "*.md")
                recursive: Whether to list subdirectories recursively
            
            Returns:
                Directory listing with file information
            """
            try:
                items = fs.list_dir(
                    path,
                    pattern=pattern if pattern else None,
                    recursive=recursive,
                )
                
                if not items:
                    return f"Directory '{path}' is empty"
                
                lines = [f"Contents of '{path}' ({len(items)} items):"]
                for item in items:
                    if item.is_dir:
                        lines.append(f"  📁 {item.name}/")
                    else:
                        size = _format_size(item.size)
                        lines.append(f"  📄 {item.name} ({size})")
                
                return "\n".join(lines)
            except Exception as e:
                return f"Error listing directory: {e}"
        
        return list_directory
    
    def _search_files_tool(self) -> BaseTool:
        fs = self
        
        @tool
        def search_files(
            pattern: str,
            path: str = ".",
            content: str = "",
        ) -> str:
            """
            Search for files by name pattern and optionally by content.
            
            Args:
                pattern: Glob pattern for filenames (e.g., "*.py", "**/*.md")
                path: Directory to search in (default: current directory)
                content: Optional text to search within files
            
            Returns:
                List of matching files
            """
            try:
                results = fs.search(
                    pattern,
                    path,
                    content_pattern=content if content else None,
                )
                
                if not results:
                    msg = f"No files matching '{pattern}'"
                    if content:
                        msg += f" containing '{content}'"
                    return msg
                
                lines = [f"Found {len(results)} files:"]
                for item in results:
                    lines.append(f"  📄 {item.path}")
                
                return "\n".join(lines)
            except Exception as e:
                return f"Error searching files: {e}"
        
        return search_files
    
    def _file_info_tool(self) -> BaseTool:
        fs = self
        
        @tool
        def file_info(path: str) -> str:
            """
            Get detailed information about a file or directory.
            
            Args:
                path: Path to the file or directory
            
            Returns:
                File/directory information
            """
            try:
                info = fs.info(path)
                
                lines = [
                    f"Path: {info.path}",
                    f"Type: {'Directory' if info.is_dir else 'File'}",
                    f"Size: {_format_size(info.size)}",
                    f"Modified: {info.modified.strftime('%Y-%m-%d %H:%M:%S')}",
                ]
                
                if info.extension:
                    lines.append(f"Extension: {info.extension}")
                
                return "\n".join(lines)
            except Exception as e:
                return f"Error getting file info: {e}"
        
        return file_info
    
    def _create_directory_tool(self) -> BaseTool:
        fs = self
        
        @tool
        def create_directory(path: str) -> str:
            """
            Create a new directory.
            
            Args:
                path: Path for the new directory
            
            Returns:
                Confirmation message
            """
            try:
                result = fs.mkdir(path)
                return f"Created directory: {result}"
            except Exception as e:
                return f"Error creating directory: {e}"
        
        return create_directory
    
    def _copy_file_tool(self) -> BaseTool:
        fs = self
        
        @tool
        def copy_file(source: str, destination: str) -> str:
            """
            Copy a file or directory to a new location.
            
            Args:
                source: Path to the source file/directory
                destination: Path to the destination
            
            Returns:
                Confirmation message
            """
            try:
                result = fs.copy(source, destination)
                return f"Copied '{source}' to '{result}'"
            except Exception as e:
                return f"Error copying: {e}"
        
        return copy_file
    
    def _move_file_tool(self) -> BaseTool:
        fs = self
        
        @tool
        def move_file(source: str, destination: str) -> str:
            """
            Move or rename a file or directory.
            
            Args:
                source: Path to the source file/directory
                destination: New path/name
            
            Returns:
                Confirmation message
            """
            try:
                result = fs.move(source, destination)
                return f"Moved '{source}' to '{result}'"
            except Exception as e:
                return f"Error moving: {e}"
        
        return move_file
    
    def _delete_file_tool(self) -> BaseTool:
        fs = self
        
        @tool
        def delete_file(path: str, recursive: bool = False) -> str:
            """
            Delete a file or directory.
            
            Args:
                path: Path to delete
                recursive: If True, delete non-empty directories
            
            Returns:
                Confirmation message
            """
            try:
                fs.delete(path, recursive=recursive)
                return f"Deleted: {path}"
            except Exception as e:
                return f"Error deleting: {e}"
        
        return delete_file


def _format_size(size: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
