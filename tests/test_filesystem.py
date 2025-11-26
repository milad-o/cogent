"""Tests for FileSystem capability."""

import tempfile
from pathlib import Path

import pytest

from agenticflow.capabilities import FileSystem


class TestFileSystemInit:
    """Test FileSystem initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        fs = FileSystem()
        
        assert fs.name == "filesystem"
        assert len(fs.allowed_paths) == 1
        assert fs.allowed_paths[0] == Path.cwd()
        assert fs.allow_write is True
        assert fs.allow_delete is False
    
    def test_custom_allowed_paths(self, tmp_path):
        """Test custom allowed paths."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        assert len(fs.allowed_paths) == 1
        assert fs.allowed_paths[0] == tmp_path
    
    def test_multiple_allowed_paths(self, tmp_path):
        """Test multiple allowed paths."""
        path1 = tmp_path / "dir1"
        path2 = tmp_path / "dir2"
        path1.mkdir()
        path2.mkdir()
        
        fs = FileSystem(allowed_paths=[str(path1), str(path2)])
        
        assert len(fs.allowed_paths) == 2
    
    def test_read_only_mode(self, tmp_path):
        """Test read-only mode."""
        fs = FileSystem(
            allowed_paths=[str(tmp_path)],
            allow_write=False,
        )
        
        assert fs.allow_write is False
    
    def test_delete_enabled(self, tmp_path):
        """Test delete enabled."""
        fs = FileSystem(
            allowed_paths=[str(tmp_path)],
            allow_delete=True,
        )
        
        assert fs.allow_delete is True


class TestFileSystemPathValidation:
    """Test path validation and security."""
    
    def test_path_within_allowed(self, tmp_path):
        """Test path within allowed directory."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        
        # Should not raise
        content = fs.read(str(test_file))
        assert content == "hello"
    
    def test_path_outside_allowed(self, tmp_path):
        """Test path outside allowed directory."""
        allowed = tmp_path / "allowed"
        forbidden = tmp_path / "forbidden"
        allowed.mkdir()
        forbidden.mkdir()
        
        fs = FileSystem(allowed_paths=[str(allowed)])
        
        test_file = forbidden / "secret.txt"
        test_file.write_text("secret")
        
        with pytest.raises(PermissionError, match="outside allowed"):
            fs.read(str(test_file))
    
    def test_denied_patterns(self, tmp_path):
        """Test denied patterns."""
        fs = FileSystem(
            allowed_paths=[str(tmp_path)],
            deny_patterns=["*.secret", "*.key"],
        )
        
        secret = tmp_path / "config.secret"
        secret.write_text("secret")
        
        with pytest.raises(PermissionError, match="denied pattern"):
            fs.read(str(secret))
    
    def test_default_deny_patterns(self, tmp_path):
        """Test default deny patterns include .env files."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        env = tmp_path / ".env"
        env.write_text("SECRET=value")
        
        with pytest.raises(PermissionError, match="denied pattern"):
            fs.read(str(env))


class TestFileSystemRead:
    """Test file reading operations."""
    
    def test_read_text_file(self, tmp_path):
        """Test reading a text file."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")
        
        content = fs.read(str(test_file))
        assert content == "Hello, World!"
    
    def test_read_multiline(self, tmp_path):
        """Test reading multiline file."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        test_file = tmp_path / "multi.txt"
        test_file.write_text("line1\nline2\nline3")
        
        content = fs.read(str(test_file))
        assert content == "line1\nline2\nline3"
    
    def test_read_nonexistent_file(self, tmp_path):
        """Test reading nonexistent file."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        with pytest.raises(FileNotFoundError):
            fs.read(str(tmp_path / "nonexistent.txt"))
    
    def test_read_directory_error(self, tmp_path):
        """Test reading a directory raises error."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        with pytest.raises(IsADirectoryError):
            fs.read(str(tmp_path))
    
    def test_read_file_too_large(self, tmp_path):
        """Test reading file exceeding max size."""
        fs = FileSystem(
            allowed_paths=[str(tmp_path)],
            max_file_size=100,
        )
        
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * 200)
        
        with pytest.raises(ValueError, match="too large"):
            fs.read(str(large_file))
    
    def test_read_bytes(self, tmp_path):
        """Test reading bytes."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")
        
        content = fs.read_bytes(str(test_file))
        assert content == b"\x00\x01\x02\x03"


class TestFileSystemWrite:
    """Test file writing operations."""
    
    def test_write_text_file(self, tmp_path):
        """Test writing a text file."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        test_file = tmp_path / "output.txt"
        fs.write(str(test_file), "Hello!")
        
        assert test_file.read_text() == "Hello!"
    
    def test_write_creates_parent_dirs(self, tmp_path):
        """Test writing creates parent directories."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        test_file = tmp_path / "nested" / "dir" / "file.txt"
        fs.write(str(test_file), "content")
        
        assert test_file.exists()
        assert test_file.read_text() == "content"
    
    def test_write_overwrite(self, tmp_path):
        """Test overwriting existing file."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        test_file = tmp_path / "existing.txt"
        test_file.write_text("old content")
        
        fs.write(str(test_file), "new content")
        
        assert test_file.read_text() == "new content"
    
    def test_write_append(self, tmp_path):
        """Test appending to file."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        test_file = tmp_path / "append.txt"
        test_file.write_text("line1\n")
        
        fs.write(str(test_file), "line2\n", append=True)
        
        assert test_file.read_text() == "line1\nline2\n"
    
    def test_write_disabled(self, tmp_path):
        """Test writing when disabled."""
        fs = FileSystem(
            allowed_paths=[str(tmp_path)],
            allow_write=False,
        )
        
        with pytest.raises(PermissionError, match="disabled"):
            fs.write(str(tmp_path / "file.txt"), "content")
    
    def test_write_bytes(self, tmp_path):
        """Test writing bytes."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        test_file = tmp_path / "binary.bin"
        fs.write_bytes(str(test_file), b"\x00\x01\x02")
        
        assert test_file.read_bytes() == b"\x00\x01\x02"


class TestFileSystemListDir:
    """Test directory listing."""
    
    def test_list_directory(self, tmp_path):
        """Test listing directory contents."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.txt").write_text("b")
        (tmp_path / "subdir").mkdir()
        
        items = fs.list_dir(str(tmp_path))
        
        assert len(items) == 3
        names = [i.name for i in items]
        assert "file1.txt" in names
        assert "file2.txt" in names
        assert "subdir" in names
    
    def test_list_with_pattern(self, tmp_path):
        """Test listing with pattern filter."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        (tmp_path / "test.py").write_text("")
        (tmp_path / "test.txt").write_text("")
        (tmp_path / "other.md").write_text("")
        
        items = fs.list_dir(str(tmp_path), pattern="*.py")
        
        assert len(items) == 1
        assert items[0].name == "test.py"
    
    def test_list_recursive(self, tmp_path):
        """Test recursive listing."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        (tmp_path / "root.txt").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "nested.txt").write_text("")
        
        items = fs.list_dir(str(tmp_path), recursive=True)
        
        names = [i.name for i in items]
        assert "root.txt" in names
        assert "nested.txt" in names
    
    def test_list_excludes_denied(self, tmp_path):
        """Test listing excludes denied patterns."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        (tmp_path / "normal.txt").write_text("")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.pyc").write_text("")
        
        items = fs.list_dir(str(tmp_path), recursive=True)
        
        names = [i.name for i in items]
        assert "normal.txt" in names
        # __pycache__ contents should be excluded
        assert "cached.pyc" not in names
    
    def test_list_sorts_dirs_first(self, tmp_path):
        """Test directories come before files."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        (tmp_path / "zfile.txt").write_text("")
        (tmp_path / "adir").mkdir()
        
        items = fs.list_dir(str(tmp_path))
        
        assert items[0].name == "adir"
        assert items[0].is_dir


class TestFileSystemSearch:
    """Test file searching."""
    
    def test_search_by_name(self, tmp_path):
        """Test searching by filename pattern."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        (tmp_path / "test.py").write_text("")
        (tmp_path / "other.py").write_text("")
        (tmp_path / "readme.md").write_text("")
        
        results = fs.search("*.py", str(tmp_path))
        
        assert len(results) == 2
    
    def test_search_recursive(self, tmp_path):
        """Test recursive search."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        (tmp_path / "root.py").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "nested.py").write_text("")
        
        results = fs.search("**/*.py", str(tmp_path))
        
        assert len(results) == 2
    
    def test_search_by_content(self, tmp_path):
        """Test searching by file content."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        (tmp_path / "match.txt").write_text("This contains NEEDLE here")
        (tmp_path / "nomatch.txt").write_text("No match here")
        
        results = fs.search("*.txt", str(tmp_path), content_pattern="NEEDLE")
        
        assert len(results) == 1
        assert results[0].name == "match.txt"
    
    def test_search_max_results(self, tmp_path):
        """Test search respects max results."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text("")
        
        results = fs.search("*.txt", str(tmp_path), max_results=5)
        
        assert len(results) == 5


class TestFileSystemOperations:
    """Test file operations (copy, move, delete)."""
    
    def test_exists(self, tmp_path):
        """Test exists check."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        existing = tmp_path / "exists.txt"
        existing.write_text("")
        
        assert fs.exists(str(existing)) is True
        assert fs.exists(str(tmp_path / "nonexistent")) is False
    
    def test_info(self, tmp_path):
        """Test file info."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        
        info = fs.info(str(test_file))
        
        assert info.name == "test.txt"
        assert info.is_dir is False
        assert info.size == 5
        assert info.extension == ".txt"
    
    def test_mkdir(self, tmp_path):
        """Test create directory."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        new_dir = tmp_path / "new" / "nested" / "dir"
        fs.mkdir(str(new_dir))
        
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_copy_file(self, tmp_path):
        """Test copying file."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest.txt"
        src.write_text("content")
        
        fs.copy(str(src), str(dst))
        
        assert src.exists()  # Original still exists
        assert dst.exists()
        assert dst.read_text() == "content"
    
    def test_copy_directory(self, tmp_path):
        """Test copying directory."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        src = tmp_path / "src_dir"
        src.mkdir()
        (src / "file.txt").write_text("content")
        
        dst = tmp_path / "dst_dir"
        fs.copy(str(src), str(dst))
        
        assert dst.exists()
        assert (dst / "file.txt").read_text() == "content"
    
    def test_move_file(self, tmp_path):
        """Test moving file."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest.txt"
        src.write_text("content")
        
        fs.move(str(src), str(dst))
        
        assert not src.exists()  # Original is gone
        assert dst.exists()
        assert dst.read_text() == "content"
    
    def test_delete_file(self, tmp_path):
        """Test deleting file."""
        fs = FileSystem(
            allowed_paths=[str(tmp_path)],
            allow_delete=True,
        )
        
        test_file = tmp_path / "todelete.txt"
        test_file.write_text("content")
        
        fs.delete(str(test_file))
        
        assert not test_file.exists()
    
    def test_delete_directory_recursive(self, tmp_path):
        """Test deleting directory recursively."""
        fs = FileSystem(
            allowed_paths=[str(tmp_path)],
            allow_delete=True,
        )
        
        dir_to_delete = tmp_path / "dir"
        dir_to_delete.mkdir()
        (dir_to_delete / "file.txt").write_text("")
        
        fs.delete(str(dir_to_delete), recursive=True)
        
        assert not dir_to_delete.exists()
    
    def test_delete_disabled(self, tmp_path):
        """Test delete when disabled."""
        fs = FileSystem(
            allowed_paths=[str(tmp_path)],
            allow_delete=False,
        )
        
        test_file = tmp_path / "protected.txt"
        test_file.write_text("")
        
        with pytest.raises(PermissionError, match="disabled"):
            fs.delete(str(test_file))


class TestFileSystemTools:
    """Test tool generation."""
    
    def test_get_tools_default(self, tmp_path):
        """Test default tool set."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        tools = fs.tools
        names = [t.name for t in tools]
        
        # Read-only tools always present
        assert "read_file" in names
        assert "list_directory" in names
        assert "search_files" in names
        assert "file_info" in names
        
        # Write tools present by default
        assert "write_file" in names
        assert "create_directory" in names
        assert "copy_file" in names
        assert "move_file" in names
        
        # Delete not present by default
        assert "delete_file" not in names
    
    def test_get_tools_read_only(self, tmp_path):
        """Test read-only tool set."""
        fs = FileSystem(
            allowed_paths=[str(tmp_path)],
            allow_write=False,
        )
        
        tools = fs.tools
        names = [t.name for t in tools]
        
        assert "read_file" in names
        assert "write_file" not in names
        assert "copy_file" not in names
    
    def test_get_tools_with_delete(self, tmp_path):
        """Test tool set with delete enabled."""
        fs = FileSystem(
            allowed_paths=[str(tmp_path)],
            allow_delete=True,
        )
        
        tools = fs.tools
        names = [t.name for t in tools]
        
        assert "delete_file" in names
    
    def test_read_file_tool(self, tmp_path):
        """Test read_file tool execution."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")
        
        tools = {t.name: t for t in fs.tools}
        result = tools["read_file"].invoke({"path": str(test_file)})
        
        assert "Hello World" in result
        assert "1 lines" in result
    
    def test_write_file_tool(self, tmp_path):
        """Test write_file tool execution."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        test_file = tmp_path / "output.txt"
        
        tools = {t.name: t for t in fs.tools}
        result = tools["write_file"].invoke({
            "path": str(test_file),
            "content": "Written content",
        })
        
        assert "Successfully wrote" in result
        assert test_file.read_text() == "Written content"
    
    def test_list_directory_tool(self, tmp_path):
        """Test list_directory tool execution."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        (tmp_path / "file.txt").write_text("")
        (tmp_path / "subdir").mkdir()
        
        tools = {t.name: t for t in fs.tools}
        result = tools["list_directory"].invoke({"path": str(tmp_path)})
        
        assert "file.txt" in result
        assert "subdir" in result
    
    def test_search_files_tool(self, tmp_path):
        """Test search_files tool execution."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        (tmp_path / "match.py").write_text("")
        (tmp_path / "other.txt").write_text("")
        
        tools = {t.name: t for t in fs.tools}
        result = tools["search_files"].invoke({
            "pattern": "*.py",
            "path": str(tmp_path),
        })
        
        assert "match.py" in result
        assert "other.txt" not in result


class TestFileSystemIntegration:
    """Integration tests for FileSystem capability."""
    
    def test_workflow_read_modify_write(self, tmp_path):
        """Test typical read-modify-write workflow."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        # Create initial file
        config = tmp_path / "config.json"
        config.write_text('{"version": 1}')
        
        # Read
        content = fs.read(str(config))
        assert '"version": 1' in content
        
        # Modify and write
        new_content = content.replace("1", "2")
        fs.write(str(config), new_content)
        
        # Verify
        assert fs.read(str(config)) == '{"version": 2}'
    
    def test_workflow_create_project_structure(self, tmp_path):
        """Test creating a project structure."""
        fs = FileSystem(allowed_paths=[str(tmp_path)])
        
        # Create directories
        fs.mkdir(str(tmp_path / "src"))
        fs.mkdir(str(tmp_path / "tests"))
        fs.mkdir(str(tmp_path / "docs"))
        
        # Create files
        fs.write(str(tmp_path / "src" / "__init__.py"), "")
        fs.write(str(tmp_path / "src" / "main.py"), "def main(): pass")
        fs.write(str(tmp_path / "tests" / "test_main.py"), "def test(): pass")
        fs.write(str(tmp_path / "README.md"), "# Project")
        
        # Verify structure
        items = fs.list_dir(str(tmp_path))
        names = [i.name for i in items]
        
        assert "src" in names
        assert "tests" in names
        assert "docs" in names
        assert "README.md" in names
    
    def test_security_traversal_attempt(self, tmp_path):
        """Test path traversal is blocked."""
        allowed = tmp_path / "allowed"
        secret = tmp_path / "secret"
        allowed.mkdir()
        secret.mkdir()
        (secret / "password.txt").write_text("supersecret")
        
        fs = FileSystem(allowed_paths=[str(allowed)])
        
        # Try various traversal attempts
        with pytest.raises(PermissionError):
            fs.read(str(allowed / ".." / "secret" / "password.txt"))
        
        with pytest.raises(PermissionError):
            fs.read(f"{allowed}/../secret/password.txt")
