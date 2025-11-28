"""
Example 14: FileSystem Capability

Demonstrates the FileSystem capability for secure, sandboxed file operations.
Shows how agents can read, write, search, and manage files within allowed directories.

Features:
- Sandboxed access with allowed paths
- Automatic deny patterns for sensitive files
- Read/write/search/list operations
- Path traversal protection
"""

import asyncio
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Programmatic Demo (no LLM needed)
# ============================================================

def programmatic_demo():
    """Demonstrate FileSystem capability without an agent."""
    print("=" * 60)
    print("📁 FileSystem Programmatic Demo")
    print("=" * 60)
    
    from agenticflow.capabilities import FileSystem
    
    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as workspace:
        workspace_path = Path(workspace)
        
        # Initialize FileSystem with sandboxed access
        fs = FileSystem(
            allowed_paths=[workspace],
            allow_write=True,
            allow_delete=True,
        )
        
        print(f"\n✓ Created FileSystem capability")
        print(f"  Allowed paths: {[str(p) for p in fs.allowed_paths]}")
        print(f"  Tools: {[t.name for t in fs.tools]}")
        
        # Create a project structure
        print("\n" + "-" * 40)
        print("📂 Creating project structure...")
        
        # Create directories
        fs.mkdir(str(workspace_path / "src"))
        fs.mkdir(str(workspace_path / "tests"))
        fs.mkdir(str(workspace_path / "docs"))
        
        # Create files
        fs.write(
            str(workspace_path / "README.md"),
            "# My Project\n\nA sample project demonstrating FileSystem capability.\n"
        )
        
        fs.write(
            str(workspace_path / "src" / "__init__.py"),
            '"""My project package."""\n\n__version__ = "1.0.0"\n'
        )
        
        fs.write(
            str(workspace_path / "src" / "main.py"),
            '''"""Main module."""

def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

def calculate(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

if __name__ == "__main__":
    print(greet("World"))
'''
        )
        
        fs.write(
            str(workspace_path / "tests" / "test_main.py"),
            '''"""Tests for main module."""
import pytest
from src.main import greet, calculate

def test_greet():
    assert greet("Alice") == "Hello, Alice!"

def test_calculate():
    assert calculate(2, 3) == 5
'''
        )
        
        fs.write(
            str(workspace_path / "config.json"),
            '{\n  "debug": true,\n  "version": "1.0.0"\n}\n'
        )
        
        print("   ✓ Created project files")
        
        # List directory contents
        print("\n" + "-" * 40)
        print("📋 Listing workspace contents:")
        items = fs.list_dir(workspace)
        for item in items:
            icon = "📁" if item.is_dir else "📄"
            print(f"   {icon} {item.name}")
        
        # Read a file
        print("\n" + "-" * 40)
        print("📖 Reading README.md:")
        content = fs.read(str(workspace_path / "README.md"))
        for line in content.strip().split("\n"):
            print(f"   {line}")
        
        # Search for Python files
        print("\n" + "-" * 40)
        print("🔍 Searching for Python files:")
        py_files = fs.search("**/*.py", workspace)
        for f in py_files:
            print(f"   📄 {f.path}")
        
        # Search by content
        print("\n" + "-" * 40)
        print('🔍 Searching for files containing "def greet":')
        matches = fs.search("**/*.py", workspace, content_pattern="def greet")
        for f in matches:
            print(f"   📄 {f.path}")
        
        # File info
        print("\n" + "-" * 40)
        print("ℹ️  File info for main.py:")
        info = fs.info(str(workspace_path / "src" / "main.py"))
        print(f"   Path: {info.path}")
        print(f"   Size: {info.size} bytes")
        print(f"   Extension: {info.extension}")
        
        # Copy file
        print("\n" + "-" * 40)
        print("📋 Copying main.py to backup.py:")
        fs.copy(
            str(workspace_path / "src" / "main.py"),
            str(workspace_path / "src" / "backup.py")
        )
        print("   ✓ Copied successfully")
        
        # Append to file
        print("\n" + "-" * 40)
        print("✏️  Appending to README.md:")
        fs.write(
            str(workspace_path / "README.md"),
            "\n## License\n\nMIT\n",
            append=True
        )
        print("   ✓ Appended license section")
        
        # Final directory listing
        print("\n" + "-" * 40)
        print("📋 Final recursive listing:")
        all_items = fs.list_dir(workspace, recursive=True)
        # Use resolved paths for comparison
        resolved_workspace = Path(workspace).resolve()
        for item in all_items:
            rel_path = Path(item.path).relative_to(resolved_workspace)
            icon = "📁" if item.is_dir else "📄"
            print(f"   {icon} {rel_path}")
        
        print("\n" + "=" * 60)


# ============================================================
# Agent Demo (requires LLM)
# ============================================================

async def agent_demo():
    """Demonstrate FileSystem capability with an agent."""
    print("🤖 Agent with FileSystem Demo")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set - skipping agent demo")
        return
    
    from agenticflow.models import ChatModel
    from agenticflow import Agent
    from agenticflow.capabilities import FileSystem
    
    # Use the examples/data directory from the project
    examples_dir = Path(__file__).parent
    data_dir = examples_dir / "data"
    
    if not data_dir.exists():
        print(f"⚠️  Data directory not found: {data_dir}")
        return
    
    model = ChatModel(model="gpt-4o-mini", temperature=0)
    
    fs = FileSystem(
        allowed_paths=[str(data_dir.resolve())],
        allow_write=False,  # Read-only for safety
    )
    
    agent = Agent(
        name="FileAssistant",
        model=model,
        instructions=(
            f"You are a file analysis assistant. "
            f"The workspace directory is: {data_dir.resolve()}\n"
            f"Help users explore and understand the files. "
            f"Always use full absolute paths when calling file tools."
        ),
        capabilities=[fs],
    )
    
    print(f"\nAgent tools: {[t.name for t in fs.tools]}")
    print(f"Data directory: {data_dir.resolve()}")
    
    # Test queries
    queries = [
        f"List the files in {data_dir.resolve()}",
        f"Read the file {data_dir.resolve()}/company_knowledge.txt and tell me who works on what project",
    ]
    
    for query in queries:
        print(f"\n❓ {query}")
        print("-" * 40)
        response = await agent.run(query, strategy="dag")
        # Clean up response
        answer = response.replace("FINAL ANSWER:", "").strip()
        print(f"💡 {answer[:600]}")


# ============================================================
# Security Demo
# ============================================================

def security_demo():
    """Demonstrate FileSystem security features."""
    print("\n" + "=" * 60)
    print("🔒 FileSystem Security Demo")
    print("=" * 60)
    
    from agenticflow.capabilities import FileSystem
    
    with tempfile.TemporaryDirectory() as workspace:
        workspace_path = Path(workspace)
        allowed = workspace_path / "allowed"
        forbidden = workspace_path / "forbidden"
        allowed.mkdir()
        forbidden.mkdir()
        
        # Create test files
        (allowed / "public.txt").write_text("public data")
        (forbidden / "secret.txt").write_text("secret data")
        (allowed / ".env").write_text("API_KEY=secret123")
        
        # Create FileSystem with only allowed directory
        fs = FileSystem(allowed_paths=[str(allowed)])
        
        print("\n1️⃣  Testing path restriction:")
        print(f"   Allowed path: {allowed}")
        
        # Can read from allowed
        content = fs.read(str(allowed / "public.txt"))
        print(f"   ✓ Can read public.txt: '{content}'")
        
        # Cannot read from forbidden
        try:
            fs.read(str(forbidden / "secret.txt"))
            print("   ✗ Should not reach here!")
        except PermissionError as e:
            print(f"   ✓ Blocked access to forbidden/secret.txt")
        
        print("\n2️⃣  Testing deny patterns (.env files):")
        try:
            fs.read(str(allowed / ".env"))
            print("   ✗ Should not reach here!")
        except PermissionError:
            print("   ✓ Blocked access to .env file (default deny pattern)")
        
        print("\n3️⃣  Testing path traversal protection:")
        try:
            # Try to escape using ../
            fs.read(str(allowed / ".." / "forbidden" / "secret.txt"))
            print("   ✗ Should not reach here!")
        except PermissionError:
            print("   ✓ Blocked path traversal attempt")
        
        print("\n4️⃣  Testing read-only mode:")
        fs_readonly = FileSystem(
            allowed_paths=[str(allowed)],
            allow_write=False,
        )
        try:
            fs_readonly.write(str(allowed / "new.txt"), "content")
            print("   ✗ Should not reach here!")
        except PermissionError:
            print("   ✓ Write blocked in read-only mode")
        
        print("\n5️⃣  Testing delete protection (default):")
        fs_no_delete = FileSystem(allowed_paths=[str(allowed)])
        try:
            fs_no_delete.delete(str(allowed / "public.txt"))
            print("   ✗ Should not reach here!")
        except PermissionError:
            print("   ✓ Delete blocked by default")
        
        print("\n✅ All security checks passed!")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    programmatic_demo()
    security_demo()
    
    print("\n" + "=" * 60)
    asyncio.run(agent_demo())
    
    print("\n" + "=" * 60)
    print("✅ Summary:")
    print("   - FileSystem provides sandboxed file access")
    print("   - Configurable allowed paths and deny patterns")
    print("   - Supports read/write/search/list/copy/move/delete")
    print("   - Built-in security: path validation, traversal protection")
    print("=" * 60)
