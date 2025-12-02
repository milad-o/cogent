"""
Shell capability - sandboxed terminal command execution.

Provides tools for running shell commands with security controls,
enabling agents to interact with the system in a controlled way.

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import Shell
    
    # Create with security restrictions
    shell = Shell(
        allowed_commands=["ls", "cat", "grep", "find", "wc"],
        blocked_commands=["rm", "sudo", "chmod"],
        allowed_paths=["/home/user/projects"],
        timeout_seconds=30,
    )
    
    agent = Agent(
        name="DevOps",
        model=model,
        capabilities=[shell],
    )
    
    # Agent can now run approved commands
    await agent.run("List all Python files in the project")
    await agent.run("Count lines of code in src/")
    ```
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agenticflow.capabilities.base import BaseCapability
from agenticflow.tools.base import tool


@dataclass
class CommandResult:
    """Result of command execution."""
    
    command: str
    stdout: str
    stderr: str
    return_code: int
    duration_ms: float
    success: bool
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timed_out: bool = False
    working_dir: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "stdout": self.stdout[:10000] + "..." if len(self.stdout) > 10000 else self.stdout,
            "stderr": self.stderr[:2000] + "..." if len(self.stderr) > 2000 else self.stderr,
            "return_code": self.return_code,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "timed_out": self.timed_out,
            "working_dir": self.working_dir,
            "truncated": len(self.stdout) > 10000 or len(self.stderr) > 2000,
        }


class SecurityError(Exception):
    """Raised when a command violates security constraints."""
    pass


class Shell(BaseCapability):
    """
    Shell capability for sandboxed command execution.
    
    Provides controlled access to shell commands with multiple layers
    of security:
    - Command allowlist/blocklist
    - Path restrictions
    - Timeout limits
    - Environment variable control
    - Dangerous pattern detection
    
    Args:
        allowed_commands: List of allowed command names (empty = all allowed with blocklist)
        blocked_commands: List of blocked command names
        allowed_paths: List of paths commands can access
        blocked_paths: List of paths commands cannot access
        timeout_seconds: Maximum execution time (default: 60)
        max_output_size: Maximum output size in bytes (default: 1MB)
        working_dir: Default working directory
        env_vars: Additional environment variables
        inherit_env: Inherit current environment (default: True)
        allow_pipes: Allow pipe (|) operations (default: True)
        allow_redirects: Allow redirect (>, >>) operations (default: False)
        allow_background: Allow background (&) operations (default: False)
        
    Tools provided:
        - run_command: Execute a shell command
        - run_script: Execute a shell script
        - get_env: Get environment variable
        - set_working_dir: Change working directory
        - list_processes: List running processes
    """
    
    @property
    def name(self) -> str:
        """Unique name for this capability."""
        return "shell"
    
    @property
    def tools(self) -> list:
        """Tools this capability provides to the agent."""
        return self.get_tools()
    
    # Dangerous commands that are always blocked
    ALWAYS_BLOCKED = {
        "rm -rf /",
        "rm -rf /*",
        "mkfs",
        "dd if=/dev/zero",
        ":(){ :|:& };:",  # Fork bomb
        "> /dev/sda",
        "chmod -R 777 /",
        "chown -R",
    }
    
    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",
        r">\s*/dev/sd",
        r"mkfs\.",
        r"dd\s+if=/dev/zero",
        r"chmod\s+-R\s+777\s+/",
        r":\(\)\s*\{",  # Fork bomb pattern
        r"\|\s*sh\b",   # Pipe to shell
        r"\|\s*bash\b",
        r"eval\s+",
        r"`.*`",  # Backtick command substitution (can be dangerous)
        r"\$\(.*\)",  # Command substitution
    ]
    
    # Default blocked commands
    DEFAULT_BLOCKED = {
        "sudo", "su", "doas",  # Privilege escalation
        "rm", "rmdir", "unlink",  # Deletion (can be allowed explicitly)
        "mkfs", "fdisk", "parted",  # Disk operations
        "shutdown", "reboot", "poweroff", "halt",  # System control
        "kill", "killall", "pkill",  # Process control
        "iptables", "ufw", "firewall-cmd",  # Firewall
        "useradd", "userdel", "usermod", "passwd",  # User management
        "crontab",  # Scheduled tasks
        "nc", "netcat", "ncat",  # Network tools (can be used maliciously)
        "curl", "wget",  # Network downloads (use WebSearch capability instead)
    }
    
    def __init__(
        self,
        allowed_commands: list[str] | None = None,
        blocked_commands: list[str] | None = None,
        allowed_paths: list[str | Path] | None = None,
        blocked_paths: list[str | Path] | None = None,
        timeout_seconds: int = 60,
        max_output_size: int = 1024 * 1024,  # 1MB
        working_dir: str | Path | None = None,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = True,
        allow_pipes: bool = True,
        allow_redirects: bool = False,
        allow_background: bool = False,
    ) -> None:
        self.allowed_commands = set(allowed_commands) if allowed_commands else set()
        self.blocked_commands = set(blocked_commands) if blocked_commands else self.DEFAULT_BLOCKED.copy()
        self.allowed_paths = [Path(p).resolve() for p in (allowed_paths or [])]
        self.blocked_paths = [Path(p).resolve() for p in (blocked_paths or ["/", "/etc", "/var", "/usr"])]
        self.timeout_seconds = timeout_seconds
        self.max_output_size = max_output_size
        self.working_dir = Path(working_dir).resolve() if working_dir else Path.cwd()
        self.env_vars = env_vars or {}
        self.inherit_env = inherit_env
        self.allow_pipes = allow_pipes
        self.allow_redirects = allow_redirects
        self.allow_background = allow_background
        
        # Validate working directory
        if self.allowed_paths:
            self._validate_path(self.working_dir)
    
    def _validate_path(self, path: Path) -> Path:
        """Validate that path is within allowed directories."""
        path = Path(path).resolve()
        
        # Check allowed paths first (if specified, path must be within allowed)
        if self.allowed_paths:
            allowed = False
            for allow in self.allowed_paths:
                try:
                    path.relative_to(allow)
                    allowed = True
                    break
                except ValueError:
                    if path == allow:
                        allowed = True
                        break
            
            if not allowed:
                msg = f"Path {path} is not within allowed directories"
                raise SecurityError(msg)
        
        # Check blocked paths (skip if path is in allowed - explicit allow wins)
        if not self.allowed_paths:
            for blocked in self.blocked_paths:
                if blocked == Path("/"):
                    # Special case: / blocks system directories but not user paths
                    # Only block if path is exactly / or is a system directory
                    if path == Path("/") or path.parts[1:2] in (("etc",), ("var",), ("usr",), ("bin",), ("sbin",), ("lib",), ("dev",), ("proc",), ("sys",)):
                        msg = f"Path {path} is a system directory"
                        raise SecurityError(msg)
                else:
                    try:
                        path.relative_to(blocked)
                        msg = f"Path {path} is within blocked directory {blocked}"
                        raise SecurityError(msg)
                    except ValueError:
                        pass
        
        return path
    
    def _validate_command(self, command: str) -> str:
        """Validate command against security rules."""
        import re
        
        # Check for always-blocked patterns
        for pattern in self.ALWAYS_BLOCKED:
            if pattern in command:
                msg = f"Command contains blocked pattern: {pattern}"
                raise SecurityError(msg)
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                msg = f"Command matches dangerous pattern"
                raise SecurityError(msg)
        
        # Check for disallowed operators
        if not self.allow_pipes and "|" in command:
            msg = "Pipe operations are not allowed"
            raise SecurityError(msg)
        
        if not self.allow_redirects and any(op in command for op in [">", ">>"]):
            msg = "Redirect operations are not allowed"
            raise SecurityError(msg)
        
        if not self.allow_background and "&" in command and not command.strip().endswith("&&"):
            # Check for background operator (but not &&)
            if re.search(r"[^&]&[^&]|[^&]&$", command):
                msg = "Background operations are not allowed"
                raise SecurityError(msg)
        
        # Extract base command
        try:
            parts = shlex.split(command)
            if not parts:
                msg = "Empty command"
                raise SecurityError(msg)
            
            base_cmd = parts[0]
            
            # Handle path-based commands
            if "/" in base_cmd:
                base_cmd = Path(base_cmd).name
            
        except ValueError as e:
            msg = f"Invalid command syntax: {e}"
            raise SecurityError(msg) from e
        
        # Check allowlist
        if self.allowed_commands and base_cmd not in self.allowed_commands:
            msg = f"Command '{base_cmd}' is not in allowed list"
            raise SecurityError(msg)
        
        # Check blocklist
        if base_cmd in self.blocked_commands:
            msg = f"Command '{base_cmd}' is blocked"
            raise SecurityError(msg)
        
        return command
    
    def _build_env(self) -> dict[str, str]:
        """Build environment for command execution."""
        if self.inherit_env:
            env = os.environ.copy()
        else:
            # Minimal environment
            env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": str(Path.home()),
                "USER": os.getenv("USER", "user"),
                "LANG": "en_US.UTF-8",
            }
        
        # Add custom env vars
        env.update(self.env_vars)
        
        # Remove potentially dangerous variables
        for key in ["LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"]:
            env.pop(key, None)
        
        return env
    
    async def _run_command(
        self,
        command: str,
        working_dir: Path | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Execute a shell command."""
        import time
        
        # Validate command
        command = self._validate_command(command)
        
        # Set working directory
        cwd = working_dir or self.working_dir
        if working_dir:
            cwd = self._validate_path(working_dir)
        
        # Build environment
        run_env = self._build_env()
        if env:
            run_env.update(env)
        
        # Set timeout (handle string conversion)
        if isinstance(timeout, str):
            timeout = int(timeout) if timeout else None
        timeout = timeout or self.timeout_seconds
        
        start_time = time.time()
        timed_out = False
        
        try:
            # Run command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=run_env,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                timed_out = True
                stdout = b""
                stderr = f"Command timed out after {timeout} seconds".encode()
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Decode output
            stdout_str = stdout.decode("utf-8", errors="replace")[:self.max_output_size]
            stderr_str = stderr.decode("utf-8", errors="replace")[:self.max_output_size]
            
            return CommandResult(
                command=command,
                stdout=stdout_str,
                stderr=stderr_str,
                return_code=process.returncode or 0,
                duration_ms=duration_ms,
                success=process.returncode == 0 and not timed_out,
                timed_out=timed_out,
                working_dir=str(cwd),
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return CommandResult(
                command=command,
                stdout="",
                stderr=str(e),
                return_code=-1,
                duration_ms=duration_ms,
                success=False,
                working_dir=str(cwd),
            )
    
    # =========================================================================
    # Tool Methods
    # =========================================================================
    
    def get_tools(self) -> list:
        """Return the tools provided by this capability."""
        
        @tool
        async def run_command(
            command: str,
            working_dir: str | None = None,
            timeout_seconds: int | None = None,
        ) -> dict[str, Any]:
            """Execute a shell command.
            
            Args:
                command: The shell command to execute
                working_dir: Working directory for the command (optional, defaults to configured working_dir)
                timeout_seconds: Maximum execution time (optional)
            
            Returns:
                Dictionary with stdout, stderr, return_code, and success status
            
            Note:
                Commands are validated against security rules before execution.
                Some commands may be blocked for security reasons.
            """
            # Resolve working directory relative to self.working_dir
            cwd = None
            if working_dir:
                wd_path = Path(working_dir)
                if not wd_path.is_absolute():
                    cwd = self.working_dir / wd_path
                else:
                    cwd = wd_path
            result = await self._run_command(command, cwd, timeout_seconds)
            return result.to_dict()
        
        @tool
        async def run_script(
            script: str,
            interpreter: str = "bash",
            working_dir: str | None = None,
            timeout_seconds: int | None = None,
        ) -> dict[str, Any]:
            """Execute a shell script.
            
            Args:
                script: The script content (multiple lines allowed)
                interpreter: Shell interpreter to use (bash, sh, zsh)
                working_dir: Working directory for the script (optional)
                timeout_seconds: Maximum execution time (optional)
            
            Returns:
                Dictionary with stdout, stderr, return_code, and success status
            """
            # Validate interpreter
            allowed_interpreters = {"bash", "sh", "zsh"}
            if interpreter not in allowed_interpreters:
                return {
                    "success": False,
                    "error": f"Interpreter must be one of: {allowed_interpreters}",
                }
            
            # Validate each line of the script
            for line in script.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        self._validate_command(line)
                    except SecurityError as e:
                        return {
                            "success": False,
                            "error": f"Script contains blocked command: {e}",
                        }
            
            # Build command with heredoc
            command = f"{interpreter} << 'EOF'\n{script}\nEOF"
            
            # Resolve working directory relative to self.working_dir
            cwd = None
            if working_dir:
                wd_path = Path(working_dir)
                if not wd_path.is_absolute():
                    cwd = self.working_dir / wd_path
                else:
                    cwd = wd_path
            result = await self._run_command(command, cwd, timeout_seconds)
            return result.to_dict()
        
        @tool
        def get_env_var(name: str) -> dict[str, Any]:
            """Get the value of an environment variable.
            
            Args:
                name: Name of the environment variable
            
            Returns:
                Dictionary with the variable name and value
            """
            env = self._build_env()
            value = env.get(name)
            
            return {
                "name": name,
                "value": value,
                "exists": value is not None,
            }
        
        @tool
        def get_working_dir() -> dict[str, Any]:
            """Get the current working directory.
            
            Returns:
                Dictionary with the current working directory path
            """
            return {
                "working_dir": str(self.working_dir),
                "exists": self.working_dir.exists(),
            }
        
        @tool
        def set_working_dir(path: str) -> dict[str, Any]:
            """Change the working directory for subsequent commands.
            
            Args:
                path: New working directory path
            
            Returns:
                Dictionary with the new working directory
            """
            try:
                new_path = self._validate_path(Path(path))
                if not new_path.exists():
                    return {
                        "success": False,
                        "error": f"Path does not exist: {path}",
                    }
                if not new_path.is_dir():
                    return {
                        "success": False,
                        "error": f"Path is not a directory: {path}",
                    }
                
                self.working_dir = new_path
                return {
                    "success": True,
                    "working_dir": str(new_path),
                }
            except SecurityError as e:
                return {
                    "success": False,
                    "error": str(e),
                }
        
        @tool
        async def which_command(command: str) -> dict[str, Any]:
            """Find the path of a command.
            
            Args:
                command: Command name to find
            
            Returns:
                Dictionary with the command path if found
            """
            result = await self._run_command(f"which {shlex.quote(command)}")
            
            if result.success and result.stdout.strip():
                return {
                    "command": command,
                    "path": result.stdout.strip(),
                    "found": True,
                }
            return {
                "command": command,
                "path": None,
                "found": False,
            }
        
        @tool
        async def list_directory(
            path: str | None = None,
            long_format: bool = False,
            show_hidden: bool = False,
        ) -> dict[str, Any]:
            """List contents of a directory.
            
            Args:
                path: Directory path (uses working dir if not specified)
                long_format: Show detailed information
                show_hidden: Show hidden files
            
            Returns:
                Dictionary with directory listing
            """
            target = path or str(self.working_dir)
            
            flags = []
            if long_format:
                flags.append("-l")
            if show_hidden:
                flags.append("-a")
            
            flag_str = " ".join(flags)
            command = f"ls {flag_str} {shlex.quote(target)}".strip()
            
            result = await self._run_command(command)
            
            return {
                "path": target,
                "contents": result.stdout,
                "success": result.success,
                "error": result.stderr if not result.success else None,
            }
        
        @tool
        def get_security_info() -> dict[str, Any]:
            """Get information about security restrictions.
            
            Returns:
                Dictionary with allowed/blocked commands and paths
            """
            return {
                "allowed_commands": list(self.allowed_commands) if self.allowed_commands else "all (except blocked)",
                "blocked_commands": list(self.blocked_commands),
                "allowed_paths": [str(p) for p in self.allowed_paths] if self.allowed_paths else "all (except blocked)",
                "blocked_paths": [str(p) for p in self.blocked_paths],
                "allow_pipes": self.allow_pipes,
                "allow_redirects": self.allow_redirects,
                "allow_background": self.allow_background,
                "timeout_seconds": self.timeout_seconds,
            }
        
        return [
            run_command,
            run_script,
            get_env_var,
            get_working_dir,
            set_working_dir,
            which_command,
            list_directory,
            get_security_info,
        ]
