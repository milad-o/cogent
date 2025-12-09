"""Tests for Spreadsheet, PDF, Browser, and Shell capabilities."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest


# =============================================================================
# Spreadsheet Capability Tests
# =============================================================================


class TestSpreadsheetCapability:
    """Tests for Spreadsheet capability."""

    def test_import(self) -> None:
        """Test that Spreadsheet can be imported."""
        from agenticflow.capabilities import Spreadsheet
        
        ss = Spreadsheet()
        assert ss is not None

    def test_get_tools(self) -> None:
        """Test that Spreadsheet provides expected tools."""
        from agenticflow.capabilities import Spreadsheet
        
        ss = Spreadsheet()
        tools = ss.get_tools()
        
        tool_names = {t.name for t in tools}
        assert "read_spreadsheet" in tool_names
        assert "write_spreadsheet" in tool_names
        assert "query_spreadsheet" in tool_names
        assert "aggregate_spreadsheet" in tool_names

    def test_read_csv(self) -> None:
        """Test reading a CSV file."""
        from agenticflow.capabilities import Spreadsheet
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV
            csv_path = Path(tmpdir) / "test.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "age", "city"])
                writer.writerow(["Alice", "30", "NYC"])
                writer.writerow(["Bob", "25", "LA"])
            
            ss = Spreadsheet(allowed_paths=[tmpdir])
            result = ss._read_csv(csv_path)
            
            assert result.row_count == 2
            assert result.columns == ["name", "age", "city"]
            assert result.data[0]["name"] == "Alice"
            assert result.data[1]["city"] == "LA"

    def test_write_csv(self) -> None:
        """Test writing a CSV file."""
        from agenticflow.capabilities import Spreadsheet
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "output.csv"
            
            ss = Spreadsheet(allowed_paths=[tmpdir])
            data = [
                {"name": "Alice", "score": 95},
                {"name": "Bob", "score": 87},
            ]
            
            result = ss._write_csv(csv_path, data)
            
            assert result.rows_written == 2
            assert csv_path.exists()
            
            # Verify content
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 2
                assert rows[0]["name"] == "Alice"

    def test_filter_data(self) -> None:
        """Test data filtering."""
        from agenticflow.capabilities import Spreadsheet
        
        ss = Spreadsheet()
        data = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
            {"name": "Charlie", "age": 35, "city": "NYC"},
        ]
        
        # Exact match filter
        result = ss._filter_data(data, {"city": "NYC"})
        assert len(result) == 2
        
        # Greater than filter
        result = ss._filter_data(data, {"age": {"$gt": 28}})
        assert len(result) == 2
        
        # Contains filter
        result = ss._filter_data(data, {"name": {"$contains": "li"}})
        assert len(result) == 2  # Alice and Charlie

    def test_aggregate_data(self) -> None:
        """Test data aggregation."""
        from agenticflow.capabilities import Spreadsheet
        
        ss = Spreadsheet()
        data = [
            {"category": "A", "value": 10},
            {"category": "A", "value": 20},
            {"category": "B", "value": 30},
        ]
        
        # Sum without grouping
        result = ss._aggregate_data(data, None, {"value": "sum"})
        assert result[0]["value_sum"] == 60
        
        # Sum with grouping
        result = ss._aggregate_data(data, ["category"], {"value": "sum"})
        assert len(result) == 2
        
        a_row = next(r for r in result if r.get("category") == "A")
        assert a_row["value_sum"] == 30

    def test_path_validation(self) -> None:
        """Test path validation."""
        from agenticflow.capabilities import Spreadsheet
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ss = Spreadsheet(allowed_paths=[tmpdir])
            
            # Valid path
            valid_path = ss._validate_path(Path(tmpdir) / "test.csv")
            assert valid_path is not None
            
            # Invalid path
            with pytest.raises(PermissionError):
                ss._validate_path("/etc/passwd")


# =============================================================================
# Browser Capability Tests
# =============================================================================


class TestBrowserCapability:
    """Tests for Browser capability."""

    def test_import(self) -> None:
        """Test that Browser can be imported."""
        from agenticflow.capabilities import Browser
        
        browser = Browser()
        assert browser is not None

    def test_get_tools(self) -> None:
        """Test that Browser provides expected tools."""
        from agenticflow.capabilities import Browser
        
        browser = Browser()
        tools = browser.get_tools()
        
        tool_names = {t.name for t in tools}
        assert "navigate" in tool_names
        assert "screenshot" in tool_names
        assert "extract_text" in tool_names
        assert "click" in tool_names
        assert "fill" in tool_names

    def test_url_validation_blocked(self) -> None:
        """Test URL validation with blocked domains."""
        from agenticflow.capabilities import Browser
        
        browser = Browser(blocked_domains=["evil.com"])
        
        # Allowed URL
        url = browser._validate_url("https://example.com")
        assert url == "https://example.com"
        
        # Blocked URL
        with pytest.raises(PermissionError):
            browser._validate_url("https://evil.com/page")

    def test_url_validation_allowed(self) -> None:
        """Test URL validation with allowed domains."""
        from agenticflow.capabilities import Browser
        
        browser = Browser(allowed_domains=["example.com"])
        
        # Allowed URL
        url = browser._validate_url("https://example.com/page")
        assert url == "https://example.com/page"
        
        # Not in allowed list
        with pytest.raises(PermissionError):
            browser._validate_url("https://other.com")

    def test_dependency_check(self) -> None:
        """Test dependency checking."""
        from agenticflow.capabilities import Browser
        
        browser = Browser()
        # Should raise ImportError if playwright not installed
        try:
            browser._require_playwright()
        except ImportError:
            pytest.skip("playwright not installed")


# =============================================================================
# Shell Capability Tests
# =============================================================================


class TestShellCapability:
    """Tests for Shell capability."""

    def test_import(self) -> None:
        """Test that Shell can be imported."""
        from agenticflow.capabilities import Shell
        
        shell = Shell()
        assert shell is not None

    def test_get_tools(self) -> None:
        """Test that Shell provides expected tools."""
        from agenticflow.capabilities import Shell
        
        shell = Shell()
        tools = shell.get_tools()
        
        tool_names = {t.name for t in tools}
        assert "run_command" in tool_names
        assert "run_script" in tool_names
        assert "get_env_var" in tool_names
        assert "get_working_dir" in tool_names
        assert "list_directory" in tool_names

    def test_command_validation_blocked(self) -> None:
        """Test command validation with blocked commands."""
        from agenticflow.capabilities import Shell
        from agenticflow.capabilities.shell import SecurityError
        
        shell = Shell()
        
        # Default blocked commands
        with pytest.raises(SecurityError):
            shell._validate_command("sudo ls")
        
        with pytest.raises(SecurityError):
            shell._validate_command("rm -rf /")

    def test_command_validation_allowed(self) -> None:
        """Test command validation with allowed commands."""
        from agenticflow.capabilities import Shell
        from agenticflow.capabilities.shell import SecurityError
        
        shell = Shell(allowed_commands=["ls", "cat", "echo"])
        
        # Allowed command
        cmd = shell._validate_command("ls -la")
        assert cmd == "ls -la"
        
        # Not in allowed list
        with pytest.raises(SecurityError):
            shell._validate_command("grep pattern file")

    def test_dangerous_patterns(self) -> None:
        """Test detection of dangerous patterns."""
        from agenticflow.capabilities import Shell
        from agenticflow.capabilities.shell import SecurityError
        
        shell = Shell(allowed_commands=[])  # Allow all for this test
        shell.blocked_commands = set()  # Clear blocked
        
        # Fork bomb
        with pytest.raises(SecurityError):
            shell._validate_command(":(){ :|:& };:")
        
        # Pipe to shell
        with pytest.raises(SecurityError):
            shell._validate_command("echo test | sh")

    def test_operator_restrictions(self) -> None:
        """Test operator restrictions."""
        from agenticflow.capabilities import Shell
        from agenticflow.capabilities.shell import SecurityError
        
        # Pipes allowed by default
        shell_with_pipes = Shell(allowed_commands=["ls", "grep"])
        cmd = shell_with_pipes._validate_command("ls | grep test")
        assert "|" in cmd
        
        # Pipes not allowed
        shell_no_pipes = Shell(allow_pipes=False)
        with pytest.raises(SecurityError):
            shell_no_pipes._validate_command("ls | grep test")
        
        # Redirects not allowed by default
        shell = Shell(allowed_commands=["echo"])
        with pytest.raises(SecurityError):
            shell._validate_command("echo test > file.txt")

    def test_path_validation(self) -> None:
        """Test path validation."""
        from agenticflow.capabilities import Shell
        from agenticflow.capabilities.shell import SecurityError
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # When using allowed_paths, working_dir must be in allowed_paths
            shell = Shell(allowed_paths=[tmpdir], working_dir=tmpdir)
            
            # Valid path (within allowed)
            valid_path = shell._validate_path(Path(tmpdir) / "subdir")
            assert valid_path is not None
            
            # Path outside allowed dirs should fail
            with pytest.raises(SecurityError):
                shell._validate_path(Path("/etc"))
        
        # Test without allowed_paths - should block system directories
        shell2 = Shell()
        with pytest.raises(SecurityError):
            shell2._validate_path(Path("/etc"))

    @pytest.mark.asyncio
    async def test_run_simple_command(self) -> None:
        """Test running a simple command."""
        from agenticflow.capabilities import Shell
        
        shell = Shell(allowed_commands=["echo"])
        result = await shell._run_command("echo hello")
        
        assert result.success
        assert "hello" in result.stdout
        assert result.return_code == 0

    @pytest.mark.asyncio
    async def test_run_command_timeout(self) -> None:
        """Test command timeout."""
        from agenticflow.capabilities import Shell
        
        shell = Shell(allowed_commands=["sleep"])
        result = await shell._run_command("sleep 10", timeout=1)
        
        assert not result.success
        assert result.timed_out

    def test_env_building(self) -> None:
        """Test environment building."""
        from agenticflow.capabilities import Shell
        
        shell = Shell(env_vars={"CUSTOM_VAR": "custom_value"})
        env = shell._build_env()
        
        assert "CUSTOM_VAR" in env
        assert env["CUSTOM_VAR"] == "custom_value"
        assert "PATH" in env
        
        # Dangerous vars should be removed
        assert "LD_PRELOAD" not in env


# =============================================================================
# Integration Tests
# =============================================================================


class TestCapabilitiesIntegration:
    """Integration tests for new capabilities."""

    def test_all_capabilities_importable(self) -> None:
        """Test that all new capabilities can be imported from main module."""
        from agenticflow.capabilities import (
            Spreadsheet,
            Browser,
            Shell,
        )
        
        assert Spreadsheet is not None
        assert Browser is not None
        assert Shell is not None

    def test_capabilities_have_get_tools(self) -> None:
        """Test that all capabilities implement get_tools."""
        from agenticflow.capabilities import (
            Spreadsheet,
            Browser,
            Shell,
        )
        
        for CapClass in [Spreadsheet, Browser, Shell]:
            cap = CapClass()
            tools = cap.get_tools()
            assert isinstance(tools, list)
            assert len(tools) > 0
