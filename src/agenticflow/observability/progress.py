"""
Progress & Output System for AgenticFlow
=========================================

A comprehensive, fully customizable output system for:
- Progress tracking with spinners, bars, and status updates
- Streaming output for async operations
- Structured logging with multiple verbosity levels
- DAG visualization in terminal
- Timeline and execution traces
- Multiple output formats (text, JSON, rich)

This module provides a unified API for all output needs,
abstracting away the complexity of formatting and streaming.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from io import StringIO
from typing import Any, TextIO, TypeVar, Generic

from agenticflow.core.utils import generate_id, now_utc


# ============================================================
# Enums and Configuration
# ============================================================

class Verbosity(IntEnum):
    """Output verbosity levels."""
    
    SILENT = 0      # No output
    MINIMAL = 1     # Only final results
    NORMAL = 2      # Key milestones
    VERBOSE = 3     # Detailed progress
    DEBUG = 4       # Everything including internals
    TRACE = 5       # Maximum detail


class OutputFormat(Enum):
    """Output format types."""
    
    TEXT = "text"           # Plain text
    RICH = "rich"           # Rich formatting (colors, Unicode)
    JSON = "json"           # JSON lines
    STRUCTURED = "structured"  # Structured data (for programmatic use)
    MINIMAL = "minimal"     # Minimal output


class ProgressStyle(Enum):
    """Progress indicator styles."""
    
    SPINNER = "spinner"     # Spinning indicator
    BAR = "bar"            # Progress bar
    DOTS = "dots"          # Dot sequence
    STEPS = "steps"        # Step counter
    PERCENT = "percent"    # Percentage
    NONE = "none"          # No progress indicator


class Theme(Enum):
    """Color themes for rich output."""
    
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    MINIMAL = "minimal"
    COLORFUL = "colorful"


@dataclass
class OutputConfig:
    """Configuration for the output system.
    
    Attributes:
        verbosity: How much detail to show.
        format: Output format type.
        progress_style: Style for progress indicators.
        theme: Color theme for rich output.
        show_timestamps: Include timestamps in output.
        show_duration: Show operation duration.
        show_agent_names: Show agent names in output.
        show_tool_names: Show tool names.
        show_dag: Visualize DAG structure.
        show_trace_ids: Show correlation IDs.
        truncate_results: Max chars for results (0 = no limit).
        stream: Output stream (default: stdout).
        indent: Indentation string.
        use_unicode: Use Unicode characters.
        use_colors: Use ANSI colors.
        spinner_chars: Characters for spinner animation.
    """
    
    verbosity: Verbosity = Verbosity.NORMAL
    format: OutputFormat = OutputFormat.RICH
    progress_style: ProgressStyle = ProgressStyle.SPINNER
    theme: Theme = Theme.DEFAULT
    show_timestamps: bool = False
    show_duration: bool = True
    show_agent_names: bool = True
    show_tool_names: bool = True
    show_dag: bool = False
    show_trace_ids: bool = False
    truncate_results: int = 200
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    indent: str = "  "
    use_unicode: bool = True
    use_colors: bool = True
    spinner_chars: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    
    @classmethod
    def minimal(cls) -> "OutputConfig":
        """Create minimal output config."""
        return cls(
            verbosity=Verbosity.MINIMAL,
            format=OutputFormat.MINIMAL,
            progress_style=ProgressStyle.NONE,
            show_timestamps=False,
            show_duration=False,
        )
    
    @classmethod
    def verbose(cls) -> "OutputConfig":
        """Create verbose output config."""
        return cls(
            verbosity=Verbosity.VERBOSE,
            format=OutputFormat.RICH,
            progress_style=ProgressStyle.SPINNER,
            show_timestamps=True,
            show_duration=True,
            show_dag=True,
        )
    
    @classmethod
    def debug(cls) -> "OutputConfig":
        """Create debug output config."""
        return cls(
            verbosity=Verbosity.DEBUG,
            format=OutputFormat.RICH,
            progress_style=ProgressStyle.STEPS,
            show_timestamps=True,
            show_duration=True,
            show_dag=True,
            show_trace_ids=True,
            truncate_results=0,
        )
    
    @classmethod
    def json(cls) -> "OutputConfig":
        """Create JSON output config."""
        return cls(
            verbosity=Verbosity.VERBOSE,
            format=OutputFormat.JSON,
            progress_style=ProgressStyle.NONE,
            show_timestamps=True,
            use_colors=False,
            use_unicode=False,
        )
    
    @classmethod
    def silent(cls) -> "OutputConfig":
        """Create silent config (no output)."""
        return cls(
            verbosity=Verbosity.SILENT,
            progress_style=ProgressStyle.NONE,
        )


# ============================================================
# Color and Styling
# ============================================================

class Colors:
    """ANSI color codes."""
    
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class Symbols:
    """Unicode symbols for output."""
    
    # Status indicators
    CHECK = "✓"
    CROSS = "✗"
    ARROW = "→"
    BULLET = "•"
    STAR = "★"
    CIRCLE = "○"
    FILLED_CIRCLE = "●"
    
    # Progress
    SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    BAR_FILLED = "█"
    BAR_EMPTY = "░"
    BAR_PARTIAL = ["▏", "▎", "▍", "▌", "▋", "▊", "▉"]
    
    # DAG/Tree
    BRANCH = "├──"
    LAST_BRANCH = "└──"
    PIPE = "│"
    CORNER = "└"
    TEE = "├"
    HORIZONTAL = "─"
    
    # Boxes
    BOX_TOP_LEFT = "┌"
    BOX_TOP_RIGHT = "┐"
    BOX_BOTTOM_LEFT = "└"
    BOX_BOTTOM_RIGHT = "┘"
    BOX_HORIZONTAL = "─"
    BOX_VERTICAL = "│"
    
    # ASCII fallbacks
    ASCII_CHECK = "[OK]"
    ASCII_CROSS = "[X]"
    ASCII_ARROW = "->"
    ASCII_BULLET = "*"
    ASCII_SPINNER = ["|", "/", "-", "\\"]
    ASCII_BAR_FILLED = "#"
    ASCII_BAR_EMPTY = "-"


class Styler:
    """Text styling utility."""
    
    def __init__(self, config: OutputConfig) -> None:
        self.config = config
        self._use_colors = config.use_colors and config.format == OutputFormat.RICH
        self._use_unicode = config.use_unicode
    
    def _wrap(self, text: str, *codes: str) -> str:
        """Wrap text in ANSI codes if colors enabled."""
        if not self._use_colors:
            return text
        return "".join(codes) + text + Colors.RESET
    
    def bold(self, text: str) -> str:
        return self._wrap(text, Colors.BOLD)
    
    def dim(self, text: str) -> str:
        return self._wrap(text, Colors.DIM)
    
    def success(self, text: str) -> str:
        return self._wrap(text, Colors.GREEN)
    
    def error(self, text: str) -> str:
        return self._wrap(text, Colors.RED)
    
    def warning(self, text: str) -> str:
        return self._wrap(text, Colors.YELLOW)
    
    def info(self, text: str) -> str:
        return self._wrap(text, Colors.CYAN)
    
    def agent(self, text: str) -> str:
        return self._wrap(text, Colors.MAGENTA, Colors.BOLD)
    
    def tool(self, text: str) -> str:
        return self._wrap(text, Colors.BLUE)
    
    def timestamp(self, text: str) -> str:
        return self._wrap(text, Colors.DIM)
    
    def sym(self, name: str) -> str:
        """Get symbol with Unicode/ASCII fallback."""
        if self._use_unicode:
            return getattr(Symbols, name, "?")
        return getattr(Symbols, f"ASCII_{name}", "?")
    
    def check(self) -> str:
        return self.success(self.sym("CHECK") if self._use_unicode else Symbols.ASCII_CHECK)
    
    def cross(self) -> str:
        return self.error(self.sym("CROSS") if self._use_unicode else Symbols.ASCII_CROSS)
    
    def arrow(self) -> str:
        return self.sym("ARROW") if self._use_unicode else Symbols.ASCII_ARROW


# ============================================================
# Event Types for Progress Tracking
# ============================================================

@dataclass
class ProgressEvent:
    """A progress event for tracking execution."""
    
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=now_utc)
    trace_id: str | None = None
    parent_id: str | None = None
    event_id: str = field(default_factory=generate_id)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.event_id,
            "type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ============================================================
# Base Renderer
# ============================================================

class BaseRenderer(ABC):
    """Base class for output renderers."""
    
    def __init__(self, config: OutputConfig) -> None:
        self.config = config
        self.styler = Styler(config)
        self._depth = 0
    
    @abstractmethod
    def render_event(self, event: ProgressEvent) -> str:
        """Render a progress event to string."""
        pass
    
    def write(self, text: str) -> None:
        """Write text to output stream."""
        if self.config.verbosity > Verbosity.SILENT:
            self.config.stream.write(text)
            self.config.stream.flush()
    
    def writeln(self, text: str = "") -> None:
        """Write text with newline."""
        self.write(text + "\n")
    
    def indent(self) -> str:
        """Get current indentation."""
        return self.config.indent * self._depth
    
    @contextmanager
    def indented(self):
        """Context manager for indentation."""
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1


class TextRenderer(BaseRenderer):
    """Plain text renderer."""
    
    def render_event(self, event: ProgressEvent) -> str:
        parts = []
        
        if self.config.show_timestamps:
            ts = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
            parts.append(f"[{ts}]")
        
        parts.append(f"{event.event_type}:")
        
        if event.data:
            data_str = " ".join(f"{k}={v}" for k, v in event.data.items())
            parts.append(data_str)
        
        return " ".join(parts)


class RichRenderer(BaseRenderer):
    """Rich text renderer with colors and symbols."""
    
    def render_event(self, event: ProgressEvent) -> str:
        parts = []
        s = self.styler
        
        # Timestamp
        if self.config.show_timestamps:
            ts = event.timestamp.strftime("%H:%M:%S")
            parts.append(s.timestamp(f"[{ts}]"))
        
        # Event type specific rendering
        event_type = event.event_type
        data = event.data
        
        if event_type == "start":
            parts.append(s.info("🚀"))
            parts.append(s.bold(data.get("title", "Starting")))
            
        elif event_type == "complete":
            parts.append(s.check())
            parts.append(s.success("Completed"))
            if self.config.show_duration and "duration_ms" in data:
                parts.append(s.dim(f"({data['duration_ms']:.0f}ms)"))
                
        elif event_type == "error":
            parts.append(s.cross())
            parts.append(s.error(f"Error: {data.get('message', 'Unknown')}"))
            
        elif event_type == "agent_start":
            agent = data.get("agent", "Unknown")
            parts.append(s.agent(f"[{agent}]"))
            parts.append("thinking...")
            
        elif event_type == "agent_complete":
            agent = data.get("agent", "Unknown")
            parts.append(s.agent(f"[{agent}]"))
            result = data.get("result", "")
            if self.config.truncate_results and len(result) > self.config.truncate_results:
                result = result[:self.config.truncate_results] + "..."
            parts.append(result)
            
        elif event_type == "tool_call":
            tool = data.get("tool", "Unknown")
            parts.append(s.arrow())
            parts.append(s.tool(tool))
            if "args" in data:
                parts.append(s.dim(f"({data['args']})"))
                
        elif event_type == "tool_result":
            parts.append(s.check())
            tool = data.get("tool", "")
            if tool:
                parts.append(s.tool(tool))
            result = str(data.get("result", ""))
            if self.config.truncate_results and len(result) > self.config.truncate_results:
                result = result[:self.config.truncate_results] + "..."
            parts.append(s.dim(result))
            
        elif event_type == "wave_start":
            wave = data.get("wave", 0)
            total = data.get("total_waves", 0)
            parallel = data.get("parallel_calls", 0)
            parts.append(s.info(f"⚡ Wave {wave}/{total}:"))
            parts.append(f"{parallel} parallel calls")
            
        elif event_type == "step":
            step = data.get("step", 0)
            total = data.get("total", 0)
            name = data.get("name", "")
            parts.append(f"[{step}/{total}]")
            parts.append(name)
            
        elif event_type == "progress":
            percent = data.get("percent", 0)
            message = data.get("message", "")
            bar = self._render_progress_bar(percent)
            parts.append(bar)
            if message:
                parts.append(message)
        
        # Resilience events
        elif event_type == "retry":
            tool = data.get("tool", "Unknown")
            attempt = data.get("attempt", 0)
            max_retries = data.get("max_retries", 0)
            delay = data.get("delay", 0)
            parts.append(s.warning("🔄"))
            parts.append(s.tool(tool))
            parts.append(f"retry {attempt}/{max_retries}")
            parts.append(s.dim(f"in {delay:.1f}s"))
            
        elif event_type == "circuit_open":
            tool = data.get("tool", "Unknown")
            reset_timeout = data.get("reset_timeout", 30)
            parts.append(s.error("⚡"))
            parts.append(s.tool(tool))
            parts.append(s.error("CIRCUIT OPEN"))
            parts.append(s.dim(f"(blocked for {reset_timeout}s)"))
            
        elif event_type == "circuit_close":
            tool = data.get("tool", "Unknown")
            parts.append(s.success("⚡"))
            parts.append(s.tool(tool))
            parts.append(s.success("circuit recovered"))
            
        elif event_type == "fallback":
            from_tool = data.get("from_tool", "?")
            to_tool = data.get("to_tool", "?")
            reason = data.get("reason", "")
            parts.append(s.warning("↩️"))
            parts.append(s.tool(from_tool))
            parts.append(s.dim("→"))
            parts.append(s.tool(to_tool))
            if reason:
                parts.append(s.dim(f"({reason})"))
                
        elif event_type == "recovery":
            tool = data.get("tool", "Unknown")
            method = data.get("method", "retry")
            parts.append(s.success("✅"))
            parts.append(s.tool(tool))
            parts.append(s.success("recovered"))
            parts.append(s.dim(f"via {method}"))
            if method == "retry" and "attempts" in data:
                parts.append(s.dim(f"({data['attempts']} attempts)"))
            elif method == "fallback" and "fallback_tool" in data:
                parts.append(s.dim(f"→ {data['fallback_tool']}"))
        
        else:
            # Generic rendering
            parts.append(f"{event_type}:")
            for k, v in data.items():
                parts.append(f"{k}={v}")
        
        return " ".join(parts)
    
    def _render_progress_bar(self, percent: float, width: int = 20) -> str:
        """Render a progress bar."""
        filled = int(width * percent / 100)
        empty = width - filled
        
        if self.styler._use_unicode:
            bar = Symbols.BAR_FILLED * filled + Symbols.BAR_EMPTY * empty
        else:
            bar = Symbols.ASCII_BAR_FILLED * filled + Symbols.ASCII_BAR_EMPTY * empty
        
        return f"[{bar}] {percent:.0f}%"


class JSONRenderer(BaseRenderer):
    """JSON lines renderer."""
    
    def render_event(self, event: ProgressEvent) -> str:
        return event.to_json()


class MinimalRenderer(BaseRenderer):
    """Minimal output renderer."""
    
    def render_event(self, event: ProgressEvent) -> str:
        if event.event_type in ("complete", "error", "result"):
            return str(event.data.get("result", event.data.get("message", "")))
        return ""


# ============================================================
# Progress Tracker
# ============================================================

class ProgressTracker:
    """
    Main progress tracking class.
    
    Provides a unified interface for all progress output needs.
    
    Example:
        >>> tracker = ProgressTracker()
        >>> 
        >>> # Simple usage
        >>> with tracker.task("Processing data"):
        ...     # do work
        ...     tracker.update("Step 1 complete")
        >>> 
        >>> # Async streaming
        >>> async for event in tracker.stream(async_operation()):
        ...     print(event)
        >>> 
        >>> # Custom config
        >>> tracker = ProgressTracker(OutputConfig.verbose())
    """
    
    def __init__(
        self,
        config: OutputConfig | None = None,
        name: str = "progress",
    ) -> None:
        """Initialize progress tracker.
        
        Args:
            config: Output configuration.
            name: Tracker name for logging.
        """
        self.config = config or OutputConfig()
        self.name = name
        self._renderer = self._create_renderer()
        self._events: list[ProgressEvent] = []
        self._start_time: datetime | None = None
        self._trace_id: str | None = None
        self._spinner_task: asyncio.Task | None = None
        self._spinner_active = False
    
    def _create_renderer(self) -> BaseRenderer:
        """Create renderer based on config."""
        renderers = {
            OutputFormat.TEXT: TextRenderer,
            OutputFormat.RICH: RichRenderer,
            OutputFormat.JSON: JSONRenderer,
            OutputFormat.MINIMAL: MinimalRenderer,
            OutputFormat.STRUCTURED: TextRenderer,
        }
        renderer_class = renderers.get(self.config.format, TextRenderer)
        return renderer_class(self.config)
    
    def _should_show(self, min_verbosity: Verbosity) -> bool:
        """Check if output should be shown at current verbosity."""
        return self.config.verbosity >= min_verbosity
    
    def _emit(self, event: ProgressEvent) -> None:
        """Emit a progress event."""
        self._events.append(event)
        
        if self._should_show(Verbosity.MINIMAL):
            output = self._renderer.render_event(event)
            if output:
                self._renderer.writeln(self._renderer.indent() + output)
    
    def start(self, title: str, **kwargs: Any) -> str:
        """Start a tracked operation.
        
        Args:
            title: Operation title.
            **kwargs: Additional event data.
            
        Returns:
            Trace ID for this operation.
        """
        self._trace_id = generate_id("trace")
        self._start_time = now_utc()
        
        event = ProgressEvent(
            event_type="start",
            data={"title": title, **kwargs},
            trace_id=self._trace_id,
        )
        self._emit(event)
        return self._trace_id
    
    def complete(self, result: Any = None, **kwargs: Any) -> None:
        """Mark operation as complete.
        
        Args:
            result: Operation result.
            **kwargs: Additional event data.
        """
        duration_ms = 0
        if self._start_time:
            duration_ms = (now_utc() - self._start_time).total_seconds() * 1000
        
        event = ProgressEvent(
            event_type="complete",
            data={"result": result, "duration_ms": duration_ms, **kwargs},
            trace_id=self._trace_id,
        )
        self._emit(event)
    
    def error(self, message: str, exception: Exception | None = None, **kwargs: Any) -> None:
        """Report an error.
        
        Args:
            message: Error message.
            exception: Optional exception.
            **kwargs: Additional event data.
        """
        data = {"message": message, **kwargs}
        if exception:
            data["exception_type"] = type(exception).__name__
            data["exception_message"] = str(exception)
        
        event = ProgressEvent(
            event_type="error",
            data=data,
            trace_id=self._trace_id,
        )
        self._emit(event)
    
    def update(self, message: str, **kwargs: Any) -> None:
        """Send a progress update.
        
        Args:
            message: Update message.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.NORMAL):
            event = ProgressEvent(
                event_type="update",
                data={"message": message, **kwargs},
                trace_id=self._trace_id,
            )
            self._emit(event)
    
    def step(self, step: int, total: int, name: str = "", **kwargs: Any) -> None:
        """Report step progress.
        
        Args:
            step: Current step number.
            total: Total steps.
            name: Step name.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.NORMAL):
            event = ProgressEvent(
                event_type="step",
                data={"step": step, "total": total, "name": name, **kwargs},
                trace_id=self._trace_id,
            )
            self._emit(event)
    
    def progress(self, percent: float, message: str = "", **kwargs: Any) -> None:
        """Report percentage progress.
        
        Args:
            percent: Progress percentage (0-100).
            message: Optional message.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.NORMAL):
            event = ProgressEvent(
                event_type="progress",
                data={"percent": percent, "message": message, **kwargs},
                trace_id=self._trace_id,
            )
            self._emit(event)
    
    def agent_start(self, agent: str, **kwargs: Any) -> None:
        """Report agent starting work.
        
        Args:
            agent: Agent name.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.NORMAL):
            event = ProgressEvent(
                event_type="agent_start",
                data={"agent": agent, **kwargs},
                trace_id=self._trace_id,
            )
            self._emit(event)
    
    def agent_complete(self, agent: str, result: str, **kwargs: Any) -> None:
        """Report agent completed.
        
        Args:
            agent: Agent name.
            result: Agent's result/output.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.NORMAL):
            event = ProgressEvent(
                event_type="agent_complete",
                data={"agent": agent, "result": result, **kwargs},
                trace_id=self._trace_id,
            )
            self._emit(event)
    
    def tool_call(self, tool: str, args: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Report tool being called.
        
        Args:
            tool: Tool name.
            args: Tool arguments.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.VERBOSE):
            event = ProgressEvent(
                event_type="tool_call",
                data={"tool": tool, "args": args or {}, **kwargs},
                trace_id=self._trace_id,
            )
            self._emit(event)
    
    def tool_result(self, tool: str, result: Any, **kwargs: Any) -> None:
        """Report tool result.
        
        Args:
            tool: Tool name.
            result: Tool result.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.VERBOSE):
            event = ProgressEvent(
                event_type="tool_result",
                data={"tool": tool, "result": result, **kwargs},
                trace_id=self._trace_id,
            )
            self._emit(event)
    
    def tool_error(self, tool: str, error: str, **kwargs: Any) -> None:
        """Report tool error.
        
        Args:
            tool: Tool name.
            error: Error message.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.MINIMAL):
            event = ProgressEvent(
                event_type="tool_error",
                data={"tool": tool, "error": error, **kwargs},
                trace_id=self._trace_id,
            )
            self._emit(event)
    
    # --- Resilience Events ---
    
    def retry(
        self,
        tool: str,
        attempt: int,
        max_retries: int,
        delay: float,
        error: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Report tool retry attempt.
        
        Args:
            tool: Tool name.
            attempt: Current attempt number.
            max_retries: Maximum retries configured.
            delay: Delay before retry in seconds.
            error: Error that triggered retry.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.NORMAL):
            event = ProgressEvent(
                event_type="retry",
                data={
                    "tool": tool,
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "delay": delay,
                    "error": error,
                    **kwargs,
                },
                trace_id=self._trace_id,
            )
            self._emit(event)
    
    def circuit_open(
        self,
        tool: str,
        failure_count: int,
        reset_timeout: float,
        **kwargs: Any,
    ) -> None:
        """Report circuit breaker opened.
        
        Args:
            tool: Tool name.
            failure_count: Number of failures that triggered opening.
            reset_timeout: Seconds until circuit attempts half-open.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.NORMAL):
            event = ProgressEvent(
                event_type="circuit_open",
                data={
                    "tool": tool,
                    "failure_count": failure_count,
                    "reset_timeout": reset_timeout,
                    **kwargs,
                },
                trace_id=self._trace_id,
            )
            self._emit(event)
    
    def circuit_close(self, tool: str, **kwargs: Any) -> None:
        """Report circuit breaker closed (recovered).
        
        Args:
            tool: Tool name.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.NORMAL):
            event = ProgressEvent(
                event_type="circuit_close",
                data={"tool": tool, **kwargs},
                trace_id=self._trace_id,
            )
            self._emit(event)
    
    def fallback(
        self,
        from_tool: str,
        to_tool: str,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Report fallback tool being used.
        
        Args:
            from_tool: Original tool that failed.
            to_tool: Fallback tool being tried.
            reason: Reason for fallback.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.NORMAL):
            event = ProgressEvent(
                event_type="fallback",
                data={
                    "from_tool": from_tool,
                    "to_tool": to_tool,
                    "reason": reason,
                    **kwargs,
                },
                trace_id=self._trace_id,
            )
            self._emit(event)
    
    def recovery(
        self,
        tool: str,
        method: str,
        attempts: int | None = None,
        fallback_tool: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Report successful recovery from failure.
        
        Args:
            tool: Original tool name.
            method: Recovery method used (retry, fallback, adapt).
            attempts: Number of attempts if recovered via retry.
            fallback_tool: Fallback tool used if recovered via fallback.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.NORMAL):
            event = ProgressEvent(
                event_type="recovery",
                data={
                    "tool": tool,
                    "method": method,
                    "attempts": attempts,
                    "fallback_tool": fallback_tool,
                    **kwargs,
                },
                trace_id=self._trace_id,
            )
            self._emit(event)

    def wave_start(self, wave: int, total_waves: int, parallel_calls: int, **kwargs: Any) -> None:
        """Report DAG wave starting.
        
        Args:
            wave: Current wave number.
            total_waves: Total waves.
            parallel_calls: Number of parallel calls in this wave.
            **kwargs: Additional event data.
        """
        if self._should_show(Verbosity.NORMAL):
            event = ProgressEvent(
                event_type="wave_start",
                data={
                    "wave": wave,
                    "total_waves": total_waves,
                    "parallel_calls": parallel_calls,
                    **kwargs,
                },
                trace_id=self._trace_id,
            )
            self._emit(event)
    
    def custom(self, event_type: str, **kwargs: Any) -> None:
        """Emit a custom event.
        
        Args:
            event_type: Event type name.
            **kwargs: Event data.
        """
        event = ProgressEvent(
            event_type=event_type,
            data=kwargs,
            trace_id=self._trace_id,
        )
        self._emit(event)
    
    @contextmanager
    def task(self, title: str, **kwargs: Any):
        """Context manager for tracking a task.
        
        Args:
            title: Task title.
            **kwargs: Additional event data.
            
        Example:
            >>> with tracker.task("Processing"):
            ...     # do work
        """
        trace_id = self.start(title, **kwargs)
        self._renderer._depth += 1
        try:
            yield trace_id
            self._renderer._depth -= 1
            self.complete()
        except Exception as e:
            self._renderer._depth -= 1
            self.error(str(e), exception=e)
            raise
    
    @asynccontextmanager
    async def async_task(self, title: str, **kwargs: Any):
        """Async context manager for tracking a task.
        
        Args:
            title: Task title.
            **kwargs: Additional event data.
        """
        trace_id = self.start(title, **kwargs)
        self._renderer._depth += 1
        try:
            yield trace_id
            self._renderer._depth -= 1
            self.complete()
        except Exception as e:
            self._renderer._depth -= 1
            self.error(str(e), exception=e)
            raise
    
    @contextmanager
    def spinner(self, message: str = "Working"):
        """Context manager showing a spinner (sync).
        
        Args:
            message: Message to show with spinner.
        """
        if (
            self.config.verbosity >= Verbosity.NORMAL
            and self.config.progress_style == ProgressStyle.SPINNER
            and self.config.format == OutputFormat.RICH
        ):
            chars = self.config.spinner_chars
            idx = 0
            done = False
            
            def spin():
                nonlocal idx
                while not done:
                    char = chars[idx % len(chars)]
                    self.config.stream.write(f"\r{char} {message}")
                    self.config.stream.flush()
                    idx += 1
                    time.sleep(0.1)
            
            import threading
            thread = threading.Thread(target=spin, daemon=True)
            thread.start()
            
            try:
                yield
            finally:
                done = True
                thread.join(timeout=0.2)
                self.config.stream.write(f"\r{self._renderer.styler.check()} {message}\n")
                self.config.stream.flush()
        else:
            yield
    
    async def stream_events(
        self,
        async_generator: AsyncIterator[dict[str, Any]],
    ) -> AsyncIterator[ProgressEvent]:
        """Stream progress events from an async generator.
        
        Args:
            async_generator: Async generator yielding state dicts.
            
        Yields:
            Progress events.
            
        Example:
            >>> async for event in tracker.stream_events(topology.stream(task)):
            ...     if event.event_type == "agent_complete":
            ...         print(f"Agent done: {event.data['agent']}")
        """
        self.start("Streaming execution")
        
        async for state in async_generator:
            # Convert state to events
            if "current_agent" in state and state.get("results"):
                results = state["results"]
                if results:
                    last = results[-1]
                    event = ProgressEvent(
                        event_type="agent_complete",
                        data={
                            "agent": last.get("agent", "Unknown"),
                            "result": last.get("thought", ""),
                        },
                        trace_id=self._trace_id,
                    )
                    self._emit(event)
                    yield event
        
        self.complete()
    
    def get_events(self) -> list[ProgressEvent]:
        """Get all recorded events."""
        return self._events.copy()
    
    def get_timeline(self) -> str:
        """Get a timeline visualization of events."""
        if not self._events:
            return "No events recorded"
        
        lines = ["Timeline:", "─" * 40]
        
        start_time = self._events[0].timestamp
        
        for event in self._events:
            delta = (event.timestamp - start_time).total_seconds()
            time_str = f"+{delta:.2f}s"
            
            if event.event_type == "start":
                symbol = "🚀"
            elif event.event_type == "complete":
                symbol = "✓"
            elif event.event_type == "error":
                symbol = "✗"
            elif event.event_type == "agent_complete":
                symbol = "👤"
            elif event.event_type == "tool_result":
                symbol = "🔧"
            else:
                symbol = "•"
            
            desc = event.data.get("title") or event.data.get("agent") or event.data.get("tool") or event.event_type
            lines.append(f"  {time_str:>8} {symbol} {desc}")
        
        return "\n".join(lines)
    
    def get_summary(self) -> dict[str, Any]:
        """Get execution summary."""
        if not self._events:
            return {}
        
        start = self._events[0].timestamp
        end = self._events[-1].timestamp
        duration = (end - start).total_seconds()
        
        event_counts: dict[str, int] = {}
        for event in self._events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        
        return {
            "total_events": len(self._events),
            "duration_seconds": duration,
            "event_counts": event_counts,
            "trace_id": self._trace_id,
        }


# ============================================================
# Callback Adapters
# ============================================================

def create_on_step_callback(
    tracker: ProgressTracker,
) -> Callable[[dict[str, Any]], None]:
    """Create an on_step callback for topology execution.
    
    Args:
        tracker: Progress tracker instance.
        
    Returns:
        Callback function for topology.run(on_step=...).
        
    Example:
        >>> tracker = ProgressTracker()
        >>> result = await topology.run(task, on_step=create_on_step_callback(tracker))
    """
    prev_results_len = 0
    
    def callback(state: dict[str, Any]) -> None:
        nonlocal prev_results_len
        
        results = state.get("results", [])
        if len(results) > prev_results_len:
            # New result added
            new_result = results[-1]
            tracker.agent_complete(
                agent=new_result.get("agent", "Unknown"),
                result=new_result.get("thought", ""),
            )
            prev_results_len = len(results)
    
    return callback


def create_executor_callback(
    tracker: ProgressTracker,
) -> Callable[[str, Any], None]:
    """Create an on_step callback for executor strategies.
    
    Args:
        tracker: Progress tracker instance.
        
    Returns:
        Callback function for agent.run(on_step=...).
    """
    def callback(step_type: str, data: Any) -> None:
        if step_type == "planning_dag":
            tracker.update("Building execution DAG...")
        elif step_type == "dag_waves":
            tracker.update(f"Plan: {data['waves']} waves, {data['total_calls']} calls")
        elif step_type == "executing_wave":
            tracker.wave_start(
                wave=data["wave"],
                total_waves=data["total_waves"],
                parallel_calls=data["parallel_calls"],
            )
        elif step_type == "act":
            tracker.tool_call(data.get("tool", ""), data.get("args"))
        elif step_type == "think":
            tracker.update(f"Thinking (iteration {data.get('iteration', 0)})...")
        elif step_type == "strategy_selected":
            tracker.update(f"Strategy: {data.get('strategy', 'unknown')}")
        else:
            tracker.custom(step_type, **data if isinstance(data, dict) else {"value": data})
    
    return callback


# ============================================================
# Convenience Functions
# ============================================================

# Global default tracker
_default_tracker: ProgressTracker | None = None


def get_tracker() -> ProgressTracker:
    """Get or create the default progress tracker."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = ProgressTracker()
    return _default_tracker


def set_tracker(tracker: ProgressTracker) -> None:
    """Set the default progress tracker."""
    global _default_tracker
    _default_tracker = tracker


def configure_output(
    verbosity: Verbosity = Verbosity.NORMAL,
    format: OutputFormat = OutputFormat.RICH,
    **kwargs: Any,
) -> ProgressTracker:
    """Configure and return a progress tracker.
    
    Args:
        verbosity: Output verbosity level.
        format: Output format.
        **kwargs: Additional OutputConfig options.
        
    Returns:
        Configured ProgressTracker.
        
    Example:
        >>> tracker = configure_output(Verbosity.VERBOSE, show_timestamps=True)
        >>> with tracker.task("My operation"):
        ...     pass
    """
    config = OutputConfig(verbosity=verbosity, format=format, **kwargs)
    tracker = ProgressTracker(config)
    set_tracker(tracker)
    return tracker


# ============================================================
# DAG Visualization
# ============================================================

def render_dag_ascii(
    nodes: list[str],
    edges: list[tuple[str, str]],
    node_status: dict[str, str] | None = None,
) -> str:
    """Render a DAG as ASCII art.
    
    Args:
        nodes: List of node names.
        edges: List of (from, to) edge tuples.
        node_status: Optional dict of node -> status.
        
    Returns:
        ASCII representation of the DAG.
    """
    if not nodes:
        return ""
    
    # Build adjacency for levels
    incoming: dict[str, set[str]] = {n: set() for n in nodes}
    outgoing: dict[str, set[str]] = {n: set() for n in nodes}
    
    for src, dst in edges:
        if src in outgoing and dst in incoming:
            outgoing[src].add(dst)
            incoming[dst].add(src)
    
    # Topological sort into levels
    levels: list[list[str]] = []
    remaining = set(nodes)
    completed: set[str] = set()
    
    while remaining:
        # Find nodes with no incomplete dependencies
        level = [
            n for n in remaining
            if all(dep in completed for dep in incoming[n])
        ]
        if not level:
            break
        levels.append(level)
        completed.update(level)
        remaining -= set(level)
    
    # Render
    lines = []
    status_symbols = {
        "pending": "○",
        "running": "◐",
        "completed": "●",
        "failed": "✗",
    }
    
    for i, level in enumerate(levels):
        level_nodes = []
        for node in level:
            status = (node_status or {}).get(node, "pending")
            symbol = status_symbols.get(status, "○")
            level_nodes.append(f"{symbol} {node}")
        
        lines.append(f"  Level {i + 1}: {' | '.join(level_nodes)}")
        
        if i < len(levels) - 1:
            lines.append("      │")
            lines.append("      ▼")
    
    return "\n".join(lines)
