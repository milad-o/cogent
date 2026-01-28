"""
Console Formatters - Rich console output for events.

Provides formatters for common event categories with colored,
human-readable output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cogent.observability.formatters.base import BaseFormatter

if TYPE_CHECKING:
    from cogent.observability.core.config import FormatConfig
    from cogent.observability.core.event import Event


# ANSI color codes
class Colors:
    """ANSI escape codes for terminal colors."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


class Styler:
    """Applies colors based on config."""

    def __init__(self, use_colors: bool = True) -> None:
        self.use_colors = use_colors

    def _wrap(self, text: str, *codes: str) -> str:
        if not self.use_colors:
            return text
        return "".join(codes) + text + Colors.RESET

    def agent(self, text: str) -> str:
        return self._wrap(text, Colors.MAGENTA)

    def tool(self, text: str) -> str:
        return self._wrap(text, Colors.YELLOW)

    def success(self, text: str) -> str:
        return self._wrap(text, Colors.GREEN)

    def error(self, text: str) -> str:
        return self._wrap(text, Colors.RED)

    def warning(self, text: str) -> str:
        return self._wrap(text, Colors.YELLOW)

    def info(self, text: str) -> str:
        return self._wrap(text, Colors.CYAN)

    def dim(self, text: str) -> str:
        return self._wrap(text, Colors.DIM)

    def bold(self, text: str) -> str:
        return self._wrap(text, Colors.BOLD)


class AgentFormatter(BaseFormatter):
    """Formats agent.* events."""

    patterns = ["agent.*"]

    def format(self, event: Event, config: FormatConfig) -> str | None:
        s = Styler(config.use_colors)
        agent_name = event.get("agent_name", "Agent")
        formatted_name = self.format_name(agent_name)

        action = event.action

        if action == "invoked":
            return f"{s.agent(formatted_name)} {s.dim('[starting]')}"

        elif action == "thinking":
            iteration = event.get("iteration", 1)
            iter_str = f" (iteration {iteration})" if iteration > 1 else ""
            return f"{s.agent(formatted_name)} {s.dim('[thinking]')}{s.dim(iter_str)}"

        elif action == "reasoning":
            thought = event.get("thought_preview", "")
            if thought:
                thought = self.truncate(thought, config.truncate or 200)
                return f"{s.agent(formatted_name)} {s.dim('[reasoning]')}\n  {s.dim(thought)}"
            return f"{s.agent(formatted_name)} {s.dim('[reasoning]')}"

        elif action == "acting":
            return f"{s.agent(formatted_name)} {s.dim('[acting]')}"

        elif action == "responded":
            duration_str = ""
            if config.show_duration and "duration_ms" in event.data:
                duration_str = f" ({self.format_duration(event.data['duration_ms'])})"

            # Token info
            tokens = event.get("tokens", {})
            token_str = ""
            if tokens and tokens.get("total"):
                total = tokens["total"]
                token_str = f" • {s.dim(f'{total} tokens')}"

            return f"{s.agent(formatted_name)} {s.success('[completed]')}{s.success(duration_str)}{token_str}"

        elif action == "error":
            error = event.get("error", "Unknown error")
            return (
                f"{s.agent(formatted_name)} {s.error('[error]')} {s.error(str(error))}"
            )

        # Default for unknown agent actions
        return f"{s.agent(formatted_name)} {s.dim(f'[{action}]')}"


class ToolFormatter(BaseFormatter):
    """Formats tool.* events."""

    patterns = ["tool.*"]

    def format(self, event: Event, config: FormatConfig) -> str | None:
        s = Styler(config.use_colors)
        tool_name = event.get("tool_name", event.get("tool", "tool"))
        agent_name = event.get("agent_name", "")
        call_id = event.get("call_id", "")
        
        # Short ID for display (first 8 chars)
        short_id = call_id[:8] if call_id else ""

        agent_prefix = f"{s.agent(self.format_name(agent_name))} " if agent_name else ""
        action = event.action

        if action == "called":
            args = event.get("args", {})
            id_str = f"{s.dim(short_id)} " if short_id else ""
            args_str = ""
            # Show args if truncate > 0 (verbose modes)
            if args and config.truncate > 0:
                # Format args as {key=value} pairs
                args_parts = [f"{k}={v!r}" for k, v in args.items()]
                args_preview = "{" + ", ".join(args_parts) + "}"
                args_preview = self.truncate(args_preview, config.truncate)
                args_str = f"\n  {s.dim(args_preview)}"
            return f"{agent_prefix}{s.dim('[tool-call]')} {id_str}{s.tool(tool_name)}{args_str}"

        elif action == "result":
            duration_str = ""
            if config.show_duration and "duration_ms" in event.data:
                duration_str = f" ({self.format_duration(event.data['duration_ms'])})"

            id_str = f"{s.dim(short_id)} " if short_id else ""
            result = event.get("result_preview", str(event.get("result", "")))
            result_str = ""
            if result:
                result = self.truncate(
                    str(result).replace("\n", " "), config.truncate or 100
                )
                # Try to format dict-like results nicely
                result_formatted = self._format_result(result)
                result_str = f"\n  {s.dim(result_formatted)}"

            return f"{agent_prefix}{s.success('[tool-result]')} {id_str}{s.tool(tool_name)}{s.success(duration_str)}{result_str}"

        elif action == "error":
            id_str = f"{s.dim(short_id)} " if short_id else ""
            error = event.get("error", "Unknown error")
            return f"{agent_prefix}{s.error('[tool-error]')} {id_str}{s.tool(tool_name)}: {s.error(str(error))}"

        return f"{agent_prefix}{s.dim(f'[tool.{action}]')} {s.tool(tool_name)}"

    def _format_result(self, result: str) -> str:
        """Format result value, handling dict-like strings."""
        import ast
        
        # Try to parse as Python literal (dict, list, etc.)
        try:
            parsed = ast.literal_eval(result)
            if isinstance(parsed, dict):
                # Format as {key='value', ...}
                parts = [f"{k}={v!r}" for k, v in parsed.items()]
                return "{" + ", ".join(parts) + "}"
            elif isinstance(parsed, (list, tuple)):
                return repr(parsed)
            else:
                return repr(parsed)
        except (ValueError, SyntaxError):
            # Not a Python literal, return as quoted string
            return repr(result)


class TaskFormatter(BaseFormatter):
    """Formats task.* events."""

    patterns = ["task.*"]

    def format(self, event: Event, config: FormatConfig) -> str | None:
        s = Styler(config.use_colors)
        action = event.action

        if action == "started":
            task = event.get("task_name", event.get("task", "task"))
            task = self.truncate(str(task), 80)
            return f"{s.success('> started')} {task}"

        elif action == "completed":
            duration_str = ""
            if config.show_duration and "duration_ms" in event.data:
                duration_str = f" ({self.format_duration(event.data['duration_ms'])})"
            return f"{s.success(f'[ok] completed{duration_str}')}"

        elif action == "failed":
            error = event.get("error", "Unknown error")
            error = self.truncate(str(error), 80)
            return f"{s.error(f'[X] failed: {error}')}"

        return f"{s.dim(f'[task.{action}]')}"


class StreamFormatter(BaseFormatter):
    """Formats stream.* events."""

    patterns = ["stream.*"]

    def format(self, event: Event, config: FormatConfig) -> str | None:
        s = Styler(config.use_colors)
        agent_name = event.get("agent_name", "Agent")
        formatted_name = self.format_name(agent_name)
        action = event.action

        if action == "start":
            model = event.get("model", "")
            model_str = f" {s.dim(f'({model})')}" if model else ""
            return f"{s.agent(formatted_name)} {s.info('> streaming...')}{model_str}"

        elif action == "token":
            # Return None to suppress individual tokens in formatted output
            # The observer handles token streaming separately
            return None

        elif action == "end":
            duration_str = ""
            if config.show_duration and "duration_ms" in event.data:
                ms = event.data["duration_ms"]
                token_count = event.get("token_count", 0)
                if token_count > 0 and ms > 0:
                    tok_per_sec = token_count / (ms / 1000)
                    duration_str = f" ({ms / 1000:.1f}s, {tok_per_sec:.0f} tok/s)"
                else:
                    duration_str = f" ({self.format_duration(ms)})"
            return f"{s.agent(formatted_name)} {s.success(f'[ok] stream complete{duration_str}')}"

        elif action == "error":
            error = event.get("error", "Stream error")
            return f"{s.agent(formatted_name)} {s.error(f'[X] stream error: {error}')}"

        return None


class DefaultFormatter(BaseFormatter):
    """Fallback formatter for unhandled events."""

    patterns = ["*"]

    def format(self, event: Event, config: FormatConfig) -> str | None:
        s = Styler(config.use_colors)
        data_preview = str(event.data)[:80] if event.data else ""
        return f"{s.dim(event.type)} {s.dim(data_preview)}"
