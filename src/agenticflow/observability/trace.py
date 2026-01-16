"""
Deep Execution Tracing - See EVERYTHING that happens.

This module provides comprehensive execution tracing that shows:
- Execution graph structure (nodes, edges, dependencies)
- Node-level state transitions
- Tool calls with full arguments and results
- Timing breakdowns at every level
- State snapshots at each step
- Real-time streaming of execution progress

Example:
    ```python
    from agenticflow import Flow, Agent
    from agenticflow.observability import ExecutionTracer

    # Create tracer with full visibility
    tracer = ExecutionTracer()

    # Attach to flow
    flow = Flow(
        name="my-flow",
        agents=[agent1, agent2],
        topology="pipeline",
    )
    flow.attach_tracer(tracer)

    # Run and see everything
    result = await flow.run("Do something")

    # Get comprehensive trace
    print(tracer.summary())
    print(tracer.graph())  # ASCII visualization
    print(tracer.timeline())  # Detailed timeline

    # Export for analysis
    tracer.export_json("trace.json")
    tracer.export_mermaid("trace.mmd")
    ```
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TextIO

from agenticflow.core.utils import generate_id, now_utc


class TraceLevel(Enum):
    """Trace detail levels."""

    MINIMAL = 1    # Only final results
    NODES = 2      # Node transitions
    TOOLS = 3      # Tool calls
    FULL = 4       # Everything including state
    DEBUG = 5      # Maximum detail

    def __ge__(self, other: TraceLevel) -> bool:
        return self.value >= other.value

    def __gt__(self, other: TraceLevel) -> bool:
        return self.value > other.value

    def __le__(self, other: TraceLevel) -> bool:
        return self.value <= other.value

    def __lt__(self, other: TraceLevel) -> bool:
        return self.value < other.value


class NodeType(Enum):
    """Types of execution nodes."""

    START = "start"
    END = "end"
    AGENT = "agent"
    TOOL = "tool"
    DECISION = "decision"
    PARALLEL = "parallel"
    JOIN = "join"
    SUBGRAPH = "subgraph"


class NodeStatus(Enum):
    """Node execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ToolTrace:
    """Trace of a tool invocation."""

    id: str = field(default_factory=generate_id)
    tool_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float = 0

    # Context
    agent_name: str | None = None
    node_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tool": self.tool_name,
            "args": self.args,
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "agent": self.agent_name,
        }


@dataclass
class NodeTrace:
    """Trace of an execution node (agent step, tool call, decision point)."""

    id: str = field(default_factory=generate_id)
    name: str = ""
    type: NodeType = NodeType.AGENT
    status: NodeStatus = NodeStatus.PENDING

    # Input/Output
    input: Any = None
    output: Any = None
    error: str | None = None

    # Structure
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float = 0

    # Nested traces
    tool_traces: list[ToolTrace] = field(default_factory=list)
    state_snapshots: list[dict] = field(default_factory=list)

    # Metadata
    agent_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "status": self.status.value,
            "input": str(self.input)[:200] if self.input else None,
            "output": str(self.output)[:500] if self.output else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "tools": [t.to_dict() for t in self.tool_traces],
            "children": self.children,
            "depends_on": self.depends_on,
            "agent": self.agent_name,
        }


@dataclass
class ExecutionSpan:
    """A span in the execution trace (like OpenTelemetry spans)."""

    id: str = field(default_factory=generate_id)
    name: str = ""
    kind: str = "internal"  # internal, agent, tool, llm, network

    # Hierarchy
    parent_id: str | None = None
    trace_id: str | None = None

    # Timing
    started_at: datetime = field(default_factory=now_utc)
    ended_at: datetime | None = None
    duration_ms: float = 0

    # Data
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None

    def end(self, status: str = "ok", error: str | None = None) -> None:
        """End the span."""
        self.ended_at = now_utc()
        self.duration_ms = (self.ended_at - self.started_at).total_seconds() * 1000
        self.status = status
        self.error = error

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": now_utc().isoformat(),
            "attributes": attributes or {},
        })

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "error": self.error,
        }


class ExecutionTracer:
    """
    Deep execution tracer - see EVERYTHING that happens.

    This is the comprehensive observability solution that tracks:

    1. **Execution Graph** - The structure of what's running
       - Nodes (agents, tools, decisions)
       - Edges (dependencies, control flow)
       - Parallel branches

    2. **Node Execution** - What each node does
       - Input/output at each step
       - Status transitions
       - Timing breakdown

    3. **Tool Calls** - Every tool invocation
       - Arguments passed
       - Results returned
       - Errors encountered

    4. **State Snapshots** - State at each point
       - Before/after each node
       - Variable values
       - Context changes

    5. **Spans** - OpenTelemetry-style spans
       - Hierarchical timing
       - Cross-cutting concerns
       - Export to tracing backends

    Example - Basic Usage:
        ```python
        tracer = ExecutionTracer()

        # Manual tracing
        with tracer.span("my_operation") as span:
            span.set_attribute("key", "value")
            result = do_something()
            span.add_event("checkpoint", {"data": result})

        # View results
        print(tracer.summary())
        ```

    Example - Flow Integration:
        ```python
        tracer = ExecutionTracer(level=TraceLevel.FULL)

        flow = Flow(agents=[...], tracer=tracer)
        await flow.run("task")

        # ASCII graph
        print(tracer.graph())

        # Timeline
        print(tracer.timeline())

        # Export
        tracer.export_json("trace.json")
        ```

    Example - Streaming:
        ```python
        tracer = ExecutionTracer(
            on_node=lambda n: print(f"Node: {n.name} -> {n.status}"),
            on_tool=lambda t: print(f"Tool: {t.tool_name}({t.args})"),
            on_span=lambda s: print(f"Span: {s.name} {s.duration_ms}ms"),
        )
        ```
    """

    def __init__(
        self,
        level: TraceLevel = TraceLevel.TOOLS,
        *,
        # Output
        stream: TextIO | None = None,
        live: bool = True,  # Print events as they happen
        colors: bool = True,

        # Callbacks
        on_node: Callable[[NodeTrace], None] | None = None,
        on_tool: Callable[[ToolTrace], None] | None = None,
        on_span: Callable[[ExecutionSpan], None] | None = None,
        on_state: Callable[[str, dict], None] | None = None,
    ):
        """
        Create an execution tracer.

        Args:
            level: Detail level for tracing.
            stream: Output stream (default: stdout).
            live: Whether to print events as they happen.
            colors: Whether to use ANSI colors.
            on_node: Callback when a node starts/completes.
            on_tool: Callback when a tool is called.
            on_span: Callback when a span ends.
            on_state: Callback when state changes.
        """
        self.level = level
        self.stream = stream or sys.stdout
        self.live = live
        self.colors = colors

        # Callbacks
        self.on_node = on_node
        self.on_tool = on_tool
        self.on_span = on_span
        self.on_state = on_state

        # Storage
        self._trace_id = generate_id()
        self._nodes: dict[str, NodeTrace] = {}
        self._tools: list[ToolTrace] = []
        self._spans: list[ExecutionSpan] = []
        self._span_stack: list[ExecutionSpan] = []
        self._edges: list[tuple[str, str, str]] = []  # (from, to, label)

        # Timing
        self._started_at: datetime | None = None
        self._ended_at: datetime | None = None

        # Counters
        self._stats: dict[str, Any] = defaultdict(int)

    # ==================== Span API (OpenTelemetry-style) ====================

    def start_span(
        self,
        name: str,
        kind: str = "internal",
        attributes: dict | None = None,
    ) -> ExecutionSpan:
        """Start a new span.

        Args:
            name: Span name.
            kind: Span kind (internal, agent, tool, llm, network).
            attributes: Initial attributes.

        Returns:
            The created span.
        """
        parent = self._span_stack[-1] if self._span_stack else None

        span = ExecutionSpan(
            name=name,
            kind=kind,
            parent_id=parent.id if parent else None,
            trace_id=self._trace_id,
            attributes=attributes or {},
        )

        self._spans.append(span)
        self._span_stack.append(span)
        self._stats["spans_started"] += 1

        if self.live and self.level >= TraceLevel.NODES:
            self._print_span_start(span)

        return span

    def end_span(
        self,
        span: ExecutionSpan | None = None,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        """End a span.

        Args:
            span: The span to end (default: current span).
            status: Final status.
            error: Error message if failed.
        """
        if span is None:
            if not self._span_stack:
                return
            span = self._span_stack.pop()
        elif span in self._span_stack:
            self._span_stack.remove(span)

        span.end(status, error)
        self._stats["spans_completed"] += 1

        if self.live and self.level >= TraceLevel.NODES:
            self._print_span_end(span)

        if self.on_span:
            self.on_span(span)

    class SpanContext:
        """Context manager for spans."""

        def __init__(self, tracer: ExecutionTracer, span: ExecutionSpan):
            self.tracer = tracer
            self.span = span

        def __enter__(self) -> ExecutionSpan:
            return self.span

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            if exc_val:
                self.tracer.end_span(self.span, "error", str(exc_val))
            else:
                self.tracer.end_span(self.span)

    def span(
        self,
        name: str,
        kind: str = "internal",
        attributes: dict | None = None,
    ) -> SpanContext:
        """Create a span context manager.

        Example:
            with tracer.span("my_operation") as s:
                s.add_event("started")
                do_work()
        """
        span = self.start_span(name, kind, attributes)
        return self.SpanContext(self, span)

    # ==================== Node API ====================

    def start_node(
        self,
        name: str,
        node_type: NodeType = NodeType.AGENT,
        *,
        input: Any = None,
        parent_id: str | None = None,
        depends_on: list[str] | None = None,
        agent_name: str | None = None,
        metadata: dict | None = None,
    ) -> NodeTrace:
        """Start tracking a node execution.

        Args:
            name: Node name.
            node_type: Type of node.
            input: Input to the node.
            parent_id: Parent node ID.
            depends_on: Dependencies.
            agent_name: Name of the agent (if agent node).
            metadata: Additional metadata.

        Returns:
            The node trace.
        """
        node = NodeTrace(
            name=name,
            type=node_type,
            status=NodeStatus.RUNNING,
            input=input,
            parent_id=parent_id,
            depends_on=depends_on or [],
            agent_name=agent_name,
            metadata=metadata or {},
            started_at=now_utc(),
        )

        self._nodes[node.id] = node
        self._stats["nodes_started"] += 1
        self._stats[f"nodes.{node_type.value}"] += 1

        # Track edges
        for dep in node.depends_on:
            self._edges.append((dep, node.id, "depends"))
        if parent_id:
            self._edges.append((parent_id, node.id, "child"))

        if self.live and self.level >= TraceLevel.NODES:
            self._print_node_start(node)

        if self.on_node:
            self.on_node(node)

        return node

    def end_node(
        self,
        node: NodeTrace,
        output: Any = None,
        status: NodeStatus = NodeStatus.COMPLETED,
        error: str | None = None,
    ) -> None:
        """End a node execution.

        Args:
            node: The node to end.
            output: Node output.
            status: Final status.
            error: Error message if failed.
        """
        node.completed_at = now_utc()
        node.duration_ms = (node.completed_at - node.started_at).total_seconds() * 1000
        node.output = output
        node.status = status
        node.error = error

        self._stats["nodes_completed"] += 1
        if status == NodeStatus.FAILED:
            self._stats["nodes_failed"] += 1

        if self.live and self.level >= TraceLevel.NODES:
            self._print_node_end(node)

        if self.on_node:
            self.on_node(node)

    def snapshot_state(self, node: NodeTrace, state: dict) -> None:
        """Capture a state snapshot for a node.

        Args:
            node: The node.
            state: State to capture.
        """
        if self.level >= TraceLevel.FULL:
            node.state_snapshots.append({
                "timestamp": now_utc().isoformat(),
                "state": state,
            })

            if self.on_state:
                self.on_state(node.id, state)

    # ==================== Tool API ====================

    def start_tool(
        self,
        tool_name: str,
        args: dict,
        *,
        agent_name: str | None = None,
        node_id: str | None = None,
    ) -> ToolTrace:
        """Start tracking a tool call.

        Args:
            tool_name: Name of the tool.
            args: Tool arguments.
            agent_name: Calling agent.
            node_id: Parent node ID.

        Returns:
            The tool trace.
        """
        trace = ToolTrace(
            tool_name=tool_name,
            args=args,
            agent_name=agent_name,
            node_id=node_id,
            started_at=now_utc(),
        )

        self._tools.append(trace)
        self._stats["tools_called"] += 1
        self._stats[f"tool.{tool_name}"] += 1

        # Add to parent node if exists
        if node_id and node_id in self._nodes:
            self._nodes[node_id].tool_traces.append(trace)

        if self.live and self.level >= TraceLevel.TOOLS:
            self._print_tool_start(trace)

        return trace

    def end_tool(
        self,
        trace: ToolTrace,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """End a tool trace.

        Args:
            trace: The tool trace.
            result: Tool result.
            error: Error message if failed.
        """
        trace.completed_at = now_utc()
        trace.duration_ms = (trace.completed_at - trace.started_at).total_seconds() * 1000
        trace.result = result
        trace.error = error

        self._stats["tools_completed"] += 1
        if error:
            self._stats["tools_failed"] += 1

        if self.live and self.level >= TraceLevel.TOOLS:
            self._print_tool_end(trace)

        if self.on_tool:
            self.on_tool(trace)

    # ==================== Query API ====================

    def get_node(self, node_id: str) -> NodeTrace | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_nodes(
        self,
        node_type: NodeType | None = None,
        status: NodeStatus | None = None,
    ) -> list[NodeTrace]:
        """Get nodes matching criteria."""
        nodes = list(self._nodes.values())
        if node_type:
            nodes = [n for n in nodes if n.type == node_type]
        if status:
            nodes = [n for n in nodes if n.status == status]
        return nodes

    def get_tools(self, tool_name: str | None = None) -> list[ToolTrace]:
        """Get tool traces."""
        if tool_name:
            return [t for t in self._tools if t.tool_name == tool_name]
        return self._tools

    def get_spans(self, kind: str | None = None) -> list[ExecutionSpan]:
        """Get spans."""
        if kind:
            return [s for s in self._spans if s.kind == kind]
        return self._spans

    # ==================== Visualization ====================

    def graph(self) -> str:
        """Get ASCII visualization of the execution graph.

        Returns:
            ASCII art representation of the graph.
        """
        lines = ["", "Execution Graph", "=" * 50]

        if not self._nodes:
            lines.append("  (no nodes)")
            return "\n".join(lines)

        # Build adjacency list
        children: dict[str, list[str]] = defaultdict(list)
        roots: list[str] = []

        for node_id, node in self._nodes.items():
            if not node.parent_id and not node.depends_on:
                roots.append(node_id)
            if node.parent_id:
                children[node.parent_id].append(node_id)
            for dep in node.depends_on:
                children[dep].append(node_id)

        # DFS print
        def print_node(node_id: str, indent: int = 0, prefix: str = "") -> None:
            node = self._nodes.get(node_id)
            if not node:
                return

            # Status indicator
            status_icon = {
                NodeStatus.PENDING: "○",
                NodeStatus.RUNNING: "●",
                NodeStatus.COMPLETED: "✓",
                NodeStatus.FAILED: "✗",
                NodeStatus.SKIPPED: "⊘",
            }.get(node.status, "?")

            # Node type
            type_str = {
                NodeType.AGENT: "🤖",
                NodeType.TOOL: "🔧",
                NodeType.DECISION: "◇",
                NodeType.PARALLEL: "⫴",
                NodeType.JOIN: "⊕",
            }.get(node.type, "□")

            # Duration
            duration_str = f" ({node.duration_ms:.0f}ms)" if node.duration_ms else ""

            lines.append(f"{prefix}{type_str} {status_icon} {node.name}{duration_str}")

            # Show tools
            if node.tool_traces and self.level >= TraceLevel.TOOLS:
                for tool in node.tool_traces:
                    tool_status = "✓" if not tool.error else "✗"
                    lines.append(f"{prefix}  └─🔧 {tool_status} {tool.tool_name} ({tool.duration_ms:.0f}ms)")

            # Children
            child_list = children.get(node_id, [])
            for i, child_id in enumerate(child_list):
                is_last = i == len(child_list) - 1
                child_prefix = prefix + ("    " if is_last else "│   ")
                lines.append(f"{prefix}{'└──' if is_last else '├──'}")
                print_node(child_id, indent + 1, child_prefix)

        for root_id in roots:
            print_node(root_id)

        return "\n".join(lines)

    def timeline(self) -> str:
        """Get timeline visualization.

        Returns:
            Timeline string.
        """
        lines = ["", "Execution Timeline", "=" * 50]

        if not self._nodes:
            lines.append("  (no events)")
            return "\n".join(lines)

        # Sort by start time
        events = []
        for node in self._nodes.values():
            if node.started_at:
                events.append(("start", node.started_at, node))
            if node.completed_at:
                events.append(("end", node.completed_at, node))

        for tool in self._tools:
            if tool.started_at:
                events.append(("tool_start", tool.started_at, tool))
            if tool.completed_at:
                events.append(("tool_end", tool.completed_at, tool))

        events.sort(key=lambda x: x[1])

        if not events:
            lines.append("  (no events)")
            return "\n".join(lines)

        start_time = events[0][1]

        for event_type, timestamp, obj in events:
            delta = (timestamp - start_time).total_seconds()
            time_str = f"+{delta:>6.2f}s"

            if event_type == "start":
                lines.append(f"  {time_str}  ▶ {obj.name} started")
            elif event_type == "end":
                status = "✓" if obj.status == NodeStatus.COMPLETED else "✗"
                lines.append(f"  {time_str}  {status} {obj.name} ({obj.duration_ms:.0f}ms)")
            elif event_type == "tool_start":
                lines.append(f"  {time_str}    → {obj.tool_name}({_truncate(str(obj.args), 50)})")
            elif event_type == "tool_end":
                status = "✓" if not obj.error else "✗"
                lines.append(f"  {time_str}    {status} {obj.tool_name} ({obj.duration_ms:.0f}ms)")

        return "\n".join(lines)

    def summary(self) -> str:
        """Get summary of the trace.

        Returns:
            Summary string.
        """
        lines = ["", "Trace Summary", "=" * 50]

        total_duration = 0
        if self._nodes:
            started = min((n.started_at for n in self._nodes.values() if n.started_at), default=None)
            ended = max((n.completed_at for n in self._nodes.values() if n.completed_at), default=None)
            if started and ended:
                total_duration = (ended - started).total_seconds() * 1000

        lines.append(f"  Trace ID: {self._trace_id[:8]}")
        lines.append(f"  Total Duration: {total_duration:.0f}ms")
        lines.append("")
        lines.append("  Nodes:")
        lines.append(f"    Started: {self._stats.get('nodes_started', 0)}")
        lines.append(f"    Completed: {self._stats.get('nodes_completed', 0)}")
        lines.append(f"    Failed: {self._stats.get('nodes_failed', 0)}")
        lines.append("")
        lines.append("  Tools:")
        lines.append(f"    Called: {self._stats.get('tools_called', 0)}")
        lines.append(f"    Completed: {self._stats.get('tools_completed', 0)}")
        lines.append(f"    Failed: {self._stats.get('tools_failed', 0)}")
        lines.append("")
        lines.append("  Spans:")
        lines.append(f"    Total: {len(self._spans)}")

        # Tool breakdown
        tool_stats = {k.replace("tool.", ""): v for k, v in self._stats.items() if k.startswith("tool.")}
        if tool_stats:
            lines.append("")
            lines.append("  Tool Calls:")
            for tool, count in sorted(tool_stats.items(), key=lambda x: -x[1]):
                lines.append(f"    {tool}: {count}")

        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        """Get statistics dict."""
        return dict(self._stats)

    # ==================== Export ====================

    def export_json(self, path: str) -> None:
        """Export trace to JSON file."""
        data = {
            "trace_id": self._trace_id,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "tools": [t.to_dict() for t in self._tools],
            "spans": [s.to_dict() for s in self._spans],
            "edges": self._edges,
            "stats": dict(self._stats),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def export_mermaid(self, path: str | None = None) -> str:
        """Export graph as Mermaid diagram.

        Args:
            path: Optional file path to write to.

        Returns:
            Mermaid diagram string.
        """
        lines = ["graph TD"]

        # Nodes
        for node_id, node in self._nodes.items():
            shape_start, shape_end = {
                NodeType.START: ("([", "])"),
                NodeType.END: ("([", "])"),
                NodeType.AGENT: ("[", "]"),
                NodeType.TOOL: ("{{", "}}"),
                NodeType.DECISION: ("{", "}"),
                NodeType.PARALLEL: ("[[", "]]"),
                NodeType.JOIN: ("((", "))"),
            }.get(node.type, ("[", "]"))

            label = f"{node.name}"
            if node.duration_ms:
                label += f"<br/>{node.duration_ms:.0f}ms"

            lines.append(f"    {node_id}{shape_start}{label}{shape_end}")

        # Edges
        for from_id, to_id, label in self._edges:
            if label:
                lines.append(f"    {from_id} -->|{label}| {to_id}")
            else:
                lines.append(f"    {from_id} --> {to_id}")

        # Style based on status
        for node_id, node in self._nodes.items():
            if node.status == NodeStatus.COMPLETED:
                lines.append(f"    style {node_id} fill:#90EE90")
            elif node.status == NodeStatus.FAILED:
                lines.append(f"    style {node_id} fill:#FFB6C1")
            elif node.status == NodeStatus.RUNNING:
                lines.append(f"    style {node_id} fill:#87CEEB")

        result = "\n".join(lines)

        if path:
            with open(path, "w") as f:
                f.write(result)

        return result

    # ==================== Clear ====================

    def clear(self) -> None:
        """Clear all trace data."""
        self._trace_id = generate_id()
        self._nodes.clear()
        self._tools.clear()
        self._spans.clear()
        self._span_stack.clear()
        self._edges.clear()
        self._stats.clear()

    # ==================== Print Helpers ====================

    def _c(self, color: str, text: str) -> str:
        """Colorize text."""
        if not self.colors:
            return text
        colors = {
            "dim": "\033[90m",
            "bold": "\033[1m",
            "green": "\033[92m",
            "red": "\033[91m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "cyan": "\033[96m",
            "reset": "\033[0m",
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def _print_span_start(self, span: ExecutionSpan) -> None:
        depth = len(self._span_stack)
        indent = "  " * depth
        self.stream.write(f"{indent}{self._c('cyan', '▶')} {span.name}\n")
        self.stream.flush()

    def _print_span_end(self, span: ExecutionSpan) -> None:
        status = self._c('green', '✓') if span.status == "ok" else self._c('red', '✗')
        self.stream.write(f"  {status} {span.name} {self._c('dim', f'({span.duration_ms:.0f}ms)')}\n")
        self.stream.flush()

    def _print_node_start(self, node: NodeTrace) -> None:
        type_icon = {
            NodeType.AGENT: "🤖",
            NodeType.TOOL: "🔧",
            NodeType.DECISION: "◇",
            NodeType.PARALLEL: "⫴",
        }.get(node.type, "▶")

        self.stream.write(f"{type_icon} {self._c('bold', node.name)} started\n")

        if node.input and self.level >= TraceLevel.FULL:
            self.stream.write(f"   {self._c('dim', 'Input:')} {_truncate(str(node.input), 100)}\n")

        self.stream.flush()

    def _print_node_end(self, node: NodeTrace) -> None:
        status = self._c('green', '✓') if node.status == NodeStatus.COMPLETED else self._c('red', '✗')
        duration = self._c('dim', f'({node.duration_ms:.0f}ms)')

        self.stream.write(f"{status} {node.name} {duration}\n")

        if node.output and self.level >= TraceLevel.FULL:
            self.stream.write(f"   {self._c('dim', 'Output:')} {_truncate(str(node.output), 200)}\n")

        if node.error:
            self.stream.write(f"   {self._c('red', 'Error:')} {node.error}\n")

        self.stream.flush()

    def _print_tool_start(self, trace: ToolTrace) -> None:
        args_str = _truncate(str(trace.args), 80)
        self.stream.write(f"   {self._c('yellow', '→')} {trace.tool_name}({args_str})\n")
        self.stream.flush()

    def _print_tool_end(self, trace: ToolTrace) -> None:
        status = self._c('green', '✓') if not trace.error else self._c('red', '✗')
        duration = self._c('dim', f'({trace.duration_ms:.0f}ms)')

        if trace.error:
            self.stream.write(f"   {status} {trace.tool_name} {duration} - {self._c('red', trace.error)}\n")
        else:
            result_str = _truncate(str(trace.result), 100) if trace.result else ""
            self.stream.write(f"   {status} {trace.tool_name} {duration} → {result_str}\n")

        self.stream.flush()


def _truncate(s: str, max_len: int) -> str:
    """Truncate string to max length."""
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."


# ============================================================
# Integration helper - wrap existing observers
# ============================================================

class TracingObserver:
    """
    Observer that integrates with ExecutionTracer.

    Use this to add deep tracing to any Flow:

        tracer = ExecutionTracer()
        observer = TracingObserver(tracer)

        flow = Flow(..., observer=observer)
    """

    def __init__(self, tracer: ExecutionTracer):
        self.tracer = tracer
        self._current_nodes: dict[str, NodeTrace] = {}
        self._current_tools: dict[str, ToolTrace] = {}

    def handle_event(self, event) -> None:
        """Handle an event from the event bus."""
        from agenticflow.observability.trace_record import TraceType

        event_type = event.type
        data = event.data

        # Agent events
        if event_type == TraceType.AGENT_THINKING:
            agent_name = data.get("agent_name", "unknown")
            node = self.tracer.start_node(
                name=agent_name,
                node_type=NodeType.AGENT,
                input=data.get("prompt_preview"),
                agent_name=agent_name,
            )
            self._current_nodes[agent_name] = node

        elif event_type == TraceType.AGENT_RESPONDED:
            agent_name = data.get("agent_name", "unknown")
            node = self._current_nodes.pop(agent_name, None)
            if node:
                output = data.get("response_preview") or data.get("response")
                self.tracer.end_node(node, output=output)

        elif event_type == TraceType.AGENT_ERROR:
            agent_name = data.get("agent_name", "unknown")
            node = self._current_nodes.pop(agent_name, None)
            if node:
                self.tracer.end_node(
                    node,
                    status=NodeStatus.FAILED,
                    error=data.get("error"),
                )

        # Tool events
        elif event_type == TraceType.TOOL_CALLED:
            tool_name = data.get("tool", "unknown")
            agent_name = data.get("agent_name")

            # Find parent node
            node_id = None
            if agent_name and agent_name in self._current_nodes:
                node_id = self._current_nodes[agent_name].id

            trace = self.tracer.start_tool(
                tool_name=tool_name,
                args=data.get("args", {}),
                agent_name=agent_name,
                node_id=node_id,
            )
            self._current_tools[tool_name] = trace

        elif event_type == TraceType.TOOL_RESULT:
            tool_name = data.get("tool", "unknown")
            trace = self._current_tools.pop(tool_name, None)
            if trace:
                self.tracer.end_tool(trace, result=data.get("result"))

        elif event_type == TraceType.TOOL_ERROR:
            tool_name = data.get("tool", "unknown")
            trace = self._current_tools.pop(tool_name, None)
            if trace:
                self.tracer.end_tool(trace, error=data.get("error"))
