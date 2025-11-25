"""Mermaid diagram generation for AgenticFlow components.

Generates Mermaid flowcharts and diagrams for:
- Individual agents with their tools
- Topologies showing agent coordination patterns
- Graph structures

Uses YAML frontmatter for configuration (not deprecated %%init%%).
"""

from __future__ import annotations

import base64
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agenticflow.agents.base import Agent
    from agenticflow.topologies.base import BaseTopology


class MermaidTheme(Enum):
    """Mermaid diagram themes."""

    DEFAULT = "default"
    DARK = "dark"
    FOREST = "forest"
    NEUTRAL = "neutral"
    BASE = "base"


class MermaidDirection(Enum):
    """Flowchart direction."""

    TOP_DOWN = "TD"
    TOP_BOTTOM = "TB"
    BOTTOM_TOP = "BT"
    LEFT_RIGHT = "LR"
    RIGHT_LEFT = "RL"


@dataclass
class MermaidConfig:
    """Configuration for Mermaid diagram rendering.

    Uses YAML frontmatter for modern Mermaid configuration.

    Attributes:
        title: Diagram title.
        theme: Color theme.
        direction: Flowchart direction.
        curve: Line curve style.
        node_spacing: Spacing between nodes.
        rank_spacing: Spacing between ranks.
        padding: Diagram padding.
        use_max_width: Whether to use maximum width.
        show_tools: Whether to show agent tools.
        show_roles: Whether to show agent roles.
        show_config: Whether to show configuration details.
        compact: Whether to use compact mode (inline tools, minimal styling).
    """

    title: str = ""
    theme: MermaidTheme = MermaidTheme.DEFAULT
    direction: MermaidDirection = MermaidDirection.TOP_DOWN
    curve: str = "basis"
    node_spacing: int = 40
    rank_spacing: int = 40
    padding: int = 8
    use_max_width: bool = True
    show_tools: bool = True
    show_roles: bool = True
    show_config: bool = False
    compact: bool = True  # Default to compact mode

    def to_frontmatter(self) -> str:
        """Generate YAML frontmatter configuration.

        Returns:
            YAML frontmatter string.
        """
        lines = ["---"]

        if self.title:
            lines.append(f"title: {self.title}")

        lines.append("config:")
        lines.append(f"  theme: {self.theme.value}")
        lines.append("  flowchart:")
        lines.append(f"    curve: {self.curve}")
        lines.append(f"    nodeSpacing: {self.node_spacing}")
        lines.append(f"    rankSpacing: {self.rank_spacing}")
        lines.append(f"    padding: {self.padding}")
        lines.append(f"    useMaxWidth: {str(self.use_max_width).lower()}")

        lines.append("---")
        return "\n".join(lines)


class MermaidRenderer:
    """Renders Mermaid diagrams to various formats."""

    MERMAID_INK_URL = "https://mermaid.ink"

    @classmethod
    def to_url(cls, mermaid_code: str, *, format: str = "svg") -> str:
        """Convert Mermaid code to a rendered image URL.

        Args:
            mermaid_code: The Mermaid diagram code.
            format: Output format ('svg' or 'png').

        Returns:
            URL to rendered diagram.
        """
        # Encode as base64
        encoded = base64.urlsafe_b64encode(mermaid_code.encode()).decode()
        return f"{cls.MERMAID_INK_URL}/{format}/{encoded}"

    @classmethod
    def to_png_url(cls, mermaid_code: str) -> str:
        """Get PNG rendering URL.

        Args:
            mermaid_code: The Mermaid diagram code.

        Returns:
            URL to PNG image.
        """
        return cls.to_url(mermaid_code, format="img")

    @classmethod
    def to_svg_url(cls, mermaid_code: str) -> str:
        """Get SVG rendering URL.

        Args:
            mermaid_code: The Mermaid diagram code.

        Returns:
            URL to SVG image.
        """
        return cls.to_url(mermaid_code, format="svg")

    @classmethod
    def draw_png(cls, mermaid_code: str) -> bytes:
        """Render Mermaid diagram as PNG bytes.

        Requires internet connection to mermaid.ink API.

        Args:
            mermaid_code: The Mermaid diagram code.

        Returns:
            PNG image bytes.

        Raises:
            ImportError: If httpx is not installed.
            RuntimeError: If rendering fails.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for PNG rendering. "
                "Install with: uv add httpx"
            )

        url = cls.to_png_url(mermaid_code)

        try:
            response = httpx.get(url, timeout=30, follow_redirects=True)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to render diagram: {e}")

    @classmethod
    def to_html(cls, mermaid_code: str) -> str:
        """Generate HTML that renders the Mermaid diagram.

        Args:
            mermaid_code: The Mermaid diagram code.

        Returns:
            HTML string with embedded diagram.
        """
        # Escape for HTML
        escaped = mermaid_code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        return f"""
<div class="mermaid">
{escaped}
</div>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
mermaid.initialize({{ startOnLoad: true }});
</script>
"""


def _escape_label(text: str) -> str:
    """Escape text for use in Mermaid labels.

    Args:
        text: Text to escape.

    Returns:
        Escaped text safe for Mermaid labels.
    """
    # Replace problematic characters
    text = text.replace('"', "&quot;")
    text = text.replace("(", "&#40;")
    text = text.replace(")", "&#41;")
    text = text.replace("[", "&#91;")
    text = text.replace("]", "&#93;")
    text = text.replace("{", "&#123;")
    text = text.replace("}", "&#125;")
    return text


def _make_node_id(name: str) -> str:
    """Create a valid Mermaid node ID from a name.

    Args:
        name: Original name.

    Returns:
        Valid node ID (alphanumeric + underscore).
    """
    # Replace non-alphanumeric with underscore
    node_id = "".join(c if c.isalnum() else "_" for c in name)
    # Ensure it doesn't start with a number
    if node_id and node_id[0].isdigit():
        node_id = "n_" + node_id
    return node_id or "node"


class AgentDiagram:
    """Generates Mermaid diagrams for agents.

    Shows the agent with its tools, role, and configuration.

    Example:
        >>> diagram = AgentDiagram(agent, config=MermaidConfig(title="My Agent"))
        >>> print(diagram.to_mermaid())
        >>> png_bytes = diagram.draw_png()
    """

    def __init__(
        self,
        agent: Agent,
        config: MermaidConfig | None = None,
    ) -> None:
        """Initialize agent diagram.

        Args:
            agent: Agent to visualize.
            config: Diagram configuration.
        """
        self.agent = agent
        self.config = config or MermaidConfig()

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram code.

        Returns:
            Mermaid flowchart code.
        """
        cfg = self.config
        lines = []

        # Add frontmatter only if needed
        if cfg.title or cfg.theme != MermaidTheme.DEFAULT:
            title = cfg.title or f"Agent: {self.agent.name}"
            lines.append(self._generate_frontmatter(title))

        # Start flowchart
        lines.append(f"flowchart {cfg.direction.value}")

        # Agent node with role
        agent_id = _make_node_id(self.agent.name)
        role = self.agent.role.value
        
        if cfg.show_roles:
            agent_label = f"{self.agent.name}\\n({role})"
        else:
            agent_label = self.agent.name

        # Use role-specific shape
        lines.append(f'    {agent_id}["{_escape_label(agent_label)}"]:::' + self._get_role_class(role))

        # Add tools - compact inline or expanded subgraph
        if cfg.show_tools and self.agent.config.tools:
            if cfg.compact:
                # Compact: single tools node with comma-separated list
                tools_str = ", ".join(self.agent.config.tools[:5])  # Limit to 5
                if len(self.agent.config.tools) > 5:
                    tools_str += f" +{len(self.agent.config.tools) - 5}"
                tools_id = f"t_{agent_id}"
                lines.append(f'    {tools_id}[/"{tools_str}"/]:::tool')
                lines.append(f"    {agent_id} --> {tools_id}")
            else:
                # Expanded: subgraph with individual tools
                lines.append(f'    subgraph tools_{agent_id} ["🛠️ Tools"]')
                lines.append("        direction LR")
                for i, tool_name in enumerate(self.agent.config.tools):
                    tool_id = f"tool_{agent_id}_{i}"
                    lines.append(f'        {tool_id}[/"{_escape_label(tool_name)}"/]:::tool')
                lines.append("    end")
                lines.append(f"    {agent_id} --> tools_{agent_id}")

        # Add config if enabled
        if cfg.show_config:
            config_id = f"cfg_{agent_id}"
            # Get model info from the model object if available
            model_info = "no model"
            if self.agent.config.model is not None:
                model_obj = self.agent.config.model
                # Try to get model name from LangChain model
                model_info = getattr(model_obj, "model_name", None) or getattr(model_obj, "model", "model") or "model"
            lines.append(f'    {config_id}{{"{model_info}"}}:::config')
            lines.append(f"    {agent_id} -.- {config_id}")

        # Add class definitions
        lines.append("")
        lines.append(self._get_class_definitions())

        return "\n".join(lines)

    def _generate_frontmatter(self, title: str) -> str:
        """Generate YAML frontmatter."""
        cfg = self.config
        lines = ["---"]
        # Quote title if it contains special YAML characters
        if any(c in title for c in ':{}[]|>&*!?#'):
            lines.append(f'title: "{title}"')
        else:
            lines.append(f"title: {title}")
        lines.append("config:")
        lines.append(f"  theme: {cfg.theme.value}")
        lines.append("  flowchart:")
        lines.append(f"    curve: {cfg.curve}")
        lines.append(f"    nodeSpacing: {cfg.node_spacing}")
        lines.append(f"    rankSpacing: {cfg.rank_spacing}")
        lines.append("---")
        return "\n".join(lines)

    def _get_role_class(self, role: str) -> str:
        """Get CSS class for agent role."""
        return {"orchestrator": "orch", "worker": "work", "planner": "plan",
                "critic": "crit", "specialist": "spec", "researcher": "res",
                "validator": "val", "assistant": "asst"}.get(role, "work")

    def _get_class_definitions(self) -> str:
        """Get compact Mermaid class definitions."""
        return """    classDef orch fill:#4a90d9,stroke:#2d5986,color:#fff
    classDef work fill:#7eb36a,stroke:#4a7a3d,color:#fff
    classDef plan fill:#e6a23c,stroke:#b87c1e,color:#fff
    classDef crit fill:#f56c6c,stroke:#c45656,color:#fff
    classDef spec fill:#9b59b6,stroke:#7b3a96,color:#fff
    classDef res fill:#3498db,stroke:#2475ad,color:#fff
    classDef val fill:#1abc9c,stroke:#16a085,color:#fff
    classDef asst fill:#95a5a6,stroke:#7f8c8d,color:#fff
    classDef tool fill:#f5f5f5,stroke:#999,color:#333
    classDef config fill:#fff3e0,stroke:#ff9800,color:#333,stroke-dasharray:3"""

    def draw_png(self) -> bytes:
        """Render diagram as PNG.

        Returns:
            PNG image bytes.
        """
        return MermaidRenderer.draw_png(self.to_mermaid())

    def get_png_url(self) -> str:
        """Get URL to PNG rendering.

        Returns:
            URL to rendered PNG.
        """
        return MermaidRenderer.to_png_url(self.to_mermaid())

    def get_svg_url(self) -> str:
        """Get URL to SVG rendering.

        Returns:
            URL to rendered SVG.
        """
        return MermaidRenderer.to_svg_url(self.to_mermaid())

    def to_html(self) -> str:
        """Generate HTML for diagram.

        Returns:
            HTML string.
        """
        return MermaidRenderer.to_html(self.to_mermaid())

    def _repr_html_(self) -> str:
        """IPython/Jupyter HTML representation.

        Returns:
            HTML for notebook display.
        """
        return self.to_html()


class TopologyDiagram:
    """Generates Mermaid diagrams for topologies.

    Shows agents, their relationships, and coordination patterns.

    Example:
        >>> diagram = TopologyDiagram(topology, config=MermaidConfig(title="My Team"))
        >>> print(diagram.to_mermaid())
        >>> png_bytes = diagram.draw_png()
    """

    def __init__(
        self,
        topology: BaseTopology,
        config: MermaidConfig | None = None,
    ) -> None:
        """Initialize topology diagram.

        Args:
            topology: Topology to visualize.
            config: Diagram configuration.
        """
        self.topology = topology
        self.config = config or MermaidConfig()

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram code.

        Returns:
            Mermaid flowchart code.
        """
        cfg = self.config
        lines = []

        # Determine topology type
        topology_type = self._get_topology_type()

        # Add frontmatter
        title = cfg.title or self.topology.config.name
        lines.append(self._generate_frontmatter(title))

        # Start flowchart
        lines.append(f"flowchart {cfg.direction.value}")

        # Generate diagram based on topology type
        if topology_type == "supervisor":
            lines.extend(self._generate_supervisor_diagram())
        elif topology_type == "pipeline":
            lines.extend(self._generate_pipeline_diagram())
        elif topology_type == "mesh":
            lines.extend(self._generate_mesh_diagram())
        elif topology_type == "hierarchical":
            lines.extend(self._generate_hierarchical_diagram())
        elif topology_type == "custom":
            lines.extend(self._generate_custom_diagram())
        else:
            lines.extend(self._generate_generic_diagram())

        # Add class definitions
        lines.append("")
        lines.append(self._get_class_definitions())

        return "\n".join(lines)

    def _generate_frontmatter(self, title: str) -> str:
        """Generate YAML frontmatter."""
        cfg = self.config
        lines = ["---"]
        # Quote title if it contains special YAML characters
        if any(c in title for c in ':{}[]|>&*!?#'):
            lines.append(f'title: "{title}"')
        else:
            lines.append(f"title: {title}")
        lines.append("config:")
        lines.append(f"  theme: {cfg.theme.value}")
        lines.append("  flowchart:")
        lines.append(f"    curve: {cfg.curve}")
        lines.append(f"    nodeSpacing: {cfg.node_spacing}")
        lines.append(f"    rankSpacing: {cfg.rank_spacing}")
        lines.append("---")
        return "\n".join(lines)

    def _get_topology_type(self) -> str:
        """Determine topology type from class name or policy."""
        # Check if it has a policy with edges for custom topologies
        if hasattr(self.topology, 'policy') and hasattr(self.topology.policy, 'rules'):
            policy = self.topology.policy
            if policy.rules:
                # Has explicit rules - could be custom or prebuilt with policy
                pass  # Continue to class name check
        
        class_name = type(self.topology).__name__.lower()
        for t in ("custom", "supervisor", "pipeline", "mesh", "hierarchical"):
            if t in class_name:
                return t
        return "generic"

    def _get_edges_from_policy(self) -> list[tuple[str, str, str]]:
        """Get edges from topology policy for accurate diagram generation.
        
        Returns:
            List of (source, target, label) tuples.
        """
        if not hasattr(self.topology, 'policy'):
            return []
        
        policy = self.topology.policy
        agent_names = list(self.topology.agents.keys())
        
        return policy.get_edges_for_diagram(agent_names)

    def _get_role_class(self, role: str) -> str:
        """Get CSS class for agent role."""
        return {"orchestrator": "orch", "worker": "work", "planner": "plan",
                "critic": "crit", "specialist": "spec", "researcher": "res",
                "validator": "val", "assistant": "asst"}.get(role, "work")

    def _make_agent_node(self, name: str, shape: str = "rect") -> str:
        """Create clean agent node."""
        agent = self.topology.agents.get(name)
        role = agent.role.value if agent else "worker"
        role_class = self._get_role_class(role)
        node_id = _make_node_id(name)
        escaped = _escape_label(name)
        
        # Shape variants - clean labels without tool counts
        if shape == "circle":
            node_def = f'{node_id}(("{escaped}")):::{role_class}'
        elif shape == "stadium":
            node_def = f'{node_id}(["{escaped}"]):::{role_class}'
        elif shape == "subroutine":
            node_def = f'{node_id}[["{escaped}"]]:::{role_class}'
        elif shape == "hex":
            node_def = f'{node_id}' + '{{{"' + escaped + '"}}}'+ f':::{role_class}'
        else:  # rect with rounded corners
            node_def = f'{node_id}["{escaped}"]:::{role_class}'
        
        return f"    {node_def}"

    def _generate_supervisor_diagram(self) -> list[str]:
        """Generate clean supervisor topology diagram with optional tools."""
        lines = []
        cfg = self.config
        
        supervisor_name = getattr(self.topology, "supervisor_name", None)
        worker_names = getattr(self.topology, "worker_names", [])
        
        if not supervisor_name:
            supervisor_name = next(iter(self.topology.agents), "supervisor")
            worker_names = [n for n in self.topology.agents if n != supervisor_name]

        sup_id = _make_node_id(supervisor_name)
        
        # Supervisor at top (circle shape) with tools as subtitle
        sup_agent = self.topology.agents.get(supervisor_name)
        if cfg.show_tools and sup_agent and sup_agent.config.tools:
            tools_str = ", ".join(sup_agent.config.tools[:3])
            if len(sup_agent.config.tools) > 3:
                tools_str += "..."
            lines.append(f'    {sup_id}((("`**{_escape_label(supervisor_name)}**\n_{tools_str}_`"))):::orch')
        else:
            lines.append(self._make_agent_node(supervisor_name, "circle"))
        
        lines.append("")
        
        # Workers in subgraph with tools shown as subtitle
        if len(worker_names) > 1:
            lines.append('    subgraph Workers[" "]')
            lines.append("        direction LR")
            for worker_name in worker_names:
                agent = self.topology.agents.get(worker_name)
                role = agent.role.value if agent else "worker"
                role_class = self._get_role_class(role)
                worker_id = _make_node_id(worker_name)
                
                if cfg.show_tools and agent and agent.config.tools:
                    tools_str = ", ".join(agent.config.tools[:3])
                    if len(agent.config.tools) > 3:
                        tools_str += "..."
                    lines.append(f'        {worker_id}["`**{_escape_label(worker_name)}**\n_{tools_str}_`"]:::{role_class}')
                else:
                    lines.append(f'        {worker_id}["{_escape_label(worker_name)}"]:::{role_class}')
            lines.append("    end")
            lines.append("")
            lines.append(f"    {sup_id} --> Workers")
        else:
            # Single worker - direct connection
            for worker_name in worker_names:
                agent = self.topology.agents.get(worker_name)
                role = agent.role.value if agent else "worker"
                role_class = self._get_role_class(role)
                worker_id = _make_node_id(worker_name)
                
                if cfg.show_tools and agent and agent.config.tools:
                    tools_str = ", ".join(agent.config.tools[:3])
                    if len(agent.config.tools) > 3:
                        tools_str += "..."
                    lines.append(f'    {worker_id}["`**{_escape_label(worker_name)}**\n_{tools_str}_`"]:::{role_class}')
                else:
                    lines.append(f'    {worker_id}["{_escape_label(worker_name)}"]:::{role_class}')
                lines.append(f"    {sup_id} --> {worker_id}")

        return lines

    def _generate_pipeline_diagram(self) -> list[str]:
        """Generate clean pipeline topology diagram with optional tools."""
        lines = []
        cfg = self.config
        stages = getattr(self.topology, "stages", list(self.topology.agents.keys()))
        
        # Define all nodes with tools as subtitle
        for i, stage_name in enumerate(stages):
            agent = self.topology.agents.get(stage_name)
            role = agent.role.value if agent else "worker"
            role_class = self._get_role_class(role)
            node_id = _make_node_id(stage_name)
            
            # Shape based on position
            if i == 0:
                shape_start, shape_end = "([", "])"  # stadium
            elif i == len(stages) - 1:
                shape_start, shape_end = "[[", "]]"  # subroutine
            else:
                shape_start, shape_end = "[", "]"  # rect
            
            if cfg.show_tools and agent and agent.config.tools:
                tools_str = ", ".join(agent.config.tools[:3])
                if len(agent.config.tools) > 3:
                    tools_str += "..."
                # Use markdown for multi-line label
                label = f'"`**{_escape_label(stage_name)}**\n_{tools_str}_`"'
            else:
                label = f'"{_escape_label(stage_name)}"'
            
            lines.append(f"    {node_id}{shape_start}{label}{shape_end}:::{role_class}")
        
        lines.append("")
        
        # Chain connections
        if len(stages) > 1:
            node_ids = [_make_node_id(n) for n in stages]
            lines.append(f"    {' --> '.join(node_ids)}")

        return lines

    def _generate_mesh_diagram(self) -> list[str]:
        """Generate clean mesh topology diagram with optional tools."""
        lines = []
        cfg = self.config
        agent_names = list(self.topology.agents.keys())
        
        # All nodes with hexagon shape, tools as subtitle
        for name in agent_names:
            agent = self.topology.agents.get(name)
            role = agent.role.value if agent else "worker"
            role_class = self._get_role_class(role)
            node_id = _make_node_id(name)
            
            if cfg.show_tools and agent and agent.config.tools:
                tools_str = ", ".join(agent.config.tools[:3])
                if len(agent.config.tools) > 3:
                    tools_str += "..."
                # Hexagon with markdown label
                lines.append(f'    {node_id}' + '{{"`**' + _escape_label(name) + '**\n_' + tools_str + '_`"}}' + f':::{role_class}')
            else:
                lines.append(f'    {node_id}' + '{{"' + _escape_label(name) + '"}}' + f':::{role_class}')
        
        lines.append("")
        
        # Mesh connections
        node_ids = [_make_node_id(n) for n in agent_names]
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                lines.append(f"    {node_ids[i]} --- {node_ids[j]}")

        return lines

    def _generate_hierarchical_diagram(self) -> list[str]:
        """Generate hierarchical topology diagram with optional tools."""
        lines = []
        cfg = self.config
        hierarchy = getattr(self.topology, "hierarchy", {})
        root = getattr(self.topology, "root", None)

        if not hierarchy or not root:
            return self._generate_generic_diagram()

        added = set()
        
        def add_node(name: str, is_root: bool = False) -> None:
            if name in added:
                return
            added.add(name)
            
            agent = self.topology.agents.get(name)
            role = agent.role.value if agent else "worker"
            role_class = self._get_role_class(role)
            node_id = _make_node_id(name)
            
            if cfg.show_tools and agent and agent.config.tools:
                tools_str = ", ".join(agent.config.tools[:3])
                if len(agent.config.tools) > 3:
                    tools_str += "..."
                if is_root:
                    lines.append(f'    {node_id}((("`**{_escape_label(name)}**\n_{tools_str}_`"))):::{role_class}')
                else:
                    lines.append(f'    {node_id}["`**{_escape_label(name)}**\n_{tools_str}_`"]:::{role_class}')
            else:
                shape = "circle" if is_root else "rect"
                lines.append(self._make_agent_node(name, shape))
            
            children = hierarchy.get(name, [])
            if children:
                parent_id = _make_node_id(name)
                for child in children:
                    add_node(child)
                    lines.append(f"    {parent_id} --> {_make_node_id(child)}")

        add_node(root, is_root=True)
        return lines

    def _generate_generic_diagram(self) -> list[str]:
        """Generate generic diagram using policy edges if available.
        
        For unknown topologies, uses policy edges or falls back to sequential.
        """
        lines = []
        cfg = self.config
        agent_names = list(self.topology.agents.keys())
        
        for name in agent_names:
            agent = self.topology.agents.get(name)
            role = agent.role.value if agent else "worker"
            role_class = self._get_role_class(role)
            node_id = _make_node_id(name)
            
            if cfg.show_tools and agent and agent.config.tools:
                tools_str = ", ".join(agent.config.tools[:3])
                if len(agent.config.tools) > 3:
                    tools_str += "..."
                lines.append(f'    {node_id}["`**{_escape_label(name)}**\n_{tools_str}_`"]:::{role_class}')
            else:
                lines.append(self._make_agent_node(name, "rect"))
        
        lines.append("")
        
        # Try to get edges from policy
        policy_edges = self._get_edges_from_policy()
        if policy_edges:
            for source, target, label in policy_edges:
                source_id = _make_node_id(source)
                target_id = _make_node_id(target)
                if label:
                    lines.append(f"    {source_id} --> |{label}|{target_id}")
                else:
                    lines.append(f"    {source_id} --> {target_id}")
        elif len(agent_names) > 1:
            # Fallback to sequential connections
            node_ids = [_make_node_id(n) for n in agent_names]
            lines.append(f"    {' --> '.join(node_ids)}")
        
        return lines

    def _generate_custom_diagram(self) -> list[str]:
        """Generate diagram for CustomTopology using explicit edges.
        
        Uses the edges defined in CustomTopology for accurate visualization.
        """
        lines = []
        cfg = self.config
        
        # Get edges from CustomTopology
        edges = getattr(self.topology, "edges", [])
        entry_point = getattr(self.topology, "entry_point", None)
        
        # Determine which agents have incoming/outgoing edges
        sources = {e.source for e in edges}
        targets = {e.target for e in edges}
        
        # Generate nodes
        for name in self.topology.agents:
            agent = self.topology.agents.get(name)
            role = agent.role.value if agent else "worker"
            role_class = self._get_role_class(role)
            node_id = _make_node_id(name)
            
            # Determine shape based on role in the flow
            is_entry = (name == entry_point)
            is_terminal = (name not in sources and name in targets)
            
            if is_entry:
                shape_start, shape_end = "([", "])"  # stadium for entry
            elif is_terminal:
                shape_start, shape_end = "[[", "]]"  # subroutine for terminal
            elif role == "orchestrator":
                shape_start, shape_end = "((", "))"  # circle for orchestrator
            else:
                shape_start, shape_end = "[", "]"  # rect for others
            
            if cfg.show_tools and agent and agent.config.tools:
                tools_str = ", ".join(agent.config.tools[:3])
                if len(agent.config.tools) > 3:
                    tools_str += "..."
                label = f'"`**{_escape_label(name)}**\n_{tools_str}_`"'
            else:
                label = f'"{_escape_label(name)}"'
            
            lines.append(f"    {node_id}{shape_start}{label}{shape_end}:::{role_class}")
        
        lines.append("")
        
        # Generate edges
        for edge in edges:
            source_id = _make_node_id(edge.source)
            target_id = _make_node_id(edge.target)
            
            if edge.bidirectional:
                arrow = " <--> "
            else:
                arrow = " --> "
            
            if edge.label:
                lines.append(f"    {source_id}{arrow}|{edge.label}|{target_id}")
            elif edge.condition:
                lines.append(f"    {source_id}{arrow}|{edge.condition}|{target_id}")
            else:
                lines.append(f"    {source_id}{arrow}{target_id}")
        
        return lines

    def _get_class_definitions(self) -> str:
        """Get compact Mermaid class definitions."""
        return """    classDef orch fill:#4a90d9,stroke:#2d5986,color:#fff
    classDef work fill:#7eb36a,stroke:#4a7a3d,color:#fff
    classDef plan fill:#e6a23c,stroke:#b87c1e,color:#fff
    classDef crit fill:#f56c6c,stroke:#c45656,color:#fff
    classDef spec fill:#9b59b6,stroke:#7b3a96,color:#fff
    classDef res fill:#3498db,stroke:#2475ad,color:#fff
    classDef val fill:#1abc9c,stroke:#16a085,color:#fff
    classDef asst fill:#95a5a6,stroke:#7f8c8d,color:#fff
    classDef tool fill:#f5f5f5,stroke:#999,color:#333"""

    def draw_png(self) -> bytes:
        """Render diagram as PNG.

        Returns:
            PNG image bytes.
        """
        return MermaidRenderer.draw_png(self.to_mermaid())

    def get_png_url(self) -> str:
        """Get URL to PNG rendering.

        Returns:
            URL to rendered PNG.
        """
        return MermaidRenderer.to_png_url(self.to_mermaid())

    def get_svg_url(self) -> str:
        """Get URL to SVG rendering.

        Returns:
            URL to rendered SVG.
        """
        return MermaidRenderer.to_svg_url(self.to_mermaid())

    def to_html(self) -> str:
        """Generate HTML for diagram.

        Returns:
            HTML string.
        """
        return MermaidRenderer.to_html(self.to_mermaid())

    def _repr_html_(self) -> str:
        """IPython/Jupyter HTML representation.

        Returns:
            HTML for notebook display.
        """
        return self.to_html()
