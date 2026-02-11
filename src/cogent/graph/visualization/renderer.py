"""Graph visualization renderers for different output formats.

This module provides functions to render knowledge graphs as Mermaid diagrams,
Graphviz DOT files, GraphML XML, Cytoscape.js JSON, and generic JSON formats.
Also supports rendering Mermaid to images (PNG, SVG) via Playwright.
"""

import html
import json
from typing import Any

from cogent.graph.models import Entity, Relationship
from cogent.graph.visualization.styles import StyleScheme, get_scheme


def entity_to_mermaid_node(entity: Entity, scheme: StyleScheme) -> str:
    """Convert an entity to a Mermaid node definition.

    Args:
        entity: Entity to convert.
        scheme: Style scheme for coloring.

    Returns:
        Mermaid node definition string.

    Example:
        >>> entity = Entity("alice", "Person", {"name": "Alice"})
        >>> node = entity_to_mermaid_node(entity, get_scheme())
        >>> print(node)  # alice["Alice (Person)"]
    """
    style = scheme.get_node_style(entity.entity_type)

    # Use name attribute if available, otherwise use ID
    label = entity.attributes.get("name", entity.id)

    # Add type to label
    display_label = f"{label} ({entity.entity_type})"

    # Determine shape brackets
    if style.shape == "rounded":
        brackets = ("(", ")")
    elif style.shape == "circle":
        brackets = ("((", "))")
    elif style.shape == "hexagon":
        brackets = ("{{", "}}")
    else:  # rectangle (default)
        brackets = ("[", "]")

    # Format: nodeId["Label"]
    return f'{entity.id}{brackets[0]}"{display_label}"{brackets[1]}'


def relationship_to_mermaid_edge(rel: Relationship, scheme: StyleScheme) -> str:
    """Convert a relationship to a Mermaid edge definition.

    Args:
        rel: Relationship to convert.
        scheme: Style scheme for edge styling.

    Returns:
        Mermaid edge definition string.

    Example:
        >>> rel = Relationship("alice", "knows", "bob")
        >>> edge = relationship_to_mermaid_edge(rel, get_scheme())
        >>> print(edge)  # alice -->|knows| bob
    """
    style = scheme.get_edge_style(rel.relation)

    # Determine arrow style
    if style.style == "dashed":
        arrow = "-.->|"
    elif style.style == "dotted":
        arrow = "-..->|"
    elif style.width >= 3:
        arrow = "==>|"  # Thick arrow
    else:
        arrow = "-->|"  # Normal arrow

    return f"{rel.source_id} {arrow}{rel.relation}| {rel.target_id}"


def to_mermaid(
    entities: list[Entity],
    relationships: list[Relationship],
    direction: str = "LR",
    group_by_type: bool = False,
    scheme: str | StyleScheme = "default",
    title: str | None = None,
) -> str:
    """Render graph as Mermaid diagram.

    Args:
        entities: List of entities to visualize.
        relationships: List of relationships to visualize.
        direction: Flow direction ("LR" for left-to-right, "TB" for top-to-bottom).
        group_by_type: If True, group entities by type in subgraphs.
        scheme: Style scheme name or StyleScheme instance.
        title: Optional diagram title.

    Returns:
        Mermaid diagram code as string.

    Example:
        >>> entities = [Entity("alice", "Person"), Entity("bob", "Person")]
        >>> rels = [Relationship("alice", "knows", "bob")]
        >>> diagram = to_mermaid(entities, rels)
        >>> print(diagram)
    """
    # Get style scheme
    if isinstance(scheme, str):
        scheme = get_scheme(scheme)

    lines = []

    # Optional title (YAML front matter)
    if title:
        lines.append("---")
        lines.append(f"title: {title}")
        lines.append("---")

    # Diagram header
    lines.append(f"graph {direction}")

    # Group entities by type if requested
    if group_by_type:
        entities_by_type: dict[str, list[Entity]] = {}
        for entity in entities:
            if entity.entity_type not in entities_by_type:
                entities_by_type[entity.entity_type] = []
            entities_by_type[entity.entity_type].append(entity)

        # Create subgraphs
        for entity_type, type_entities in entities_by_type.items():
            lines.append(f"    subgraph {entity_type}")
            for entity in type_entities:
                node_def = entity_to_mermaid_node(entity, scheme)
                lines.append(f"        {node_def}")
            lines.append("    end")
    else:
        # Add all nodes
        for entity in entities:
            node_def = entity_to_mermaid_node(entity, scheme)
            lines.append(f"    {node_def}")

    # Add edges
    for rel in relationships:
        edge_def = relationship_to_mermaid_edge(rel, scheme)
        lines.append(f"    {edge_def}")

    # Add styling (color definitions)
    entity_types = {e.entity_type for e in entities}
    for entity_type in entity_types:
        style_def = scheme.get_node_style(entity_type)
        # Mermaid class styling
        lines.append(
            f"    classDef {entity_type}Style fill:{style_def.color},"
            f"stroke:{style_def.border_color},color:{style_def.text_color}"
        )

    # Apply styles to nodes
    for entity in entities:
        lines.append(f"    class {entity.id} {entity.entity_type}Style")

    return "\n".join(lines)


def to_graphviz(
    entities: list[Entity],
    relationships: list[Relationship],
    scheme: str | StyleScheme = "default",
    title: str | None = None,
) -> str:
    """Render graph as Graphviz DOT format.

    Args:
        entities: List of entities to visualize.
        relationships: List of relationships to visualize.
        scheme: Style scheme name or StyleScheme instance.
        title: Optional diagram title.

    Returns:
        DOT format string for Graphviz.

    Example:
        >>> entities = [Entity("alice", "Person")]
        >>> dot = to_graphviz(entities, [])
        >>> # Can be rendered with: dot -Tpng -o graph.png
    """
    # Get style scheme
    if isinstance(scheme, str):
        scheme = get_scheme(scheme)

    lines = []

    # Graph header
    lines.append("digraph G {")
    lines.append("    rankdir=LR;")
    lines.append('    node [fontname="Arial"];')

    # Optional title
    if title:
        escaped_title = title.replace('"', '\\"')
        lines.append('    labelloc="t";')
        lines.append(f'    label="{escaped_title}";')

    # Add nodes
    for entity in entities:
        style = scheme.get_node_style(entity.entity_type)
        label = entity.attributes.get("name", entity.id)
        display_label = f"{label}\\n({entity.entity_type})"

        # Escape quotes in label
        display_label = display_label.replace('"', '\\"')

        # Determine shape
        shape_map = {
            "rounded": "box",
            "circle": "circle",
            "rectangle": "box",
            "hexagon": "hexagon",
        }
        shape = shape_map.get(style.shape, "box")
        rounded = ",style=rounded" if style.shape == "rounded" else ""

        lines.append(
            f'    "{entity.id}" [label="{display_label}", '
            f'fillcolor="{style.color}", '
            f'color="{style.border_color}", '
            f'fontcolor="{style.text_color}", '
            f'shape={shape}, style=filled{rounded}];'
        )

    # Add edges
    for rel in relationships:
        style = scheme.get_edge_style(rel.relation)

        # Style mapping
        style_attr = {
            "solid": "solid",
            "dashed": "dashed",
            "dotted": "dotted",
        }.get(style.style, "solid")

        lines.append(
            f'    "{rel.source_id}" -> "{rel.target_id}" '
            f'[label="{rel.relation}", '
            f'color="{style.color}", '
            f'penwidth={style.width}, '
            f'style={style_attr}];'
        )

    lines.append("}")

    return "\n".join(lines)


def to_graphml(
    entities: list[Entity],
    relationships: list[Relationship],
    title: str | None = None,
) -> str:
    """Render graph as GraphML XML format.

    GraphML is an XML-based format for graphs that can be imported into
    tools like yEd, Gephi, and Cytoscape.

    Args:
        entities: List of entities to visualize.
        relationships: List of relationships to visualize.
        title: Optional diagram title.

    Returns:
        GraphML XML string.

    Example:
        >>> entities = [Entity("alice", "Person")]
        >>> xml = to_graphml(entities, [])
    """
    lines = []

    # XML header
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
        'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">'
    )

    # Define attributes
    lines.append('    <key id="d0" for="node" attr.name="type" attr.type="string"/>')
    lines.append('    <key id="d1" for="node" attr.name="name" attr.type="string"/>')
    lines.append(
        '    <key id="d2" for="edge" attr.name="relation" attr.type="string"/>'
    )

    # Graph element
    lines.append('    <graph id="G" edgedefault="directed">')

    # Optional title
    if title:
        escaped = html.escape(title)
        lines.append(f'        <desc>{escaped}</desc>')

    # Add nodes
    for entity in entities:
        lines.append(f'        <node id="{html.escape(entity.id)}">')
        lines.append(
            f'            <data key="d0">{html.escape(entity.entity_type)}</data>'
        )

        name = entity.attributes.get("name", entity.id)
        lines.append(f'            <data key="d1">{html.escape(str(name))}</data>')

        lines.append("        </node>")

    # Add edges
    for i, rel in enumerate(relationships):
        edge_id = f"e{i}"
        lines.append(
            f'        <edge id="{edge_id}" '
            f'source="{html.escape(rel.source_id)}" '
            f'target="{html.escape(rel.target_id)}">'
        )
        lines.append(f'            <data key="d2">{html.escape(rel.relation)}</data>')
        lines.append("        </edge>")

    lines.append("    </graph>")
    lines.append("</graphml>")

    return "\n".join(lines)


def save_diagram(
    content: str,
    file_path: str,
    format: str = "mermaid",
) -> None:
    """Save diagram content to a file.

    Args:
        content: Diagram content (Mermaid, DOT, GraphML, or JSON).
        file_path: Path to save the file.
        format: Output format ("mermaid", "graphviz", "graphml", "cytoscape", "json").

    Raises:
        ValueError: If format is not recognized.

    Example:
        >>> diagram = to_mermaid(entities, relationships)
        >>> save_diagram(diagram, "graph.mmd", format="mermaid")
    """
    # Add appropriate file extension if missing
    extensions = {
        "mermaid": ".mmd",
        "graphviz": ".dot",
        "graphml": ".graphml",
        "cytoscape": ".json",
        "json": ".json",
    }

    if format not in extensions:
        raise ValueError(
            f"Unknown format: {format}. Use: {list(extensions.keys())}"
        )

    # Ensure extension
    if not file_path.endswith(extensions[format]):
        file_path = file_path + extensions[format]

    # Write file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def to_cytoscape_json(
    entities: list[Entity],
    relationships: list[Relationship],
) -> str:
    """Render graph as Cytoscape.js JSON format.

    Cytoscape.js is a popular JavaScript library for graph visualization in web browsers.
    This format can be used with Cytoscape Desktop, Cytoscape.js, and other tools.

    Args:
        entities: List of entities to visualize.
        relationships: List of relationships to visualize.

    Returns:
        JSON string in Cytoscape.js format.

    Example:
        >>> entities = [Entity("alice", "Person"), Entity("bob", "Person")]
        >>> rels = [Relationship("alice", "knows", "bob")]
        >>> json_str = to_cytoscape_json(entities, rels)
        >>> import json
        >>> data = json.loads(json_str)
        >>> print(data["elements"]["nodes"][0])
    """
    elements = {"nodes": [], "edges": []}

    # Add nodes (entities)
    for entity in entities:
        node = {
            "data": {
                "id": entity.id,
                "label": entity.attributes.get("name", entity.id),
                "type": entity.entity_type,
                **entity.attributes,  # Include all attributes
            }
        }
        elements["nodes"].append(node)

    # Add edges (relationships)
    edge_id = 0
    for rel in relationships:
        edge = {
            "data": {
                "id": f"e{edge_id}",
                "source": rel.source_id,
                "target": rel.target_id,
                "label": rel.relation,
                "relation": rel.relation,
                **rel.attributes,  # Include all attributes
            }
        }
        elements["edges"].append(edge)
        edge_id += 1

    return json.dumps({"elements": elements}, indent=2)


def to_json_graph(
    entities: list[Entity],
    relationships: list[Relationship],
) -> str:
    """Render graph as generic JSON Graph Format.

    This is a simple, standardized JSON format for representing graphs.
    See: http://jsongraphformat.info/

    Args:
        entities: List of entities to visualize.
        relationships: List of relationships to visualize.

    Returns:
        JSON string in JSON Graph Format.

    Example:
        >>> entities = [Entity("alice", "Person", {"age": 30})]
        >>> rels = [Relationship("alice", "knows", "bob")]
        >>> json_str = to_json_graph(entities, rels)
    """
    nodes = []
    edges = []

    # Add nodes
    for entity in entities:
        node = {
            "id": entity.id,
            "label": entity.attributes.get("name", entity.id),
            "metadata": {
                "type": entity.entity_type,
                **entity.attributes,
            },
        }
        nodes.append(node)

    # Add edges
    for rel in relationships:
        edge = {
            "source": rel.source_id,
            "target": rel.target_id,
            "relation": rel.relation,
            "metadata": rel.attributes,
        }
        edges.append(edge)

    graph = {
        "graph": {
            "directed": True,
            "type": "knowledge_graph",
            "nodes": nodes,
            "edges": edges,
        }
    }

    return json.dumps(graph, indent=2)


async def render_mermaid_to_image(
    mermaid_code: str,
    output_path: str,
    format: str = "png",
    width: int = 1920,
    height: int = 1080,
) -> None:
    """Render Mermaid diagram to image using Mermaid CLI.

    This function requires Mermaid CLI to be installed via npm:
        npm install -g @mermaid-js/mermaid-cli

    Or using your system package manager:
        brew install mermaid-cli  # macOS
        apt install mermaid        # Debian/Ubuntu

    Args:
        mermaid_code: Mermaid diagram code.
        output_path: Path to save the image.
        format: Output format ("png", "svg", "pdf").
        width: Viewport width in pixels.
        height: Viewport height in pixels.

    Raises:
        FileNotFoundError: If Mermaid CLI (mmdc) is not installed.
        RuntimeError: If rendering fails.

    Example:
        >>> diagram = to_mermaid(entities, relationships)
        >>> await render_mermaid_to_image(diagram, "graph.png")
    """
    import asyncio
    import tempfile
    from pathlib import Path

    # Validate format
    if format not in ("png", "svg", "pdf"):
        raise ValueError(f"format must be 'png', 'svg', or 'pdf', got: {format}")

    # Create temporary file for Mermaid code
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mmd", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(mermaid_code)
        tmp_path = tmp.name

    try:
        # Build mmdc command
        cmd = [
            "mmdc",
            "-i", tmp_path,
            "-o", output_path,
            "-w", str(width),
            "-H", str(height),
            "-t", "default",  # Use default (light) theme to respect custom styles
        ]

        # Add background color for better visibility
        if format == "png":
            cmd.extend(["-b", "white"])

        # Run Mermaid CLI
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            if "command not found" in error_msg or "not found" in error_msg:
                raise FileNotFoundError(
                    "Mermaid CLI (mmdc) not found. Install with: npm install -g @mermaid-js/mermaid-cli"
                )
            raise RuntimeError(f"Mermaid rendering failed: {error_msg}")

    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)

