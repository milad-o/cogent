"""Graph visualization renderers for different output formats.

This module provides functions to render knowledge graphs as Mermaid diagrams.
Supports rendering Mermaid to images (PNG, SVG, PDF) via Mermaid CLI (mmdc).

For interactive visualizations, use PyVis (HTML) or gravis (2D/3D web).
"""

import html
import json
from typing import Any

from cogent.graph.models import Entity, Relationship
from cogent.graph.visualization.styles import StyleScheme, get_scheme


def _safe_mermaid_id(raw_id: str) -> str:
    safe = []
    for ch in raw_id:
        if ch.isalnum() or ch == "_":
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def entity_to_mermaid_node(
    entity: Entity,
    scheme: StyleScheme,
    node_id: str | None = None,
) -> str:
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
    safe_id = node_id or _safe_mermaid_id(entity.id)
    return f'{safe_id}{brackets[0]}"{display_label}"{brackets[1]}'


def relationship_to_mermaid_edge(
    rel: Relationship,
    scheme: StyleScheme,
    source_id: str | None = None,
    target_id: str | None = None,
) -> str:
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

    safe_source = source_id or _safe_mermaid_id(rel.source_id)
    safe_target = target_id or _safe_mermaid_id(rel.target_id)
    return f"{safe_source} {arrow}{rel.relation}| {safe_target}"


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
        group_by_type: If True, group entities by type in subgraphs for better organization.
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

    # Use flowchart instead of graph for better layout
    lines.append(f"flowchart {direction}")

    safe_ids = {entity.id: _safe_mermaid_id(entity.id) for entity in entities}

    # Group entities by type if requested
    if group_by_type:
        # Create subgraphs for each entity type
        entities_by_type: dict[str, list[Entity]] = {}
        for entity in entities:
            if entity.entity_type not in entities_by_type:
                entities_by_type[entity.entity_type] = []
            entities_by_type[entity.entity_type].append(entity)

        # Add subgraphs
        for entity_type, type_entities in sorted(entities_by_type.items()):
            lines.append(f"    subgraph {entity_type}")
            for entity in type_entities:
                node_def = entity_to_mermaid_node(
                    entity, scheme, node_id=safe_ids[entity.id]
                )
                lines.append(f"        {node_def}")
            lines.append(f"    end")
    else:
        # Add all nodes without grouping
        for entity in entities:
            node_def = entity_to_mermaid_node(
                entity, scheme, node_id=safe_ids[entity.id]
            )
            lines.append(f"    {node_def}")

    # Add edges
    for rel in relationships:
        edge_def = relationship_to_mermaid_edge(
            rel,
            scheme,
            source_id=safe_ids.get(rel.source_id, _safe_mermaid_id(rel.source_id)),
            target_id=safe_ids.get(rel.target_id, _safe_mermaid_id(rel.target_id)),
        )
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
        lines.append(
            f"    class {safe_ids[entity.id]} {entity.entity_type}Style"
        )

    return "\n".join(lines)


def save_diagram(
    content: str,
    file_path: str,
    format: str = "mermaid",
) -> None:
    """Save diagram content to a file.

    Args:
        content: Diagram content (Mermaid).
        file_path: Path to save the file.
        format: Output format ("mermaid").

    Raises:
        ValueError: If format is not recognized.

    Example:
        >>> diagram = to_mermaid(entities, relationships)
        >>> save_diagram(diagram, "graph.mmd", format="mermaid")
    """
    # Add appropriate file extension if missing
    extensions = {
        "mermaid": ".mmd",
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


def to_pyvis(
    entities: list[Entity],
    relationships: list[Relationship],
    *,
    height: str = "600px",
    width: str = "100%",
    physics_config: dict[str, Any] | None = None,
    entity_color: str | None = None,
    relationship_color: str = "#2B7CE9",
    notebook: bool = False,
    directed: bool = True,
    scheme: str | StyleScheme = "default",
    color_by_type: bool = True,
    show_type_in_label: bool = True,
) -> Any:
    """
    Convert entities and relationships to a PyVis Network object.

    Creates an interactive force-directed graph with drag, zoom, and hover capabilities.
    Perfect for interactive exploration, notebooks, and presentations.

    Args:
        entities: List of Entity objects to visualize
        relationships: List of Relationship objects to visualize
        height: Canvas height (e.g., "600px", "100vh")
        width: Canvas width (e.g., "100%", "800px")
        physics_config: Custom physics configuration for layout
        entity_color: Default color for entities (hex). If None and color_by_type=True,
            colors are assigned from the style scheme.
        relationship_color: Color for relationship edges (hex)
        notebook: True if running in Jupyter notebook
        directed: Whether to show directed edges (arrows)
        scheme: Style scheme for coloring nodes by type
        color_by_type: If True, color nodes by their entity type using the scheme
        show_type_in_label: If True, include entity type in node labels

    Returns:
        PyVis Network object (call .save_graph(path) to export)

    Raises:
        ImportError: If networkx or pyvis not installed

    Example:
        ```python
        from cogent.graph.visualization import to_pyvis

        # Create network with type-based coloring
        net = to_pyvis(entities, relationships, color_by_type=True)
        net.save_graph("graph.html")

        # Custom physics
        net = to_pyvis(
            entities,
            relationships,
            physics_config={
                "barnesHut": {
                    "gravitationalConstant": -3000,
                    "centralGravity": 0.5,
                    "springLength": 200
                }
            }
        )
        ```
    """
    try:
        import networkx as nx
        from pyvis.network import Network
    except ImportError as e:
        raise ImportError(
            "PyVis and NetworkX required for interactive visualization. "
            "Install with: uv add networkx pyvis"
        ) from e

    # Get style scheme
    if isinstance(scheme, str):
        scheme = get_scheme(scheme)

    # Build NetworkX graph
    G = nx.DiGraph() if directed else nx.Graph()

    # Map for PyVis shapes from Mermaid shapes
    shape_map = {
        "rectangle": "box",
        "rounded": "box",
        "circle": "circle",
        "hexagon": "hexagon",
    }

    # Add entities as nodes
    for entity in entities:
        attrs = entity.attributes or {}
        
        # Get node style from scheme
        node_style = scheme.get_node_style(entity.entity_type)
        
        # Determine node color
        if color_by_type:
            node_color = node_style.color
            border_color = node_style.border_color
        else:
            node_color = entity_color or "#7BE382"
            border_color = "#333333"
        
        # Build label: prefer "name" attribute, fallback to ID
        name = attrs.get("name", entity.id)
        if show_type_in_label:
            label = f"{name}\n({entity.entity_type})"
        else:
            label = name
        
        # Build tooltip
        tooltip_lines = [f"<b>{entity.entity_type}</b>: {entity.id}"]
        for k, v in attrs.items():
            tooltip_lines.append(f"<i>{k}</i>: {v}")
        tooltip_html = "<br>".join(tooltip_lines)
        
        # Determine shape
        pyvis_shape = shape_map.get(node_style.shape, "dot")

        G.add_node(
            entity.id,
            title=tooltip_html,
            color=node_color,  # Simple string color for PyVis
            borderWidth=2,
            borderWidthSelected=3,
            label=label,
            shape=pyvis_shape,
            group=entity.entity_type,  # For grouping/legend
        )

    # Add relationships as edges
    for rel in relationships:
        edge_style = scheme.get_edge_style(rel.relation)
        
        # Determine edge width
        width = max(1, edge_style.width)
        
        # Determine edge style (dashes for dashed/dotted)
        dashes = edge_style.style in ("dashed", "dotted")
        
        G.add_edge(
            rel.source_id,
            rel.target_id,
            label=rel.relation,
            title=f"<b>{rel.relation}</b>",
            color=relationship_color,
            width=width,
            dashes=dashes,
        )

    # Create PyVis network
    net = Network(
        height=height,
        width=width,
        notebook=notebook,
        directed=directed,
        cdn_resources="remote",
    )
    net.from_nx(G)

    # Manually set node colors (from_nx doesn't preserve the color attribute properly)
    # Build a mapping of node_id -> color from our entities
    node_color_map = {}
    for entity in entities:
        node_style = scheme.get_node_style(entity.entity_type)
        if color_by_type:
            node_color_map[entity.id] = node_style.color
        else:
            node_color_map[entity.id] = entity_color or "#7BE382"
    
    # Apply colors to PyVis nodes
    for node in net.nodes:
        if node["id"] in node_color_map:
            node["color"] = node_color_map[node["id"]]

    # Apply physics configuration
    default_physics = {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 150,
            }
        },
        "edges": {
            "font": {"size": 12, "align": "middle"},
            "arrows": {"to": {"enabled": True}},
        },
    }

    if physics_config:
        # Merge user config with defaults
        config = default_physics.copy()
        config.update(physics_config)
    else:
        config = default_physics

    net.set_options(f"var options = {json.dumps(config)}")

    return net


def to_gravis(
    entities: list[Entity],
    relationships: list[Relationship],
    *,
    mode: str = "2d",
    renderer: str = "vis",
    node_size_data: str | None = None,
    node_label_data: str | None = None,
    edge_curvature: float = 0.0,
    zoom_factor: float = 0.75,
    show_node_label: bool = True,
    show_edge_label: bool = False,
    layout_algorithm: str | None = None,
    graph_height: int = 450,
    **kwargs: Any,
) -> Any:
    """
    Convert entities and relationships to interactive gravis visualization.

    Creates web-based interactive 2D or 3D visualizations using d3.js, vis.js,
    or three.js. Supports rich metadata, images in nodes, and flexible styling.

    Args:
        entities: List of Entity objects to visualize
        relationships: List of Relationship objects to visualize
        mode: Visualization mode - "2d" or "3d"
        renderer: Rendering engine:
            - "d3": d3.js-based 2D visualization
            - "vis": vis.js-based 2D visualization (default, force-directed)
            - "three": three.js-based 3D visualization
        node_size_data: Entity attribute to map to node size (e.g., "degree")
        node_label_data: Entity attribute to use for labels (default: "label")
        edge_curvature: Curvature of edges (0.0 = straight, higher = more curved)
        zoom_factor: Initial zoom level (default: 0.75)
        show_node_label: Whether to show node labels
        show_edge_label: Whether to show edge labels (relationship types)
        layout_algorithm: Force layout algorithm (vis.js only):
            - "barnesHut" (default)
            - "forceAtlas2Based"
            - "repulsion"
            - "hierarchicalRepulsion"
        graph_height: Height of visualization in pixels (default: 450)
        **kwargs: Additional gravis parameters (node_size_factor, etc.)

    Returns:
        gravis Figure object with methods:
            - .display() - Open in browser
            - .export_html(path) - Save as standalone HTML
            - .export_svg(path) - Save as SVG (2D only)
            - .export_png(path) - Save as PNG (requires Selenium)
            - .export_jpg(path) - Save as JPG (requires Selenium)

    Raises:
        ImportError: If gravis or networkx not installed

    Example:
        ```python
        from cogent.graph.visualization import to_gravis

        # Basic 2D interactive (vis.js)
        fig = to_gravis(entities, relationships)
        fig.display()  # Opens in browser

        # 3D visualization
        fig = to_gravis(entities, relationships, mode="3d", renderer="three")
        fig.export_html("graph_3d.html")

        # d3.js with custom styling
        fig = to_gravis(
            entities,
            relationships,
            renderer="d3",
            node_size_data="degree",
            edge_curvature=0.3,
            zoom_factor=0.8
        )
        fig.export_svg("graph.svg")

        # From KnowledgeGraph
        entities = kg.get_all_entities()
        relationships = kg.get_all_relationships()
        fig = to_gravis(entities, relationships)
        fig.display()
        ```
    """
    try:
        import gravis as gv
        import networkx as nx
    except ImportError as e:
        raise ImportError(
            "gravis and NetworkX required for interactive web visualization. "
            "Install with: uv add gravis"
        ) from e

    # Build NetworkX graph
    G = nx.DiGraph()

    # Add entities as nodes with all attributes
    for entity in entities:
        node_attrs = {
            "entity_type": entity.entity_type,
            "label": entity.attributes.get("name", entity.id),
            **entity.attributes,
        }
        G.add_node(entity.id, **node_attrs)

    # Add relationships as edges
    for rel in relationships:
        edge_label = rel.relation if show_edge_label else ""
        G.add_edge(
            rel.source_id,
            rel.target_id,
            relation=rel.relation,
            label=edge_label,
        )

    # Prepare kwargs for gravis
    gravis_kwargs: dict[str, Any] = {
        "graph_height": graph_height,
        "zoom_factor": zoom_factor,
        "show_node_label": show_node_label,
        "show_edge_label": show_edge_label,
    }

    # Data source mappings
    if node_size_data:
        gravis_kwargs["node_size_data_source"] = node_size_data
    if node_label_data:
        gravis_kwargs["node_label_data_source"] = node_label_data
    else:
        gravis_kwargs["node_label_data_source"] = "label"

    # Rendering options
    if edge_curvature != 0.0:
        gravis_kwargs["edge_curvature"] = edge_curvature

    # Layout algorithm (vis.js only)
    if renderer == "vis" and layout_algorithm:
        gravis_kwargs["layout_algorithm"] = layout_algorithm

    # Merge additional kwargs
    gravis_kwargs.update(kwargs)

    # Select renderer and create figure
    if mode == "3d" or renderer == "three":
        fig = gv.three(G, **gravis_kwargs)
    elif renderer == "d3":
        fig = gv.d3(G, **gravis_kwargs)
    else:  # vis (default)
        fig = gv.vis(G, **gravis_kwargs)

    return fig
