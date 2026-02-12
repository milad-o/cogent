"""Graph Visualization Demo - All Formats

Demonstrates all visualization capabilities with type-based styling:
1. Mermaid diagrams (text-based, GitHub/docs friendly, type grouping)
2. PyVis interactive HTML (force-directed, physics, type colors)
3. gravis interactive 2D/3D web visualizations (d3.js/vis.js/three.js)

Uses a Solar System knowledge graph to showcase:
- Type-based node coloring (Planet, Moon, Star, Phenomenon)
- Grouped visualization (subgraphs by entity type)
- Multiple renderers for different use cases
- Consistent color schemes across all formats
"""

import asyncio
from pathlib import Path

from cogent.graph import Entity, Graph, Relationship
from cogent.graph.visualization.styles import StyleScheme, NodeStyle, EdgeStyle


class SolarSystemScheme(StyleScheme):
    """Custom color scheme for Solar System entities."""
    
    def __init__(self) -> None:
        """Initialize with celestial body colors."""
        super().__init__()
        
        self.node_styles = {
            "Star": NodeStyle(
                shape="circle",
                color="#FFD700",  # Gold
                border_color="#FFA500",
                text_color="#000000",
            ),
            "Planet": NodeStyle(
                shape="circle",
                color="#90CAF9",  # Blue
                border_color="#1976D2",
                text_color="#000000",
            ),
            "Moon": NodeStyle(
                shape="circle",
                color="#E0E0E0",  # Light gray
                border_color="#757575",
                text_color="#000000",
            ),
            "Phenomenon": NodeStyle(
                shape="hexagon",
                color="#FFCC80",  # Orange
                border_color="#F57C00",
                text_color="#000000",
            ),
        }
        
        self.edge_styles = {
            "orbits": EdgeStyle(color="#1976D2", width=2, style="solid"),
            "located_on": EdgeStyle(color="#F57C00", width=2, style="dashed"),
            "surrounds": EdgeStyle(color="#9C27B0", width=2, style="dashed"),
        }


async def build_sample_graph() -> Graph:
    """Build a Solar System knowledge graph for visualization."""
    graph = Graph()

    # Star
    await graph.add_entity("sun", "Star", name="Sun", type_class="G-type")

    # Planets
    await graph.add_entity("earth", "Planet", name="Earth", type_class="Terrestrial")
    await graph.add_entity("mars", "Planet", name="Mars", type_class="Terrestrial")
    await graph.add_entity("jupiter", "Planet", name="Jupiter", type_class="Gas Giant")
    await graph.add_entity("saturn", "Planet", name="Saturn", type_class="Gas Giant")

    # Moons
    await graph.add_entity("moon", "Moon", name="Luna")
    await graph.add_entity("europa", "Moon", name="Europa")
    await graph.add_entity("titan", "Moon", name="Titan")

    # Phenomena
    await graph.add_entity("great_red_spot", "Phenomenon", name="Great Red Spot")
    await graph.add_entity("saturn_rings", "Phenomenon", name="Rings of Saturn")

    # Relationships
    await graph.add_relationship("earth", "orbits", "sun")
    await graph.add_relationship("mars", "orbits", "sun")
    await graph.add_relationship("jupiter", "orbits", "sun")
    await graph.add_relationship("saturn", "orbits", "sun")

    await graph.add_relationship("moon", "orbits", "earth")
    await graph.add_relationship("europa", "orbits", "jupiter")
    await graph.add_relationship("titan", "orbits", "saturn")

    await graph.add_relationship("great_red_spot", "located_on", "jupiter")
    await graph.add_relationship("saturn_rings", "surrounds", "saturn")

    return graph


async def demo_all_visualizations():
    """Demonstrate all visualization formats."""
    print("=" * 60)
    print("🌌 Solar System - Graph Visualization Demo")
    print("=" * 60)

    # Build sample graph
    print("\n🪐 Building Solar System knowledge graph...")
    graph = await build_sample_graph()

    # Get data for visualization
    entities = await graph.get_all_entities()
    relationships = await graph.get_relationships()
    print(f"✅ Solar System Graph: {len(entities)} celestial bodies, {len(relationships)} relationships\n")

    # Output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Clean up old visualization files
    for old_file in output_dir.glob("graph_gravis_*.html"):
        old_file.unlink()
    
    # Create custom color scheme for solar system
    scheme = SolarSystemScheme()

    # === 1. Mermaid Diagram (with type grouping) ===
    print("1️⃣  Mermaid Diagram (text-based, grouped by type)")
    print("-" * 40)
    from cogent.graph.visualization import to_mermaid, render_mermaid_to_image

    mermaid_code = to_mermaid(
        entities,
        relationships,
        direction="LR",
        group_by_type=True,  # Group nodes into subgraphs by type
        scheme=scheme,  # Use solar system colors
    )
    mermaid_path = output_dir / "graph.mmd"
    mermaid_path.write_text(mermaid_code)
    print(f"✅ Saved: {mermaid_path.name}")
    
    # Render to SVG
    svg_path = output_dir / "graph.svg"
    await render_mermaid_to_image(mermaid_code, str(svg_path), format="svg")
    print(f"✅ Rendered: {svg_path.name}")
    
    # Render to PNG
    png_path = output_dir / "graph.png"
    await render_mermaid_to_image(mermaid_code, str(png_path), format="png")
    print(f"✅ Rendered: {png_path.name}")
    
    print("   - Renders in GitHub markdown, VS Code, Notion")
    print("   - Type-based subgraphs (Star, Planet, Moon, Phenomenon)")
    print("   - Requires: npm install -g @mermaid-js/mermaid-cli\n")

    # === 2. PyVis Interactive HTML (with type colors) ===
    print("2️⃣  PyVis Interactive (force-directed, type colors)")
    print("-" * 40)
    from cogent.graph.visualization import to_pyvis

    net = to_pyvis(
        entities,
        relationships,
        height="750px",
        width="100%",
        directed=True,
        color_by_type=True,  # Color nodes by entity type
        show_type_in_label=True,  # Show type in node labels
        scheme=scheme,  # Use solar system colors
    )
    pyvis_path = output_dir / "graph_pyvis.html"
    net.save_graph(str(pyvis_path))
    
    # Fix PyVis width/height CSS bug (it outputs "2" instead of "100%")
    import re
    html_content = pyvis_path.read_text()
    html_content = re.sub(r"width:\s*\d+;", "width: 100%;", html_content)
    html_content = re.sub(r"height:\s*(\d+)px;", r"height: \1px;", html_content)
    pyvis_path.write_text(html_content)
    
    print(f"✅ Saved: {pyvis_path.name}")
    print("   - Drag nodes, zoom, pan")
    print("   - Physics simulation (attractive/repulsive forces)")
    print("   - Type-based colors (Star, Planet, Moon, Phenomenon)")
    print(f"   - Open: file://{pyvis_path.absolute()}\n")

    # === 3. gravis Interactive 2D/3D (with type colors) ===
    print("3️⃣  gravis Interactive Web (d3/vis.js/three.js, type colors)")
    print("-" * 40)
    from cogent.graph.visualization import to_gravis

    # 2D vis.js (force-directed with drag/zoom)
    fig_vis = to_gravis(
        entities,
        relationships,
        mode="2d",
        renderer="vis",
        show_node_label=True,
        show_edge_label=True,
        graph_height=750,
        color_by_type=True,  # Type-based colors
        scheme=scheme,  # Use solar system colors
    )
    gravis_2d_path = output_dir / "graph_gravis_2d.html"
    fig_vis.export_html(str(gravis_2d_path))
    print(f"✅ 2D vis.js → {gravis_2d_path.name}")
    print("   - Drag nodes, zoom, pan, hover tooltips")
    print("   - Force-directed physics simulation")

    # 2D d3.js (good for SVG export)
    fig_d3 = to_gravis(
        entities,
        relationships,
        mode="2d",
        renderer="d3",
        show_node_label=True,
        show_edge_label=False,  # Cleaner without edge labels
        zoom_factor=0.8,
        color_by_type=True,
        scheme=scheme,  # Use solar system colors
        edge_curvature=0.2,  # Curved edges for better visibility
    )
    gravis_d3_path = output_dir / "graph_gravis_d3.html"
    fig_d3.export_html(str(gravis_d3_path))
    print(f"✅ 2D d3.js  → {gravis_d3_path.name}")
    print("   - d3.js-based, smooth animations, curved edges")

    # 3D three.js (interactive 3D exploration)
    fig_3d = to_gravis(
        entities,
        relationships,
        mode="3d",
        renderer="three",
        show_node_label=True,
        show_edge_label=False,  # Cleaner in 3D without edge labels
        color_by_type=True,
        scheme=scheme,  # Use solar system colors
    )
    gravis_3d_path = output_dir / "graph_gravis_3d.html"
    fig_3d.export_html(str(gravis_3d_path))
    print(f"✅ 3D three.js → {gravis_3d_path.name}")
    print(f"\n   Open: file://{gravis_2d_path.absolute()}\n")


if __name__ == "__main__":
    asyncio.run(demo_all_visualizations())
