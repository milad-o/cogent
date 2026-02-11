"""Graph Visualization Demo - All Formats

Demonstrates all visualization capabilities:
1. Mermaid diagrams (text-based, GitHub/docs friendly)
2. PyVis interactive HTML (force-directed, physics simulation)
3. iplotx publication-quality plots (PDF/PNG/SVG, multiple layouts)
4. gravis interactive 2D/3D web visualizations (d3.js/vis.js/three.js)
5. Graphviz DOT (for external tools)
6. GraphML/Cytoscape (exchange formats)
"""

import asyncio
from pathlib import Path

from cogent.graph import Entity, Graph, Relationship


async def build_sample_graph() -> Graph:
    """Build a small knowledge graph for visualization."""
    graph = Graph()

    # Create a simple org chart + tech stack
    # People
    await graph.add_entity("alice", "Person", name="Alice Chen", role="CTO")
    await graph.add_entity("bob", "Person", name="Bob Kumar", role="Team Lead")
    await graph.add_entity("carol", "Person", name="Carol Martinez", role="Engineer")
    await graph.add_entity("dave", "Person", name="Dave Lee", role="Engineer")

    # Teams
    await graph.add_entity("platform", "Team", name="Platform Team", size=4)
    await graph.add_entity("ml", "Team", name="ML Team", size=3)

    # Technologies
    await graph.add_entity("python", "Technology", name="Python", category="Language")
    await graph.add_entity(
        "kubernetes", "Technology", name="Kubernetes", category="Infrastructure"
    )
    await graph.add_entity(
        "postgres", "Technology", name="PostgreSQL", category="Database"
    )

    # Relationships
    await graph.add_relationship("bob", "reports_to", "alice")
    await graph.add_relationship("carol", "reports_to", "bob")
    await graph.add_relationship("dave", "reports_to", "bob")

    await graph.add_relationship("bob", "leads", "platform")
    await graph.add_relationship("alice", "oversees", "ml")

    await graph.add_relationship("carol", "expert_in", "python")
    await graph.add_relationship("dave", "expert_in", "kubernetes")
    await graph.add_relationship("bob", "expert_in", "postgres")

    await graph.add_relationship("platform", "uses", "kubernetes")
    await graph.add_relationship("platform", "uses", "postgres")
    await graph.add_relationship("ml", "uses", "python")

    return graph


async def demo_all_visualizations():
    """Demonstrate all visualization formats."""
    print("=" * 60)
    print("Graph Visualization Demo - All Formats")
    print("=" * 60)

    # Build sample graph
    print("\n📊 Building sample knowledge graph...")
    graph = await build_sample_graph()

    # Get data for visualization
    entities = await graph.get_all_entities()
    relationships = await graph.get_relationships()
    print(f"✅ Graph: {len(entities)} entities, {len(relationships)} relationships\n")

    # Output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # === 1. Mermaid Diagram ===
    print("1️⃣  Mermaid Diagram (text-based)")
    print("-" * 40)
    from cogent.graph.visualization import to_mermaid

    mermaid_code = to_mermaid(entities, relationships, direction="LR")
    mermaid_path = output_dir / "graph.mmd"
    mermaid_path.write_text(mermaid_code)
    print(f"✅ Saved: {mermaid_path.name}")
    print("   - Renders in GitHub markdown, VS Code, Notion")
    print("   - Convert to image: mmdc -i graph.mmd -o graph.svg\n")

    # === 2. PyVis Interactive HTML ===
    print("2️⃣  PyVis Interactive (force-directed)")
    print("-" * 40)
    from cogent.graph.visualization import to_pyvis

    net = to_pyvis(
        entities,
        relationships,
        height="750px",
        width="100%",
        directed=True,
    )
    pyvis_path = output_dir / "graph_pyvis.html"
    net.save_graph(str(pyvis_path))
    print(f"✅ Saved: {pyvis_path.name}")
    print("   - Drag nodes, zoom, pan")
    print("   - Physics simulation (attractive/repulsive forces)")
    print(f"   - Open: file://{pyvis_path.absolute()}\n")

    # === 3. iplotx Publication-Quality ===
    print("3️⃣  iplotx Publication Plots (matplotlib)")
    print("-" * 40)
    from cogent.graph.visualization import to_iplotx

    layouts = {
        "spring": "Force-directed spring layout",
        "circular": "Circular/radial layout",
        "hierarchical": "Tree/hierarchical layout",
    }

    for layout, desc in layouts.items():
        fig = to_iplotx(
            entities,
            relationships,
            layout=layout,
            figsize=(12, 8),
            node_color={
                "Person": "#90CAF9",
                "Team": "#FFCC80",
                "Technology": "#C8E6C9",
            },
            title=f"Knowledge Graph ({layout})",
            font_size=11,
        )

        # Save in multiple formats
        for fmt in ["pdf", "png", "svg"]:
            path = output_dir / f"graph_{layout}.{fmt}"
            fig.savefig(str(path), dpi=300, bbox_inches="tight")

        print(f"✅ {layout:12} → graph_{layout}.{{pdf,png,svg}}")
        print(f"   {desc}")

    print()

    # === 4. gravis Interactive 2D/3D ===
    print("4️⃣  gravis Interactive Web (d3/vis.js/three.js)")
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
        zoom_factor=0.8,
    )
    gravis_d3_path = output_dir / "graph_gravis_d3.html"
    fig_d3.export_html(str(gravis_d3_path))
    print(f"✅ 2D d3.js  → {gravis_d3_path.name}")
    print("   - d3.js-based, smooth animations")

    # 3D three.js (interactive 3D exploration)
    fig_3d = to_gravis(
        entities,
        relationships,
        mode="3d",
        renderer="three",
        show_node_label=True,
    )
    gravis_3d_path = output_dir / "graph_gravis_3d.html"
    fig_3d.export_html(str(gravis_3d_path))
    print(f"✅ 3D three.js → {gravis_3d_path.name}")
    print("   - Rotate, zoom, fly-through graph in 3D")
    print(f"   - Open: file://{gravis_2d_path.absolute()}\n")

    # === 5. Graphviz DOT ===
    print("5️⃣  Graphviz DOT (for external tools)")
    print("-" * 40)
    from cogent.graph.visualization import to_graphviz

    dot_code = to_graphviz(entities, relationships)
    dot_path = output_dir / "graph.dot"
    dot_path.write_text(dot_code)
    print(f"✅ Saved: {dot_path.name}")
    print("   - Use with: dot -Tpng graph.dot -o graph.png")
    print("   - Professional hierarchical layouts\n")

    # === 6. GraphML/Cytoscape ===
    print("6️⃣  GraphML & Cytoscape JSON (exchange formats)")
    print("-" * 40)
    from cogent.graph.visualization import to_cytoscape_json, to_graphml

    # GraphML (XML format for Gephi, yEd, etc.)
    graphml_code = to_graphml(entities, relationships)
    graphml_path = output_dir / "graph.graphml"
    graphml_path.write_text(graphml_code)
    print(f"✅ GraphML → {graphml_path.name}")
    print("   - Import into Gephi, yEd, Cytoscape Desktop")

    # Cytoscape.js JSON
    cytoscape_json = to_cytoscape_json(entities, relationships)
    cytoscape_path = output_dir / "graph_cytoscape.json"
    cytoscape_path.write_text(cytoscape_json)
    print(f"✅ Cytoscape.js → {cytoscape_path.name}")
    print("   - Use with Cytoscape.js web applications\n")

    # === Summary ===
    print("=" * 60)
    print("📁 All files saved to:", output_dir)
    print("=" * 60)
    print("\n📊 Visualization Summary:")
    print("   - Mermaid: Text-based, GitHub/docs friendly")
    print("   - PyVis: Quick interactive HTML (force-directed)")
    print("   - iplotx: Publication PDFs/PNGs (multiple layouts)")
    print("   - gravis: Advanced 2D/3D interactive web")
    print("   - Graphviz: Professional layouts (requires binary)")
    print("   - GraphML/Cytoscape: Import into other tools")
    print("\n✨ Pure Python (no system dependencies):")
    print("   → PyVis, iplotx, gravis")
    print("\n🔧 Requires external binaries:")
    print("   → Mermaid CLI (mmdc), Graphviz (dot)")


if __name__ == "__main__":
    asyncio.run(demo_all_visualizations())
