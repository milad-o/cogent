"""Graph Visualization Demo - All Formats"""

import asyncio
from pathlib import Path

from cogent.graph import Graph
from cogent.graph.visualization.styles import create_scheme_from_entities


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
    graph = await build_sample_graph()
    entities = await graph.get_all_entities()
    relationships = await graph.get_relationships()
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    for old_file in output_dir.glob("graph_gravis_*.html"):
        old_file.unlink()
    
    scheme = create_scheme_from_entities(entities)
    
    # Mermaid
    from cogent.graph.visualization import to_mermaid, render_mermaid_to_image
    
    mermaid_code = to_mermaid(entities, relationships, direction="LR", group_by_type=True, scheme=scheme)
    (output_dir / "graph.mmd").write_text(mermaid_code)
    
    markdown_content = f"""# Solar System Knowledge Graph

```mermaid
{mermaid_code}
```
"""
    (output_dir / "graph.md").write_text(markdown_content)
    
    try:
        await render_mermaid_to_image(mermaid_code, str(output_dir / "graph.svg"), format="svg")
        await render_mermaid_to_image(mermaid_code, str(output_dir / "graph.png"), format="png")
    except FileNotFoundError:
        pass
    
    # PyVis
    from cogent.graph.visualization import to_pyvis
    import re
    
    net = to_pyvis(entities, relationships, height="750px", width="100%", directed=True, 
                   color_by_type=True, show_type_in_label=True, scheme=scheme)
    pyvis_path = output_dir / "graph_pyvis.html"
    net.save_graph(str(pyvis_path))
    
    html_content = pyvis_path.read_text()
    html_content = re.sub(r"width:\s*\d+;", "width: 100%;", html_content)
    html_content = re.sub(r"height:\s*(\d+)px;", r"height: \1px;", html_content)
    pyvis_path.write_text(html_content)
    
    # gravis
    try:
        from cogent.graph.visualization import to_gravis
        
        fig_vis = to_gravis(entities, relationships, mode="2d", renderer="vis", 
                           show_node_label=True, show_edge_label=True, graph_height=750,
                           color_by_type=True, scheme=scheme)
        fig_vis.export_html(str(output_dir / "graph_gravis_2d.html"))
        
        fig_d3 = to_gravis(entities, relationships, mode="2d", renderer="d3",
                          show_node_label=True, show_edge_label=False, zoom_factor=0.8,
                          color_by_type=True, scheme=scheme, edge_curvature=0.2)
        fig_d3.export_html(str(output_dir / "graph_gravis_d3.html"))
        
        fig_3d = to_gravis(entities, relationships, mode="3d", renderer="three",
                          show_node_label=True, show_edge_label=False,
                          color_by_type=True, scheme=scheme)
        fig_3d.export_html(str(output_dir / "graph_gravis_3d.html"))
    except (UnicodeEncodeError, Exception):
        pass
    
    print(f"Generated visualizations in: {output_dir.resolve()}")


if __name__ == "__main__":
    asyncio.run(demo_all_visualizations())
