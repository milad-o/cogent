# Graph Visualization Examples

Demonstrations of Cogent's knowledge graph visualization capabilities.

## 🎨 Visualization Demo

**[visualization_demo.py](visualization_demo.py)** - Complete demo showcasing all visualization formats:
- Mermaid diagrams (text-based, GitHub/docs friendly)
- PyVis interactive HTML (force-directed networks)
- gravis 2D/3D visualizations (d3.js/vis.js/three.js)

### Run the Demo

```bash
uv run python examples/graph/visualization_demo.py
```

All outputs are saved to [`output/`](output/) directory.

### 👀 View Example Outputs

**[See the generated visualizations →](output/README.md)**

The output directory contains example visualizations of a Solar System knowledge graph with clickable links to view the interactive HTML files.

## 📊 Supported Formats

| Format | Best For | Interactive | GitHub Friendly |
|--------|----------|-------------|-----------------|
| **Mermaid** | Documentation, GitHub READMEs | No | ✅ Renders automatically |
| **PyVis** | Quick prototyping, force-directed layouts | Yes | Via htmlpreview |
| **gravis** | Production dashboards, 3D exploration | Yes | Via htmlpreview |

## 🔧 Usage in Your Code

```python
from cogent.graph import Graph
from cogent.graph.visualization import to_mermaid, to_pyvis, to_gravis

# Build your knowledge graph
graph = Graph()
await graph.add_entity("earth", "Planet", name="Earth")
await graph.add_entity("sun", "Star", name="Sun")
await graph.add_relationship("earth", "orbits", "sun")

# Get entities and relationships
entities = await graph.get_all_entities()
relationships = await graph.get_relationships()

# Mermaid (markdown/GitHub)
mermaid = to_mermaid(entities, relationships)
Path("graph.md").write_text(f"```mermaid\n{mermaid}\n```")

# PyVis (interactive HTML)
net = to_pyvis(entities, relationships)
net.save_graph("graph.html")

# gravis (2D/3D)
fig = to_gravis(entities, relationships, mode="3d", renderer="three")
fig.export_html("graph_3d.html")
```

See the [full documentation](../../docs/graph.md) for more details.
