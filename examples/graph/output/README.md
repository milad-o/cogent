# Graph Visualization Demo Outputs

Example outputs from the graph visualization demo showcasing different visualization formats.

## 📊 View Interactive Visualizations

### PyVis Network (Force-Directed)
**[🔗 View PyVis Visualization](https://raw.githack.com/milad-o/cogent/main/examples/graph/output/graph_pyvis.html)**

### gravis 2D/3D Interactive

- **[🔗 View 2D vis.js](https://raw.githack.com/milad-o/cogent/main/examples/graph/output/graph_gravis_2d.html)**
- **[🔗 View 2D d3.js](https://raw.githack.com/milad-o/cogent/main/examples/graph/output/graph_gravis_d3.html)**
- **[🔗 View 3D three.js](https://raw.githack.com/milad-o/cogent/main/examples/graph/output/graph_gravis_3d.html)**

## 📝 Mermaid Diagram

**[📖 View Mermaid Diagram](graph.md)** - Renders automatically in GitHub

## 🚀 Generate Your Own

```bash
cd examples/graph
uv run python visualization_demo.py
```

## 🔧 Files in This Directory

| File | Format | Description |
|------|--------|-------------|
| `graph.md` | Markdown + Mermaid | GitHub/VS Code friendly |
| `graph.mmd` | Mermaid text | Raw Mermaid code |
| `graph_pyvis.html` | PyVis HTML | Interactive force-directed network |
| `graph_gravis_2d.html` | gravis vis.js | 2D interactive with physics |
| `graph_gravis_d3.html` | gravis d3.js | 2D with smooth animations |
| `graph_gravis_3d.html` | gravis three.js | 3D interactive exploration |
