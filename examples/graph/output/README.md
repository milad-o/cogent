# Graph Visualization Demo Outputs

This directory contains example outputs from the graph visualization demo showcasing different visualization formats.

## 📊 View Interactive Visualizations

GitHub doesn't render HTML files directly. Use these links to view the interactive visualizations:

### PyVis Network (Force-Directed)
- **[🔗 View PyVis Visualization](https://htmlpreview.github.io/?https://github.com/milad-o/cogent/blob/main/examples/graph/output/graph_pyvis.html)**
- Drag nodes, zoom, pan
- Physics simulation with attractive/repulsive forces
- Type-based colors

### gravis 2D/3D Interactive

- **[🔗 View 2D vis.js](https://htmlpreview.github.io/?https://github.com/milad-o/cogent/blob/main/examples/graph/output/graph_gravis_2d.html)**  
  Force-directed with drag/zoom/hover tooltips

- **[🔗 View 2D d3.js](https://htmlpreview.github.io/?https://github.com/milad-o/cogent/blob/main/examples/graph/output/graph_gravis_d3.html)**  
  Smooth animations with curved edges

- **[🔗 View 3D three.js](https://htmlpreview.github.io/?https://github.com/milad-o/cogent/blob/main/examples/graph/output/graph_gravis_3d.html)**  
  Interactive 3D exploration

## 📝 Mermaid Diagram

The Mermaid diagram renders automatically in GitHub:

**[📖 View Mermaid Diagram](graph.md)** - Click to see the rendered Solar System knowledge graph

Or view the raw Mermaid code: [graph.mmd](graph.mmd)

## 🚀 Generate Your Own

Run the demo locally to generate visualizations for your own knowledge graphs:

```bash
cd examples/graph
uv run python visualization_demo.py
```

All outputs will be saved to this directory.

## 🔧 Files in This Directory

| File | Format | Description |
|------|--------|-------------|
| `graph.md` | Markdown + Mermaid | GitHub/VS Code friendly, renders automatically |
| `graph.mmd` | Mermaid text | Raw Mermaid diagram code |
| `graph_pyvis.html` | PyVis HTML | Interactive force-directed network |
| `graph_gravis_2d.html` | gravis vis.js | 2D interactive with physics |
| `graph_gravis_d3.html` | gravis d3.js | 2D with smooth animations |
| `graph_gravis_3d.html` | gravis three.js | 3D interactive exploration |

## 📚 Alternative Viewing Methods

### Method 1: htmlpreview.github.io (Used Above)
Simple, no setup required. The links above use this service.

### Method 2: raw.githack.com (CDN with Caching)
Replace the repository path in this template:
```
https://raw.githack.com/milad-o/cogent/main/examples/graph/output/[filename].html
```

### Method 3: Clone and Open Locally
```bash
git clone https://github.com/milad-o/cogent.git
cd cogent/examples/graph/output
# Open any .html file in your browser
```

### Method 4: GitHub Pages (For Production)
Enable GitHub Pages in repository settings to serve HTML files directly.
