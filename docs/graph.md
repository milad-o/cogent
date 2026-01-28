# Graph Module

The `cogent.graph` module provides visualization for agents and their structure.

## Overview

Visualize any agent as a diagram:

```python
from cogent import Agent

agent = Agent(name="assistant", model=model, tools=[search, write])

# Get graph view
view = agent.graph()

# Render in any format
print(view.mermaid())   # Mermaid code
print(view.ascii())     # Terminal-friendly
print(view.dot())       # Graphviz DOT
print(view.url())       # mermaid.ink URL
print(view.html())      # Embeddable HTML

# Save to file
view.save("diagram.png")
view.save("diagram.svg")
```

---

## GraphView API

All entities return a `GraphView` from their `.graph()` method:

### Rendering Methods

```python
view = agent.graph()

# Mermaid diagram code
mermaid_code = view.mermaid()

# ASCII art for terminal
ascii_art = view.ascii()

# Graphviz DOT format
dot_code = view.dot()

# mermaid.ink URL (shareable)
url = view.url()

# Embeddable HTML
html = view.html()
```

### Saving Diagrams

```python
# Auto-detect format from extension
view.save("diagram.png")    # PNG image
view.save("diagram.svg")    # SVG vector
view.save("diagram.mmd")    # Mermaid source
view.save("diagram.dot")    # Graphviz DOT
view.save("diagram.html")   # HTML page
```

---

## Agent Graphs

Visualize agent structure:

```python
from cogent import Agent
from cogent.capabilities import WebSearch, FileSystem

agent = Agent(
    name="researcher",
    model=model,
    tools=[search, analyze, summarize],
    capabilities=[WebSearch(), FileSystem()],
)

view = agent.graph()
print(view.mermaid())
```

---

## Configuration

### GraphConfig

```python
from cogent.graph import GraphConfig, GraphTheme, GraphDirection

config = GraphConfig(
    direction=GraphDirection.TOP_DOWN,
    theme=GraphTheme.DEFAULT,
    show_tools=True,
    show_capabilities=True,
)

view = agent.graph(config=config)
```

### GraphDirection

| Direction | Description |
|-----------|-------------|
| `TOP_DOWN` | Vertical, top to bottom |
| `LEFT_RIGHT` | Horizontal, left to right |
| `BOTTOM_UP` | Vertical, bottom to top |
| `RIGHT_LEFT` | Horizontal, right to left |

### GraphTheme

| Theme | Description |
|-------|-------------|
| `DEFAULT` | Standard colors |
| `DARK` | Dark background |
| `FOREST` | Green tones |
| `NEUTRAL` | Grayscale |

---

## Execution Graphs

Visualize execution traces:

```python
from cogent.observability import ExecutionTracer

tracer = ExecutionTracer()
result = await agent.run("Query", tracer=tracer)

# Get execution graph
view = tracer.graph()
print(view.mermaid())
```

---

## Interactive Viewing

```python
view = agent.graph()

# Open mermaid.ink in default browser
view.open()

# Or get URL to share
url = view.url()
```

---

## API Reference

### GraphView Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `mermaid()` | `str` | Mermaid diagram code |
| `ascii()` | `str` | ASCII art diagram |
| `dot()` | `str` | Graphviz DOT code |
| `url()` | `str` | mermaid.ink URL |
| `html()` | `str` | Embeddable HTML |
| `save(path)` | `None` | Save to file |
| `open()` | `None` | Open in browser |
