# Graph Module

The `agenticflow.graph` module provides unified visualization for agents, topologies, and flows.

## Overview

Visualize any entity as a diagram:

```python
from agenticflow import Agent

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
print(mermaid_code)
# graph TD
#   assistant[assistant]
#   search[🔧 search]
#   write[🔧 write]
#   assistant --> search
#   assistant --> write

# ASCII art for terminal
ascii_art = view.ascii()
print(ascii_art)
# ┌─────────────┐
# │  assistant  │
# └──────┬──────┘
#    ┌───┴───┐
#    ▼       ▼
# [search] [write]

# Graphviz DOT format
dot_code = view.dot()

# mermaid.ink URL (shareable)
url = view.url()
print(url)  # https://mermaid.ink/img/...

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

# Explicit format
view.save("output", format="png")
```

---

## Agent Graphs

Visualize agent structure:

```python
from agenticflow import Agent
from agenticflow.capabilities import RAG, WebSearch

agent = Agent(
    name="researcher",
    model=model,
    tools=[search, analyze, summarize],
    capabilities=[RAG(retriever), WebSearch()],
)

view = agent.graph()
print(view.mermaid())
```

Output:
```mermaid
graph TD
    researcher[🤖 researcher]
    
    subgraph Tools
        search[🔧 search]
        analyze[🔧 analyze]
        summarize[🔧 summarize]
    end
    
    subgraph Capabilities
        RAG[📚 RAG]
        WebSearch[🌐 WebSearch]
    end
    
    researcher --> Tools
    researcher --> Capabilities
```

---

## Topology Graphs

Visualize multi-agent coordination:

### Supervisor

```python
from agenticflow.topologies import Supervisor

topology = Supervisor(
    coordinator=manager,
    workers=[analyst, writer, reviewer],
)

view = topology.graph()
print(view.mermaid())
```

Output:
```mermaid
graph TD
    manager[👔 manager]
    analyst[🤖 analyst]
    writer[🤖 writer]
    reviewer[🤖 reviewer]
    
    manager -->|delegates| analyst
    manager -->|delegates| writer
    manager -->|delegates| reviewer
```

### Pipeline

```python
from agenticflow.topologies import Pipeline

topology = Pipeline(stages=[researcher, writer, editor])

view = topology.graph()
print(view.mermaid())
```

Output:
```mermaid
graph LR
    researcher[🤖 researcher] --> writer[🤖 writer] --> editor[🤖 editor]
```

### Mesh

```python
from agenticflow.topologies import Mesh

topology = Mesh(agents=[analyst1, analyst2, analyst3])

view = topology.graph()
print(view.mermaid())
```

Output:
```mermaid
graph TD
    analyst1[🤖 analyst1]
    analyst2[🤖 analyst2]
    analyst3[🤖 analyst3]
    
    analyst1 <-->|round 1| analyst2
    analyst1 <-->|round 1| analyst3
    analyst2 <-->|round 1| analyst3
```

---

## Flow Graphs

Visualize complex flows:

```python
from agenticflow import Flow

flow = Flow(
    name="content-pipeline",
    agents=[researcher, writer, editor],
    topology="pipeline",
)

view = flow.graph()
print(view.mermaid())
```

---

## Configuration

### GraphConfig

```python
from agenticflow.graph import GraphConfig, GraphTheme, GraphDirection

config = GraphConfig(
    direction=GraphDirection.TOP_DOWN,  # or LEFT_RIGHT
    theme=GraphTheme.DEFAULT,           # or DARK, FOREST, NEUTRAL
    show_tools=True,
    show_capabilities=True,
    show_interceptors=False,
    node_spacing=50,
    rank_spacing=100,
)

view = agent.graph(config=config)
```

### GraphDirection

| Direction | Description |
|-----------|-------------|
| `TOP_DOWN` | Vertical, top to bottom (TD) |
| `LEFT_RIGHT` | Horizontal, left to right (LR) |
| `BOTTOM_UP` | Vertical, bottom to top (BU) |
| `RIGHT_LEFT` | Horizontal, right to left (RL) |

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
from agenticflow.observability import ExecutionTracer

tracer = ExecutionTracer()
result = await agent.run("Query", tracer=tracer)

# Get execution graph
view = tracer.graph()
print(view.mermaid())
```

Output:
```mermaid
graph TD
    start([Start])
    llm1[LLM Call]
    tool1[search]
    llm2[LLM Call]
    end_node([End])
    
    start --> llm1
    llm1 -->|tool decision| tool1
    tool1 --> llm2
    llm2 --> end_node
    
    style start fill:#90EE90
    style end_node fill:#90EE90
    style tool1 fill:#FFE4B5
```

---

## ASCII Rendering

Terminal-friendly diagrams:

```python
view = topology.graph()
print(view.ascii())

# Output:
# ┌─────────────┐
# │   manager   │
# └──────┬──────┘
#        │
#   ┌────┼────┐
#   ▼    ▼    ▼
# [ana] [wri] [rev]
```

### DAG ASCII Rendering

```python
from agenticflow.observability import render_dag_ascii

dag = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": ["D"],
    "D": [],
}

print(render_dag_ascii(dag))
```

---

## Interactive Viewing

Open in browser:

```python
view = agent.graph()

# Open mermaid.ink in default browser
view.open()

# Or get URL to share
url = view.url()
print(f"View at: {url}")
```

---

## Integration

### Jupyter Notebooks

```python
from agenticflow import Agent

agent = Agent(name="assistant", model=model)
view = agent.graph()

# Display inline in notebook
view.display()

# Or use IPython display
from IPython.display import HTML
HTML(view.html())
```

### VS Code

```python
# Save as .mmd file for Mermaid preview extension
view.save("diagram.mmd")
```

### Documentation

```markdown
# My Agent Architecture

```mermaid
{view.mermaid()}
```
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
| `display()` | `None` | Display in notebook |

### Configuration Classes

| Class | Description |
|-------|-------------|
| `GraphConfig` | Diagram configuration |
| `GraphTheme` | Color themes |
| `GraphDirection` | Diagram direction |
