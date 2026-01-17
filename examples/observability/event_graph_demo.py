"""
Event Graph Visualization Demo

Demonstrates how to use observer.event_graph() to visualize event-driven flows.
"""

import asyncio
import sys
from pathlib import Path

# Add examples to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import Agent
from agenticflow.flow.reactive import ReactiveFlow  # Use ReactiveFlow directly
from agenticflow.observability import Observer


async def demo_basic_graph() -> None:
    """Demo 1: Basic event graph without timing."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Event Graph")
    print("=" * 80)

    model = get_model()
    
    # Create agents
    classifier = Agent(
        name="Classifier",
        model=model,
        system_prompt="Classify the task type.",
    )
    
    processor = Agent(
        name="Processor",
        model=model,
        system_prompt="Process the classified task.",
    )
    
    finalizer = Agent(
        name="Finalizer",
        model=model,
        system_prompt="Finalize the processing.",
    )

    observer = Observer.trace()
    flow = ReactiveFlow(observer=observer)
    
    flow.register(classifier, on="task.created", emits="task.classified")
    flow.register(processor, on="task.classified", emits="processing.done")
    flow.register(finalizer, on="processing.done", emits="flow.done")

    await flow.run("Analyze customer feedback", initial_event="task.created")

    # Generate graph
    print("\n📊 Event Flow Graph:")
    print("-" * 80)
    graph = observer.event_graph()
    print(graph)
    print("-" * 80)


async def demo_graph_with_timing() -> None:
    """Demo 2: Event graph with timing annotations."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Event Graph with Timing")
    print("=" * 80)

    model = get_model()
    
    researcher = Agent(
        name="Researcher",
        model=model,
        system_prompt="Research the topic briefly.",
    )
    
    writer = Agent(
        name="Writer",
        model=model,
        system_prompt="Write a summary based on research.",
    )

    observer = Observer.trace()
    flow = ReactiveFlow(observer=observer)
    
    flow.register(researcher, on="research.start", emits="research.done")
    flow.register(writer, on="research.done", emits="flow.done")

    await flow.run("Quantum computing", initial_event="research.start")

    # Generate graph with timing
    print("\n📊 Event Flow Graph (with timing):")
    print("-" * 80)
    graph = observer.event_graph(include_timing=True)
    print(graph)
    print("-" * 80)


async def demo_complex_flow() -> None:
    """Demo 3: Complex multi-branch flow."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Complex Multi-Branch Flow")
    print("=" * 80)

    model = get_model()
    
    # Router that determines priority
    router = Agent(
        name="Router",
        model=model,
        system_prompt="Determine if task is urgent (mention of 'critical', 'urgent', 'emergency') or normal priority. Respond with just 'urgent' or 'normal'.",
    )
    
    urgent_handler = Agent(
        name="UrgentHandler",
        model=model,
        system_prompt="Handle urgent tasks immediately with high priority response.",
    )
    
    normal_handler = Agent(
        name="NormalHandler",
        model=model,
        system_prompt="Process normal priority tasks with standard response.",
    )
    
    reviewer = Agent(
        name="Reviewer",
        model=model,
        system_prompt="Review completed work and provide feedback.",
    )

    observer = Observer.trace()
    flow = ReactiveFlow(observer=observer)
    
    # Simple routing flow
    flow.register(router, on="task.created", emits="task.urgent")  # Simplified for demo
    flow.register(urgent_handler, on="task.urgent", emits="task.handled")
    flow.register(reviewer, on="task.handled", emits="flow.done")

    await flow.run("Critical security issue!", initial_event="task.created")

    # Generate graph
    print("\n📊 Event Flow Graph (multi-branch):")
    print("-" * 80)
    graph = observer.event_graph(include_timing=True)
    print(graph)
    print("-" * 80)


async def demo_export_graph() -> None:
    """Demo 4: Export graph to file."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Export Graph to File")
    print("=" * 80)

    from pathlib import Path
    
    model = get_model()
    
    agent1 = Agent(name="Agent1", model=model, system_prompt="Step 1")
    agent2 = Agent(name="Agent2", model=model, system_prompt="Step 2")
    agent3 = Agent(name="Agent3", model=model, system_prompt="Step 3")

    observer = Observer.trace()
    flow = ReactiveFlow(observer=observer)
    
    flow.register(agent1, on="start", emits="step1.done")
    flow.register(agent2, on="step1.done", emits="step2.done")
    flow.register(agent3, on="step2.done", emits="flow.done")

    await flow.run("Process this", initial_event="start")

    # Export to file
    graph = observer.event_graph(include_timing=True)
    output_file = Path("flow_graph.mmd")
    output_file.write_text(graph)
    
    print(f"\n✅ Graph exported to: {output_file.absolute()}")
    print("\n💡 You can:")
    print("  1. View in VS Code with Mermaid extension")
    print("  2. Paste into https://mermaid.live")
    print("  3. Include in documentation")
    print("\n📄 Graph content:")
    print("-" * 80)
    print(graph)
    print("-" * 80)


async def main() -> None:
    """Run all demos."""
    await demo_basic_graph()
    await demo_graph_with_timing()
    await demo_complex_flow()
    await demo_export_graph()
    
    print("\n\n" + "=" * 80)
    print("✅ All event graph demos completed!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • observer.event_graph() generates Mermaid diagrams")
    print("  • Use include_timing=True to see reactor performance")
    print("  • Export to .mmd files for documentation")
    print("  • Great for debugging reactive flows")
    print("  • Events shown as rounded boxes, reactors as rectangles")
    print("\n📚 See docs/observability.md for complete guide")


if __name__ == "__main__":
    asyncio.run(main())
