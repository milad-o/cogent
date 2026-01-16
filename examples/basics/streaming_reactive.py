#!/usr/bin/env python3
"""
Streaming Reactions in ReactiveFlow
===================================

Demonstrates real-time token-by-token streaming from agents in event-driven flows.

Unlike regular ReactiveFlow.run() which waits for complete responses,
run_streaming() yields tokens as they're generated, providing:
- Real-time feedback during agent processing
- Better UX with progressive output
- Lower perceived latency
- Ability to show which agent is currently active

Key Features:
1. Basic streaming - see tokens as they arrive
2. Multi-agent streaming - track which agent is speaking
3. Event-driven chaining with streaming
4. Stream progress indicators

Prerequisites:
    - Streaming-capable model (OpenAI, Anthropic, etc.)
    - Set API key: export OPENAI_API_KEY=your-key

Run:
    uv run python examples/reactive/streaming.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model


# =============================================================================
# Demo 1: Basic Streaming
# =============================================================================

async def basic_streaming():
    """Demo 1: Basic streaming with ReactiveFlow."""
    print("\n" + "=" * 70)
    print("Demo 1: Basic Streaming")
    print("=" * 70)
    
    from agenticflow import Agent
    from agenticflow.reactive import ReactiveFlow, react_to
    
    # Create agent with streaming-capable model
    assistant = Agent(
        name="assistant",
        model=get_model(),
        system_prompt="You are a helpful assistant. Be concise.",
    )
    
    # Create reactive flow
    flow = ReactiveFlow()
    flow.register(assistant, [react_to("task.created")])
    
    print("\n📝 Task: Explain streaming in 2 sentences")
    print("-" * 70)
    print()
    
    # Stream execution - tokens arrive in real-time
    async for chunk in flow.run_streaming("Explain streaming in 2 sentences"):
        print(chunk.content, end="", flush=True)
        
        if chunk.is_final:
            print()  # Newline after completion
    
    print()
    print("✅ Streaming complete!")


# =============================================================================
# Demo 2: Multi-Agent Streaming
# =============================================================================

async def multi_agent_streaming():
    """Demo 2: Streaming with multiple agents - track who's speaking."""
    print("\n" + "=" * 70)
    print("Demo 2: Multi-Agent Streaming")
    print("=" * 70)
    
    from agenticflow import Agent
    from agenticflow.reactive import ReactiveFlow, react_to
    
    # Create multiple agents
    researcher = Agent(
        name="researcher",
        model=get_model(),
        system_prompt="You research topics. Provide 2-3 key facts.",
    )
    
    writer = Agent(
        name="writer",
        model=get_model(),
        system_prompt="You write engaging summaries. Keep it to 2 sentences.",
    )
    
    # Create flow with chained agents
    flow = ReactiveFlow()
    flow.register(researcher, [react_to("task.created").emits("researcher.completed")])
    flow.register(writer, [react_to("researcher.completed")])
    
    print("\n📝 Task: Research and summarize quantum computing")
    print("-" * 70)
    
    current_agent = None
    
    async for chunk in flow.run_streaming("Research and summarize quantum computing"):
        # Show agent name when it changes
        if chunk.agent_name != current_agent:
            if current_agent is not None:
                print()  # Newline before new agent
            print(f"\n🤖 [{chunk.agent_name}]:", end=" ")
            current_agent = chunk.agent_name
        
        print(chunk.content, end="", flush=True)
        
        if chunk.is_final:
            print()  # Newline after agent completes
    
    print()
    print("✅ Multi-agent streaming complete!")


# =============================================================================
# Demo 3: Progress Indicators
# =============================================================================

async def streaming_with_progress():
    """Demo 3: Show progress indicators during streaming."""
    print("\n" + "=" * 70)
    print("Demo 3: Streaming with Progress Indicators")
    print("=" * 70)
    
    from agenticflow import Agent
    from agenticflow.reactive import ReactiveFlow, react_to
    
    # Create agents for a 3-stage pipeline
    analyzer = Agent(
        name="analyzer",
        model=get_model(),
        system_prompt="Analyze the problem. List 2 key points.",
    )
    
    planner = Agent(
        name="planner",
        model=get_model(),
        system_prompt="Create a solution plan. 2-3 steps.",
    )
    
    executor = Agent(
        name="executor",
        model=get_model(),
        system_prompt="Provide final solution. Be concise.",
    )
    
    # Create pipeline flow
    flow = ReactiveFlow()
    flow.register(analyzer, [react_to("task.created").emits("analyzed")])
    flow.register(planner, [react_to("analyzed").emits("planned")])
    flow.register(executor, [react_to("planned")])
    
    print("\n📝 Task: Build a web scraper")
    print("-" * 70)
    
    # Track progress
    agents_completed = 0
    total_agents = 3
    current_agent = None
    
    async for chunk in flow.run_streaming("Build a web scraper"):
        # Update progress when agent changes
        if chunk.agent_name != current_agent:
            if current_agent is not None:
                agents_completed += 1
            
            current_agent = chunk.agent_name
            progress = f"[{agents_completed + 1}/{total_agents}]"
            print(f"\n\n{progress} 🤖 {chunk.agent_name}:")
            print("-" * 70)
        
        print(chunk.content, end="", flush=True)
    
    print("\n")
    print("=" * 70)
    print("✅ Pipeline streaming complete!")


# =============================================================================
# Demo 4: Conditional Streaming
# =============================================================================

async def conditional_streaming():
    """Demo 4: Different agents stream based on event data."""
    print("\n" + "=" * 70)
    print("Demo 4: Conditional Streaming (Event-Driven Routing)")
    print("=" * 70)
    
    from agenticflow import Agent
    from agenticflow.reactive import ReactiveFlow, react_to
    
    # Create specialized agents
    python_expert = Agent(
        name="python_expert",
        model=get_model(),
        system_prompt="You are a Python expert. Provide Python advice.",
    )
    
    js_expert = Agent(
        name="js_expert",
        model=get_model(),
        system_prompt="You are a JavaScript expert. Provide JS advice.",
    )
    
    general_agent = Agent(
        name="general",
        model=get_model(),
        system_prompt="You are a general programming assistant.",
    )
    
    # Create flow with conditional triggers
    flow = ReactiveFlow()
    flow.register(
        python_expert,
        [react_to("question.asked").when(lambda e: "python" in str(e.data).lower())],
    )
    flow.register(
        js_expert,
        [react_to("question.asked").when(lambda e: "javascript" in str(e.data).lower())],
    )
    flow.register(
        general_agent,
        [react_to("question.asked").when(
            lambda e: "python" not in str(e.data).lower() 
            and "javascript" not in str(e.data).lower()
        )],
    )
    
    # Test multiple questions
    questions = [
        ("How do I use list comprehensions in Python?", {"language": "python"}),
        ("What are JavaScript promises?", {"language": "javascript"}),
        ("What is version control?", {"language": "general"}),
    ]
    
    for question, data in questions:
        print(f"\n📝 Question: {question}")
        print("-" * 70)
        
        async for chunk in flow.run_streaming(
            question,
            initial_event="question.asked",
            initial_data=data,
        ):
            if chunk.content:  # Skip empty chunks
                print(chunk.content, end="", flush=True)
        
        print("\n")
    
    print("✅ Conditional streaming complete!")


# =============================================================================
# Demo 5: Error Handling in Streaming
# =============================================================================

async def streaming_error_handling():
    """Demo 5: Graceful error handling during streaming."""
    print("\n" + "=" * 70)
    print("Demo 5: Error Handling in Streaming")
    print("=" * 70)
    
    from agenticflow import Agent
    from agenticflow.reactive import ReactiveFlow, react_to
    
    # Create agent that might fail
    assistant = Agent(
        name="assistant",
        model=get_model(),
        system_prompt="You are a helpful assistant.",
    )
    
    flow = ReactiveFlow()
    flow.register(assistant, [react_to("task.created")])
    
    print("\n📝 Task: Streaming with potential errors")
    print("-" * 70)
    
    try:
        async for chunk in flow.run_streaming("Explain error handling"):
            print(chunk.content, end="", flush=True)
            
            # Check for errors in metadata
            if chunk.metadata and chunk.metadata.get("error"):
                print(f"\n⚠️  Error detected: {chunk.metadata['error']}")
            
            if chunk.is_final:
                if chunk.finish_reason == "error":
                    print("\n❌ Streaming ended with error")
                else:
                    print("\n✅ Streaming completed successfully")
    
    except Exception as e:
        print(f"\n❌ Exception during streaming: {e}")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all streaming demos."""
    print("\n" + "=" * 70)
    print("Streaming Reactions in ReactiveFlow")
    print("=" * 70)
    
    try:
        await basic_streaming()
        await asyncio.sleep(1)
        
        await multi_agent_streaming()
        await asyncio.sleep(1)
        
        await streaming_with_progress()
        await asyncio.sleep(1)
        
        await conditional_streaming()
        await asyncio.sleep(1)
        
        await streaming_error_handling()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
