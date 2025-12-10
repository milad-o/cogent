#!/usr/bin/env python3
"""
Example 19: Streaming LLM Responses
===================================

This example demonstrates AgenticFlow's streaming capabilities,
which enable real-time token-by-token output from LLM providers.

Streaming improves user experience by showing output as it's generated
rather than waiting for the complete response.

Key Features Demonstrated:
1. Simple streaming with stream=True parameter
2. Agent-level default streaming
3. Low-level think() with streaming
4. run() streaming with conversation history
5. Structured event streaming with stream_events()
6. Callbacks for handling stream events

Prerequisites:
    - An LLM provider with streaming support (OpenAI, Anthropic, etc.)
    - Set your API key: export OPENAI_API_KEY=your-key

Run:
    uv run python examples/19_streaming.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model as config_get_model, settings

# For demo without real API - check if any provider is configured
MOCK_MODE = not any([
    settings.openai_api_key,
    settings.groq_api_key,
    settings.anthropic_api_key,
    settings.gemini_api_key,
])


# =============================================================================
# Mock Model Factory for Demo (simulates streaming)
# =============================================================================

def create_mock_model(response_text: str):
    """
    Create a mock model that works with Agent's type checking.
    
    This creates a native BaseChatModel subclass that
    simulates streaming by yielding token chunks with delays.
    """
    from collections.abc import AsyncIterator, Iterator
    from typing import Any
    
    from agenticflow.models.base import BaseChatModel
    from agenticflow.core.messages import AIMessage
    
    class MockChatModel(BaseChatModel):
        """Mock chat model for testing/demo with streaming support."""
        
        def __init__(self, response: str):
            self._response = response
        
        @property
        def model_name(self) -> str:
            return "mock-streaming"
        
        def invoke(
            self,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> AIMessage:
            return AIMessage(content=self._response)
        
        async def ainvoke(
            self,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> AIMessage:
            return AIMessage(content=self._response)
        
        def stream(
            self,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> Iterator[AIMessage]:
            """Synchronous streaming implementation."""
            import time
            tokens = []
            i = 0
            text = self._response
            while i < len(text):
                chunk_len = min(4, len(text) - i)
                tokens.append(text[i:i + chunk_len])
                i += chunk_len
            
            for token in tokens:
                time.sleep(0.02)
                yield AIMessage(content=token)
        
        async def astream(
            self,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> AsyncIterator[AIMessage]:
            """Async streaming implementation."""
            tokens = []
            i = 0
            text = self._response
            while i < len(text):
                chunk_len = min(4, len(text) - i)
                tokens.append(text[i:i + chunk_len])
                i += chunk_len
            
            for token in tokens:
                await asyncio.sleep(0.02)
                yield AIMessage(content=token)
    
    return MockChatModel(response=response_text)


def get_model(response_for_mock: str = "Hello!"):
    """Get model - mock for demo, real for production."""
    if MOCK_MODE:
        return create_mock_model(response_for_mock)
    else:
        return config_get_model()


# =============================================================================
# Demo Functions
# =============================================================================

async def demo_basic_streaming():
    """Demo 1: Simple streaming with stream=True parameter."""
    print("\n" + "=" * 60)
    print("Demo 1: Simple Streaming (stream=True)")
    print("=" * 60)
    
    from agenticflow import Agent
    
    model = get_model(
        "Streaming is a powerful feature that allows you to see "
        "the response as it's being generated, token by token. "
        "This improves the user experience significantly!"
    )
    
    # Create agent normally (non-streaming by default)
    agent = Agent(
        name="StreamingAssistant",
        model=model,
        system_prompt="You are a helpful assistant. Keep responses concise.",
    )
    
    print("\n# Using stream=True on run():")
    print("-" * 40)
    
    # Simply add stream=True to enable streaming!
    async for chunk in agent.run("Explain streaming in 2 sentences.", stream=True):
        print(chunk.content, end="", flush=True)
    
    print("\n" + "-" * 40)
    print("✅ Streaming complete!")


async def demo_default_streaming():
    """Demo 2: Agent-level default streaming."""
    print("\n" + "=" * 60)
    print("Demo 2: Agent-Level Default Streaming")
    print("=" * 60)
    
    from agenticflow import Agent
    
    model = get_model(
        "I'm a streaming agent by default! Every response I give "
        "will stream token by token automatically."
    )
    
    # Create agent with streaming enabled by default
    agent = Agent(
        name="StreamingByDefault",
        model=model,
        stream=True,  # <-- Enable streaming as default!
    )
    
    print("\n# Agent with stream=True default:")
    print("-" * 40)
    
    # No need to specify stream=True - it's the default!
    async for chunk in agent.run("Hello! Are you a streaming agent?"):
        print(chunk.content, end="", flush=True)
    
    print("\n" + "-" * 40)
    
    # Can still override to get non-streaming response
    if MOCK_MODE:
        agent._model = create_mock_model("Yes, but I can return complete responses too!")
    
    print("\n# Override with stream=False:")
    result = await agent.run("Give me a one-liner.", stream=False)
    print(result)
    
    print("\n✅ Agent-level streaming works!")


async def demo_think_stream():
    """Demo 3: Low-level streaming with think(stream=True)."""
    print("\n" + "=" * 60)
    print("Demo 3: Token Streaming with think(stream=True)")
    print("=" * 60)
    
    from agenticflow import Agent
    
    model = get_model(
        "Using think(stream=True) gives you low-level LLM access "
        "without the agentic loop."
    )
    
    agent = Agent(
        name="ThinkingAgent",
        model=model,
        system_prompt="You are a helpful assistant. Keep responses concise.",
    )
    
    print("\nStreaming with think(stream=True):")
    print("-" * 40)
    
    async for chunk in agent.think("Explain streaming in one sentence.", stream=True):
        print(chunk.content, end="", flush=True)
    
    print("\n" + "-" * 40)
    print("✅ Streaming complete!")


async def demo_conversation_streaming():
    """Demo 4: Streaming with conversation history using run()."""
    print("\n" + "=" * 60)
    print("Demo 4: Chat Streaming with History")
    print("=" * 60)
    
    from agenticflow import Agent
    
    model = get_model("Nice to meet you, Alice! How can I help you today?")
    
    agent = Agent(
        name="ChatAssistant",
        model=model,
        system_prompt="You are a friendly assistant. Remember user details.",
    )
    
    thread_id = "demo-thread-001"
    
    # First message
    print("\n[User]: Hi, my name is Alice!")
    print("[Assistant]: ", end="", flush=True)
    
    async for chunk in agent.run("Hi, my name is Alice!", stream=True, thread_id=thread_id):
        print(chunk.content, end="", flush=True)
    print()
    
    # For mock mode, create new model for second response
    # (In real mode, same model handles both)
    if MOCK_MODE:
        agent._model = create_mock_model(
            "Yes, I remember - your name is Alice! What can I do for you?"
        )
    
    # Second message - should remember context
    print("\n[User]: What's my name?")
    print("[Assistant]: ", end="", flush=True)
    
    async for chunk in agent.run("What's my name?", stream=True, thread_id=thread_id):
        print(chunk.content, end="", flush=True)
    print()
    
    print("\n✅ Chat streaming with memory works!")


async def demo_event_streaming():
    """Demo 5: Structured event streaming."""
    print("\n" + "=" * 60)
    print("Demo 5: Structured Event Streaming")
    print("=" * 60)
    
    from agenticflow import Agent, StreamEventType
    
    model = get_model("Here's what I found: The answer is 42. That's all!")
    
    agent = Agent(
        name="EventStreamer",
        model=model,
    )
    
    print("\nEvent stream:")
    print("-" * 40)
    
    async for event in agent.stream_events("What is 6 times 7?"):
        if event.type == StreamEventType.STREAM_START:
            print(f"🚀 Stream started (agent: {event.metadata.get('agent_name')})")
        elif event.type == StreamEventType.TOKEN:
            print(event.content, end="", flush=True)
        elif event.type == StreamEventType.TOOL_CALL_START:
            print(f"\n🔧 Tool call: {event.tool_name}")
        elif event.type == StreamEventType.STREAM_END:
            print(f"\n✅ Stream ended ({event.metadata.get('token_count')} chunks)")
        elif event.type == StreamEventType.ERROR:
            print(f"\n❌ Error: {event.error}")
    
    print("-" * 40)


async def demo_callbacks():
    """Demo 6: Using callbacks for stream handling."""
    print("\n" + "=" * 60)
    print("Demo 6: Stream Callbacks")
    print("=" * 60)
    
    from agenticflow import (
        Agent,
        PrintStreamCallback,
        CollectorStreamCallback,
    )
    
    model = get_model("Callbacks let you process tokens without writing a loop!")
    
    agent = Agent(
        name="CallbackDemo",
        model=model,
    )
    
    # Using PrintStreamCallback for automatic printing
    print("\n1. Using PrintStreamCallback:")
    print("-" * 40)
    
    callback = PrintStreamCallback(
        prefix="🤖 ",
        suffix="\n",
        end="",
        flush=True,
    )
    
    async for chunk in agent.think("Say something short.", stream=True):
        callback.on_token(chunk.content)
    callback.on_stream_end("")
    
    # Using CollectorStreamCallback to collect output
    print("\n2. Using CollectorStreamCallback:")
    print("-" * 40)
    
    if MOCK_MODE:
        agent._model = create_mock_model("This response will be collected!")
    
    collector = CollectorStreamCallback()
    
    async for chunk in agent.think("Give me a brief response.", stream=True):
        collector.on_token(chunk.content)
        print(".", end="", flush=True)  # Progress indicator
    
    print(f"\n\nCollected: {collector.get_full_response()!r}")
    print(f"Token count: {len(collector._tokens)}")


async def demo_print_stream_helper():
    """Demo 7: Using the print_stream() helper."""
    print("\n" + "=" * 60)
    print("Demo 7: print_stream() Helper")
    print("=" * 60)
    
    from agenticflow import Agent, print_stream
    
    model = get_model("The print_stream helper makes it easy to stream and capture!")
    
    agent = Agent(
        name="PrintHelper",
        model=model,
    )
    
    print("\nUsing print_stream() helper:")
    print("-" * 40)
    
    # print_stream handles both display and collection
    response = await print_stream(
        agent.think("Explain print_stream in one sentence.", stream=True),
        prefix="📝 ",
        suffix="\n",
    )
    
    print(f"\nFull response captured: {len(response)} characters")


async def demo_collect_stream():
    """Demo 8: Collecting stream without display."""
    print("\n" + "=" * 60)
    print("Demo 8: Collecting Stream Silently")
    print("=" * 60)
    
    from agenticflow import Agent, collect_stream
    
    model = get_model("This is collected silently without printing anything.")
    
    agent = Agent(
        name="Collector",
        model=model,
    )
    
    print("\nCollecting stream silently...")
    
    # collect_stream gathers all output without printing
    content, tool_calls = await collect_stream(
        agent.think("Give me a secret message.", stream=True)
    )
    
    print(f"✅ Collected {len(content)} characters")
    print(f"📝 Content: {content!r}")
    print(f"🔧 Tool calls: {len(tool_calls)}")


async def demo_custom_processing():
    """Demo 7: Custom stream processing."""
    print("\n" + "=" * 60)
    print("Demo 7: Custom Stream Processing")
    print("=" * 60)
    
    from agenticflow import Agent
    
    model = get_model(
        "Custom processing allows you to transform tokens as they arrive. "
        "You could uppercase them, count words, or do anything else!"
    )
    
    agent = Agent(
        name="CustomProcessor",
        model=model,
    )
    
    print("\n1. Uppercase transformation:")
    print("-" * 40)
    
    async for chunk in agent.think("Say hello in lowercase.", stream=True):
        # Transform each chunk as it arrives
        print(chunk.content.upper(), end="", flush=True)
    print()
    
    print("\n2. Word counting during stream:")
    print("-" * 40)
    
    if MOCK_MODE:
        agent._model = create_mock_model(
            "One two three four five six seven eight nine ten."
        )
    
    buffer = ""
    word_count = 0
    
    async for chunk in agent.think("Count to ten with words.", stream=True):
        buffer += chunk.content
        # Count complete words
        while " " in buffer:
            word, buffer = buffer.split(" ", 1)
            if word.strip():
                word_count += 1
        print(chunk.content, end="", flush=True)
    
    # Count remaining buffer as a word if non-empty
    if buffer.strip():
        word_count += 1
    
    print(f"\n\n📊 Word count: {word_count}")


async def demo_streaming_comparison():
    """Demo 8: Compare streaming vs non-streaming performance."""
    print("\n" + "=" * 60)
    print("Demo 8: Streaming vs Non-Streaming")
    print("=" * 60)
    
    import time
    from agenticflow import Agent
    
    response_text = (
        "This is a longer response to demonstrate the difference "
        "between streaming and non-streaming. With streaming, you see "
        "each token as it arrives. Without streaming, you wait for "
        "the entire response before seeing anything."
    )
    model = get_model(response_text)
    
    agent = Agent(
        name="Comparison",
        model=model,
    )
    
    prompt = "Write a 3-sentence response about AI."
    
    # Non-streaming
    print("\n1. Non-streaming (wait for complete response):")
    print("-" * 40)
    
    start = time.time()
    response = await agent.think(prompt)
    non_stream_time = time.time() - start
    
    print(response)
    print(f"\n⏱️  Time to first character: {non_stream_time:.2f}s")
    print(f"⏱️  Total time: {non_stream_time:.2f}s")
    
    # Streaming
    print("\n2. Streaming (tokens arrive progressively):")
    print("-" * 40)
    
    start = time.time()
    first_token_time = None
    
    async for chunk in agent.think(prompt, stream=True):
        if first_token_time is None:
            first_token_time = time.time() - start
        print(chunk.content, end="", flush=True)
    
    total_time = time.time() - start
    
    print(f"\n\n⏱️  Time to first character: {first_token_time:.3f}s")
    print(f"⏱️  Total time: {total_time:.2f}s")
    
    print("\n✅ Streaming provides faster perceived response!")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all streaming demos."""
    print("=" * 60)
    print("AgenticFlow Streaming Demonstration")
    print("=" * 60)
    
    if MOCK_MODE:
        print("\n⚠️  Running in MOCK MODE (no API key detected)")
        print("   Set OPENAI_API_KEY for real LLM streaming")
    else:
        print("\n🔑 Using real OpenAI API")
    
    await demo_basic_streaming()      # Demo 1: Simple stream=True
    await demo_default_streaming()    # Demo 2: Agent-level default
    await demo_think_stream()              # Demo 3: think(stream=True)
    await demo_conversation_streaming()    # Demo 4: run() with thread_id
    await demo_event_streaming()           # Demo 5: Structured events
    await demo_callbacks()            # Demo 6: Callbacks
    await demo_print_stream_helper()  # Demo 7: print_stream()
    await demo_collect_stream()       # Demo 8: collect_stream()
    
    print("\n✅ All Streaming demos completed!")


if __name__ == "__main__":
    asyncio.run(main())
