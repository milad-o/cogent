#!/usr/bin/env python3
"""
Example: Observability with Extended Thinking.

Demonstrates how thinking/reasoning content and token counts
are displayed in the observability output when using an Agent.
"""

import asyncio

from cogent import Agent
from cogent.models.anthropic import AnthropicChat
from cogent.observability import Observer


async def main():
    print("=" * 60)
    print("Agent Observability with Extended Thinking")
    print("=" * 60)
    print()
    
    # Create observer at debug level to see LLM events
    observer = Observer(level="debug")
    
    # Create model with extended thinking
    llm = AnthropicChat(
        thinking_budget=2000,
        max_tokens=4000,
    )
    
    # Create agent with observer attached
    agent = Agent(
        name="ThinkingAgent",
        model=llm,
        instructions="You are a helpful assistant that thinks through problems carefully.",
        observer=observer,  # Attach observer for full observability
    )
    
    # Make a request that triggers thinking
    prompt = "What is 127 * 893? Think through it step by step."
    
    print(f"Prompt: {prompt}")
    print()
    print("-" * 60)
    print("Agent Events (debug level):")
    print("-" * 60)
    
    response = await agent.run(prompt)
    
    print()
    print("-" * 60)
    print("Response:")
    print("-" * 60)
    print(response.content)
    
    print()
    print("-" * 60)
    print("Token Usage:")
    print("-" * 60)
    if response.metadata and response.metadata.tokens:
        t = response.metadata.tokens
        print(f"  Prompt tokens:     {t.prompt_tokens}")
        print(f"  Completion tokens: {t.completion_tokens}")
        print(f"  Total tokens:      {t.total_tokens}")
        if t.reasoning_tokens:
            print(f"  Reasoning tokens:  {t.reasoning_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
