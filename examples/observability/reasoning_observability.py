#!/usr/bin/env python3
"""
Example: Observability with OpenAI Reasoning.

Demonstrates how reasoning tokens are displayed in observability
when using OpenAI o3-mini with reasoning effort.
"""

import asyncio

from cogent import Agent
from cogent.models.openai import OpenAIChat
from cogent.observability import Observer


async def main():
    print("=" * 60)
    print("Agent Observability with OpenAI Reasoning")
    print("=" * 60)
    print()

    # Create observer at debug level to see LLM events
    observer = Observer(level="debug")

    # Create model with reasoning effort
    llm = OpenAIChat(
        model="o3-mini",
        reasoning_effort="medium",
    )

    # Create agent with observer attached
    agent = Agent(
        name="ReasoningAgent",
        model=llm,
        instructions="You are a helpful assistant that reasons through problems.",
        observer=observer,
    )

    # Make a request that triggers reasoning
    prompt = "What is the sum of the first 10 prime numbers?"

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
