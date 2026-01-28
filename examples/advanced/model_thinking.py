#!/usr/bin/env python3
"""
Example: Model-Level Thinking Features.

Demonstrates the Extended Thinking (Anthropic), Thinking Config (Gemini),
and Reasoning Config (OpenAI) features for enhanced reasoning.

These features enable models to "think" before responding, significantly
improving performance on complex reasoning tasks like math, logic, and analysis.
"""

import asyncio


# =============================================================================
# Anthropic Extended Thinking
# =============================================================================
async def anthropic_thinking_example():
    """Demonstrate Anthropic Extended Thinking.
    
    Extended Thinking allows Claude to reason through problems step-by-step
    before providing a final answer.
    
    Supported models:
    - claude-opus-4-* (most capable)
    - claude-sonnet-4-* (balanced)
    - claude-3-7-sonnet-* (previous gen)
    - claude-3-5-sonnet-20241022
    """
    from cogent.models.anthropic import AnthropicChat
    
    print("=" * 60)
    print("Anthropic Extended Thinking")
    print("=" * 60)
    
    # Method 1: Direct initialization with thinking_budget
    # Note: max_tokens must be > thinking_budget, we recommend at least 2x
    llm = AnthropicChat(
        thinking_budget=2000,  # Token budget for thinking (min 1024)
        max_tokens=4000,       # Must be > thinking_budget
    )
    
    # Method 2: Use with_thinking() for fluent configuration
    # llm = AnthropicChat(max_tokens=16000).with_thinking(budget=8000)
    
    response = await llm.ainvoke(
        "What is 127 * 893? Show your work step by step."
    )
    
    # Access the thinking process
    if hasattr(response, "thinking"):
        print("\n📝 Thinking Process:")
        print("-" * 40)
        thinking = response.thinking
        print(thinking[:500] + "..." if len(thinking) > 500 else thinking)
    
    print("\n✅ Final Answer:")
    print("-" * 40)
    print(response.content)
    
    # Token usage including reasoning
    if response.metadata.tokens:
        print("\n📊 Token Usage:")
        print(f"  Prompt: {response.metadata.tokens.prompt_tokens}")
        print(f"  Completion: {response.metadata.tokens.completion_tokens}")
        if response.metadata.tokens.reasoning_tokens:
            print(f"  Reasoning: {response.metadata.tokens.reasoning_tokens}")


# =============================================================================
# Gemini Thinking Config
# =============================================================================
async def gemini_thinking_example():
    """Demonstrate Gemini Thinking Config.
    
    Gemini 2.5 models support thinking_budget for controlled reasoning depth.
    
    Supported models:
    - gemini-2.5-pro-preview-05-06
    - gemini-2.5-flash-preview-05-20
    """
    from cogent.models.gemini import GeminiChat
    
    print("\n" + "=" * 60)
    print("Gemini Thinking Config")
    print("=" * 60)
    
    llm = GeminiChat(
        model="gemini-2.5-flash-preview-05-20",
        thinking_budget=8192,
        include_thoughts=True,
    )
    
    response = await llm.ainvoke(
        "Explain why the sky appears blue during the day."
    )
    
    # Access thought summary if available
    if hasattr(response, "thoughts"):
        print("\n💭 Thought Summary:")
        print("-" * 40)
        thoughts = response.thoughts
        print(thoughts[:500] + "..." if len(thoughts) > 500 else thoughts)
    
    print("\n✅ Response:")
    print("-" * 40)
    print(response.content)


# =============================================================================
# OpenAI Reasoning Config (o-series models)
# =============================================================================
async def openai_reasoning_example():
    """Demonstrate OpenAI reasoning configuration.
    
    o-series models support reasoning_effort parameter.
    
    Supported models:
    - o3-mini (supports reasoning_effort: low/medium/high)
    - o1, o1-mini
    """
    from cogent.models.openai import OpenAIChat
    
    print("\n" + "=" * 60)
    print("OpenAI Reasoning Config")
    print("=" * 60)
    
    llm = OpenAIChat(
        model="o3-mini",
        reasoning_effort="high",  # low, medium, or high
    )
    
    response = await llm.ainvoke(
        "A train travels at 60 mph for 2 hours, then 80 mph for 1.5 hours. "
        "What is the average speed for the entire journey?"
    )
    
    print("\n✅ Response:")
    print("-" * 40)
    print(response.content)
    
    if response.metadata.tokens:
        print("\n📊 Token Usage:")
        print(f"  Prompt: {response.metadata.tokens.prompt_tokens}")
        print(f"  Completion: {response.metadata.tokens.completion_tokens}")
        if response.metadata.tokens.reasoning_tokens:
            print(f"  Reasoning: {response.metadata.tokens.reasoning_tokens}")


# =============================================================================
# xAI Grok Reasoning
# =============================================================================
async def xai_reasoning_example():
    """Demonstrate xAI Grok reasoning capabilities.
    
    Grok 4 models have built-in reasoning. Token usage includes
    reasoning_tokens automatically.
    
    Supported models:
    - grok-4 (flagship)
    - grok-4-1-fast (fast agentic)
    """
    from cogent.models.xai import XAIChat
    
    print("\n" + "=" * 60)
    print("xAI Grok Reasoning")
    print("=" * 60)
    
    llm = XAIChat(model="grok-4-1-fast")
    
    response = await llm.ainvoke(
        "If you have 3 red balls and 2 blue balls, what is the "
        "probability of drawing 2 red balls in a row without replacement?"
    )
    
    print("\n✅ Response:")
    print("-" * 40)
    print(response.content)
    
    if response.metadata.tokens:
        print("\n📊 Token Usage:")
        print(f"  Prompt: {response.metadata.tokens.prompt_tokens}")
        print(f"  Completion: {response.metadata.tokens.completion_tokens}")
        if response.metadata.tokens.reasoning_tokens:
            print(f"  Reasoning: {response.metadata.tokens.reasoning_tokens}")


# =============================================================================
# Main
# =============================================================================
async def main():
    """Run thinking/reasoning examples."""
    print("\n🧠 Model-Level Thinking Features")
    print("=" * 60)
    print("Demonstrating thinking/reasoning features across providers.")
    print("Note: Requires API keys for respective providers.")
    print("=" * 60)
    
    # Uncomment the examples you want to run:
    
    # await anthropic_thinking_example()
    # await gemini_thinking_example()
    # await openai_reasoning_example()
    # await xai_reasoning_example()
    
    print("\n✅ Examples available. Uncomment the ones you want to run.")


if __name__ == "__main__":
    asyncio.run(main())
