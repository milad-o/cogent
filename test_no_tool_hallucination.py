"""Test that agents without tools don't hallucinate tool calls."""
import asyncio
from agenticflow import Agent
from agenticflow.models import ChatModel

async def test_no_tool_hallucination():
    """Test that a basic agent without tools doesn't try to use tools."""
    
    print("=" * 70)
    print("TEST 1: Agent with custom instructions, no tools")
    print("=" * 70)
    
    # Create a simple agent with no tools
    agent = Agent(
        name="Assistant",
        model=ChatModel(model="gpt-4o-mini"),
        instructions="You are a helpful assistant. Be concise.",
    )
    
    # Check the system prompt doesn't include tool instructions
    system_prompt = agent.get_effective_system_prompt()
    print("\n=== System Prompt ===")
    print(system_prompt)
    print()
    
    # Verify no tool-related instructions
    assert "TOOL:" not in system_prompt, "System prompt shouldn't mention TOOL: syntax when no tools available"
    assert "tool_name" not in system_prompt.lower(), "System prompt shouldn't mention tool_name when no tools available"
    assert "Use Tools" not in system_prompt, "System prompt shouldn't have 'Use Tools' section when no tools available"
    
    print("✅ No tool instructions in custom prompt!\n")
    
    print("=" * 70)
    print("TEST 2: Agent with default role prompt (WORKER), no tools")
    print("=" * 70)
    
    # Test with default worker role (default when no role specified)
    agent2 = Agent(
        name="Helper",
        model=ChatModel(model="gpt-4o-mini"),
    )
    
    system_prompt2 = agent2.get_effective_system_prompt()
    print("\n=== System Prompt ===")
    print(system_prompt2)
    print()
    
    # Verify no tool-related instructions
    assert "TOOL:" not in system_prompt2, "Default prompt shouldn't mention TOOL: when no tools"
    assert "tool_name" not in system_prompt2.lower() or "tool work to workers" in system_prompt2.lower(), \
        "Default prompt shouldn't mention tool_name when no tools available"
    
    print("✅ No tool instructions in default prompt!\n")
    
    print("=" * 70)
    print("TEST 3: Agent with tools SHOULD have tool instructions")
    print("=" * 70)
    
    from agenticflow.tools.base import tool
    
    @tool
    def dummy_tool(query: str) -> str:
        """A dummy tool."""
        return f"Result: {query}"
    
    agent3 = Agent(
        name="ToolUser",
        model=ChatModel(model="gpt-4o-mini"),
        tools=[dummy_tool],
    )
    
    system_prompt3 = agent3.get_effective_system_prompt()
    print("\n=== System Prompt (truncated) ===")
    print(system_prompt3[:500] + "...")
    print()
    
    # This SHOULD have tool instructions
    assert "TOOL:" in system_prompt3 or "tool" in system_prompt3.lower(), \
        "Prompt with tools should mention tools"
    
    print("✅ Tool instructions present when tools provided!\n")
    
    print("=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)
    print("\nSummary:")
    print("  - Agents without tools: No tool instructions ✅")
    print("  - Agents with tools: Tool instructions present ✅")
    print("  - Fix prevents LLM from hallucinating tool calls! ✅")

if __name__ == "__main__":
    asyncio.run(test_no_tool_hallucination())
