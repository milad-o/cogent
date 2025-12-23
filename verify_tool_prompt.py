"""Verify tool instructions are properly included when tools are provided."""
import asyncio
from agenticflow import Agent
from agenticflow.models import ChatModel
from agenticflow.tools.base import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

async def main():
    agent_with_tools = Agent(
        name="Researcher",
        model=ChatModel(model="gpt-4o-mini"),
        tools=[search],
    )
    
    prompt = agent_with_tools.get_effective_system_prompt()
    
    print("=" * 70)
    print("FULL SYSTEM PROMPT WITH TOOLS")
    print("=" * 70)
    print(prompt)
    print()
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    # Check for tool instructions
    has_tool_section = "Use Tools" in prompt or "TOOL:" in prompt
    has_tool_list = "search:" in prompt or "Available Tools" in prompt
    
    print(f"✅ Has tool usage instructions: {has_tool_section}")
    print(f"✅ Has tool list/descriptions: {has_tool_list}")
    
    if not has_tool_section:
        print("\n⚠️  WARNING: Tool instructions not found in prompt!")
    if not has_tool_list:
        print("\n⚠️  WARNING: Tool list not found in prompt!")

if __name__ == "__main__":
    asyncio.run(main())
