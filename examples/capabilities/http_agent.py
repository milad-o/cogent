"""HTTP Client capability - agent integration demo.

Shows HTTPClient being used by an agent to fetch and analyze API data.

Run with: uv run python examples/capabilities/http_agent_demo.py
"""

import asyncio

from cogent import Agent
from cogent.capabilities.http_client import HTTPClient


async def main():
    """Demonstrate HTTPClient being used by an agent."""
    print("\n" + "=" * 60)
    print("HTTP CLIENT CAPABILITY - AGENT DEMO")
    print("=" * 60)
    
    agent = Agent(
        name="API Research Agent",
        model="gpt-4o-mini",
        capabilities=[HTTPClient()],
        system_prompt="""You are an API research assistant.
        
Use HTTP tools to fetch data from APIs and analyze the results.
Always check response status and extract key information.""",
    )
    
    print("\n" + "=" * 60)
    print("SCENARIO: Fetch and Analyze GitHub User Data")
    print("=" * 60)
    
    task = """Fetch information about the GitHub user 'torvalds' using the GitHub API:
https://api.github.com/users/torvalds

Then tell me:
1. Their real name
2. Number of public repos
3. Number of followers
4. Their bio
5. When they joined GitHub"""
    
    print(f"\nAgent task: {task}\n")
    print("-" * 60)
    
    result = await agent.run(task)
    
    print(f"\nAgent response:\n{result.content}")
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
