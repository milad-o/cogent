"""API Tester capability - agent integration demo.

Shows APITester being used by an agent to test and validate APIs.

Run with: uv run python examples/capabilities/api_tester_agent_demo.py
"""

import asyncio

from cogent import Agent
from cogent.capabilities.api_tester import APITester


async def main():
    """Demonstrate APITester being used by an agent."""
    print("\n" + "=" * 60)
    print("API TESTER CAPABILITY - AGENT DEMO")
    print("=" * 60)
    
    agent = Agent(
        name="API Testing Agent",
        model="gpt-4o-mini",
        capabilities=[APITester(base_url="https://jsonplaceholder.typicode.com")],
        system_prompt="""You are an API testing specialist.

Use API testing tools to validate endpoints, check schemas, and measure performance.
Provide clear test reports with pass/fail status.""",
    )
    
    print("\n" + "=" * 60)
    print("SCENARIO: JSONPlaceholder API Health Check")
    print("=" * 60)
    
    task = """Run a health check on the JSONPlaceholder API. Test these endpoints:

1. GET /posts - should return 200
2. GET /users/1 - should return 200 with fields: id, name, email
3. GET /posts/1 - should return 200
4. POST /posts - should return 201

Then run a performance test on /posts with 20 concurrent requests.

Provide a summary: overall health status, any failures, and performance metrics."""
    
    print(f"\nAgent task: {task}\n")
    print("-" * 60)
    
    result = await agent.run(task)
    
    print(f"\nAgent response:\n{result.content}")
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
