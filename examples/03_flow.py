"""
Demo: Code Review Team

Supervisor coordinates reviewers for parallel code analysis.

Usage:
    uv run python examples/03_flow.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from agenticflow import Agent, AgentRole, Flow, TopologyPattern

load_dotenv()


@tool
def check_security(code: str) -> str:
    """Check code for security vulnerabilities."""
    return "Security: No SQL injection or XSS vulnerabilities found."


@tool
def check_performance(code: str) -> str:
    """Analyze code for performance issues."""
    return "Performance: Consider caching repeated database calls."


@tool
def check_style(code: str) -> str:
    """Check code style and best practices."""
    return "Style: Line 42 exceeds 88 characters. Missing type hints on 3 functions."


async def main():
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"))

    lead = Agent(
        name="Lead",
        model=model,
        role=AgentRole.SUPERVISOR,
        description="Coordinates review and synthesizes findings",
    )
    security = Agent(name="SecurityReviewer", model=model, tools=[check_security])
    perf = Agent(name="PerfReviewer", model=model, tools=[check_performance])
    style = Agent(name="StyleReviewer", model=model, tools=[check_style])

    flow = Flow(
        name="code-review",
        agents=[lead, security, perf, style],
        topology=TopologyPattern.SUPERVISOR,
        supervisor=lead,
    )

    result = await flow.run("Review this Python function for a web API endpoint")
    print(result["results"][-1]["thought"])


if __name__ == "__main__":
    asyncio.run(main())
