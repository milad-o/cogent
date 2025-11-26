"""
Demo: Agent Roles & Delegation

Demonstrates role-specific agents and delegation patterns.

Features:
- Role factory methods: as_supervisor(), as_worker(), as_reviewer()
- Built-in role prompts and behaviors
- Delegation parsing

Usage:
    uv run python examples/07_roles.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agenticflow import Agent, Flow, FlowObserver
from agenticflow.agent import parse_delegation, get_role_prompt
from agenticflow.core.enums import AgentRole

load_dotenv()


async def demo_role_factories():
    """Show role-specific factory methods."""
    print("\n--- Role Factory Methods ---")
    
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    # Create role-specific agents
    supervisor = Agent.as_supervisor(
        name="Manager",
        model=model,
        workers=["Analyst", "Writer"],
    )
    print(f"  Supervisor: {supervisor.name} (role={supervisor.role})")
    
    worker = Agent.as_worker(
        name="Analyst",
        model=model,
    )
    print(f"  Worker: {worker.name} (role={worker.role})")
    
    critic = Agent.as_critic(
        name="QA",
        model=model,
    )
    print(f"  Critic: {critic.name} (role={critic.role})")
    
    planner = Agent.as_planner(
        name="Strategist",
        model=model,
    )
    print(f"  Planner: {planner.name} (role={planner.role})")


async def demo_role_prompts():
    """Show built-in role prompts."""
    print("\n--- Built-in Role Prompts ---")
    
    for role in [AgentRole.SUPERVISOR, AgentRole.WORKER, AgentRole.CRITIC, AgentRole.PLANNER]:
        prompt = get_role_prompt(role)
        # Show first 80 chars
        preview = prompt[:80].replace("\n", " ") + "..."
        print(f"  {role.value}: {preview}")


async def demo_delegation():
    """Show delegation parsing."""
    print("\n--- Delegation Parsing ---")
    
    # Supervisor delegation format
    responses = [
        "DELEGATE TO Analyst: Research market trends for AI",
        "FINAL ANSWER: The analysis is complete.",
        "ROUTE TO Writer: Please summarize these findings",
    ]
    
    for response in responses:
        delegation = parse_delegation(response)
        if delegation:
            print(f"  Input: {response[:50]}...")
            print(f"    → Action: {delegation.action}, Target: {delegation.target}, Task: {delegation.task[:30]}...")
        else:
            print(f"  Input: {response[:50]}... → No delegation found")


async def demo_supervisor_flow():
    """Show supervisor coordinating workers."""
    print("\n--- Supervisor Flow ---")
    
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    # Create team
    supervisor = Agent.as_supervisor(
        name="Manager",
        model=model,
        workers=["Researcher", "Writer"],
        instructions="Coordinate the team to analyze and summarize the topic.",
    )
    
    researcher = Agent.as_worker(
        name="Researcher",
        model=model,
        instructions="Research and provide facts.",
    )
    
    writer = Agent.as_worker(
        name="Writer",
        model=model,
        instructions="Write clear summaries.",
    )
    
    # Run with supervisor topology
    flow = Flow(
        name="team",
        agents=[supervisor, researcher, writer],
        topology="supervisor",
        observer=FlowObserver.normal(),
    )
    
    result = await flow.run("Topic: Benefits of remote work")
    print(f"\n  Result: {result.results[-1]['thought'][:200]}...")


async def demo_review_flow():
    """Show critic checking work."""
    print("\n--- Review Flow ---")
    
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    writer = Agent.as_worker(
        name="Writer",
        model=model,
        instructions="Write a short paragraph about the topic.",
    )
    
    critic = Agent.as_critic(
        name="QA",
        model=model,
        instructions="Review for clarity and accuracy. Suggest improvements.",
    )
    
    # Pipeline: Writer → Critic
    flow = Flow(
        name="review",
        agents=[writer, critic],
        topology="pipeline",
        observer=FlowObserver.normal(),
    )
    
    result = await flow.run("Topic: Cloud computing")
    print(f"\n  Writer output + Review: {result.results[-1]['thought'][:200]}...")


async def main():
    print("\n" + "=" * 50)
    print("  Agent Roles & Delegation Demo")
    print("=" * 50)
    
    await demo_role_factories()
    await demo_role_prompts()
    await demo_delegation()
    await demo_supervisor_flow()
    await demo_review_flow()
    
    print("\n" + "=" * 50)
    print("✅ Done!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
