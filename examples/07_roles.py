"""
Demo: Clean 4-Role System

Demonstrates the simplified role system with only 4 distinct roles.

The Role System:
┌─────────────┬────────────┬──────────────┬───────────────┐
│ Role        │ can_finish │ can_delegate │ can_use_tools │
├─────────────┼────────────┼──────────────┼───────────────┤
│ WORKER      │     ❌     │      ❌      │      ✅       │
│ SUPERVISOR  │     ✅     │      ✅      │      ❌       │
│ AUTONOMOUS  │     ✅     │      ❌      │      ✅       │
│ REVIEWER    │     ✅     │      ❌      │      ❌       │
└─────────────┴────────────┴──────────────┴───────────────┘

Usage:
    uv run python examples/07_roles.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agenticflow import Agent, Flow, FlowObserver
from agenticflow.agent import parse_delegation, get_role_prompt, get_role_behavior
from agenticflow.core.enums import AgentRole

load_dotenv()


async def demo_role_system():
    """Show the clean 4-role system."""
    print("\n--- The 4 Roles ---")
    
    for role in AgentRole:
        behavior = get_role_behavior(role)
        print(f"\n  {role.value.upper()}:")
        print(f"    can_finish:    {'✅' if behavior.can_finish else '❌'}")
        print(f"    can_delegate:  {'✅' if behavior.can_delegate else '❌'}")
        print(f"    can_use_tools: {'✅' if behavior.can_use_tools else '❌'}")


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
    print(f"\n  Agent.as_supervisor() → {supervisor.role.value}")
    print(f"    Can finish: {supervisor.can_finish}, Can delegate: {supervisor.can_delegate}")
    
    worker = Agent.as_worker(
        name="Analyst",
        model=model,
    )
    print(f"\n  Agent.as_worker() → {worker.role.value}")
    print(f"    Can finish: {worker.can_finish}, Can use tools: {worker.can_use_tools}")
    
    critic = Agent.as_critic(
        name="QA",
        model=model,
    )
    print(f"\n  Agent.as_critic() → {critic.role.value}")  # Uses REVIEWER
    print(f"    Can finish: {critic.can_finish}, Can use tools: {critic.can_use_tools}")


async def demo_role_prompts():
    """Show built-in role prompts."""
    print("\n--- Built-in Role Prompts ---")
    
    for role in AgentRole:
        prompt = get_role_prompt(role)
        # Show first 80 chars
        preview = prompt.strip()[:80].replace("\n", " ") + "..."
        print(f"\n  {role.value}:")
        print(f"    {preview}")


async def demo_delegation():
    """Show delegation parsing."""
    print("\n--- Delegation Parsing ---")
    
    # Supervisor delegation format
    responses = [
        "DELEGATE TO Analyst: Research market trends for AI",
        "FINAL ANSWER: The analysis is complete.",
        "ROUTE TO Writer: Please summarize these findings",
        "I'm thinking about this problem...",  # No command
    ]
    
    for response in responses:
        delegation = parse_delegation(response)
        print(f"\n  Input: \"{response[:50]}{'...' if len(response) > 50 else ''}\"")
        if delegation:
            print(f"    ✅ {delegation.action.upper()}", end="")
            if delegation.target:
                print(f" → {delegation.target}", end="")
            print(f": {delegation.task[:40]}...")
        else:
            print(f"    ❌ No command detected")


async def demo_supervisor_flow():
    """Show supervisor coordinating workers."""
    print("\n--- Supervisor Flow Pattern ---")
    print("  Pattern: SUPERVISOR ↔ [WORKER, WORKER]")
    print("  SUPERVISOR delegates and finishes; WORKERs do tool work")
    
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    # Create team
    supervisor = Agent.as_supervisor(
        name="Manager",
        model=model,
        workers=["Researcher", "Writer"],
        instructions="Coordinate the team to analyze the topic. Delegate to workers, then synthesize.",
    )
    
    researcher = Agent.as_worker(
        name="Researcher",
        model=model,
        instructions="Research and provide key facts.",
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
    
    result = await flow.run("Benefits of remote work")
    print(f"\n  Result: {result.results[-1]['thought'][:200]}...")


async def demo_review_flow():
    """Show reviewer checking work."""
    print("\n--- Pipeline with Review Pattern ---")
    print("  Pattern: WORKER → REVIEWER")
    print("  WORKER does work; REVIEWER approves/rejects and finishes")
    
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    writer = Agent.as_worker(
        name="Writer",
        model=model,
        instructions="Write a short paragraph about the topic.",
    )
    
    reviewer = Agent.as_critic(
        name="QA",
        model=model,
        instructions="Review for clarity. Give FINAL ANSWER when approved.",
    )
    
    # Pipeline: Writer → Reviewer
    flow = Flow(
        name="review",
        agents=[writer, reviewer],
        topology="pipeline",
        observer=FlowObserver.normal(),
    )
    
    result = await flow.run("Cloud computing benefits")
    print(f"\n  Final output: {result.results[-1]['thought'][:200]}...")


async def demo_autonomous():
    """Show autonomous agent working independently."""
    print("\n--- Autonomous Pattern ---")
    print("  Pattern: Single AUTONOMOUS agent")
    print("  Can use tools AND finish (perfect for solo tasks)")
    
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    assistant = Agent(
        name="Assistant",
        model=model,
        role=AgentRole.AUTONOMOUS,
        instructions="Help the user with their request. Give FINAL ANSWER when done.",
    )
    
    flow = Flow(
        name="solo",
        agents=[assistant],
        topology="single",
        observer=FlowObserver.normal(),
    )
    
    result = await flow.run("What's 2+2?")
    print(f"\n  Answer: {result.results[-1]['thought'][:200]}...")


async def main():
    print("\n" + "=" * 60)
    print("  Clean 4-Role System Demo")
    print("=" * 60)
    
    await demo_role_system()
    await demo_role_factories()
    await demo_role_prompts()
    await demo_delegation()
    await demo_supervisor_flow()
    await demo_review_flow()
    await demo_autonomous()
    
    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
