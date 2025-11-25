#!/usr/bin/env python3
"""Example: Real topology execution with OpenAI.

This example demonstrates all topology types working with real LLM calls.
Requires OPENAI_API_KEY in .env file.
"""

import asyncio
import os
from dotenv import load_dotenv

from agenticflow.agents import Agent, AgentConfig
from agenticflow.core.enums import AgentRole
from agenticflow.events import EventBus
from agenticflow.topologies import (
    TopologyConfig,
    TopologyPolicy,
    SupervisorTopology,
    PipelineTopology,
    MeshTopology,
    BaseTopology,
)

# Load environment variables
load_dotenv()


def create_real_agent(
    name: str,
    role: AgentRole,
    system_prompt: str,
    event_bus: EventBus,
    tools: list[str] | None = None,
) -> Agent:
    """Create a real agent with OpenAI model."""
    config = AgentConfig(
        name=name,
        role=role,
        model_name="gpt-4o-mini",  # Use mini for cost efficiency
        system_prompt=system_prompt,
        temperature=0.7,
        tools=tools or [],
    )
    return Agent(config=config, event_bus=event_bus)


async def test_supervisor_topology():
    """Test supervisor topology with real agents."""
    print("\n" + "=" * 60)
    print("TEST 1: SUPERVISOR TOPOLOGY")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Create agents
    manager = create_real_agent(
        name="Manager",
        role=AgentRole.ORCHESTRATOR,
        system_prompt="""You are a project manager coordinating a content team.
Your job is to:
1. Analyze the task and decide which team member should work on it
2. Review their work and provide feedback
3. Say "FINISH" when the task is complete

Available team members:
- Researcher: Gathers information and facts
- Writer: Creates content based on research

Be concise. After reviewing work from team members, decide if more work is needed or if we can finish.""",
        event_bus=event_bus,
    )
    
    researcher = create_real_agent(
        name="Researcher",
        role=AgentRole.WORKER,
        system_prompt="""You are a researcher. When given a topic:
1. Provide 2-3 key facts or insights about the topic
2. Keep your response brief (2-3 sentences)
3. Focus on interesting, lesser-known information""",
        event_bus=event_bus,
    )
    
    writer = create_real_agent(
        name="Writer",
        role=AgentRole.WORKER,
        system_prompt="""You are a creative writer. When given information:
1. Create a brief, engaging paragraph (3-4 sentences)
2. Make it interesting and accessible
3. Use the research provided as your foundation""",
        event_bus=event_bus,
    )
    
    topology = SupervisorTopology(
        config=TopologyConfig(name="content-team", max_iterations=5),
        agents=[manager, researcher, writer],
        supervisor_name="Manager",
    )
    
    print("\n📋 Task: Write a fun fact about octopuses")
    print("-" * 40)
    
    result = await topology.run("Write a fun fact about octopuses")
    
    print(f"\n✅ Completed in {result.iteration} iterations")
    print("\n📝 Results:")
    for r in result.results:
        print(f"\n[{r['agent']}]:")
        print(f"  {r['thought'][:200]}...")


async def test_pipeline_topology():
    """Test pipeline topology with real agents."""
    print("\n" + "=" * 60)
    print("TEST 2: PIPELINE TOPOLOGY")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Create pipeline stages
    brainstormer = create_real_agent(
        name="Brainstormer",
        role=AgentRole.WORKER,
        system_prompt="""You are a creative brainstormer. Given a topic:
1. Generate 3 creative angles or ideas
2. Keep each idea to one sentence
3. Be unique and interesting""",
        event_bus=event_bus,
    )
    
    developer = create_real_agent(
        name="Developer",
        role=AgentRole.WORKER,
        system_prompt="""You are a content developer. Given ideas from brainstorming:
1. Pick the most interesting idea
2. Expand it into 2-3 sentences
3. Add supporting details""",
        event_bus=event_bus,
    )
    
    polisher = create_real_agent(
        name="Polisher",
        role=AgentRole.WORKER,
        system_prompt="""You are a content polisher. Given developed content:
1. Refine the language for clarity and engagement
2. Fix any awkward phrasing
3. Make it publication-ready
Output ONLY the final polished content.""",
        event_bus=event_bus,
    )
    
    topology = PipelineTopology(
        config=TopologyConfig(name="content-pipeline", max_iterations=10),
        agents=[brainstormer, developer, polisher],
        stages=["Brainstormer", "Developer", "Polisher"],
    )
    
    print("\n📋 Task: Create content about AI in healthcare")
    print("-" * 40)
    
    result = await topology.run("Create content about AI in healthcare")
    
    print(f"\n✅ Completed in {result.iteration} iterations")
    print("\n📝 Pipeline Output:")
    for r in result.results:
        print(f"\n[{r['agent']}]:")
        print(f"  {r['thought'][:300]}...")


async def test_mesh_topology():
    """Test mesh topology with real agents."""
    print("\n" + "=" * 60)
    print("TEST 3: MESH TOPOLOGY (Collaborative)")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Create collaborative agents
    analyst = create_real_agent(
        name="Analyst",
        role=AgentRole.WORKER,
        system_prompt="""You are a data analyst in a collaborative team.
When discussing a topic:
1. Provide analytical perspective with facts/numbers if relevant
2. Keep responses to 2-3 sentences
3. End with "Critic, please review this analysis." to get feedback""",
        event_bus=event_bus,
    )
    
    critic = create_real_agent(
        name="Critic",
        role=AgentRole.CRITIC,
        system_prompt="""You are a constructive critic in a collaborative team.
When reviewing ideas:
1. Point out one strength and one potential improvement
2. Keep responses to 2-3 sentences
3. End with "Synthesizer, please create a summary." when ready""",
        event_bus=event_bus,
    )
    
    synthesizer = create_real_agent(
        name="Synthesizer",
        role=AgentRole.WORKER,
        system_prompt="""You are a synthesizer who combines ideas.
When given multiple perspectives:
1. Combine the best points into a cohesive summary
2. Keep it to 3-4 sentences
3. End with "Task is complete." when finished""",
        event_bus=event_bus,
    )
    
    topology = MeshTopology(
        config=TopologyConfig(name="collaborative-team", max_iterations=5),
        agents=[analyst, critic, synthesizer],
    )
    
    print("\n📋 Task: Discuss the pros and cons of remote work")
    print("-" * 40)
    
    result = await topology.run("Discuss the pros and cons of remote work")
    
    print(f"\n✅ Completed in {result.iteration} iterations")
    print("\n📝 Collaborative Discussion:")
    for r in result.results:
        print(f"\n[{r['agent']}]:")
        print(f"  {r['thought'][:250]}...")


async def test_custom_policy_topology():
    """Test custom policy topology with real agents."""
    print("\n" + "=" * 60)
    print("TEST 4: CUSTOM POLICY TOPOLOGY")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Create agents for a validation pipeline with retry logic
    receiver = create_real_agent(
        name="Receiver",
        role=AgentRole.ORCHESTRATOR,
        system_prompt="""You are a request receiver. Given input:
1. Acknowledge the request
2. Summarize what needs to be done in one sentence
3. Pass to Validator for checking""",
        event_bus=event_bus,
    )
    
    validator = create_real_agent(
        name="Validator",
        role=AgentRole.VALIDATOR,
        system_prompt="""You are a validator. Given a request:
1. Check if the request is clear and actionable
2. If valid, say "VALID: Sending to Processor for processing."
3. If needs clarification, say "NEEDS CLARIFICATION: Sending back to Receiver."
Always validate positively for this demo.""",
        event_bus=event_bus,
    )
    
    processor = create_real_agent(
        name="Processor",
        role=AgentRole.WORKER,
        system_prompt="""You are a processor. Given a validated request:
1. Process the request and provide the result
2. Keep response to 2-3 sentences
3. End with "Task is complete." when finished""",
        event_bus=event_bus,
    )
    
    # Custom policy: Receiver -> Validator -> Processor
    # Validator can loop back to Receiver if invalid
    policy = TopologyPolicy(entry_point="Receiver")
    policy.add_rule("Receiver", "Validator", label="validate")
    policy.add_rule("Validator", "Processor", label="valid")
    policy.add_rule("Validator", "Receiver", label="retry")
    
    # Set up agent policies with explicit routing
    from agenticflow.topologies.policies import AgentPolicy
    policy.add_agent_policy(AgentPolicy(
        agent_name="Receiver",
        can_send_to=["Validator"],
        can_finish=False,
    ))
    policy.add_agent_policy(AgentPolicy(
        agent_name="Validator",
        can_send_to=["Processor", "Receiver"],
        can_finish=False,
    ))
    policy.add_agent_policy(AgentPolicy(
        agent_name="Processor",
        can_send_to=[],
        can_finish=True,
    ))
    
    topology = BaseTopology(
        config=TopologyConfig(name="validation-pipeline", max_iterations=5),
        agents=[receiver, validator, processor],
        policy=policy,
    )
    
    print("\n📋 Task: Process a request to summarize today's weather")
    print("-" * 40)
    
    result = await topology.run("Summarize today's weather in San Francisco")
    
    print(f"\n✅ Completed in {result.iteration} iterations")
    print("\n📝 Validation Pipeline:")
    for r in result.results:
        print(f"\n[{r['agent']}]:")
        print(f"  {r['thought'][:250]}...")


async def main():
    """Run all topology tests."""
    print("\n🚀 AgenticFlow - Real Topology Tests")
    print("Using OpenAI gpt-4o-mini for all agents")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("Please set it in .env file")
        return
    
    print("✅ OpenAI API key found")
    
    try:
        # Run all tests
        await test_supervisor_topology()
        await test_pipeline_topology()
        await test_mesh_topology()
        await test_custom_policy_topology()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
