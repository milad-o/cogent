"""
A2A Delegation in Topologies

Demonstrates how agents can delegate tasks to specialists
using declarative configuration across different topology patterns.
"""

import asyncio
from agenticflow import Agent
from agenticflow.topologies import Supervisor, Pipeline, Mesh, AgentConfig


# Example 1: Supervisor with Delegation
async def supervisor_with_delegation():
    """Coordinator delegates to workers, workers can request help from specialists."""
    print("\n=== Example 1: Supervisor with Delegation ===\n")
    
    # Define agents
    project_manager = Agent(
        name="project_manager",
        system_prompt="""You are a project manager coordinating a content creation team.
        Analyze tasks and delegate to appropriate team members.
        Use delegate_to when you need work from your team."""
    )
    
    researcher = Agent(
        name="researcher",
        system_prompt="""You are a researcher who gathers information.
        When you receive delegated tasks, complete them and reply with results.
        You can delegate to fact_checker for verification."""
    )
    
    writer = Agent(
        name="writer",
        system_prompt="""You are a content writer.
        When you receive delegated tasks, complete them and reply with results."""
    )
    
    fact_checker = Agent(
        name="fact_checker",
        system_prompt="""You are a fact checker who verifies information accuracy.
        Reply with verification results when delegated tasks."""
    )
    
    # Create supervisor topology with delegation config
    supervisor = Supervisor(
        coordinator=AgentConfig(
            agent=project_manager,
            role="project_manager",
            can_delegate=["researcher", "writer"]  # PM can delegate to team
        ),
        workers=[
            AgentConfig(
                agent=researcher,
                role="researcher",
                can_reply=True,  # Can respond to delegated tasks
                can_delegate=["fact_checker"]  # Can delegate to specialist
            ),
            AgentConfig(
                agent=writer,
                role="writer",
                can_reply=True
            ),
            AgentConfig(
                agent=fact_checker,
                role="fact_checker",
                can_reply=True  # Specialist handles delegated requests
            ),
        ],
        parallel=False  # Sequential for this example
    )
    
    # Tools are auto-injected based on can_delegate/can_reply:
    # - project_manager gets: delegate_to(target: researcher|writer)
    # - researcher gets: reply_with_result, delegate_to(target: fact_checker)
    # - writer gets: reply_with_result
    # - fact_checker gets: reply_with_result
    
    result = await supervisor.run(
        "Create a fact-checked article about quantum computing breakthroughs in 2024"
    )
    
    print(f"Final output: {result.output}")
    print(f"\nAgent outputs: {result.agent_outputs}")
    print(f"Execution order: {result.execution_order}")


# Example 2: Pipeline with Delegation
async def pipeline_with_delegation():
    """Pipeline stages can delegate to specialists when needed."""
    print("\n=== Example 2: Pipeline with Delegation ===\n")
    
    # Define pipeline agents
    data_collector = Agent(
        name="data_collector",
        system_prompt="""You collect raw data for analysis.
        Gather information from various sources."""
    )
    
    data_analyzer = Agent(
        name="data_analyzer",
        system_prompt="""You analyze data and generate insights.
        You can delegate complex statistical analysis to the specialist."""
    )
    
    report_writer = Agent(
        name="report_writer",
        system_prompt="""You write clear, professional reports based on analysis."""
    )
    
    statistician = Agent(
        name="statistician",
        system_prompt="""You are a statistical analysis specialist.
        Reply with detailed statistical analysis when delegated."""
    )
    
    # Create pipeline with delegation
    pipeline = Pipeline(
        stages=[
            AgentConfig(
                agent=data_collector,
                role="data_collection"
            ),
            AgentConfig(
                agent=data_analyzer,
                role="data_analysis",
                can_delegate=["statistician"]  # Can delegate to specialist
            ),
            AgentConfig(
                agent=report_writer,
                role="report_writing"
            ),
            AgentConfig(
                agent=statistician,
                role="statistician",
                can_reply=True  # Handles delegated statistical tasks
            ),
        ]
    )
    
    # Tools auto-injected:
    # - data_analyzer gets: delegate_to(target: statistician)
    # - statistician gets: reply_with_result
    
    result = await pipeline.run(
        "Analyze customer satisfaction survey data and create a report"
    )
    
    print(f"Final output: {result.output}")
    print(f"\nExecution order: {result.execution_order}")


# Example 3: Mesh with Delegation
async def mesh_with_delegation():
    """Mesh collaboration where agents can delegate to external specialists."""
    print("\n=== Example 3: Mesh with Delegation ===\n")
    
    # Define mesh participants
    business_analyst = Agent(
        name="business_analyst",
        system_prompt="""You analyze business implications and strategy.
        You can delegate financial analysis to the finance specialist."""
    )
    
    tech_analyst = Agent(
        name="tech_analyst",
        system_prompt="""You analyze technical feasibility and architecture.
        Build on others' insights."""
    )
    
    ux_analyst = Agent(
        name="ux_analyst",
        system_prompt="""You analyze user experience and usability.
        Build on others' insights."""
    )
    
    finance_specialist = Agent(
        name="finance_specialist",
        system_prompt="""You provide detailed financial analysis.
        Reply with financial insights when delegated."""
    )
    
    # Create mesh with delegation
    mesh = Mesh(
        agents=[
            AgentConfig(
                agent=business_analyst,
                role="business_perspective",
                can_delegate=["finance_specialist"]  # Can delegate financial analysis
            ),
            AgentConfig(
                agent=tech_analyst,
                role="technical_perspective"
            ),
            AgentConfig(
                agent=ux_analyst,
                role="user_perspective"
            ),
            AgentConfig(
                agent=finance_specialist,
                role="finance_specialist",
                can_reply=True  # Handles delegated financial requests
            ),
        ],
        max_rounds=2,
        require_consensus=False
    )
    
    # Tools auto-injected:
    # - business_analyst gets: delegate_to(target: finance_specialist)
    # - finance_specialist gets: reply_with_result
    
    result = await mesh.run(
        "Evaluate the viability of launching a new mobile banking app"
    )
    
    print(f"Final output: {result.output}")
    print(f"\nRounds completed: {result.rounds}")


# Example 4: Dynamic Delegation Policy
async def dynamic_delegation_policy():
    """Demonstrate flexible delegation policies."""
    print("\n=== Example 4: Dynamic Delegation Policy ===\n")
    
    # Define agents
    senior_dev = Agent(
        name="senior_dev",
        system_prompt="""You are a senior developer who can delegate to any team member.
        Analyze tasks and distribute work efficiently."""
    )
    
    junior_dev = Agent(
        name="junior_dev",
        system_prompt="""You are a junior developer.
        Complete delegated tasks and reply with results.
        You can only delegate to the code_reviewer."""
    )
    
    code_reviewer = Agent(
        name="code_reviewer",
        system_prompt="""You review code for quality and best practices.
        Reply with review results."""
    )
    
    tester = Agent(
        name="tester",
        system_prompt="""You write and run tests.
        Reply with test results."""
    )
    
    # Create supervisor with hierarchical delegation
    supervisor = Supervisor(
        coordinator=AgentConfig(
            agent=senior_dev,
            role="senior_developer",
            can_delegate=True  # True = can delegate to ANY worker
        ),
        workers=[
            AgentConfig(
                agent=junior_dev,
                role="junior_developer",
                can_reply=True,
                can_delegate=["code_reviewer"]  # Limited delegation
            ),
            AgentConfig(
                agent=code_reviewer,
                role="code_reviewer",
                can_reply=True
            ),
            AgentConfig(
                agent=tester,
                role="tester",
                can_reply=True
            ),
        ]
    )
    
    # Delegation policies:
    # - senior_dev can delegate to: junior_dev, code_reviewer, tester (all workers)
    # - junior_dev can delegate to: code_reviewer only
    # - code_reviewer can only reply (no delegation)
    # - tester can only reply (no delegation)
    
    result = await supervisor.run(
        "Implement a new user authentication feature with tests"
    )
    
    print(f"Final output: {result.output}")


# Example 5: Cross-Topology Delegation
async def cross_topology_delegation():
    """Demonstrate delegation across different topology patterns."""
    print("\n=== Example 5: Cross-Topology Delegation ===\n")
    
    # This example shows how the same agents can participate in different
    # topologies with different delegation policies
    
    researcher = Agent(
        name="researcher",
        system_prompt="You research topics thoroughly."
    )
    
    analyst = Agent(
        name="analyst",
        system_prompt="You analyze data and provide insights."
    )
    
    writer = Agent(
        name="writer",
        system_prompt="You write clear, engaging content."
    )
    
    # Same agents, different topologies and policies
    
    # Pipeline: No delegation needed (linear flow)
    pipeline = Pipeline(
        stages=[
            AgentConfig(agent=researcher, role="research"),
            AgentConfig(agent=analyst, role="analyze"),
            AgentConfig(agent=writer, role="write"),
        ]
    )
    
    # Supervisor: Analyst coordinates, can delegate to researcher
    supervisor = Supervisor(
        coordinator=AgentConfig(
            agent=analyst,
            role="coordinator",
            can_delegate=["researcher", "writer"]
        ),
        workers=[
            AgentConfig(agent=researcher, role="researcher", can_reply=True),
            AgentConfig(agent=writer, role="writer", can_reply=True),
        ]
    )
    
    # Mesh: All collaborate, analyst can delegate to specialist
    specialist = Agent(
        name="specialist",
        system_prompt="You provide expert domain knowledge."
    )
    
    mesh = Mesh(
        agents=[
            AgentConfig(agent=researcher, role="research_perspective"),
            AgentConfig(
                agent=analyst,
                role="analysis_perspective",
                can_delegate=["specialist"]
            ),
            AgentConfig(agent=writer, role="writing_perspective"),
            AgentConfig(agent=specialist, role="specialist", can_reply=True),
        ],
        max_rounds=2
    )
    
    print("Same agents, different delegation policies per topology:")
    print("- Pipeline: No delegation (sequential)")
    print("- Supervisor: Analyst delegates to team")
    print("- Mesh: Analyst can delegate to external specialist")


async def main():
    """Run all delegation examples."""
    await supervisor_with_delegation()
    await pipeline_with_delegation()
    await mesh_with_delegation()
    await dynamic_delegation_policy()
    await cross_topology_delegation()


if __name__ == "__main__":
    asyncio.run(main())
