"""Coordination patterns - simple async implementations.

Four fundamental patterns for multi-agent collaboration:
- Supervisor: One agent coordinates and delegates to workers
- Pipeline: Sequential processing A → B → C
- Mesh: All agents collaborate with rounds until consensus
- Hierarchical: Tree structure with delegation levels
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .core import AgentConfig, BaseTopology, TopologyResult, TopologyType

if TYPE_CHECKING:
    from ..agent import Agent
    from ..memory import TeamMemory


@dataclass
class Supervisor(BaseTopology):
    """One coordinator agent delegates to and synthesizes from workers.

    The supervisor:
    1. Analyzes the task and decides which workers to use
    2. Delegates subtasks to workers (parallel or sequential)
    3. Collects results and synthesizes final output

    Example:
        >>> supervisor = Supervisor(
        ...     coordinator=AgentConfig(agent=manager, role="project manager"),
        ...     workers=[
        ...         AgentConfig(agent=researcher, role="research"),
        ...         AgentConfig(agent=writer, role="writing"),
        ...     ]
        ... )
        >>> result = await supervisor.run("Create a market analysis report")
    """

    coordinator: AgentConfig
    """The supervising agent that coordinates work."""

    workers: list[AgentConfig] = field(default_factory=list)
    """Worker agents that perform delegated tasks."""

    parallel: bool = True
    """Whether to run workers in parallel (True) or sequential (False)."""

    topology_type: TopologyType = field(default=TopologyType.SUPERVISOR, init=False)

    async def run(
        self,
        task: str,
        *,
        team_memory: TeamMemory | None = None,
    ) -> TopologyResult:
        """Execute supervisor coordination pattern."""
        agent_outputs: dict[str, str] = {}
        execution_order: list[str] = []
        coord_name = self.coordinator.name or "coordinator"

        # Report coordinator starting
        if team_memory:
            await team_memory.report_status(coord_name, "planning")

        # Step 1: Coordinator analyzes task and creates delegation plan
        worker_names = [w.name for w in self.workers]
        planning_prompt = f"""You are coordinating a team to accomplish this task:

TASK: {task}

AVAILABLE TEAM MEMBERS:
{chr(10).join(f"- {w.name}: {w.role or 'general'}" for w in self.workers)}

Analyze the task and create clear, specific subtasks for each team member.
Format your response as:
## Plan
[Your analysis of how to approach the task]

## Assignments
{chr(10).join(f"### {w.name}" + chr(10) + "[Specific instructions for this team member]" for w in self.workers)}
"""

        plan_response = await self.coordinator.agent.run(planning_prompt)
        agent_outputs[coord_name] = plan_response
        execution_order.append(coord_name)

        # Share plan in team memory
        if team_memory:
            await team_memory.share_result(coord_name, {"plan": plan_response})
            await team_memory.report_status(coord_name, "delegating")

        # Step 2: Workers execute their assignments
        async def run_worker(worker: AgentConfig, plan: str) -> tuple[str, str]:
            w_name = worker.name or "worker"
            
            # Report worker starting
            if team_memory:
                await team_memory.report_status(w_name, "working")
            
            worker_prompt = f"""You are working on a team task.

ORIGINAL TASK: {task}

COORDINATOR'S PLAN:
{plan}

YOUR ROLE: {worker.role or "team member"}

Complete your assigned portion of the work. Be thorough and specific."""

            result = await worker.agent.run(worker_prompt)
            
            # Share result and report completion
            if team_memory:
                await team_memory.share_result(w_name, {"output": result})
                await team_memory.report_status(w_name, "done")
            
            return w_name, result

        if self.parallel:
            # Run all workers in parallel
            tasks = [run_worker(w, plan_response) for w in self.workers]
            results = await asyncio.gather(*tasks)
            for name, output in results:
                agent_outputs[name] = output
                execution_order.append(name)
        else:
            # Run workers sequentially
            for worker in self.workers:
                name, output = await run_worker(worker, plan_response)
                agent_outputs[name] = output
                execution_order.append(name)

        # Step 3: Coordinator synthesizes results
        if team_memory:
            await team_memory.report_status(coord_name, "synthesizing")

        synthesis_prompt = f"""You coordinated a team on this task:

ORIGINAL TASK: {task}

TEAM RESULTS:
{chr(10).join(f"### {name}" + chr(10) + output for name, output in agent_outputs.items() if name != coord_name)}

Synthesize these results into a comprehensive final deliverable.
Ensure quality, coherence, and completeness."""

        final_output = await self.coordinator.agent.run(synthesis_prompt)
        agent_outputs[f"{coord_name}_synthesis"] = final_output
        execution_order.append(f"{coord_name}_synthesis")

        # Report completion
        if team_memory:
            await team_memory.share_result(
                coord_name, {"synthesis": final_output, "final": True}
            )
            await team_memory.report_status(coord_name, "done")

        return TopologyResult(
            output=final_output,
            agent_outputs=agent_outputs,
            execution_order=execution_order,
            rounds=1,
            metadata={"pattern": "supervisor", "parallel": self.parallel},
        )

    def get_agents(self) -> list[AgentConfig]:
        return [self.coordinator, *self.workers]


@dataclass
class Pipeline(BaseTopology):
    """Sequential processing where each agent's output feeds the next.

    The pipeline pattern is ideal for workflows like:
    - Research → Draft → Edit → Polish
    - Extract → Transform → Load
    - Analyze → Plan → Execute → Review

    Example:
        >>> pipeline = Pipeline(
        ...     stages=[
        ...         AgentConfig(agent=researcher, role="gather information"),
        ...         AgentConfig(agent=writer, role="draft content"),
        ...         AgentConfig(agent=editor, role="polish and refine"),
        ...     ]
        ... )
        >>> result = await pipeline.run("Create a technical blog post")
    """

    stages: list[AgentConfig] = field(default_factory=list)
    """Ordered list of agents to process sequentially."""

    topology_type: TopologyType = field(default=TopologyType.PIPELINE, init=False)

    async def run(
        self,
        task: str,
        *,
        team_memory: TeamMemory | None = None,
    ) -> TopologyResult:
        """Execute pipeline pattern: A → B → C."""
        agent_outputs: dict[str, str] = {}
        execution_order: list[str] = []

        current_input = task

        for i, stage in enumerate(self.stages):
            name = stage.name or f"stage_{i}"
            is_first = i == 0
            is_last = i == len(self.stages) - 1

            # Report stage starting
            if team_memory:
                await team_memory.report_status(name, "working")

            if is_first:
                prompt = f"""You are the first stage in a processing pipeline.

TASK: {task}

YOUR ROLE: {stage.role or "process the input"}

Process this task and prepare output for the next stage."""
            else:
                prompt = f"""You are stage {i + 1} in a processing pipeline.

ORIGINAL TASK: {task}

PREVIOUS STAGE OUTPUT:
{current_input}

YOUR ROLE: {stage.role or "continue processing"}

{"Produce the final deliverable." if is_last else "Process and pass to next stage."}"""

            result = await stage.agent.run(prompt)
            agent_outputs[name] = result
            execution_order.append(name)
            current_input = result

            # Share result and report completion
            if team_memory:
                await team_memory.share_result(
                    name, {"output": result, "stage": i, "final": is_last}
                )
                await team_memory.report_status(name, "done")

        return TopologyResult(
            output=current_input,  # Last stage output is final
            agent_outputs=agent_outputs,
            execution_order=execution_order,
            rounds=1,
            metadata={"pattern": "pipeline", "stages": len(self.stages)},
        )

    def get_agents(self) -> list[AgentConfig]:
        return list(self.stages)


@dataclass
class Mesh(BaseTopology):
    """All agents collaborate with visibility into each other's work.

    The mesh pattern runs multiple rounds where each agent can see
    and build upon others' contributions until consensus or max rounds.

    Ideal for:
    - Brainstorming and ideation
    - Peer review and refinement
    - Collaborative problem-solving
    - Multi-perspective analysis

    Example:
        >>> mesh = Mesh(
        ...     agents=[
        ...         AgentConfig(agent=analyst1, role="business perspective"),
        ...         AgentConfig(agent=analyst2, role="technical perspective"),
        ...         AgentConfig(agent=analyst3, role="user perspective"),
        ...     ],
        ...     max_rounds=3,
        ... )
        >>> result = await mesh.run("Evaluate this product idea")
    """

    agents: list[AgentConfig] = field(default_factory=list)
    """All collaborating agents."""

    max_rounds: int = 3
    """Maximum collaboration rounds."""

    synthesizer: AgentConfig | None = None
    """Optional dedicated agent to synthesize final output."""

    topology_type: TopologyType = field(default=TopologyType.MESH, init=False)

    async def run(
        self,
        task: str,
        *,
        team_memory: TeamMemory | None = None,
    ) -> TopologyResult:
        """Execute mesh collaboration pattern."""
        agent_outputs: dict[str, str] = {}
        execution_order: list[str] = []

        round_history: list[dict[str, str]] = []

        for round_num in range(1, self.max_rounds + 1):
            round_outputs: dict[str, str] = {}

            # Build context from previous rounds
            history_context = ""
            if round_history:
                history_context = "\n\nPREVIOUS ROUNDS:\n"
                for i, round_data in enumerate(round_history, 1):
                    history_context += f"\n--- Round {i} ---\n"
                    for name, output in round_data.items():
                        history_context += f"\n{name}:\n{output}\n"

            # All agents contribute in parallel
            async def get_contribution(
                agent_cfg: AgentConfig,
                round_n: int = round_num,
            ) -> tuple[str, str]:
                name = agent_cfg.name or "agent"

                # Report agent starting this round
                if team_memory:
                    await team_memory.report_status(name, f"round_{round_n}")

                if round_n == 1:
                    prompt = f"""You are collaborating with a team on this task.

TASK: {task}

YOUR PERSPECTIVE: {agent_cfg.role or "general"}

Provide your initial analysis and contribution."""
                else:
                    prompt = f"""You are in round {round_n} of team collaboration.

TASK: {task}
{history_context}

YOUR PERSPECTIVE: {agent_cfg.role or "general"}

Review your colleagues' contributions and provide your updated analysis.
Build on good ideas, offer corrections, and add new insights."""

                result = await agent_cfg.agent.run(prompt)

                # Share result in team memory
                if team_memory:
                    await team_memory.share_result(
                        name,
                        {"output": result, "round": round_n},
                    )

                return name, result

            tasks = [get_contribution(a) for a in self.agents]
            results = await asyncio.gather(*tasks)

            for name, output in results:
                round_outputs[name] = output
                agent_outputs[f"{name}_round{round_num}"] = output
                execution_order.append(f"{name}_round{round_num}")

            round_history.append(round_outputs)

        # Synthesize final output
        synthesis_context = "\n\nALL CONTRIBUTIONS:\n"
        for i, round_data in enumerate(round_history, 1):
            synthesis_context += f"\n--- Round {i} ---\n"
            for name, output in round_data.items():
                synthesis_context += f"\n{name}:\n{output}\n"

        if self.synthesizer:
            synth_name = self.synthesizer.name or "synthesizer"
            
            if team_memory:
                await team_memory.report_status(synth_name, "synthesizing")

            synthesis_prompt = f"""Synthesize the team's collaborative work.

TASK: {task}
{synthesis_context}

Create a comprehensive final output that incorporates the best insights
from all contributors and all rounds of refinement."""

            final_output = await self.synthesizer.agent.run(synthesis_prompt)
            agent_outputs[synth_name] = final_output
            execution_order.append(synth_name)

            if team_memory:
                await team_memory.share_result(
                    synth_name, {"synthesis": final_output, "final": True}
                )
                await team_memory.report_status(synth_name, "done")
        else:
            # Use first agent as synthesizer
            synth = self.agents[0]
            synth_name = synth.name or "synthesizer"

            if team_memory:
                await team_memory.report_status(synth_name, "synthesizing")

            synthesis_prompt = f"""As a final step, synthesize all contributions.

TASK: {task}
{synthesis_context}

Create a comprehensive final output that represents the team's consensus."""

            final_output = await synth.agent.run(synthesis_prompt)
            agent_outputs[f"{synth_name}_synthesis"] = final_output
            execution_order.append(f"{synth_name}_synthesis")

            if team_memory:
                await team_memory.share_result(
                    synth_name, {"synthesis": final_output, "final": True}
                )
                await team_memory.report_status(synth_name, "done")

        return TopologyResult(
            output=final_output,
            agent_outputs=agent_outputs,
            execution_order=execution_order,
            rounds=self.max_rounds,
            metadata={"pattern": "mesh", "agents": len(self.agents)},
        )

    def get_agents(self) -> list[AgentConfig]:
        agents = list(self.agents)
        if self.synthesizer:
            agents.append(self.synthesizer)
        return agents


@dataclass
class Hierarchical(BaseTopology):
    """Tree structure with manager agents delegating to sub-teams.

    Organizes agents in a hierarchy where:
    - Root manager receives the task
    - Managers delegate to their subordinates
    - Results flow back up the tree

    Ideal for:
    - Large organizations with specializations
    - Complex tasks requiring decomposition
    - Projects with clear ownership structure

    Example:
        >>> hierarchical = Hierarchical(
        ...     root=AgentConfig(agent=ceo, role="executive decision maker"),
        ...     structure={
        ...         "ceo": [
        ...             AgentConfig(agent=tech_lead, role="technical lead"),
        ...             AgentConfig(agent=product_lead, role="product lead"),
        ...         ],
        ...         "tech_lead": [
        ...             AgentConfig(agent=dev1, role="backend developer"),
        ...             AgentConfig(agent=dev2, role="frontend developer"),
        ...         ],
        ...     }
        ... )
        >>> result = await hierarchical.run("Build a new feature")
    """

    root: AgentConfig
    """The root/top-level manager agent."""

    structure: dict[str, list[AgentConfig]] = field(default_factory=dict)
    """Mapping of manager names to their direct reports."""

    topology_type: TopologyType = field(default=TopologyType.HIERARCHICAL, init=False)

    async def run(
        self,
        task: str,
        *,
        team_memory: TeamMemory | None = None,
    ) -> TopologyResult:
        """Execute hierarchical delegation pattern."""
        agent_outputs: dict[str, str] = {}
        execution_order: list[str] = []

        async def delegate(
            manager: AgentConfig,
            subtask: str,
            depth: int = 0,
        ) -> str:
            """Recursively delegate through the hierarchy."""
            manager_name = manager.name or "manager"
            subordinates = self.structure.get(manager_name, [])

            if not subordinates:
                # Leaf node - just do the work
                if team_memory:
                    await team_memory.report_status(manager_name, "working")

                prompt = f"""Complete this task:

TASK: {subtask}

YOUR ROLE: {manager.role or "team member"}

Provide a complete and thorough response."""

                result = await manager.agent.run(prompt)
                agent_outputs[manager_name] = result
                execution_order.append(manager_name)

                if team_memory:
                    await team_memory.share_result(manager_name, {"output": result})
                    await team_memory.report_status(manager_name, "done")

                return result

            # Manager with subordinates - delegate and synthesize
            if team_memory:
                await team_memory.report_status(manager_name, "delegating")

            sub_names = [s.name for s in subordinates]
            delegation_prompt = f"""You are a manager delegating work to your team.

TASK: {subtask}

YOUR TEAM:
{chr(10).join(f"- {s.name}: {s.role or 'team member'}" for s in subordinates)}

Create specific subtasks for each team member.
Format as:
### {sub_names[0]}
[Specific subtask for this person]

{"".join(f"### {name}" + chr(10) + "[Specific subtask for this person]" + chr(10) + chr(10) for name in sub_names[1:])}"""

            plan = await manager.agent.run(delegation_prompt)
            agent_outputs[f"{manager_name}_plan"] = plan
            execution_order.append(f"{manager_name}_plan")

            if team_memory:
                await team_memory.share_result(manager_name, {"plan": plan})

            # Delegate to subordinates in parallel
            async def run_subordinate(sub: AgentConfig) -> tuple[str, str]:
                sub_name = sub.name or "subordinate"
                # Extract this subordinate's subtask from plan (simplified)
                sub_task = f"{subtask}\n\nYour specific assignment from {manager_name}:\n{plan}"
                result = await delegate(sub, sub_task, depth + 1)
                return sub_name, result

            tasks = [run_subordinate(s) for s in subordinates]
            results = await asyncio.gather(*tasks)
            sub_results = dict(results)

            # Manager synthesizes subordinate work
            if team_memory:
                await team_memory.report_status(manager_name, "synthesizing")

            synthesis_prompt = f"""Synthesize your team's work.

ORIGINAL TASK: {subtask}

TEAM RESULTS:
{chr(10).join(f"### {name}" + chr(10) + output for name, output in sub_results.items())}

Create a cohesive deliverable from your team's contributions."""

            synthesis = await manager.agent.run(synthesis_prompt)
            agent_outputs[f"{manager_name}_synthesis"] = synthesis
            execution_order.append(f"{manager_name}_synthesis")

            if team_memory:
                await team_memory.share_result(
                    manager_name, {"synthesis": synthesis, "depth": depth}
                )
                await team_memory.report_status(manager_name, "done")

            return synthesis

        final_output = await delegate(self.root, task)

        return TopologyResult(
            output=final_output,
            agent_outputs=agent_outputs,
            execution_order=execution_order,
            rounds=1,
            metadata={
                "pattern": "hierarchical",
                "depth": self._calculate_depth(),
            },
        )

    def _calculate_depth(self) -> int:
        """Calculate the depth of the hierarchy."""
        if not self.structure:
            return 1

        def get_depth(name: str, visited: set) -> int:
            if name in visited:
                return 0
            visited.add(name)
            subordinates = self.structure.get(name, [])
            if not subordinates:
                return 1
            return 1 + max(get_depth(s.name, visited) for s in subordinates)

        return get_depth(self.root.name, set())

    def get_agents(self) -> list[AgentConfig]:
        agents = [self.root]
        for subordinates in self.structure.values():
            agents.extend(subordinates)
        return agents


# Convenience factory functions


def supervisor(
    coordinator: Agent,
    workers: list[Agent],
    *,
    parallel: bool = True,
    roles: dict[str, str] | None = None,
) -> Supervisor:
    """Create a Supervisor topology with minimal boilerplate.

    Example:
        >>> topology = supervisor(
        ...     coordinator=manager_agent,
        ...     workers=[researcher, writer, editor],
        ...     roles={"researcher": "research", "writer": "draft", "editor": "polish"}
        ... )
    """
    roles = roles or {}
    return Supervisor(
        coordinator=AgentConfig(
            agent=coordinator,
            role=roles.get(getattr(coordinator, "name", ""), "coordinator"),
        ),
        workers=[
            AgentConfig(
                agent=w,
                role=roles.get(getattr(w, "name", ""), None),
            )
            for w in workers
        ],
        parallel=parallel,
    )


def pipeline(stages: list[Agent], *, roles: list[str] | None = None) -> Pipeline:
    """Create a Pipeline topology with minimal boilerplate.

    Example:
        >>> topology = pipeline(
        ...     stages=[researcher, writer, editor],
        ...     roles=["gather info", "draft content", "polish"]
        ... )
    """
    roles = roles or []
    return Pipeline(
        stages=[
            AgentConfig(agent=a, role=roles[i] if i < len(roles) else None)
            for i, a in enumerate(stages)
        ]
    )


def mesh(
    agents: list[Agent],
    *,
    max_rounds: int = 3,
    synthesizer: Agent | None = None,
    roles: list[str] | None = None,
) -> Mesh:
    """Create a Mesh topology with minimal boilerplate.

    Example:
        >>> topology = mesh(
        ...     agents=[analyst1, analyst2, analyst3],
        ...     max_rounds=2,
        ...     roles=["business", "technical", "user experience"]
        ... )
    """
    roles = roles or []
    return Mesh(
        agents=[
            AgentConfig(agent=a, role=roles[i] if i < len(roles) else None)
            for i, a in enumerate(agents)
        ],
        max_rounds=max_rounds,
        synthesizer=AgentConfig(agent=synthesizer) if synthesizer else None,
    )
