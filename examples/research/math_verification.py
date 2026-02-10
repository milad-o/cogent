"""Math Problem Solving: Single vs Multi-Agent Comparison.

This script compares single-agent and multi-agent approaches for solving GSM8K-style
math problems with verifiable outcomes across multiple LLM providers.

**Approaches:**
- **Single Agent**: One agent with all math tools
- **Multi-Agent**: Specialist agents (arithmetic, growth, equations) coordinated by an orchestrator

**Providers tested:**
- OpenAI (GPT-4o, GPT-4o-mini)
- xAI (Grok-4)
- Anthropic (Claude Sonnet 4)
- Mistral (Mistral Large)
- Google Gemini (Gemini 2.5 Pro)

Usage:
    # Test specific model
    uv run python examples/research/math_verification.py --model gpt4

    # Test all models
    uv run python examples/research/math_verification.py --all

    # List available models
    uv run python examples/research/math_verification.py --list
"""

import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, Field

from cogent import Agent, Observer, tool
from cogent.agent.resilience import ResilienceConfig

# Configure matplotlib for SVG output and better styling
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


# =============================================================================
# Tool Definitions
# =============================================================================


@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression safely.

    Args:
        expression: Math expression (e.g., "50 * 3", "200 + 150", "1000 - 105")

    Returns:
        The numerical result
    """
    allowed_chars = set("0123456789+-*/()., ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError(f"Expression contains invalid characters: {expression}")

    try:
        result = eval(expression)
        return float(result)
    except Exception as e:
        raise ValueError(f"Failed to evaluate '{expression}': {e}")


@tool
def calculate_percentage(value: float, percentage: float) -> float:
    """Calculate a percentage of a value.

    Args:
        value: The base value
        percentage: The percentage (e.g., 40 for 40%)

    Returns:
        The calculated amount
    """
    return (value * percentage) / 100


@tool
def calculate_growth_rate(initial: float, final: float) -> float:
    """Calculate the growth rate between two values.

    Args:
        initial: Starting value
        final: Ending value

    Returns:
        Growth rate as a percentage
    """
    if initial == 0:
        raise ValueError("Initial value cannot be zero")
    return ((final - initial) / initial) * 100


@tool
def calculate_compound_growth(initial: float, rates: list[float]) -> float:
    """Calculate final value after applying multiple growth rates sequentially.

    Args:
        initial: Starting value
        rates: List of growth rates as percentages (e.g., [15, 23, -8, 31])

    Returns:
        Final value after all growth applied
    """
    value = initial
    for rate in rates:
        value = value * (1 + rate / 100)
    return value


@tool
def calculate_average(values: list[float]) -> float:
    """Calculate the arithmetic mean of a list of values.

    Args:
        values: List of numbers

    Returns:
        The average value
    """
    if not values:
        raise ValueError("Cannot calculate average of empty list")
    return sum(values) / len(values)


@tool
def solve_linear_equation(a: float, b: float) -> float:
    """Solve linear equation: ax + b = 0

    Args:
        a: Coefficient of x
        b: Constant term

    Returns:
        Solution for x
    """
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be zero in linear equation")
    return -b / a


# =============================================================================
# Problem Definitions
# =============================================================================


@dataclass
class MathProblem:
    """A math problem with a verifiable answer."""

    question: str
    ground_truth: float
    description: str


PROBLEMS = [
    MathProblem(
        question="""A company's revenue was $2.4M at the start of the year.
It grew 15% in Q1, 23% in Q2, fell 8% in Q3, then grew 31% in Q4.
What was the final revenue? Also calculate the average quarterly growth rate.""",
        ground_truth=4091412.96,
        description="Compound growth",
    ),
    MathProblem(
        question="""A store had 150 apples. On Monday, they sold 40% of them.
On Tuesday, they received a shipment that increased their stock by 60 apples.
On Wednesday, they sold 25% of their current stock.
How many apples remain?""",
        ground_truth=112.5,
        description="Multi-step percentages",
    ),
    MathProblem(
        question="""A carnival booth made $50 per day selling popcorn.
It made three times as much selling cotton candy each day.
Over 5 days, the booth earned revenue but had to pay $30 rent and $75 for ingredients.
What was the profit after expenses?""",
        ground_truth=895.0,
        description="Revenue & expenses",
    ),
]


# =============================================================================
# Structured Output Schema
# =============================================================================


class MathSolution(BaseModel):
    """Structured output for math problem solutions."""

    reasoning: str = Field(
        description="Step-by-step explanation of how you solved the problem"
    )
    final_answer: float = Field(description="The final numerical answer to the problem")


# =============================================================================
# Model Configurations
# =============================================================================

MODELS = {
    "gpt4": {
        "name": "GPT-4o",
        "orchestrator": "gpt-4o",
        "specialist": "gpt-4o-mini",
    },
    "grok": {
        "name": "xAI Grok-4",
        "orchestrator": "grok-4",
        "specialist": "grok-4-1-fast",
        "resilience": ResilienceConfig(
            timeout_seconds=180.0,  # 3 minutes for slower responses
        ),
    },
    "claude": {
        "name": "Claude Sonnet 4",
        "orchestrator": "claude-sonnet-4",
        "specialist": "claude-sonnet-4",
    },
    "mistral": {
        "name": "Mistral Large",
        "orchestrator": "mistral-large-latest",
        "specialist": "mistral-small-latest",
        "resilience": ResilienceConfig(
            timeout_seconds=90.0,
        ),
    },
    "gemini": {
        "name": "Gemini 2.5 Pro",
        "orchestrator": "gemini-2.5-pro",
        "specialist": "gemini-2.5-flash",
    },
}


# =============================================================================
# Agent Factory
# =============================================================================


def create_agents(model_config: dict) -> tuple[Agent, Agent]:
    """Create single and multi-agent systems for a model."""
    orchestrator_model = model_config["orchestrator"]
    specialist_model = model_config["specialist"]

    # Get resilience config if specified (for slow models like Grok)
    resilience = model_config.get("resilience")

    # Create observer for detailed visibility
    observer = Observer(level="progress")

    # Single agent with all tools
    single_agent = Agent(
        name="MathSolver",
        model=orchestrator_model,
        instructions="""You are an expert mathematics problem solver.

Solve problems step-by-step using the available tools.
Show your reasoning clearly, then provide the final numerical answer.""",
        tools=[
            calculator,
            calculate_percentage,
            calculate_growth_rate,
            calculate_compound_growth,
            calculate_average,
            solve_linear_equation,
        ],
        observer=observer,
        resilience=resilience,
    )

    # Specialists
    arithmetic_expert = Agent(
        name="ArithmeticExpert",
        model=specialist_model,
        instructions="""You are an arithmetic specialist.

Use the calculator and percentage tools to perform calculations accurately.
Show each step of your work clearly.""",
        tools=[calculator, calculate_percentage],
        resilience=resilience,
    )

    growth_expert = Agent(
        name="GrowthExpert",
        model=specialist_model,
        instructions="""You are a growth and statistics specialist.

Use growth rate and compound growth tools to analyze changes over time.
Calculate averages when needed.""",
        tools=[calculate_growth_rate, calculate_compound_growth, calculate_average],
        resilience=resilience,
    )

    equation_expert = Agent(
        name="EquationExpert",
        model=specialist_model,
        instructions="""You are an equation solving specialist.

Solve linear equations using the provided tool.
Explain the solution process.""",
        tools=[solve_linear_equation],
        resilience=resilience,
    )

    # Orchestrator
    multi_agent_orchestrator = Agent(
        name="MathOrchestrator",
        model=orchestrator_model,
        instructions="""You are a math problem orchestrator.

Break down complex problems and delegate to specialists:
- ArithmeticExpert: Basic calculations, percentages
- GrowthExpert: Growth rates, compound growth, averages
- EquationExpert: Solving equations

Synthesize their results into a final answer with clear reasoning.""",
        subagents=[arithmetic_expert, growth_expert, equation_expert],
        resilience=resilience,
    )

    return single_agent, multi_agent_orchestrator


# =============================================================================
# Evaluation
# =============================================================================


@dataclass
class EvaluationResult:
    """Results from solving a problem."""

    problem_description: str
    ground_truth: float
    agent_answer: float | None
    reasoning: str
    duration: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tool_calls: int
    correct: bool


async def evaluate_approach(agent: Agent, problem: MathProblem) -> EvaluationResult:
    """Evaluate one problem with one agent."""
    start = time.time()
    response = await agent.run(problem.question, output=MathSolution)
    duration = time.time() - start

    # Get structured output
    from cogent.agent.output import StructuredResult

    if isinstance(response.content, StructuredResult):
        solution = response.content.data
    else:
        solution = response.content

    answer = solution.final_answer if isinstance(solution, MathSolution) else None
    reasoning = (
        solution.reasoning
        if isinstance(solution, MathSolution)
        else "No solution provided"
    )

    # Check correctness (allow small floating point tolerance)
    correct = answer is not None and abs(answer - problem.ground_truth) < 0.01

    # Count tool calls (including subagent tool calls)
    tool_calls = len(response.tool_calls)
    if response.subagent_responses:
        for sub_resp in response.subagent_responses:
            tool_calls += len(sub_resp.tool_calls)

    # Get token counts (some providers may not report tokens)
    if response.metadata.tokens:
        input_tokens = response.metadata.tokens.prompt_tokens
        output_tokens = response.metadata.tokens.completion_tokens
        total_tokens = response.metadata.tokens.total_tokens
    else:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

    return EvaluationResult(
        problem_description=problem.description,
        ground_truth=problem.ground_truth,
        agent_answer=answer,
        reasoning=reasoning,
        duration=duration,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        tool_calls=tool_calls,
        correct=correct,
    )


async def run_evaluation(model_key: str) -> tuple[str, pd.DataFrame]:
    """Run evaluation for a specific model."""
    model_config = MODELS[model_key]
    model_name = model_config["name"]

    print(f"\n{'=' * 60}")
    print(f"Testing: {model_name}")
    print(f"{'=' * 60}")

    # Create agents
    single_agent, multi_agent = create_agents(model_config)
    print("✓ Agents created")

    # Run all evaluations concurrently
    tasks = []
    for problem in PROBLEMS:
        tasks.append(evaluate_approach(single_agent, problem))
        tasks.append(evaluate_approach(multi_agent, problem))

    results = await asyncio.gather(*tasks)

    # Extract results
    single_results = [results[i] for i in range(0, len(results), 2)]
    multi_results = [results[i] for i in range(1, len(results), 2)]

    # Create detailed results DataFrame
    df_detailed = pd.DataFrame(
        {
            "Problem": [f"P{i}: {p.description}" for i, p in enumerate(PROBLEMS, 1)],
            "Ground Truth": [p.ground_truth for p in PROBLEMS],
            "Single Answer": [r.agent_answer for r in single_results],
            "Single ✓": ["✓" if r.correct else "✗" for r in single_results],
            "Single Tokens": [r.total_tokens for r in single_results],
            "Single In": [r.input_tokens for r in single_results],
            "Single Out": [r.output_tokens for r in single_results],
            "Single Tools": [r.tool_calls for r in single_results],
            "Single Time": [r.duration for r in single_results],
            "Multi Answer": [r.agent_answer for r in multi_results],
            "Multi ✓": ["✓" if r.correct else "✗" for r in multi_results],
            "Multi Tokens": [r.total_tokens for r in multi_results],
            "Multi In": [r.input_tokens for r in multi_results],
            "Multi Out": [r.output_tokens for r in multi_results],
            "Multi Tools": [r.tool_calls for r in multi_results],
            "Multi Time": [r.duration for r in multi_results],
        }
    )

    print("\n✓ Evaluation complete!")
    print(df_detailed.to_string(index=False))

    # Save visualization
    save_visualization(
        model_key, model_name, df_detailed, single_results, multi_results
    )

    return model_name, df_detailed


def save_visualization(
    model_key: str,
    model_name: str,
    df_detailed: pd.DataFrame,
    single_results: list,
    multi_results: list,
) -> None:
    """Create and save visualization for a model."""
    # Calculate metrics
    single_correct = (df_detailed["Single ✓"] == "✓").sum()
    multi_correct = (df_detailed["Multi ✓"] == "✓").sum()
    single_total_tokens = df_detailed["Single Tokens"].sum()
    multi_total_tokens = df_detailed["Multi Tokens"].sum()
    single_input_tokens = df_detailed["Single In"].sum()
    multi_input_tokens = df_detailed["Multi In"].sum()
    single_output_tokens = df_detailed["Single Out"].sum()
    multi_output_tokens = df_detailed["Multi Out"].sum()
    single_time = df_detailed["Single Time"].sum()
    multi_time = df_detailed["Multi Time"].sum()
    single_tools = df_detailed["Single Tools"].sum()
    multi_tools = df_detailed["Multi Tools"].sum()

    # Create visualization
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, hspace=0.1, wspace=0.15)

    # 1. Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    accuracy_data = pd.DataFrame(
        {
            "Approach": ["Single", "Multi"],
            "Accuracy": [
                single_correct / len(PROBLEMS) * 100,
                multi_correct / len(PROBLEMS) * 100,
            ],
        }
    )
    sns.barplot(
        data=accuracy_data,
        x="Approach",
        y="Accuracy",
        hue="Approach",
        palette=["#3498db", "#e74c3c"],
        legend=False,
        ax=ax1,
    )
    ax1.set_ylim(0, 110)
    ax1.set_title("Accuracy", fontsize=9)
    ax1.tick_params(labelsize=8)

    # 2. Token Usage
    ax2 = fig.add_subplot(gs[0, 1])
    token_data = pd.DataFrame(
        {
            "Approach": ["Single", "Single", "Multi", "Multi"],
            "Type": ["Input", "Output", "Input", "Output"],
            "Tokens": [
                single_input_tokens,
                single_output_tokens,
                multi_input_tokens,
                multi_output_tokens,
            ],
        }
    )
    sns.barplot(data=token_data, x="Approach", y="Tokens", hue="Type", ax=ax2)
    ax2.set_title("Token Usage (Input/Output)", fontsize=9)
    ax2.legend(loc="upper right", fontsize=7)
    ax2.tick_params(labelsize=8)

    # 3. Tool Calls
    ax3 = fig.add_subplot(gs[0, 2])
    tool_data = pd.DataFrame(
        {"Approach": ["Single", "Multi"], "Tools": [single_tools, multi_tools]}
    )
    sns.barplot(
        data=tool_data,
        x="Approach",
        y="Tools",
        hue="Approach",
        palette=["#3498db", "#e74c3c"],
        legend=False,
        ax=ax3,
    )
    ax3.set_title("Total Tool Calls", fontsize=9)
    ax3.tick_params(labelsize=8)

    # 4. Time
    ax4 = fig.add_subplot(gs[1, 0])
    time_data = pd.DataFrame(
        {"Approach": ["Single", "Multi"], "Time": [single_time, multi_time]}
    )
    sns.barplot(
        data=time_data,
        x="Approach",
        y="Time",
        hue="Approach",
        palette=["#3498db", "#e74c3c"],
        legend=False,
        ax=ax4,
    )
    ax4.set_title("Total Time (seconds)", fontsize=9)
    ax4.tick_params(labelsize=8)

    # 5. Per-Problem Tokens
    ax5 = fig.add_subplot(gs[1, 1])
    problem_data = []
    for i, (single, multi) in enumerate(zip(single_results, multi_results), 1):
        problem_data.append(
            {"Problem": f"P{i}", "Approach": "Single", "Tokens": single.total_tokens}
        )
        problem_data.append(
            {"Problem": f"P{i}", "Approach": "Multi", "Tokens": multi.total_tokens}
        )
    problem_df = pd.DataFrame(problem_data)
    sns.barplot(data=problem_df, x="Problem", y="Tokens", hue="Approach", ax=ax5)
    ax5.set_title("Tokens per Problem", fontsize=9)
    ax5.legend(loc="upper right", fontsize=7)
    ax5.tick_params(labelsize=8)

    # 6. Efficiency Averages
    ax6 = fig.add_subplot(gs[1, 2])
    efficiency_data = {
        "Approach": ["Single", "Multi"],
        "Tokens/Problem": [
            single_total_tokens / len(PROBLEMS),
            multi_total_tokens / len(PROBLEMS),
        ],
        "Time/Problem (s)": [
            single_time / len(PROBLEMS),
            multi_time / len(PROBLEMS),
        ],
    }
    efficiency_df = pd.DataFrame(efficiency_data)
    efficiency_df.set_index("Approach")[["Tokens/Problem"]].plot(
        kind="bar", ax=ax6, color=["#3498db", "#e74c3c"], legend=False, width=0.7
    )
    ax6.set_title("Avg Tokens/Problem", fontsize=9)
    ax6.set_ylabel("Tokens", fontsize=8)
    ax6.tick_params(labelsize=8)
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)

    plt.suptitle(f"Single vs Multi-Agent: {model_name}", fontsize=11, fontweight="bold")

    # Save to file
    output_dir = Path(__file__).parent
    output_file = output_dir / f"results_{model_key}.svg"
    plt.savefig(output_file, format="svg", bbox_inches="tight")
    print(f"\n✓ Visualization saved: {output_file}")
    plt.close()


# =============================================================================
# CLI
# =============================================================================


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Math verification benchmark")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Model to test (default: all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        for key, config in MODELS.items():
            print(f"  {key:10s} - {config['name']}")
        return

    if args.all:
        models_to_test = list(MODELS.keys())
    elif args.model:
        models_to_test = [args.model]
    else:
        # Default: test all
        models_to_test = list(MODELS.keys())

    print(f"\nTesting {len(models_to_test)} model(s): {', '.join(models_to_test)}")

    # Run evaluations
    all_results = {}
    for model_key in models_to_test:
        try:
            model_name, df = await run_evaluation(model_key)
            all_results[model_name] = df
        except Exception as e:
            print(f"\n✗ Error testing {MODELS[model_key]['name']}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Completed {len(all_results)}/{len(models_to_test)} evaluations")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
