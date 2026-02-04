"""Math problem solving: Single vs Multi-Agent with verifiable outcomes.

Run:
    uv run python examples/research/math_verification.py
"""

import asyncio
import time
from dataclasses import dataclass

from pydantic import BaseModel, Field

from cogent import Agent, Observer, tool


# ============================================================================
# TOOLS: Math operations with exact computation
# ============================================================================

@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: Math expression (e.g., "50 * 3", "200 + 150", "1000 - 105")
    
    Returns:
        The numerical result
    
    Examples:
        calculator("50 * 3") -> 150.0
        calculator("(1000 - 105) / 5") -> 179.0
    """
    # Safe evaluation - only allow numbers and basic operators
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
    
    Examples:
        calculate_percentage(100, 25) -> 25.0
        calculate_percentage(50, 40) -> 20.0
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
    
    Examples:
        calculate_growth_rate(100, 150) -> 50.0 (50% growth)
        calculate_growth_rate(100, 92) -> -8.0 (8% decline)
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
    
    Examples:
        calculate_compound_growth(100, [10, 20]) -> 132.0
        calculate_compound_growth(2400000, [15, 23, -8, 31]) -> 4091412.96
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
    
    Examples:
        calculate_average([10, 20, 30]) -> 20.0
        calculate_average([15, 23, -8, 31]) -> 15.25
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
    
    Examples:
        solve_linear_equation(2, -10) -> 5.0  (2x - 10 = 0, x = 5)
        solve_linear_equation(3, 9) -> -3.0   (3x + 9 = 0, x = -3)
    """
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be zero in linear equation")
    return -b / a


# ============================================================================
# PROBLEMS: GSM8K-style with exact answers
# ============================================================================

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
        ground_truth=4091412.96,  # Final revenue
        description="Compound growth with multiple periods"
    ),
    
    MathProblem(
        question="""A store had 150 apples. On Monday, they sold 40% of them.
On Tuesday, they received a shipment that increased their stock by 60 apples.
On Wednesday, they sold 25% of their current stock.
How many apples remain?""",
        ground_truth=81.0,  # Final apples: (150 - 60) + 60 = 150, then 150 - 37.5 = 112.5 wait...
        # Let me recalculate: 150 - (150*0.4) = 150 - 60 = 90
        # Then 90 + 60 = 150
        # Then 150 - (150*0.25) = 150 - 37.5 = 112.5
        # Hmm, should be whole apples... let me recalculate with floor
        # Actually: 150 * 0.4 = 60 sold, 90 remain
        # 90 + 60 = 150
        # 150 * 0.25 = 37.5, so 150 - 37.5 = 112.5... but can't have half apple
        # I'll use 112.5 as the mathematical answer
        description="Multi-step arithmetic with percentages"
    ),
    
    MathProblem(
        question="""A carnival booth made $50 per day selling popcorn.
It made three times as much selling cotton candy each day.
Over 5 days, the booth earned revenue but had to pay $30 rent and $75 for ingredients.
What was the profit after expenses?""",
        ground_truth=895.0,  # (50 + 150) * 5 - 105 = 1000 - 105 = 895
        description="GSM8K example problem"
    ),
]

# Fix the apples problem ground truth
PROBLEMS[1].ground_truth = 112.5  # 150 - 60 + 60 - 37.5


# ============================================================================
# STRUCTURED OUTPUT
# ============================================================================

class MathSolution(BaseModel):
    """Structured output for math problem solutions."""
    reasoning: str = Field(description="Step-by-step explanation of how you solved the problem")
    final_answer: float = Field(description="The final numerical answer to the problem")


# ============================================================================
# EVALUATION
# ============================================================================

@dataclass
class EvaluationResult:
    """Results from solving a problem."""
    problem_description: str
    ground_truth: float
    agent_answer: float | None
    reasoning: str
    duration: float
    tokens: int
    tool_calls: int
    correct: bool


def is_correct(answer: float | None, ground_truth: float) -> bool:
    """Check if answer exactly matches ground truth."""
    if answer is None:
        return False
    return answer == ground_truth


async def evaluate_approach(agent: Agent, problem: MathProblem) -> EvaluationResult:
    """Evaluate one problem with one agent."""
    start = time.time()
    response = await agent.run(problem.question, output=MathSolution)
    duration = time.time() - start
    
    # Get structured output - it's wrapped in StructuredResult
    from cogent.agent.output import StructuredResult
    if isinstance(response.content, StructuredResult):
        solution = response.content.data
    else:
        solution = response.content
    
    answer = solution.final_answer if isinstance(solution, MathSolution) else None
    reasoning = solution.reasoning if isinstance(solution, MathSolution) else "No solution provided"
    
    correct = is_correct(answer, problem.ground_truth)
    
    # Count tool calls (including subagent tool calls)
    tool_calls = len(response.tool_calls)
    if response.subagent_responses:
        for sub_resp in response.subagent_responses:
            tool_calls += len(sub_resp.tool_calls)
    
    return EvaluationResult(
        problem_description=problem.description,
        ground_truth=problem.ground_truth,
        agent_answer=answer,
        reasoning=reasoning,
        duration=duration,
        tokens=response.metadata.tokens.total_tokens,
        tool_calls=tool_calls,
        correct=correct,
    )


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Compare single agent vs multi-agent on verifiable math problems."""
    
    observer = Observer(level="progress")
    
    # ========================================================================
    # SINGLE AGENT: One agent with all tools
    # ========================================================================
    
    single_agent = Agent(
        name="MathSolver",
        model="gemini:gemini-2.5-pro",
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
    )
    
    # ========================================================================
    # MULTI-AGENT: Specialists + Orchestrator
    # ========================================================================
    
    # Arithmetic specialist
    arithmetic_expert = Agent(
        name="ArithmeticExpert",
        model="gemini:gemini-2.5-flash",
        instructions="""You are an arithmetic specialist.
        
Use the calculator and percentage tools to perform calculations accurately.
Show each step of your work clearly.""",
        tools=[calculator, calculate_percentage],
    )
    
    # Growth/statistics specialist
    growth_expert = Agent(
        name="GrowthExpert",
        model="gemini:gemini-2.5-flash",
        instructions="""You are a growth and statistics specialist.
        
Use growth rate and compound growth tools to analyze changes over time.
Calculate averages when needed.""",
        tools=[calculate_growth_rate, calculate_compound_growth, calculate_average],
    )
    
    # Equation solver specialist
    equation_expert = Agent(
        name="EquationExpert",
        model="gemini:gemini-2.5-flash",
        instructions="""You are an equation solving specialist.
        
Solve linear equations using the provided tool.
Explain the solution process.""",
        tools=[solve_linear_equation],
    )
    
    # Orchestrator that delegates to specialists
    multi_agent_orchestrator = Agent(
        name="MathOrchestrator",
        model="gemini:gemini-2.5-pro",
        instructions="""You are a math problem orchestrator.
        
Break down complex problems and delegate to specialists:
- ArithmeticExpert: Basic calculations, percentages
- GrowthExpert: Growth rates, compound growth, averages
- EquationExpert: Solving equations

Synthesize their results into a final answer with clear reasoning.""",
        subagents=[arithmetic_expert, growth_expert, equation_expert],
        observer=observer,
    )
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("MATH PROBLEM SOLVING: Single vs Multi-Agent")
    print("=" * 70)
    
    single_results = []
    multi_results = []
    
    for i, problem in enumerate(PROBLEMS, 1):
        print(f"\n{'=' * 70}")
        print(f"PROBLEM {i}: {problem.description}")
        print(f"{'=' * 70}")
        print(f"Ground Truth: {problem.ground_truth:,.2f}\n")
        
        # Single agent
        print("──────────────────────────────────────────────────────────────────────")
        print("SINGLE AGENT (all tools)")
        print("──────────────────────────────────────────────────────────────────────")
        result_single = await evaluate_approach(single_agent, problem)
        single_results.append(result_single)
        
        status = "✓ CORRECT" if result_single.correct else "✗ WRONG"
        print(f"Status: {status}")
        print(f"Duration: {result_single.duration:.1f}s")
        print(f"Tokens: {result_single.tokens:,}")
        print(f"Tool Calls: {result_single.tool_calls}")
        
        # Multi-agent
        print("\n──────────────────────────────────────────────────────────────────────")
        print("MULTI-AGENT (specialist delegation)")
        print("──────────────────────────────────────────────────────────────────────")
        result_multi = await evaluate_approach(multi_agent_orchestrator, problem)
        multi_results.append(result_multi)
        
        status = "✓ CORRECT" if result_multi.correct else "✗ WRONG"
        print(f"Status: {status}")
        print(f"Duration: {result_multi.duration:.1f}s")
        print(f"Tokens: {result_multi.tokens:,}")
        print(f"Tool Calls: {result_multi.tool_calls}")
    
    # ========================================================================
    # SUMMARY COMPARISON
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY: Single vs Multi-Agent")
    print("=" * 70)
    
    # Calculate totals
    single_correct = sum(1 for r in single_results if r.correct)
    multi_correct = sum(1 for r in multi_results if r.correct)
    
    single_tokens = sum(r.tokens for r in single_results)
    multi_tokens = sum(r.tokens for r in multi_results)
    
    single_time = sum(r.duration for r in single_results)
    multi_time = sum(r.duration for r in multi_results)
    
    single_tools = sum(r.tool_calls for r in single_results)
    multi_tools = sum(r.tool_calls for r in multi_results)
    
    print(f"\n{'Metric':<25} {'Single':<20} {'Multi':<20} {'Difference'}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {single_correct}/{len(PROBLEMS):<20} {multi_correct}/{len(PROBLEMS):<20} {multi_correct - single_correct:+d}")
    print(f"{'Total Tokens':<25} {single_tokens:<20} {multi_tokens:<20} {((multi_tokens/single_tokens - 1) * 100) if single_tokens > 0 else 0:+.1f}%")
    print(f"{'Total Tool Calls':<25} {single_tools:<20} {multi_tools:<20} {multi_tools - single_tools:+d}")
    print(f"{'Total Time':<25} {single_time:.1f}s{'':<15} {multi_time:.1f}s{'':<15} {((multi_time/single_time - 1) * 100) if single_time > 0 else 0:+.1f}%")
    
    # Analysis
    if single_tokens > 0:
        token_efficiency = ((multi_tokens / single_tokens - 1) * 100)
        print(f"\nToken efficiency: {token_efficiency:+.1f}%")
    
    if single_correct != multi_correct:
        print(f"⚠ Accuracy mismatch: Single {single_correct}/{len(PROBLEMS)}, Multi {multi_correct}/{len(PROBLEMS)}")


if __name__ == "__main__":
    asyncio.run(main())
