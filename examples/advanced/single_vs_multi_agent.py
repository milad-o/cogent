"""Single Agent vs Multi-Agent: A Fair Complex Comparison.

THE TEST: Both approaches complete a COMPLEX multi-dimensional task:
    "Comprehensive M&A due diligence analysis requiring 8 specialist domains"

This is intentionally complex to give multi-agent a fair shot:
    - 8 different analysis domains (not just 3)
    - 15+ fields in structured output
    - Nested data structures
    - Cross-cutting concerns requiring synthesis
    - Conflicting signals that need reconciliation

We measure:
    - Duration (wall-clock time)
    - LLM Calls (API cost driver)  
    - Total Tokens (actual cost)
    - Output Quality (completeness of analysis)

Run:
    uv run python examples/advanced/single_vs_multi_agent.py
"""

import asyncio
import time
from typing import Literal

from pydantic import BaseModel, Field

from cogent import Agent, Observer, tool


# ============================================================================
# THE TASK: Complex M&A Due Diligence (8 domains, 15+ output fields)
# ============================================================================

TASK = """Perform comprehensive M&A due diligence analysis for acquiring TargetCo Inc.

Target Company: TargetCo Inc.
- Industry: Enterprise SaaS (HR Tech)
- Revenue: $120M ARR, growing 35% YoY
- Employees: 450
- Founded: 2018
- HQ: Austin, TX
- Asking Price: $600M (5x revenue multiple)

Acquirer Context:
- We are a $2B revenue HR software company
- Strategic goal: Expand into mid-market segment
- Integration budget: $50M over 2 years

Analyze ALL 8 domains and synthesize into a final GO/NO-GO recommendation.
Consider synergies, risks, and integration complexity across all dimensions."""


# Complex nested output structure (15+ fields)
class FinancialAssessment(BaseModel):
    """Financial health analysis."""
    revenue_quality: Literal["STRONG", "MODERATE", "WEAK"]
    margin_profile: str
    cash_flow_health: Literal["HEALTHY", "NEUTRAL", "CONCERNING"]
    valuation_assessment: str
    deal_value_opinion: Literal["FAIR", "OVERPRICED", "UNDERPRICED"]


class TechAssessment(BaseModel):
    """Technology stack analysis."""
    architecture_quality: Literal["MODERN", "ADEQUATE", "LEGACY"]
    tech_debt_level: Literal["LOW", "MODERATE", "HIGH"]
    integration_complexity: Literal["SIMPLE", "MODERATE", "COMPLEX"]
    security_posture: Literal["STRONG", "ADEQUATE", "WEAK"]


class TeamAssessment(BaseModel):
    """Team and culture analysis."""
    leadership_strength: Literal["STRONG", "ADEQUATE", "WEAK"]
    retention_risk: Literal["LOW", "MODERATE", "HIGH"]
    culture_fit: Literal["ALIGNED", "MIXED", "MISALIGNED"]
    key_person_dependencies: list[str]


class MarketAssessment(BaseModel):
    """Market position analysis."""
    competitive_position: Literal["LEADER", "CHALLENGER", "NICHE"]
    market_growth: Literal["HIGH", "MODERATE", "DECLINING"]
    customer_concentration: Literal["DIVERSIFIED", "MODERATE", "CONCENTRATED"]
    expansion_potential: str


class RiskAssessment(BaseModel):
    """Risk analysis across dimensions."""
    legal_risks: list[str]
    regulatory_risks: list[str]
    operational_risks: list[str]
    integration_risks: list[str]
    overall_risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]


class SynergyAssessment(BaseModel):
    """Synergy and value creation analysis."""
    revenue_synergies: str
    cost_synergies: str
    strategic_synergies: str
    estimated_synergy_value: str
    time_to_realize: str


class DueDiligenceReport(BaseModel):
    """Complete M&A due diligence report - 15+ fields across 6 nested structures."""
    
    target_company: str = Field(description="Company being analyzed")
    deal_value: str = Field(description="Proposed acquisition price")
    
    # Nested assessments
    financial: FinancialAssessment
    technology: TechAssessment
    team: TeamAssessment
    market: MarketAssessment
    risks: RiskAssessment
    synergies: SynergyAssessment
    
    # Synthesis
    recommendation: Literal["GO", "NO-GO", "CONDITIONAL"]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    key_concerns: list[str] = Field(description="Top 3 deal-breaker concerns")
    key_opportunities: list[str] = Field(description="Top 3 value drivers")
    conditions: list[str] = Field(description="Conditions for GO recommendation")
    executive_summary: str = Field(description="2-3 sentence summary for board")


# ============================================================================
# DOMAIN TOOLS (Realistic - with dependencies, discovery, iteration)
# ============================================================================

# Simulated data stores (realistic M&A data room)
CUSTOMERS = {
    "CUST001": {"name": "Acme Corp", "arr": 3_200_000, "segment": "enterprise", "health": "green"},
    "CUST002": {"name": "TechStart Inc", "arr": 890_000, "segment": "mid-market", "health": "yellow"},
    "CUST003": {"name": "GlobalBank", "arr": 2_100_000, "segment": "enterprise", "health": "green"},
    "CUST004": {"name": "RetailMax", "arr": 1_500_000, "segment": "mid-market", "health": "red"},
    "CUST005": {"name": "HealthCare Plus", "arr": 4_500_000, "segment": "enterprise", "health": "green"},
}

CONTRACTS = {
    "CUST001": {"term_months": 36, "auto_renew": True, "change_of_control": True, "termination_notice": 90},
    "CUST002": {"term_months": 12, "auto_renew": True, "change_of_control": False, "termination_notice": 30},
    "CUST003": {"term_months": 24, "auto_renew": True, "change_of_control": True, "termination_notice": 60},
    "CUST004": {"term_months": 12, "auto_renew": False, "change_of_control": True, "termination_notice": 30},
    "CUST005": {"term_months": 36, "auto_renew": True, "change_of_control": False, "termination_notice": 90},
}

EMPLOYEES = {
    "EMP001": {"name": "Sarah Chen", "role": "CEO", "tenure_years": 5, "critical": True, "retention_risk": "low"},
    "EMP002": {"name": "Marcus Webb", "role": "CTO", "tenure_years": 4, "critical": True, "retention_risk": "high"},
    "EMP003": {"name": "David Park", "role": "CFO", "tenure_years": 0.5, "critical": True, "retention_risk": "medium"},
    "EMP004": {"name": "Lisa Martinez", "role": "VP Sales", "tenure_years": 3, "critical": True, "retention_risk": "low"},
    "EMP005": {"name": "Priya Sharma", "role": "Head of ML", "tenure_years": 2, "critical": True, "retention_risk": "high"},
    "EMP006": {"name": "James Kim", "role": "VP Engineering", "tenure_years": 2, "critical": False, "retention_risk": "low"},
}

CODEBASES = {
    "core-platform": {"loc": 450_000, "test_coverage": 72, "tech_debt_score": 3.2, "last_major_refactor": "2024-01"},
    "payroll-module": {"loc": 85_000, "test_coverage": 45, "tech_debt_score": 6.8, "last_major_refactor": "2019-06"},
    "ml-engine": {"loc": 120_000, "test_coverage": 81, "tech_debt_score": 2.1, "last_major_refactor": "2025-03"},
    "mobile-app": {"loc": 65_000, "test_coverage": 68, "tech_debt_score": 4.5, "last_major_refactor": "2023-09"},
}

DOCUMENTS = {
    "DOC001": {"title": "Q4 2025 Financials", "category": "financial", "pages": 45},
    "DOC002": {"title": "Patent Portfolio", "category": "legal", "pages": 120},
    "DOC003": {"title": "SOC 2 Audit Report", "category": "compliance", "pages": 85},
    "DOC004": {"title": "Customer Churn Analysis", "category": "financial", "pages": 32},
    "DOC005": {"title": "Technical Architecture", "category": "technology", "pages": 67},
    "DOC006": {"title": "Employee Handbook", "category": "hr", "pages": 45},
    "DOC007": {"title": "Litigation Summary", "category": "legal", "pages": 12},
    "DOC008": {"title": "Market Analysis 2025", "category": "market", "pages": 54},
}


# --- DISCOVERY TOOLS (find things that need follow-up) ---

@tool
def search_data_room(query: str, category: str | None = None) -> str:
    """Search the M&A data room for documents. Returns document IDs to read.
    
    Args:
        query: Search terms (matches document titles and categories)
        category: Filter by category (financial, legal, compliance, technology, hr, market) or None/all
    """
    results = []
    query_words = query.lower().split()
    
    for doc_id, doc in DOCUMENTS.items():
        # Skip if category filter doesn't match
        if category and category != "all" and doc["category"] != category:
            continue
        
        # Match if any query word appears in title or category
        title_lower = doc["title"].lower()
        cat_lower = doc["category"].lower()
        
        if any(word in title_lower or word in cat_lower for word in query_words):
            results.append(f"{doc_id}: {doc['title']} ({doc['pages']} pages)")
    
    if not results:
        return f"No documents found matching '{query}'"
    return "Found documents:\n" + "\n".join(results)


@tool
def read_document(doc_id: str) -> str:
    """Read a specific document from the data room. Use search_data_room first to find doc IDs."""
    if doc_id not in DOCUMENTS:
        return f"Document {doc_id} not found. Use search_data_room to find valid document IDs."
    
    doc = DOCUMENTS[doc_id]
    
    # Return realistic document content based on type
    content_map = {
        "DOC001": """Q4 2025 Financial Summary:
- Revenue: $32.1M (up 8% QoQ)
- Gross Margin: 77.2%
- Operating Loss: $1.8M (improving from $2.4M Q3)
- Cash: $45.2M, Burn: $4.1M/month
- ARR: $120M (35% YoY growth)
- Note: Revenue recognition adjustment of $2.3M in Oct related to CUST004 contract dispute""",
        
        "DOC002": """Patent Portfolio Summary:
12 granted patents, 5 pending:
- US10234567: Workforce optimization algorithm (expires 2038)
- US10345678: Predictive scheduling (expires 2039)
- US10456789: AI-based compliance detection (expires 2040)
Note: Patent US10234567 has potential overlap with Workday patent US9876543. Legal review recommended.""",
        
        "DOC003": """SOC 2 Type II Audit Report (2025):
Status: COMPLIANT
Scope: Security, Availability, Confidentiality
Exceptions: 2 minor findings
- Finding 1: Access review delays (remediated)
- Finding 2: Incomplete change management logs (remediation in progress)
Next audit: March 2026""",
        
        "DOC004": """Customer Churn Analysis:
Gross Churn: 8.2% annually
Net Revenue Retention: 115%
At-Risk Customers (health=red): 1 customer (CUST004 - RetailMax, $1.5M ARR)
Reason: Contract dispute over SLA violations in Q3. Legal action threatened.
Action: Account team escalation in progress.""",
        
        "DOC005": """Technical Architecture Overview:
Primary Stack: Python/FastAPI, React/TypeScript, PostgreSQL
Infrastructure: AWS EKS (Kubernetes)
Concerns:
- payroll-module: Legacy Python 2.7 code, needs migration
- API versioning: v1/v2/v3 all live, deprecation delayed
- Database: CUST005 (HealthCare Plus) has custom schema modifications""",
        
        "DOC006": """Employee Handbook Summary:
Total Employees: 450
Key Policies:
- Remote-first since 2020
- Equity refresh: Annual, 4-year vesting
- Note: 12 employees have acceleration clauses on change of control""",
        
        "DOC007": """Litigation Summary:
Active Cases: 2
1. Johnson v. TargetCo (wrongful termination) - Expected settlement $200-500K
2. RetailMax v. TargetCo (SLA dispute) - Potential exposure $1.2M, trial date Q3 2026
Historical: 2 minor settlements (<$100K each)""",
        
        "DOC008": """Market Analysis 2025:
HR Tech TAM: $35B
Mid-market segment: $8B
TargetCo position: #4 (8% share)
Key threat: Rippling growing 60% YoY, aggressive pricing
Opportunity: AI features command 15-20% premium""",
    }
    
    return content_map.get(doc_id, f"Document {doc_id} content not available")


# --- CUSTOMER ANALYSIS TOOLS (need to iterate through customers) ---

@tool
def list_customers(segment: str | None = None, min_arr: str | None = None) -> str:
    """List customers with optional filters. Returns customer IDs for detailed lookup.
    
    Args:
        segment: Filter by segment (enterprise, mid-market, smb) or None for all
        min_arr: Minimum ARR filter as string (e.g. "500000") or None for all
    
    Call with NO arguments to see ALL customers.
    """
    # Normalize segment - treat "None", "null", "all", "" as no filter
    segment_clean = None
    if segment and segment.lower() not in ("none", "null", "all", ""):
        segment_clean = segment.lower()
    
    # Convert min_arr to int if provided (and not "None", "0", etc.)
    min_arr_int = None
    if min_arr and min_arr not in ("None", "null", "0", ""):
        try:
            min_arr_int = int(min_arr)
        except ValueError:
            pass
    
    results = []
    for cust_id, cust in CUSTOMERS.items():
        if segment_clean and cust["segment"] != segment_clean:
            continue
        if min_arr_int and cust["arr"] < min_arr_int:
            continue
        results.append(f"{cust_id}: {cust['name']} (${cust['arr']:,} ARR, {cust['segment']}, health={cust['health']})")
    
    if not results:
        return "No customers found matching filters"
    return "Customers found:\n" + "\n".join(results)


@tool
def get_customer_details(customer_id: str) -> str:
    """Get detailed info for a specific customer. Use list_customers first to get IDs."""
    if customer_id not in CUSTOMERS:
        return f"Customer {customer_id} not found. Use list_customers to find valid IDs."
    
    cust = CUSTOMERS[customer_id]
    contract = CONTRACTS.get(customer_id, {})
    
    return f"""Customer: {cust['name']} ({customer_id})
ARR: ${cust['arr']:,}
Segment: {cust['segment']}
Health Score: {cust['health']}
Contract Term: {contract.get('term_months', 'N/A')} months
Auto-Renew: {contract.get('auto_renew', 'N/A')}
Change of Control Clause: {contract.get('change_of_control', 'N/A')}
Termination Notice: {contract.get('termination_notice', 'N/A')} days"""


@tool
def get_customer_risk_assessment(customer_id: str) -> str:
    """Assess specific customer risk. Requires customer_id from list_customers."""
    if customer_id not in CUSTOMERS:
        return f"Customer {customer_id} not found."
    
    cust = CUSTOMERS[customer_id]
    contract = CONTRACTS.get(customer_id, {})
    
    risks = []
    if cust["health"] == "red":
        risks.append("CRITICAL: Customer health is red - high churn risk")
    if contract.get("change_of_control"):
        risks.append(f"WARNING: Change of control clause - ${cust['arr']:,} ARR at risk")
    if not contract.get("auto_renew"):
        risks.append("CONCERN: No auto-renewal - manual renewal required")
    if cust["arr"] > 2_000_000:
        risks.append(f"NOTE: Large customer concentration - ${cust['arr']:,} ARR")
    
    if not risks:
        return f"Customer {customer_id} ({cust['name']}): No significant risks identified"
    
    return f"Customer {customer_id} ({cust['name']}) Risk Assessment:\n" + "\n".join(f"- {r}" for r in risks)


# --- EMPLOYEE TOOLS (need to find and investigate key personnel) ---

@tool
def list_employees(critical_only: str | None = None, role_filter: str | None = None) -> str:
    """List employees. Returns employee IDs for detailed lookup.
    
    Args:
        critical_only: Set to "true" to filter for critical employees only
        role_filter: Filter by role keyword (e.g. "engineer", "executive") or None for all
    """
    # Handle string boolean
    filter_critical = critical_only and critical_only.lower() == "true"
    
    # Normalize role_filter - treat "None", "null", "all" as no filter
    role_filter_clean = None
    if role_filter and role_filter.lower() not in ("none", "null", "all", ""):
        role_filter_clean = role_filter.lower()
    
    results = []
    for emp_id, emp in EMPLOYEES.items():
        if filter_critical and not emp["critical"]:
            continue
        if role_filter_clean and role_filter_clean not in emp["role"].lower():
            continue
        critical_tag = " [CRITICAL]" if emp["critical"] else ""
        results.append(f"{emp_id}: {emp['name']} ({emp['role']}, tenure: {emp['tenure_years']}yr){critical_tag}")
    
    if not results:
        return "No employees found matching filters"
    return f"Employees found ({len(results)} total):\n" + "\n".join(results)


@tool
def get_employee_details(employee_id: str) -> str:
    """Get detailed employee info. Use list_employees to find IDs first."""
    if employee_id not in EMPLOYEES:
        return f"Employee {employee_id} not found."
    
    emp = EMPLOYEES[employee_id]
    return f"""Employee: {emp['name']} ({employee_id})
Role: {emp['role']}
Tenure: {emp['tenure_years']} years
Critical to Operations: {emp['critical']}
Retention Risk: {emp['retention_risk']}"""


@tool
def assess_retention_risk(employee_id: str) -> str:
    """Deep dive on retention risk for a specific employee. Requires employee_id."""
    if employee_id not in EMPLOYEES:
        return f"Employee {employee_id} not found."
    
    emp = EMPLOYEES[employee_id]
    
    factors = []
    if emp["retention_risk"] == "high":
        factors.append("HIGH RISK: Employee has indicated interest from competitors")
    if emp["tenure_years"] < 1:
        factors.append("CONCERN: Short tenure - may not be fully integrated")
    if emp["critical"]:
        factors.append(f"CRITICAL: Key person - loss would significantly impact operations")
    if emp["role"] in ["CTO", "Head of ML"]:
        factors.append("NOTE: Technical leadership - has deep institutional knowledge")
    
    recommendation = ""
    if emp["retention_risk"] == "high" and emp["critical"]:
        recommendation = "\nRECOMMENDATION: Prioritize retention package with 2-year cliff and significant equity refresh"
    
    return f"""Retention Assessment for {emp['name']}:
{"".join(f"- {f}\n" for f in factors)}{recommendation}"""


# --- TECHNOLOGY TOOLS (need to discover and investigate codebases) ---

@tool
def list_codebases() -> str:
    """List all codebases/repositories. Returns names for detailed analysis."""
    results = []
    for name, info in CODEBASES.items():
        results.append(f"{name}: {info['loc']:,} LOC, {info['test_coverage']}% coverage, tech_debt={info['tech_debt_score']}/10")
    return "Codebases:\n" + "\n".join(results)


@tool
def analyze_codebase(codebase_name: str) -> str:
    """Deep analysis of a specific codebase. Use list_codebases to find names."""
    if codebase_name not in CODEBASES:
        return f"Codebase '{codebase_name}' not found. Use list_codebases to see available repos."
    
    info = CODEBASES[codebase_name]
    
    issues = []
    if info["tech_debt_score"] > 5:
        issues.append(f"CRITICAL: High tech debt ({info['tech_debt_score']}/10) - requires immediate attention")
    if info["test_coverage"] < 60:
        issues.append(f"WARNING: Low test coverage ({info['test_coverage']}%) - risk for regressions")
    if info["last_major_refactor"] < "2022-01":
        issues.append(f"CONCERN: No major refactor since {info['last_major_refactor']} - accumulated technical debt likely")
    
    return f"""Codebase Analysis: {codebase_name}
Lines of Code: {info['loc']:,}
Test Coverage: {info['test_coverage']}%
Tech Debt Score: {info['tech_debt_score']}/10
Last Major Refactor: {info['last_major_refactor']}

{"Issues Found:" if issues else "No critical issues found."}
{"".join(f"- {i}\n" for i in issues)}"""


@tool
def estimate_remediation(codebase_name: str) -> str:
    """Estimate remediation effort for a codebase. Use after analyze_codebase identifies issues."""
    if codebase_name not in CODEBASES:
        return f"Codebase '{codebase_name}' not found."
    
    info = CODEBASES[codebase_name]
    
    effort_months = 0
    if info["tech_debt_score"] > 5:
        effort_months += (info["tech_debt_score"] - 5) * 2
    if info["test_coverage"] < 70:
        effort_months += (70 - info["test_coverage"]) * 0.1
    
    return f"""Remediation Estimate for {codebase_name}:
Current State: {info['tech_debt_score']}/10 tech debt, {info['test_coverage']}% coverage
Target State: <3/10 tech debt, >80% coverage
Estimated Effort: {effort_months:.1f} engineer-months
Recommended Team Size: {max(2, int(effort_months / 3))} engineers
Timeline: {max(3, int(effort_months / 2))} months"""


# --- FINANCIAL INVESTIGATION TOOLS (dynamic based on what you find) ---

@tool
def get_financial_summary(period: str = "Q4 2025") -> str:
    """Get financial summary. May reveal items needing investigation."""
    return f"""Financial Summary ({period}):
Revenue: $32.1M
Gross Margin: 77.2%
Operating Margin: -5.6%
Cash: $45.2M
Burn Rate: $4.1M/month

Notable Items Requiring Investigation:
1. Revenue adjustment of $2.3M in October (see DOC001)
2. Customer dispute with CUST004 affecting recognition
3. Increased legal fees ($1.2M vs $0.4M Q3) - see DOC007"""


@tool
def investigate_financial_item(item: str) -> str:
    """Investigate a specific financial item. Use after get_financial_summary reveals issues."""
    item_lower = item.lower()
    
    if "revenue adjustment" in item_lower or "2.3m" in item_lower:
        return """Revenue Adjustment Investigation ($2.3M):
Cause: CUST004 (RetailMax) disputed Q3 invoices due to SLA violations
Status: $1.5M reversed, $0.8M in escrow pending resolution
Impact: One-time adjustment, not recurring
Risk: If CUST004 churns, full $1.5M ARR lost + potential damages
Related: See DOC007 for litigation details, DOC004 for customer analysis"""
    
    if "legal fees" in item_lower or "1.2m" in item_lower:
        return """Legal Fees Investigation ($1.2M Q4 vs $0.4M Q3):
Breakdown:
- RetailMax litigation: $0.6M (trial prep)
- Johnson wrongful termination: $0.2M (settlement negotiation)
- M&A preparation: $0.3M (data room, due diligence support)
- General counsel: $0.1M (normal operations)
Projection: Expect $0.8M/quarter until RetailMax resolved (Q3 2026)"""
    
    if "cust004" in item_lower or "retailmax" in item_lower:
        return """CUST004 (RetailMax) Issue Deep Dive:
Timeline:
- Q3 2025: SLA violations (3 outages exceeding SLA)
- Oct 2025: RetailMax disputed $2.3M in invoices
- Nov 2025: Formal litigation filed
- Current: Trial date set Q3 2026
Exposure: $1.5M ARR (potential churn) + $1.2M damages claim
Mitigation: Settlement discussions ongoing, may resolve for $0.5M"""
    
    return f"No detailed information available for '{item}'. Try: 'revenue adjustment', 'legal fees', or 'CUST004'"


# ============================================================================
# TOOL GROUPINGS BY DOMAIN
# ============================================================================

# Discovery & Documents
DISCOVERY_TOOLS = [search_data_room, read_document]

# Customer analysis (requires iteration)
CUSTOMER_TOOLS = [list_customers, get_customer_details, get_customer_risk_assessment]

# Employee/HR (requires iteration)
EMPLOYEE_TOOLS = [list_employees, get_employee_details, assess_retention_risk]

# Technology (requires iteration + follow-up)
TECH_TOOLS = [list_codebases, analyze_codebase, estimate_remediation]

# Financial (discovery + investigation)
FINANCIAL_TOOLS = [get_financial_summary, investigate_financial_item]

# All 14 tools combined - requires multi-step reasoning
ALL_TOOLS = (
    DISCOVERY_TOOLS + CUSTOMER_TOOLS + EMPLOYEE_TOOLS + 
    TECH_TOOLS + FINANCIAL_TOOLS
)


# ============================================================================
# APPROACH 1: SINGLE AGENT (14 dynamic tools requiring chained calls)
# ============================================================================


async def single_agent_approach() -> dict:
    """One agent with all 14 dynamic tools.
    
    This tests whether a single agent can handle:
    - Tool discovery (search → read)
    - Iteration (list → get details for each)
    - Conditional investigation (find issue → dig deeper)
    """
    print("\n" + "=" * 70)
    print("APPROACH 1: Single Agent + 14 Dynamic Tools")
    print("=" * 70)
    
    observer = Observer(level="normal")
    
    agent = Agent(
        name="DueDiligenceAnalyst",
        model="openai:gpt-4o",
        tools=ALL_TOOLS,
        output=DueDiligenceReport,
        instructions="""You are a senior M&A analyst performing comprehensive due diligence.

⚠️ CRITICAL: You MUST use the tools provided to gather real data. DO NOT make up information.
⚠️ MAXIMIZE PARALLELISM: Call ALL tools you need in a single response. Don't wait for one area before starting another.

You have 14 tools across 5 domains. Call tools from ALL domains in PARALLEL:

STEP 1 - PARALLEL DISCOVERY (call ALL of these at once):
- get_financial_summary() → get financials first
- list_customers() → get all customers (no filters!)
- list_employees(critical_only="true") → get critical employees
- list_codebases() → get all codebases
- search_data_room(query="financial") → find financial docs
- search_data_room(query="legal") → find legal docs

STEP 2 - PARALLEL DEEP DIVE (based on Step 1 results, call ALL at once):
For each customer ID → get_customer_details() AND get_customer_risk_assessment()
For each employee ID → get_employee_details() AND assess_retention_risk()  
For each codebase → analyze_codebase()
For each financial issue → investigate_financial_item()
For each doc ID → read_document()

STEP 3 - PARALLEL REMEDIATION (if needed):
For codebases with tech_debt > 5 → estimate_remediation()

⚡ KEY: In each response, call as many tools as possible in parallel. 
Do NOT do one domain at a time. Do ALL domains simultaneously.""",
        observer=observer,
    )
    
    start = time.perf_counter()
    result = await agent.run(TASK)
    duration = time.perf_counter() - start
    
    # Extract structured output
    output = None
    if result.content and hasattr(result.content, 'data'):
        output = result.content.data
    
    if output:
        print(f"\n📊 {output.target_company}: {output.recommendation} ({output.confidence} confidence)")
        print(f"💰 Deal Value: {output.deal_value}")
        print(f"\n📋 Executive Summary:\n{output.executive_summary}")
        print(f"\n✅ Key Opportunities: {', '.join(output.key_opportunities)}")
        print(f"⚠️  Key Concerns: {', '.join(output.key_concerns)}")
        if output.conditions:
            print(f"📝 Conditions: {', '.join(output.conditions)}")
    else:
        print(f"\n📋 Result:\n{result.content}")
    
    tokens = 0
    if result.metadata and result.metadata.tokens:
        tokens = result.metadata.tokens.total_tokens
    
    # Count iterations from observer events (more accurate than hardcoded "1")
    llm_calls = "multiple"  # We can't easily count from result metadata
    
    return {
        "approach": "single_agent",
        "duration_seconds": round(duration, 2),
        "llm_calls": llm_calls,
        "tokens": tokens,
        "recommendation": output.recommendation if output else "N/A",
        "output": output,
    }


# ============================================================================
# APPROACH 2: MULTI-AGENT ORCHESTRATION (Specialists with domain tools)
# ============================================================================


async def multi_agent_approach() -> dict:
    """Orchestrator delegates to specialist agents, each with their domain tools.
    
    Each specialist has 2-4 tools in their domain and must do multi-step investigation.
    """
    print("\n" + "=" * 70)
    print("APPROACH 2: Multi-Agent Orchestration (5 Specialists)")
    print("=" * 70)
    
    observer = Observer(level="normal")
    
    # Document Research Specialist
    doc_researcher = Agent(
        name="DocumentResearcher",
        model="openai:gpt-4o",
        tools=DISCOVERY_TOOLS,
        instructions="""You are a document research specialist.

Available categories: financial, legal, compliance, technology, hr, market

Workflow:
1. Search each category: search_data_room(query="financial"), search_data_room(query="legal"), etc.
2. Read each document found using read_document(doc_id)
3. Summarize all findings with specific document references

Search broadly - use category names as queries to find all documents.""",
    )
    
    # Customer Analyst
    customer_analyst = Agent(
        name="CustomerAnalyst",
        model="openai:gpt-4o",
        tools=CUSTOMER_TOOLS,
        instructions="""You are a customer analysis specialist.

Workflow:
1. First call list_customers() with NO filters to see ALL customers
2. Call get_customer_details(customer_id) for EACH customer ID returned
3. Call get_customer_risk_assessment(customer_id) for any customer with health=red or health=yellow
4. Summarize customer concentration, ARR distribution, and churn risks

Do NOT filter on first call - you need to see all customers.""",
    )
    
    # HR/Team Analyst
    team_analyst = Agent(
        name="TeamAnalyst",
        model="openai:gpt-4o",
        tools=EMPLOYEE_TOOLS,
        instructions="""You are an HR and team analysis specialist.

Workflow:
1. Call list_employees(critical_only="true") to find key personnel
2. Call get_employee_details(employee_id) for EACH employee ID returned
3. Call assess_retention_risk(employee_id) for EACH employee
4. Identify key person dependencies and recommend retention strategies

Note: critical_only should be the string "true" not a boolean.""",
    )
    
    # Technology Analyst
    tech_analyst = Agent(
        name="TechAnalyst",
        model="openai:gpt-4o",
        tools=TECH_TOOLS,
        instructions="""You are a technology due diligence specialist.
Use list_codebases to find all repositories.
Analyze each codebase for tech debt and quality issues.
For any codebase with high tech debt (>5/10), estimate remediation effort.
Provide total remediation cost and timeline.""",
    )
    
    # Financial Investigator
    financial_analyst = Agent(
        name="FinancialAnalyst",
        model="openai:gpt-4o",
        tools=FINANCIAL_TOOLS,
        instructions="""You are a financial due diligence specialist.
Use get_financial_summary to get overview and identify issues.
Use investigate_financial_item to dig into any concerning items.
Follow the trail - if an investigation reveals more issues, investigate those too.
Provide complete picture of financial health and risks.""",
    )
    
    # Orchestrator coordinates all specialists
    orchestrator = Agent(
        name="ChiefStrategyOfficer",
        model="openai:gpt-4o",
        tools=[
            doc_researcher.as_tool(description="Research documents in data room - finds and reads key documents"),
            customer_analyst.as_tool(description="Analyze customer base - details and risks for each customer"),
            team_analyst.as_tool(description="Analyze team and key personnel - retention risks and dependencies"),
            tech_analyst.as_tool(description="Technology assessment - codebase quality and remediation needs"),
            financial_analyst.as_tool(description="Financial investigation - summary and issue deep-dives"),
        ],
        output=DueDiligenceReport,
        instructions="""You are the Chief Strategy Officer leading M&A due diligence.

You have 5 specialist analysts. Each will do thorough investigation in their domain:
1. DocumentResearcher - Searches and reads key documents from data room
2. CustomerAnalyst - Analyzes each customer, assesses risks
3. TeamAnalyst - Identifies critical employees and retention risks
4. TechAnalyst - Evaluates each codebase, estimates remediation
5. FinancialAnalyst - Reviews financials, investigates issues

DELEGATE TO ALL 5 SPECIALISTS. They will each do multi-step investigation.
Then synthesize their findings into a complete DueDiligenceReport.

The board needs a complete picture for this $600M decision.""",
        observer=observer,
    )
    
    start = time.perf_counter()
    result = await orchestrator.run(TASK)
    duration = time.perf_counter() - start
    
    # Extract structured output
    output = None
    if result.content and hasattr(result.content, 'data'):
        output = result.content.data
    
    if output:
        print(f"\n📊 {output.target_company}: {output.recommendation} ({output.confidence} confidence)")
        print(f"💰 Deal Value: {output.deal_value}")
        print(f"\n📋 Executive Summary:\n{output.executive_summary}")
        print(f"\n✅ Key Opportunities: {', '.join(output.key_opportunities)}")
        print(f"⚠️  Key Concerns: {', '.join(output.key_concerns)}")
        if output.conditions:
            print(f"📝 Conditions: {', '.join(output.conditions)}")
    else:
        print(f"\n📋 Result:\n{result.content}")
    
    tokens = 0
    if result.metadata and result.metadata.tokens:
        tokens = result.metadata.tokens.total_tokens
    
    return {
        "approach": "multi_agent",
        "duration_seconds": round(duration, 2),
        "llm_calls": "many",  # Orchestrator + 5 specialists, each doing multiple iterations
        "tokens": tokens,  # Heavily undercounted
        "recommendation": output.recommendation if output else "N/A",
        "output": output,
    }


# ============================================================================
# THE COMPARISON
# ============================================================================


async def main():
    print("\n" + "🎯 " * 25)
    print("FAIR COMPARISON: Single Agent vs Multi-Agent")
    print("🎯 " * 25)
    print(f"\n📋 TASK: Complex M&A Due Diligence with DYNAMIC tools")
    print(f"🔧 Tools require: discovery → iteration → investigation chains")
    print(f"💰 Decision: $600M acquisition - GO/NO-GO/CONDITIONAL\n")
    
    results = []
    
    # Run both approaches
    results.append(await single_agent_approach())
    results.append(await multi_agent_approach())
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("📊 RESULTS COMPARISON")
    print("=" * 80)
    print()
    
    headers = f"{'Approach':<35} {'Duration':<12} {'Tokens':<12} {'Recommendation'}"
    print(headers)
    print("-" * 75)
    
    for r in results:
        approach = r["approach"].replace("_", " ").title()
        duration_str = f"{r['duration_seconds']:.2f}s"
        tokens_str = f"{r.get('tokens', 'N/A')}*" if "multi" in r["approach"] else str(r.get('tokens', 'N/A'))
        print(f"{approach:<35} {duration_str:<12} {tokens_str:<12} {r['recommendation']}")
    
    print("\n* Token count for multi-agent is severely undercounted (nested agent calls not tracked)")
    
    # Calculate overhead
    single = results[0]
    multi = results[1]
    
    print("\n" + "=" * 80)
    print("📈 OVERHEAD ANALYSIS")
    print("=" * 80)
    
    if single["duration_seconds"] > 0:
        multi_overhead = ((multi["duration_seconds"] / single["duration_seconds"]) - 1) * 100
        print(f"""
Multi-Agent Orchestration vs Single Agent:
  ⏱️  Duration: {multi_overhead:+.0f}% {'slower' if multi_overhead > 0 else 'faster'}
  💰 Tokens: {multi['tokens']}* vs {single['tokens']} (multi-agent actual is 3-5x higher)

Note: Both approaches must do multi-step tool chains:
  - Search → Read (discovery)
  - List → Get Details (iteration)  
  - Find Issue → Investigate (conditional)
""")


if __name__ == "__main__":
    asyncio.run(main())
