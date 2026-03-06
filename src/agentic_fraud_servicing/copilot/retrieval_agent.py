"""Fast data retrieval agent via Tool Gateway.

Fetches transactions, auth logs, and customer profiles for the current case
using the three @function_tool wrappers from copilot/context.py. Returns a
structured RetrievalResult summarizing what was found and any data gaps.
"""

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

from agentic_fraud_servicing.copilot.context import (
    CopilotContext,
    tool_fetch_customer_profile,
    tool_lookup_transactions,
    tool_query_auth_logs,
)
from agentic_fraud_servicing.gateway.tool_gateway import ToolGateway

# System prompt for the retrieval agent
RETRIEVAL_INSTRUCTIONS = """\
You are a fast data retrieval specialist for AMEX card dispute servicing.
Your role is to gather all relevant data for a case using the available tools.

**Available tools**:
1. **tool_lookup_transactions** — Fetches all TRANSACTION-type evidence nodes
   for the current case, with PAN fields masked. Use this to retrieve the
   disputed and surrounding transactions.
2. **tool_query_auth_logs** — Fetches all AUTH_EVENT-type evidence nodes.
   Use this to check authentication methods, failed attempts, device changes,
   and login history around the time of disputed transactions.
3. **tool_fetch_customer_profile** — Fetches the CUSTOMER-type evidence node.
   Use this to get the customer's profile, recent account changes, and risk
   indicators.

**Instructions**:
- Call ALL three tools to gather comprehensive data for the case.
- Summarize what was retrieved in plain language.
- Identify data gaps — for example, if no auth events exist for a disputed
  transaction period, or if the customer profile is missing.
- Do NOT fabricate data. Only report what the tools return.

Respond with structured output only.
"""


class RetrievalResult(BaseModel):
    """Structured output from the retrieval agent.

    Attributes:
        transactions: Retrieved transaction records (PAN-masked).
        auth_events: Retrieved authentication event records.
        customer_profile: Customer profile if found, None otherwise.
        retrieval_summary: Brief summary of what was retrieved.
        data_gaps: Notable data gaps identified during retrieval.
    """

    transactions: list[dict] = Field(default_factory=list)
    auth_events: list[dict] = Field(default_factory=list)
    customer_profile: dict | None = None
    retrieval_summary: str = ""
    data_gaps: list[str] = Field(default_factory=list)


# Agent instance with tools and structured output
retrieval_agent = Agent(
    name="fast_retrieval",
    instructions=RETRIEVAL_INSTRUCTIONS,
    tools=[tool_lookup_transactions, tool_query_auth_logs, tool_fetch_customer_profile],
    output_type=AgentOutputSchema(RetrievalResult, strict_json_schema=False),
)


async def run_retrieval(
    case_id: str,
    call_id: str,
    gateway: ToolGateway,
    model_provider: ModelProvider,
) -> RetrievalResult:
    """Run the retrieval agent to fetch all available data for a case.

    Args:
        case_id: The case identifier to retrieve data for.
        call_id: The current call identifier.
        gateway: ToolGateway instance for mediated data access.
        model_provider: LLM model provider for inference.

    Returns:
        RetrievalResult with fetched data and gap analysis.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    copilot_ctx = CopilotContext(
        case_id=case_id,
        call_id=call_id,
        gateway=gateway,
    )

    try:
        result = await Runner.run(
            retrieval_agent,
            input="Retrieve all available data for the current case.",
            context=copilot_ctx,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Retrieval agent failed: {exc}") from exc
