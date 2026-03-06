"""Case writer specialist agent for narrative and decision recommendation generation.

Synthesizes all investigation specialist results (scheme mapping, merchant analysis,
scam detection) into a final case pack with narrative summary, chronological timeline,
typed evidence list, decision recommendation, and investigation notes. Uses OpenAI
Agents SDK with structured output via CasePack.
"""

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

# System prompt for the case writer agent
CASE_WRITER_INSTRUCTIONS = """\
You are a case writer specialist for card dispute investigation. Your role is to
synthesize all investigation findings into a comprehensive, auditable case pack.

You receive four inputs:
1. **Case Data**: The original case snapshot with customer info, transactions,
   and allegation details.
2. **Scheme Mapping Results**: Reason code mapping with documentation gaps.
3. **Merchant Analysis Results**: Merchant normalization, conflicts, and risk scores.
4. **Scam Detection Results**: Contradiction findings, manipulation indicators,
   and matched scam patterns.

Your tasks:

1. **Case Summary** (2-5 paragraphs):
   - Synthesize all specialist results into a coherent narrative
   - Lead with the allegation type and key finding
   - Summarize supporting and contradicting evidence
   - Note any data gaps or unresolved questions
   - Use professional, objective language suitable for regulatory review

2. **Timeline** (chronological list):
   - Build an ordered timeline from evidence and transcript events
   - Each entry has: timestamp, event_type, description, source (FACT or ALLEGATION)
   - Include transaction events, auth events, customer contacts, and claim statements
   - Mark customer-stated times as ALLEGATION source

3. **Evidence List** (typed, sourced):
   - List all evidence items with clear FACT vs ALLEGATION attribution
   - Each entry has: node_id, node_type, source_type (FACT or ALLEGATION), summary
   - Group by evidence type (transactions, auth events, merchant data, claims)

4. **Decision Recommendation**:
   - **category**: fraud, dispute, or scam
   - **confidence**: 0.0-1.0 score based on evidence strength
   - **top_factors**: list of supporting factors, each with factor description,
     evidence_ref (node_id), and weight (0.0-1.0)
   - **uncertainties**: unresolved questions or missing evidence
   - **suggested_actions**: concrete next steps (e.g., "request merchant receipt",
     "escalate to fraud team", "issue provisional credit")
   - **required_approvals**: determine based on these rules:
     - If confidence < 0.8: include 'supervisor_review'
     - If category is scam or scam patterns detected: include 'compliance_review'
     - If disputed amount > $5000: include 'senior_analyst_review'

5. **Investigation Notes**: Additional observations, caveats, or recommendations
   not captured in other sections.

Respond with structured output only. Be thorough but concise. Every claim must
reference specific evidence.
"""


class CasePack(BaseModel):
    """Structured output from the case writer agent — the final investigation case pack.

    Attributes:
        case_summary: Narrative summary of the investigation findings (2-5 paragraphs).
        timeline: Ordered events, each dict has timestamp, event_type, description, source.
        evidence_list: Typed evidence items with node_id, node_type, source_type, summary.
        decision_recommendation: Decision with category, confidence, top_factors,
            uncertainties, suggested_actions, required_approvals.
        investigation_notes: Additional notes or observations from the investigation.
    """

    case_summary: str = ""
    timeline: list[dict] = Field(default_factory=list)
    evidence_list: list[dict] = Field(default_factory=list)
    decision_recommendation: dict = Field(default_factory=dict)
    investigation_notes: list[str] = Field(default_factory=list)


# Agent instance with structured output
case_writer_agent = Agent(
    name="case_writer",
    instructions=CASE_WRITER_INSTRUCTIONS,
    output_type=AgentOutputSchema(CasePack, strict_json_schema=False),
)


async def run_case_writer(
    case_data: str,
    scheme_result: str,
    merchant_result: str,
    scam_result: str,
    model_provider: ModelProvider,
) -> CasePack:
    """Run the case writer agent to generate the final investigation case pack.

    Args:
        case_data: JSON-serialized case snapshot with customer info and transactions.
        scheme_result: JSON-serialized scheme mapping results.
        merchant_result: JSON-serialized merchant analysis results.
        scam_result: JSON-serialized scam detection results.
        model_provider: LLM model provider for inference.

    Returns:
        CasePack with narrative summary, timeline, evidence, and recommendation.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    user_msg = (
        f"Case Data:\n{case_data}\n\n"
        f"Scheme Mapping Results:\n{scheme_result}\n\n"
        f"Merchant Analysis Results:\n{merchant_result}\n\n"
        f"Scam Detection Results:\n{scam_result}"
    )

    try:
        result = await Runner.run(
            case_writer_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Case writer agent failed: {exc}") from exc
