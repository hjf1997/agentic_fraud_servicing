"""Scheme mapper specialist agent for reason code mapping and documentation gap analysis.

Maps fraud/dispute/scam allegations to card network reason codes (AMEX, Visa,
Mastercard) and identifies documentation gaps needed to support the claim under
the matched reason code. Uses OpenAI Agents SDK with structured output via
SchemeMappingResult.
"""

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

# System prompt for the scheme mapper agent
SCHEME_MAPPER_INSTRUCTIONS = """\
You are a scheme mapping specialist for card dispute servicing. Your role is to
map fraud/dispute/scam allegations to the correct card network reason codes and
identify documentation gaps.

Your tasks:
1. **Map to reason codes**: Based on the allegation type and specific claims,
   identify all applicable card network reason codes. Cover these networks:

   **AMEX Reason Codes**:
   - C08: Goods/Services Not Received — cardmember paid but goods/services
     were not delivered.
   - C18: "No Show" or CARDeposit Cancelled — reservation cancelled but charged.
   - C28: Cancelled Recurring Billing — recurring charge after cancellation.
   - C32: Goods/Services Damaged or Defective — goods received in poor condition.
   - FR2: Fraud Full Recourse — unauthorized transaction, full liability on merchant.
   - FR4: Immediate Chargeback — fraud with no prior authorization attempt.
   - FR6: Partial Immediate Chargeback — partial fraud recovery.

   **Visa Reason Codes**:
   - 10.4: Other Fraud — Card Absent Environment — fraud in card-not-present.
   - 10.5: Visa Fraud Monitoring Program — identified via fraud monitoring.
   - 13.1: Merchandise/Services Not Received — paid but not delivered.
   - 13.2: Cancelled Recurring Transaction — recurring after cancellation.
   - 13.3: Not as Described or Defective — goods differ from description.

   **Mastercard Reason Codes**:
   - 4837: No Cardholder Authorization — unauthorized transaction.
   - 4853: Cardholder Dispute — goods/services not as described or not received.
   - 4863: Cardholder Does Not Recognize — unrecognized transaction.
   - 4871: Chip/PIN Liability Shift — chip transaction fraud liability.

2. **Rank by confidence**: Assign a match_confidence (0.0-1.0) to each reason
   code based on how well it fits the specific claims and evidence.

3. **Identify the primary reason code**: Select the best-matching reason code
   and its network as the primary match.

4. **Identify documentation gaps**: For the primary reason code, list the
   specific documents or evidence still needed to support the claim:
   - Fraud: police report, signed affidavit, device/IP logs, card status
   - Dispute (not received): tracking info, delivery confirmation, merchant
     correspondence
   - Dispute (defective): photos of damage, return receipt, repair estimates
   - Scam: communication records, transaction authorization proof, timeline
     of manipulation events

5. **Provide analysis summary**: Explain the mapping rationale in 2-3 sentences.

Respond with structured output only. Be precise about reason codes and their
applicability. If the allegation is ambiguous, still provide best-effort mapping
with lower confidence values.
"""


class SchemeMappingResult(BaseModel):
    """Structured output from the scheme mapper agent.

    Attributes:
        reason_codes: Matched reason codes with network, code, description,
            and match_confidence for each.
        primary_reason_code: The best-matching reason code string.
        primary_network: The network for the primary reason code (AMEX/VISA/MC).
        documentation_gaps: Required documents/evidence still missing.
        analysis_summary: Brief explanation of the mapping rationale.
    """

    reason_codes: list[dict] = Field(default_factory=list)
    primary_reason_code: str = ""
    primary_network: str = ""
    documentation_gaps: list[str] = Field(default_factory=list)
    analysis_summary: str = ""


# Agent instance with structured output
scheme_mapper_agent = Agent(
    name="scheme_mapper",
    instructions=SCHEME_MAPPER_INSTRUCTIONS,
    output_type=AgentOutputSchema(SchemeMappingResult, strict_json_schema=False),
)


async def run_scheme_mapping(
    case_summary: str,
    allegation_type: str,
    claims: list[str],
    evidence_summary: str,
    model_provider: ModelProvider,
) -> SchemeMappingResult:
    """Run the scheme mapper agent to map allegations to reason codes.

    Args:
        case_summary: Summary of the case and its context.
        allegation_type: The classified allegation type (fraud/dispute/scam).
        claims: Specific claim statements extracted from the transcript.
        evidence_summary: Summary of available evidence (transactions, auth events, etc.).
        model_provider: LLM model provider for inference.

    Returns:
        SchemeMappingResult with matched reason codes and documentation gaps.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    # Build user message combining all case context
    claims_text = "\n".join(f"  - {claim}" for claim in claims) if claims else "  (none)"
    user_msg = (
        f"Case Summary:\n{case_summary}\n\n"
        f"Allegation Type: {allegation_type}\n\n"
        f"Specific Claims:\n{claims_text}\n\n"
        f"Available Evidence:\n{evidence_summary}"
    )

    try:
        result = await Runner.run(
            scheme_mapper_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Scheme mapper agent failed: {exc}") from exc
