"""Scheme mapper specialist agent for reason code mapping and documentation gap analysis.

Maps fraud/dispute/scam allegations to card network reason codes (AMEX, Visa,
Mastercard) and identifies documentation gaps needed to support the claim under
the matched reason code. Uses OpenAI Agents SDK with structured output via
SchemeMappingResult.
"""

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

from agentic_fraud_servicing.models.enums import INVESTIGATION_CATEGORIES_REFERENCE

# System prompt for the scheme mapper agent
SCHEME_MAPPER_INSTRUCTIONS = f"""\
You are a scheme mapping specialist for card dispute servicing. Your role is to
map investigation findings to the correct card network reason codes and identify
documentation gaps, using the four-category investigation framework.

## Investigation Categories Reference

{INVESTIGATION_CATEGORIES_REFERENCE}

## Reason Code Mapping by Investigation Category

Map the investigation category (what the system concludes) to applicable reason
codes. The allegation type (what the CM claims) may differ from the investigation
category.

### THIRD_PARTY_FRAUD — Unauthorized transaction by external criminal

**AMEX Reason Codes**:
- FR2: Fraud Full Recourse — unauthorized transaction, full liability on merchant.
- FR4: Immediate Chargeback — fraud with no prior authorization attempt.
- FR6: Partial Immediate Chargeback — partial fraud recovery.

**Visa Reason Codes**:
- 10.4: Other Fraud — Card Absent Environment — fraud in card-not-present.
- 10.5: Visa Fraud Monitoring Program — identified via fraud monitoring.

**Mastercard Reason Codes**:
- 4837: No Cardholder Authorization — unauthorized transaction.
- 4863: Cardholder Does Not Recognize — unrecognized transaction.
- 4871: Chip/PIN Liability Shift — chip transaction fraud liability.

### FIRST_PARTY_FRAUD — Cardmember misrepresentation (friendly fraud)

First-party fraud claims are typically DENIED — no valid chargeback reason code
applies because the cardmember authorized the transaction. The claim should be
flagged for the Special Investigations Unit (SIU) or internal fraud team.

Set primary_reason_code to "DENIED" and primary_network to "N/A" when
the investigation concludes FIRST_PARTY_FRAUD. Documentation gaps should focus
on evidence needed to support the denial decision (chip+PIN logs, delivery
proof, merchant confirmation, behavioral analysis report).

### SCAM — CM authorized but was deceived by external scammer

Scam cases have limited chargeback recovery options. Applicable codes depend on
whether the merchant was complicit or the transaction involved stolen credentials:
- FR2 (AMEX): possible if merchant was complicit in the scam.
- 10.4 (Visa): possible if credentials were stolen via phishing.
- 4837 (MC): possible if authorization was obtained through social engineering.

In most pure scam cases, recovery is limited and the focus shifts to
customer remediation and scam pattern reporting.

### DISPUTE — Legitimate merchant performance complaint

**AMEX Reason Codes**:
- C08: Goods/Services Not Received — cardmember paid but goods/services
  were not delivered.
- C18: "No Show" or CARDeposit Cancelled — reservation cancelled but charged.
- C28: Cancelled Recurring Billing — recurring charge after cancellation.
- C32: Goods/Services Damaged or Defective — goods received in poor condition.

**Visa Reason Codes**:
- 13.1: Merchandise/Services Not Received — paid but not delivered.
- 13.2: Cancelled Recurring Transaction — recurring after cancellation.
- 13.3: Not as Described or Defective — goods differ from description.

**Mastercard Reason Codes**:
- 4853: Cardholder Dispute — goods/services not as described or not received.

## Your Tasks

1. **Determine investigation category**: Based on the evidence and specialist
   findings, identify which of the four categories best fits.

2. **Map to reason codes**: Select applicable reason codes for the identified
   category. Assign a match_confidence (0.0-1.0) to each.

3. **Identify the primary reason code**: Select the best-matching reason code
   and its network. For FIRST_PARTY_FRAUD, use "DENIED" / "N/A".

4. **Identify documentation gaps**: For the primary reason code, list specific
   documents or evidence still needed:
   - THIRD_PARTY_FRAUD: police report, signed affidavit, device/IP logs, card status
   - FIRST_PARTY_FRAUD: chip+PIN transaction logs, delivery proof with signature,
     merchant confirmation of legitimate purchase, behavioral analysis report
   - SCAM: communication records, transaction authorization proof, timeline of
     manipulation events, scam pattern report
   - DISPUTE: tracking info, delivery confirmation, merchant correspondence,
     photos of damage, return receipt

5. **Provide analysis summary**: Explain the mapping rationale in 2-3 sentences,
   including why the investigation category was selected.

Respond with structured output only. Be precise about reason codes and their
applicability. If the category is ambiguous, still provide best-effort mapping
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
