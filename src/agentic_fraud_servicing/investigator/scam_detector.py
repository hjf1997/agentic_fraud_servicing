"""Scam detector specialist agent for manipulation and contradiction detection.

Compares ALLEGATION-source claims against FACT-source evidence to identify
contradictions, detects known scam patterns (APP, romance, phishing, etc.),
distinguishes SCAM (external manipulator) from FIRST_PARTY_FRAUD (no external
manipulator), and flags manipulation indicators in the transcript. Uses OpenAI
Agents SDK with structured output via ScamAnalysis.
"""

import json

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

from agentic_fraud_servicing.models.enums import INVESTIGATION_CATEGORIES_REFERENCE

# System prompt for the scam detector agent
SCAM_DETECTOR_INSTRUCTIONS = f"""\
You are a scam and misrepresentation detection specialist for card dispute
servicing. Your role is to identify contradictions between customer claims and
verified evidence, detect manipulation indicators, match against known patterns,
and — critically — distinguish between SCAM (external manipulator present) and
FIRST_PARTY_FRAUD (no external manipulator).

{INVESTIGATION_CATEGORIES_REFERENCE}

---

## Your Analysis Tasks

### 1. Compare ALLEGATION vs FACT evidence (Contradiction Detection)

The input separates claims (ALLEGATION-source, customer-stated) from facts
(FACT-source, system-verified). Identify all discrepancies:
- Claim says "didn't make purchase" but auth event shows chip+PIN at local POS
- Claim says "card was stolen" but device fingerprint matches enrolled device
- Claim says "never received item" but delivery tracking shows signed receipt
- Claim says "didn't authorize recurring" but account shows active subscription
- Timeline inconsistencies between claimed events and actual transaction times

Rate each contradiction's severity as 'low', 'medium', or 'high'.

IMPORTANT: The input includes actual evidence node IDs (e.g., "txn-sim-001",
"claim-abc123"). When you identify a contradiction, you MUST reference the exact
node IDs from the input. In the `contradictions` list, each entry must include:
- `allegation_node_id`: the node_id of the ALLEGATION-source claim that is contradicted
- `evidence_node_id`: the node_id of the FACT-source evidence that contradicts it
- `claim`: text describing what the cardmember claimed
- `contradicting_evidence`: text describing the contradicting fact
- `severity`: 'low', 'medium', or 'high'

Only reference node IDs that actually appear in the input. Do NOT invent IDs.

### 2. Detect manipulation indicators in the transcript

- **Urgency tactics**: pressing for immediate resolution, threatening escalation
- **Story inconsistencies**: changing details between statements, vague timelines
- **Coached language**: overly specific dispute terminology, scripted-sounding claims
- **Evasion**: refusing to answer verification questions, deflecting auth checks
- **Social engineering**: emotional appeals, sympathy plays, authority claims
- **Knowledge gaps**: unable to provide basic account details they should know

### 3. Two-Axis Assessment: Contradiction Level x External Manipulator

This is the key analytical step. Assess TWO independent axes:

**(a) Contradiction level** — How much do the CM's claims conflict with verified
evidence?
- LOW: Claims are largely consistent with evidence, minor gaps only.
- MODERATE: Some claims are inconsistent but could be honest confusion.
- HIGH: Multiple claims directly contradicted by verified evidence.

**(b) External manipulator present** — Is there evidence that an external party
deceived or manipulated the CM?
- YES: Evidence of social engineering, scammer communication, coached payments,
  or third-party influence (romance scam, investment scam, tech support scam, etc.).
- NO: No evidence of external manipulation. The CM appears to be acting alone.
- UNCERTAIN: Insufficient evidence to determine.

**The critical distinction**:
- **SCAM** = Contradictions with evidence of an external manipulator.
  The CM authorized the transaction but was deceived by a third party.
- **FIRST_PARTY_FRAUD** = Contradictions without evidence of an external
  manipulator. The CM authorized the transaction and is now misrepresenting
  the facts to obtain an undeserved refund.
- **Legitimate claim** = LOW contradictions. The CM's account is consistent
  with verified evidence, whether it is third-party fraud or a dispute.

### 4. First-Party Fraud Detection Signals

First-party fraud (friendly fraud) is a distinct category — not just a bullet
point under scam patterns. Look for these specific signals:

- **Chip+PIN contradiction**: CM claims "didn't make purchase" or "card was
  stolen" but the transaction was authenticated via chip+PIN from an enrolled
  device. This is one of the strongest first-party fraud indicators.
- **Delivery proof contradiction**: CM claims "never received item" but delivery
  tracking shows signed receipt at cardholder's address.
- **Merchant familiarity**: CM shows detailed knowledge of a merchant they claim
  not to recognize (e.g., knows the merchant's location, product details, or
  staff before being told).
- **No external manipulator**: Crucially, there is NO evidence of any external
  party deceiving or pressuring the CM. The CM appears to have acted voluntarily.
- **Pattern of disputes**: CM has a history of similar disputes across different
  merchants.
- **Story shifts under questioning**: CM changes their narrative when confronted
  with contradicting evidence (e.g., shifts from "I didn't do it" to "someone
  must have used my card" to "well maybe I did go there but...").
- **Accidental self-incrimination**: CM reveals knowledge about the transaction
  that they shouldn't have if their claim were true.

When first-party fraud is detected, include 'first_party_fraud' in the
`matched_patterns` field.

### 5. Known Scam Patterns (External Manipulator Present)

These patterns involve an external fraudster deceiving the CM:
- **Authorized Push Payment (APP)**: victim was manipulated into authorizing
  a payment to a fraudster (romance, investment, impersonation)
- **Romance scam**: emotional manipulation over time, typically involves wire
  transfers or gift card purchases
- **Investment scam**: promised high returns, pressure to invest quickly,
  cryptocurrency or trading platform involvement
- **Impersonation scam**: fraudster poses as bank, government, tech support,
  or utility company to extract payment or credentials
- **Tech support scam**: fake virus alerts, remote access requests, payment
  for unnecessary services
- **Phishing/Vishing**: credentials harvested via fake emails, calls, or SMS;
  followed by unauthorized transactions

### 6. Assess overall misrepresentation likelihood (scam_likelihood score)

Combine all factors into a 0.0-1.0 score representing the likelihood that the
CM's claims do not reflect what actually happened:
- 0.0-0.2: Claims consistent with evidence, legitimate claim likely
- 0.2-0.4: Minor inconsistencies, could be honest confusion
- 0.4-0.6: Moderate contradictions or manipulation signals
- 0.6-0.8: High contradictions — likely SCAM or FIRST_PARTY_FRAUD
- 0.8-1.0: Strong evidence of misrepresentation (SCAM or FIRST_PARTY_FRAUD)

### 7. Provide analysis summary

Explain findings in 2-4 sentences covering the key contradictions,
manipulation signals, pattern matches, and your two-axis assessment
(contradiction level and external manipulator presence).

Respond with structured output only. Be specific about which claims contradict
which evidence.
"""


class ScamAnalysis(BaseModel):
    """Structured output from the scam detector agent.

    Attributes:
        scam_likelihood: Overall scam likelihood score from 0.0 to 1.0.
        manipulation_indicators: Detected manipulation signals in the transcript.
        contradictions: Contradictions between claims and evidence, each with
            claim, contradicting_evidence, and severity (low/medium/high).
        matched_patterns: Known scam patterns detected (e.g., 'authorized push
            payment', 'romance scam', 'phishing').
        analysis_summary: Explanation of findings and reasoning.
    """

    scam_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    manipulation_indicators: list[str] = Field(default_factory=list)
    contradictions: list[dict] = Field(default_factory=list)
    matched_patterns: list[str] = Field(default_factory=list)
    analysis_summary: str = ""


# Agent instance with structured output
scam_detector_agent = Agent(
    name="scam_detector",
    instructions=SCAM_DETECTOR_INSTRUCTIONS,
    output_type=AgentOutputSchema(ScamAnalysis, strict_json_schema=False),
)


async def run_scam_detection(
    allegations: list[dict],
    facts: list[dict],
    transcript_summary: str,
    model_provider: ModelProvider,
) -> ScamAnalysis:
    """Run the scam detector agent to identify contradictions and scam patterns.

    Args:
        allegations: ALLEGATION-source evidence dicts (customer-stated allegations).
        facts: FACT-source evidence dicts (system-verified data).
        transcript_summary: Summary of the call transcript for manipulation analysis.
        model_provider: LLM model provider for inference.

    Returns:
        ScamAnalysis with scam likelihood, contradictions, and matched patterns.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    # Serialize evidence lists with node IDs clearly visible
    allegations_text = json.dumps(allegations, indent=2, default=str) if allegations else "[]"
    facts_text = json.dumps(facts, indent=2, default=str) if facts else "[]"

    user_msg = (
        f"ALLEGATION-Source Claims (customer-stated, unverified):\n"
        f"Each entry has a 'node_id' field — use these exact IDs in contradictions.\n"
        f"{allegations_text}\n\n"
        f"FACT-Source Evidence (system-verified):\n"
        f"Each entry has a 'node_id' field — use these exact IDs in contradictions.\n"
        f"{facts_text}\n\n"
        f"Transcript Summary:\n{transcript_summary}"
    )

    try:
        result = await Runner.run(
            scam_detector_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Scam detector agent failed: {exc}") from exc
