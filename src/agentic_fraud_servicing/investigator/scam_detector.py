"""Scam detector specialist agent for manipulation and contradiction detection.

Compares ALLEGATION-source claims against FACT-source evidence to identify
contradictions, detects known scam patterns (APP, romance, phishing, etc.),
and flags manipulation indicators in the transcript. Uses OpenAI Agents SDK
with structured output via ScamAnalysis.
"""

import json

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

# System prompt for the scam detector agent
SCAM_DETECTOR_INSTRUCTIONS = """\
You are a scam detection specialist for card dispute servicing. Your role is to
identify contradictions between customer claims and verified evidence, detect
manipulation indicators, and match against known scam patterns.

Your tasks:

1. **Compare ALLEGATION vs FACT evidence**: The input separates claims
   (ALLEGATION-source, customer-stated) from facts (FACT-source, system-verified).
   Look for discrepancies:
   - Claim says "didn't make purchase" but auth event shows chip+PIN at local POS
   - Claim says "card was stolen" but device fingerprint matches enrolled device
   - Claim says "never received item" but delivery tracking shows signed receipt
   - Claim says "didn't authorize recurring" but account shows active subscription
   - Timeline inconsistencies between claimed events and actual transaction times

2. **Detect manipulation indicators** in the transcript:
   - **Urgency tactics**: pressing for immediate resolution, threatening escalation
   - **Story inconsistencies**: changing details between statements, vague timelines
   - **Coached language**: overly specific dispute terminology, scripted-sounding claims
   - **Evasion**: refusing to answer verification questions, deflecting auth checks
   - **Social engineering**: emotional appeals, sympathy plays, authority claims
   - **Knowledge gaps**: unable to provide basic account details they should know

3. **Match against known scam patterns**:
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
   - **First-party fraud**: cardholder made the purchase but claims otherwise
     to get a refund (friendly fraud)

4. **Assess overall scam likelihood**: Combine all factors into a 0.0-1.0 score:
   - 0.0-0.2: Very unlikely scam, claims consistent with evidence
   - 0.2-0.4: Low scam indicators, minor inconsistencies
   - 0.4-0.6: Moderate concern, some contradictions or manipulation signals
   - 0.6-0.8: High concern, multiple contradictions or clear manipulation
   - 0.8-1.0: Very high concern, strong evidence of scam pattern

5. **Provide analysis summary**: Explain findings in 2-4 sentences covering the
   key contradictions, manipulation signals, and pattern matches.

Respond with structured output only. Be specific about which claims contradict
which evidence. Rate contradiction severity as 'low', 'medium', or 'high'.
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
    claims: list[dict],
    facts: list[dict],
    transcript_summary: str,
    model_provider: ModelProvider,
) -> ScamAnalysis:
    """Run the scam detector agent to identify contradictions and scam patterns.

    Args:
        claims: ALLEGATION-source evidence dicts (customer-stated claims).
        facts: FACT-source evidence dicts (system-verified data).
        transcript_summary: Summary of the call transcript for manipulation analysis.
        model_provider: LLM model provider for inference.

    Returns:
        ScamAnalysis with scam likelihood, contradictions, and matched patterns.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    # Serialize evidence lists to readable text
    claims_text = json.dumps(claims, indent=2) if claims else "[]"
    facts_text = json.dumps(facts, indent=2) if facts else "[]"

    user_msg = (
        f"ALLEGATION-Source Claims (customer-stated, unverified):\n{claims_text}\n\n"
        f"FACT-Source Evidence (system-verified):\n{facts_text}\n\n"
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
