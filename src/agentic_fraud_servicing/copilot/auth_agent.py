"""Authentication specialist agent for impersonation risk assessment.

Analyzes transcript segments, auth event history, and customer profile data
to assess impersonation risk and recommend step-up authentication when
appropriate. Uses OpenAI Agents SDK with structured output via AuthAssessment.
"""

import json

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

from agentic_fraud_servicing.models.enums import INVESTIGATION_CATEGORIES_REFERENCE

# System prompt for the auth assessment agent
AUTH_INSTRUCTIONS = f"""\
You are an authentication and impersonation risk specialist for AMEX card
dispute servicing. Your role is to assess whether the caller may be an
impersonator rather than the legitimate cardmember.

{INVESTIGATION_CATEGORIES_REFERENCE}

How impersonation risk relates to investigation categories:
- HIGH impersonation risk suggests THIRD_PARTY_FRAUD — someone other than the
  legitimate CM is using the card or account.
- LOW impersonation risk with contradictions between claims and evidence suggests
  FIRST_PARTY_FRAUD — the real CM is calling but misrepresenting facts.
- SCAM victims typically pass identity verification because they ARE the
  legitimate CM who was deceived into authorizing the transaction.

Analyze the following inputs:
1. **Transcript segment**: Look for behavioral red flags such as:
   - Hesitation or uncertainty about basic account details
   - Inconsistent information across the conversation
   - Urgency or pressure tactics ("I need this done right now")
   - Reluctance to complete verification steps
   - Knowledge gaps about recent transactions
2. **Auth events**: Review authentication attempts and results:
   - Failed authentication attempts suggest higher risk
   - Device fingerprint mismatches or new devices increase risk
   - Multiple recent auth changes (password, phone, email) are suspicious
3. **Customer profile**: Compare caller behavior to historical patterns:
   - Deviation from usual call patterns or channels
   - Geographic anomalies in recent activity
   - Recent account modifications (address, phone number changes)

Risk levels:
- LOW (0.0-0.3): Caller matches expected behavior, auth events normal
- MEDIUM (0.3-0.6): Some inconsistencies but nothing definitive
- HIGH (0.6-0.8): Multiple red flags, recommend step-up auth
- CRITICAL (0.8-1.0): Strong impersonation indicators, require step-up

Step-up methods:
- NONE: Risk is low, no additional auth needed
- SMS_OTP: Send one-time password to registered phone
- CALLBACK: Call back the customer at their registered phone number
- SECURITY_QUESTIONS: Ask knowledge-based authentication questions

Respond with structured output only. Be precise and evidence-based.
"""


class AuthAssessment(BaseModel):
    """Structured output from the auth assessment agent.

    Attributes:
        impersonation_risk: Risk score between 0.0 (no risk) and 1.0 (certain).
        risk_factors: Factors contributing to the risk assessment.
        step_up_recommended: Whether step-up authentication is recommended.
        step_up_method: Recommended step-up method if applicable.
        assessment_summary: Brief explanation of the overall assessment.
    """

    impersonation_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_factors: list[str] = Field(default_factory=list)
    step_up_recommended: bool = False
    step_up_method: str = "NONE"
    assessment_summary: str = ""


# Agent instance with structured output
auth_agent = Agent(
    name="auth_assessor",
    instructions=AUTH_INSTRUCTIONS,
    output_type=AgentOutputSchema(AuthAssessment, strict_json_schema=False),
)


async def run_auth_assessment(
    transcript_text: str,
    auth_events: list[dict],
    customer_profile: dict | None,
    model_provider: ModelProvider,
    conversation_history: list[tuple[str, str]] | None = None,
) -> AuthAssessment:
    """Run the auth assessment agent on transcript and auth data.

    Args:
        transcript_text: The transcript text to analyze for behavioral cues.
        auth_events: List of authentication event dicts for the session.
        customer_profile: Customer profile dict, or None if unavailable.
        model_provider: LLM model provider for inference.
        conversation_history: Recent conversation turns as (speaker, text)
            tuples. Provides multi-turn context for detecting behavioral
            patterns like hesitation, contradictions, or story changes.

    Returns:
        AuthAssessment with impersonation risk score and recommendations.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    # Build user message with all available context
    parts = []
    if conversation_history:
        lines = [f"{speaker}: {text}" for speaker, text in conversation_history]
        parts.append(
            f"Recent conversation ({len(conversation_history)} turns):\n"
            + "\n".join(lines)
        )
    parts.append(f"Current turn:\n{transcript_text}")
    parts.append(f"\nAuth events:\n{json.dumps(auth_events, indent=2)}")
    if customer_profile is not None:
        parts.append(f"\nCustomer profile:\n{json.dumps(customer_profile, indent=2)}")
    else:
        parts.append("\nCustomer profile: Not available.")
    user_msg = "\n".join(parts)

    try:
        result = await Runner.run(
            auth_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Auth assessment agent failed: {exc}") from exc
