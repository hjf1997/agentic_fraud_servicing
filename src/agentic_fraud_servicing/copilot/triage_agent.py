"""Triage specialist agent for claim extraction and category classification.

Analyzes transcript segments to extract specific claims, classify the allegation
type (fraud/dispute/scam), and detect category shifts from previous assessments.
Uses OpenAI Agents SDK with structured output via TriageResult.
"""

from agents import Agent, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

from agentic_fraud_servicing.models.enums import AllegationType

# System prompt for the triage agent
TRIAGE_INSTRUCTIONS = """\
You are a triage specialist for AMEX card dispute servicing. Your role is to
analyze transcript segments from calls between cardmembers and contact center
professionals (CCPs).

Your tasks:
1. **Extract claims**: Identify specific claim statements the cardmember makes
   about disputed transactions, unauthorized charges, or scam encounters.
2. **Classify allegation type**: Determine the category based on AMEX definitions:
   - FRAUD: Unauthorized transactions the cardmember did not make or authorize.
     Indicators: "I didn't make this purchase", "someone used my card",
     "I don't recognize this charge".
   - DISPUTE: Authorized transactions with a problem (wrong amount, defective
     goods, service not received, duplicate charge).
     Indicators: "I paid but never received", "they charged me twice",
     "the amount is wrong", "the product was defective".
   - SCAM: The cardmember was manipulated into authorizing a transaction
     through deception (phishing, impersonation, social engineering).
     Indicators: "they told me to send money", "I thought it was legitimate",
     "they pretended to be from my bank", "I was tricked".
3. **Detect category shifts**: If a previous classification was provided,
   determine whether the new transcript evidence suggests a different category.
4. **Identify key phrases**: Extract the specific phrases from the transcript
   that support your classification.

Respond with structured output only. Be precise and evidence-based.
If the transcript is ambiguous or insufficient to classify, set allegation_type
to null and confidence to a low value.
"""


class TriageResult(BaseModel):
    """Structured output from the triage agent.

    Attributes:
        claims: Extracted claim statements from the transcript segment.
        allegation_type: Classified category or None if unclear.
        confidence: Classification confidence between 0.0 and 1.0.
        category_shift_detected: True if category shifted from previous.
        key_phrases: Phrases from the transcript that drove classification.
    """

    claims: list[str] = Field(default_factory=list)
    allegation_type: AllegationType | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    category_shift_detected: bool = False
    key_phrases: list[str] = Field(default_factory=list)


# Agent instance with structured output
triage_agent = Agent(
    name="triage",
    instructions=TRIAGE_INSTRUCTIONS,
    output_type=TriageResult,
)


async def run_triage(
    transcript_text: str,
    previous_type: AllegationType | None,
    model_provider: ModelProvider,
) -> TriageResult:
    """Run the triage agent on a transcript segment.

    Args:
        transcript_text: The transcript text to analyze.
        previous_type: Previous allegation classification, if any.
        model_provider: LLM model provider for inference.

    Returns:
        TriageResult with extracted claims and classification.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    # Build user message with context about previous classification
    parts = [f"Transcript segment:\n{transcript_text}"]
    if previous_type is not None:
        parts.append(
            f"\nPrevious classification: {previous_type.value}. "
            "Evaluate whether the new evidence supports or shifts this category."
        )
    else:
        parts.append("\nNo previous classification exists. Classify from scratch.")
    user_msg = "\n".join(parts)

    try:
        result = await Runner.run(
            triage_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Triage agent failed: {exc}") from exc
