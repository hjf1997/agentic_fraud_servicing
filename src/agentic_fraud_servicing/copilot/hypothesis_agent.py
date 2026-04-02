"""Hypothesis scoring specialist agent for 4-category investigation assessment.

Receives all accumulated context (allegations, auth assessment, retrieved evidence,
conversation history, previous scores) and produces a holistic probability
distribution across the 4 investigation categories via LLM reasoning. Replaces
the formulaic weighted-average approach used previously in the orchestrator.
"""

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel

from agentic_fraud_servicing.models.enums import INVESTIGATION_CATEGORIES_REFERENCE

# --- Output model ---


class HypothesisAssessment(BaseModel):
    """Structured output from the hypothesis scoring agent.

    Attributes:
        scores: Probability distribution across 4 investigation categories.
            Keys: THIRD_PARTY_FRAUD, FIRST_PARTY_FRAUD, SCAM, DISPUTE.
            Values between 0.0 and 1.0, summing to approximately 1.0.
        reasoning: Per-category explanation of the score assignment.
            Same 4 keys as scores.
        contradictions: Detected contradictions between CM allegations and evidence.
        assessment_summary: Overall assessment of the current situation.
    """

    scores: dict[str, float] = {
        "THIRD_PARTY_FRAUD": 0.25,
        "FIRST_PARTY_FRAUD": 0.25,
        "SCAM": 0.25,
        "DISPUTE": 0.25,
    }
    reasoning: dict[str, str] = {
        "THIRD_PARTY_FRAUD": "",
        "FIRST_PARTY_FRAUD": "",
        "SCAM": "",
        "DISPUTE": "",
    }
    contradictions: list[str] = []
    assessment_summary: str = ""


# --- System prompt ---

HYPOTHESIS_INSTRUCTIONS = f"""\
You are a hypothesis scoring specialist for AMEX card dispute investigation.
Your job is to assess the probability of each investigation category given ALL
accumulated evidence, allegations, and conversation context.

{INVESTIGATION_CATEGORIES_REFERENCE}

## Your Input

You receive the following context each turn:

1. **Accumulated Allegations** — Structured allegations extracted from the conversation
   so far, each with an AllegationDetailType (e.g., UNRECOGNIZED_TRANSACTION, CARD_POSSESSION,
   GOODS_NOT_RECEIVED) and extracted entities (amounts, merchants, dates).
2. **Auth Assessment** — Impersonation risk score, risk factors, and step-up
   auth recommendations from the authentication specialist.
3. **Retrieved Evidence** — Verified system data: transactions, authentication
   events, customer profile, device enrollment, delivery proofs.
4. **Current Hypothesis Scores** — The previous turn's probability distribution
   across the 4 categories. Use these as a Bayesian prior to update.
5. **Conversation Summary** — Running summary of the call so far.

## Scoring Rules

1. **Score all 4 categories holistically.** Do not focus only on the most likely
   category. Every category must receive a reasoned score.
2. **Scores should approximate a probability distribution** — they should sum to
   roughly 1.0. Small deviations are acceptable but avoid scores that sum to
   more than 1.2 or less than 0.8.
3. **Use previous scores as a prior.** Update them based on new evidence rather
   than scoring from scratch each turn. Scores should shift gradually unless
   strong contradictory evidence emerges.
4. **FIRST_PARTY_FRAUD is cross-cutting.** Any allegation type (FRAUD, DISPUTE,
   SCAM) can turn out to be first-party fraud. Always evaluate this category
   regardless of what the CM alleges.

## Key Reasoning Patterns

Apply these evidence-to-hypothesis mappings:

- **Chip+PIN auth from enrolled device contradicts "unauthorized" allegation** →
  Strongly increase FIRST_PARTY_FRAUD. If the CM's own enrolled device was used
  with chip+PIN, the transaction was very likely authorized by the CM.

- **CM alleges card lost/stolen but device was enrolled and used recently** →
  Increase FIRST_PARTY_FRAUD. A recently enrolled, actively used device
  contradicts a lost/stolen narrative.

- **Evidence of external manipulator** (coached language, urgency from a third
  party, social-engineering patterns, payment to unfamiliar recipient) →
  Increase SCAM. The presence of an identifiable external deceiver distinguishes
  scam from first-party fraud.

- **Contradictions WITHOUT an external manipulator** → FIRST_PARTY_FRAUD, not
  SCAM. If the CM's story contradicts evidence but there is no sign of an
  external scammer, the CM is likely misrepresenting.

- **CM story consistent + auth logs show unfamiliar device/location** →
  Increase THIRD_PARTY_FRAUD. When the CM's account shows genuinely suspicious
  activity from unknown devices, the unauthorized fraud hypothesis strengthens.

- **CM complaint about merchant service, no fraud indicators** →
  Increase DISPUTE. When there are no contradictions or fraud signals and the
  issue is about goods/services/billing, this is a merchant dispute.

- **Signed delivery proof contradicts "never received" allegation** →
  Increase FIRST_PARTY_FRAUD. Verified delivery evidence directly contradicts
  the CM's goods-not-received allegation.

- **CM accidentally reveals merchant familiarity before being told** →
  Increase FIRST_PARTY_FRAUD. Knowledge of merchant details that should be
  unknown to a fraud victim suggests the CM made the purchase.

- **Story shifts or inconsistencies when confronted with evidence** →
  Increase FIRST_PARTY_FRAUD. Changing the narrative under pressure is a
  behavioral red flag for misrepresentation.

- **CM authorized a third party who made the disputed transactions** →
  Decrease THIRD_PARTY_FRAUD. If the CM granted access (employee,
  family, delegate, agency), the transactions are not unauthorized — increase
  DISPUTE (billing issue) or FIRST_PARTY_FRAUD (CM denies knowledge despite
  granting access).

- **High impersonation risk from auth assessment** →
  Increase THIRD_PARTY_FRAUD. If the caller may not be the real cardholder,
  the account may have been taken over.

## Output Format

Provide your assessment as structured output with:
- scores: dict with exactly 4 keys (THIRD_PARTY_FRAUD, FIRST_PARTY_FRAUD, SCAM,
  DISPUTE), each a float between 0.0 and 1.0
- reasoning: dict with the same 4 keys, each a brief explanation (1-3 sentences)
- contradictions: list of detected contradictions between allegations and evidence
- assessment_summary: 2-4 sentence overall assessment
"""


# --- Agent instance ---

hypothesis_agent = Agent(
    name="hypothesis",
    instructions=HYPOTHESIS_INSTRUCTIONS,
    output_type=AgentOutputSchema(HypothesisAssessment, strict_json_schema=False),
)


# --- Runner wrapper ---


async def run_hypothesis(
    allegations_summary: str,
    auth_summary: str,
    evidence_summary: str,
    current_scores: dict[str, float],
    conversation_summary: str,
    model_provider: ModelProvider,
) -> HypothesisAssessment:
    """Run the hypothesis agent to score investigation categories.

    Args:
        allegations_summary: Formatted allegations with types and entities.
        auth_summary: Auth assessment text (impersonation risk, risk factors).
        evidence_summary: Retrieved evidence text (transactions, auth events).
        current_scores: Previous hypothesis scores (4-key dict).
        conversation_summary: Running summary of the call so far.
        model_provider: LLM model provider for inference.

    Returns:
        HypothesisAssessment with updated scores, reasoning, and contradictions.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    # Format previous scores for the prompt
    scores_text = ", ".join(f"{k}: {v:.2f}" for k, v in current_scores.items())

    user_msg = (
        f"## Accumulated Allegations\n{allegations_summary}\n\n"
        f"## Auth Assessment\n{auth_summary}\n\n"
        f"## Retrieved Evidence\n{evidence_summary}\n\n"
        f"## Current Hypothesis Scores\n{scores_text}\n\n"
        f"## Conversation Summary\n{conversation_summary}"
    )

    try:
        result = await Runner.run(
            hypothesis_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Hypothesis agent failed: {exc}") from exc
