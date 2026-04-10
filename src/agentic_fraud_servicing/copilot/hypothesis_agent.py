"""Hypothesis scoring arbitrator for 5-category investigation assessment.

Synthesizes three category specialist outputs (dispute, scam, fraud) into a
holistic probability distribution across the 5 investigation categories.
First-party fraud is detected cross-cuttingly by the arbitrator — it has no
dedicated specialist. UNABLE_TO_DETERMINE absorbs probability mass when
evidence is insufficient. Specialists are run externally by the orchestrator.
"""

from __future__ import annotations

from agents import Agent, AgentOutputSchema, ModelProvider
from agents.run_config import RunConfig
from pydantic import BaseModel, Field, field_validator

from agentic_fraud_servicing.copilot.hypothesis_specialists import (
    SpecialistAssessment,
)
from agentic_fraud_servicing.providers.retry import run_with_retry

# --- Output model ---


class HypothesisAssessment(BaseModel):
    """Structured output from the hypothesis scoring arbitrator.

    Attributes:
        scores: Probability distribution across 5 investigation categories.
            Keys: THIRD_PARTY_FRAUD, FIRST_PARTY_FRAUD, SCAM, DISPUTE,
            UNABLE_TO_DETERMINE. Values between 0.0 and 1.0, summing to
            approximately 1.0.
        reasoning: Per-category explanation of the score assignment.
            Same 5 keys as scores.
        contradictions: Detected contradictions between CM allegations and evidence.
        assessment_summary: Overall assessment of the current situation.
        specialist_assessments: Specialist outputs from this turn, carried
            forward to the next turn for incremental reasoning. NOT produced
            by the LLM — set programmatically after the arbitrator returns.
    """

    scores: dict[str, float] = {
        "THIRD_PARTY_FRAUD": 0.20,
        "FIRST_PARTY_FRAUD": 0.20,
        "SCAM": 0.20,
        "DISPUTE": 0.20,
        "UNABLE_TO_DETERMINE": 0.20,
    }
    reasoning: dict[str, str] = {
        "THIRD_PARTY_FRAUD": "",
        "FIRST_PARTY_FRAUD": "",
        "SCAM": "",
        "DISPUTE": "",
        "UNABLE_TO_DETERMINE": "",
    }
    contradictions: list[str] = []
    assessment_summary: str = ""
    specialist_assessments: dict[str, SpecialistAssessment] = Field(
        default_factory=dict, exclude=True
    )

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning_values(cls, v: dict) -> dict[str, str]:
        """Coerce non-string values in reasoning dict to strings.

        LLMs sometimes nest contradictions or specialist_assessments inside
        the reasoning dict as list/dict values instead of keeping them as
        separate top-level fields. Convert to strings so validation passes.
        """
        if not isinstance(v, dict):
            return v
        return {k: str(val) if not isinstance(val, str) else val for k, val in v.items()}


# --- System prompt ---

HYPOTHESIS_INSTRUCTIONS = """\
You are a hypothesis scoring arbitrator for AMEX card dispute investigation.
You synthesize assessments from three category specialists (Dispute, Scam,
Third-Party Fraud) into a final 5-category probability distribution.

## Your Input

You receive the following context each turn:

1. **Specialist Assessments** — Three independent evaluations, each with a
   likelihood score, policy-grounded reasoning, supporting/contradicting
   evidence, and policy citations. Specialists score likelihood based on
   currently available evidence only (not speculation about offline evidence).
   They evaluate their own category in isolation and cite specific policy
   documents.
2. **Auth Assessment** — Impersonation risk score, risk factors, and step-up
   auth recommendations from the authentication specialist.
3. **Accumulated Allegations** — What the cardmember claims, with detail types
   and extracted entities. Needed for cross-cutting first-party fraud detection.
4. **Current Hypothesis Scores** — The previous turn's probability distribution
   across the 5 categories. Use these as a Bayesian prior to update.
5. **Previous Reasoning Trace** — Your own per-category reasoning from the
   last assessment turn. Use this to ground your update: identify what changed
   and explain how it shifts each score.

## Scoring Rules

1. **Produce a 5-category distribution.** The three specialists cover Dispute,
   Scam, and Third-Party Fraud. You must also score FIRST_PARTY_FRAUD and
   UNABLE_TO_DETERMINE — these are your unique responsibilities as the
   arbitrator.

2. **Scores should approximate a probability distribution** — they should sum
   to roughly 1.0. Small deviations are acceptable but avoid scores that sum
   to more than 1.2 or less than 0.8.

3. **Use previous scores as a prior.** Compare the current specialist outputs
   against the previous reasoning trace. Scores should shift gradually unless
   strong contradictory evidence emerges. Explain the delta for each category.

4. **Weigh specialist assessments critically.** Specialist likelihood scores
   reflect how well currently available evidence fits their category — they do
   not account for evidence that could be collected offline after case opening.
   Your scores should also be grounded in available evidence only. Consider
   whether another specialist's contradicting evidence undermines a high score.
   Look at the full picture across all three assessments.

5. **Allegations are not evidence.** CM claims establish which hypotheses to
   investigate, but cannot by themselves move scores. Only system evidence,
   specialist-cited policy findings, or contradictions should shift scores.

6. **Repetition is not new evidence.** If the previous reasoning trace already
   accounted for an allegation, restating it is not grounds for score changes.

7. **Score UNABLE_TO_DETERMINE based on evidence sufficiency.** This category
   absorbs probability mass when evidence is insufficient to distinguish
   between the four real investigation categories. Assign high
   UNABLE_TO_DETERMINE when:
   - It is an early assessment turn and limited evidence has been gathered.
   - Multiple categories remain plausible and no distinguishing evidence
     differentiates them (e.g., specialist likelihoods are all moderate
     and close to each other).
   - Specialists report significant evidence gaps, particularly for items
     that can only be collected offline after case opening (e.g., merchant
     records, delivery proof, device forensics).
   - The CM's narrative is consistent with multiple categories and critical
     distinguishing evidence is unavailable.

   Decrease UNABLE_TO_DETERMINE as:
   - The conversation progresses and specialist evidence gaps are filled.
   - One or more categories become clearly dominant with supporting evidence.
   - Distinguishing evidence emerges that separates categories.

   UNABLE_TO_DETERMINE is NOT an investigation outcome. It signals "more
   information needed" — as the call progresses, probability mass should
   flow from UNABLE_TO_DETERMINE into the real categories as evidence
   accumulates.

## First-Party Fraud Detection (Your Unique Role)

FIRST_PARTY_FRAUD has no specialist — it is always an investigation finding
detected by you through cross-specialist analysis. Score it based on:

- **All specialists report low likelihood**: If dispute, scam, and fraud
  specialists all find the evidence doesn't fit their category well, the
  remaining probability mass should flow to first-party fraud.

- **Contradicting evidence without external manipulator**: If specialists
  flag contradicting evidence (e.g., CM claims unauthorized but chip+PIN
  from enrolled device) AND the scam specialist finds no evidence of an
  external deceiver, this strongly indicates first-party fraud.

- **Auth assessment shows CM's own credentials**: If the auth assessment
  confirms the CM's device/credentials were used for the disputed
  transactions, increase first-party fraud.

- **Inconsistencies across specialist findings**: If the dispute specialist
  finds the CM did receive goods, the fraud specialist finds the CM's
  device was used, and the scam specialist finds no external manipulator —
  the convergence of contradictions points to first-party fraud.

- **ANY allegation type can be first-party fraud**: A CM claiming fraud,
  dispute, or scam can all turn out to be first-party fraud. Never rule it
  out based on what the CM alleges.

- **Distinguish from UNABLE_TO_DETERMINE**: First-party fraud requires
  positive evidence of contradictions. If evidence is simply missing (not
  contradictory), the mass belongs in UNABLE_TO_DETERMINE, not
  FIRST_PARTY_FRAUD.

## Output Format

Provide your assessment as structured output with:
- scores: dict with exactly 5 keys (THIRD_PARTY_FRAUD, FIRST_PARTY_FRAUD,
  SCAM, DISPUTE, UNABLE_TO_DETERMINE), each a float between 0.0 and 1.0
- reasoning: dict with the same 5 keys, each a brief explanation (1-3
  sentences) grounded in specialist findings and evidence
- contradictions: list of detected contradictions between allegations and
  evidence (consolidate from specialist findings + your own analysis)
- assessment_summary: 2-4 sentence overall assessment
"""


# --- Agent instance ---

hypothesis_agent = Agent(
    name="hypothesis",
    instructions=HYPOTHESIS_INSTRUCTIONS,
    output_type=AgentOutputSchema(HypothesisAssessment, strict_json_schema=False),
)


# --- Formatting helpers ---


def _format_specialist_for_arbitrator(
    assessments: dict[str, SpecialistAssessment],
) -> str:
    """Format specialist outputs into the arbitrator's user message section."""
    _LABELS = {
        "DISPUTE": "Dispute Specialist",
        "SCAM": "Scam Specialist",
        "THIRD_PARTY_FRAUD": "Fraud Specialist (Third-Party)",
    }
    parts: list[str] = []
    for category in ("DISPUTE", "SCAM", "THIRD_PARTY_FRAUD"):
        a = assessments.get(category)
        if a is None:
            parts.append(f"### {_LABELS[category]}\nNot available.")
            continue
        supporting = ", ".join(a.supporting_evidence) if a.supporting_evidence else "none"
        contradicting = ", ".join(a.contradicting_evidence) if a.contradicting_evidence else "none"
        citations = (
            "\n".join(f"  - {c}" for c in a.policy_citations)
            if a.policy_citations
            else "  none"
        )
        parts.append(
            f"### {_LABELS[category]}\n"
            f"Likelihood: {a.likelihood:.2f}\n"
            f"Reasoning: {a.reasoning}\n"
            f"Supporting evidence: {supporting}\n"
            f"Contradicting evidence: {contradicting}\n"
            f"Policy citations:\n{citations}"
        )
    return "\n\n".join(parts)


# --- Runner wrapper ---


async def run_arbitrator(
    specialist_assessments: dict[str, SpecialistAssessment],
    allegations_summary: str,
    auth_summary: str,
    current_scores: dict[str, float],
    model_provider: ModelProvider,
    previous_reasoning: HypothesisAssessment | None = None,
) -> HypothesisAssessment:
    """Run the arbitrator to score investigation categories.

    Takes pre-computed specialist assessments (run externally by the
    orchestrator) and synthesizes them into the final 5-category probability
    distribution.

    Args:
        specialist_assessments: Specialist outputs keyed by category
            (DISPUTE, SCAM, THIRD_PARTY_FRAUD).
        allegations_summary: Formatted allegations with types and entities.
        auth_summary: Auth assessment text (impersonation risk, risk factors).
        current_scores: Previous hypothesis scores (5-key dict).
        model_provider: LLM model provider for inference.
        previous_reasoning: Full HypothesisAssessment from the last successful
            run. Provides the reasoning trace for Bayesian updating.

    Returns:
        HypothesisAssessment with updated scores, reasoning, and
        contradictions.

    Raises:
        RuntimeError: If the arbitrator agent SDK call fails.
    """
    # 1. Format arbitrator user message
    scores_text = ", ".join(f"{k}: {v:.2f}" for k, v in current_scores.items())

    prev_reasoning_text = "First assessment — no prior reasoning."
    if previous_reasoning is not None:
        reasoning_lines = [
            f"- {k}: {v}" for k, v in previous_reasoning.reasoning.items() if v
        ]
        parts = []
        if reasoning_lines:
            parts.append("Per-category reasoning:\n" + "\n".join(reasoning_lines))
        if previous_reasoning.contradictions:
            parts.append(
                "Contradictions: " + "; ".join(previous_reasoning.contradictions)
            )
        if previous_reasoning.assessment_summary:
            parts.append(f"Summary: {previous_reasoning.assessment_summary}")
        if parts:
            prev_reasoning_text = "\n".join(parts)

    user_msg = (
        f"## Specialist Assessments\n\n"
        f"{_format_specialist_for_arbitrator(specialist_assessments)}\n\n"
        f"## Auth Assessment\n{auth_summary}\n\n"
        f"## Accumulated Allegations\n{allegations_summary}\n\n"
        f"## Current Hypothesis Scores\n{scores_text}\n\n"
        f"## Previous Reasoning Trace\n{prev_reasoning_text}"
    )

    # 2. Run arbitrator
    try:
        result = await run_with_retry(
            hypothesis_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Hypothesis agent failed: {exc}") from exc
