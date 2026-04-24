"""Hypothesis scoring arbitrator for 5-category investigation assessment.

Synthesizes three category specialist outputs (dispute, scam, fraud) into a
holistic probability distribution across the 5 investigation categories.
First-party fraud is detected cross-cuttingly by the arbitrator — it has no
dedicated specialist. UNABLE_TO_DETERMINE absorbs probability mass when
evidence is insufficient. Specialists are run externally by the orchestrator.
"""

from __future__ import annotations

import logging

from agents import Agent, AgentOutputSchema, ModelProvider
from agents.run_config import RunConfig
from pydantic import BaseModel, Field, field_validator

from agentic_fraud_servicing.copilot.hypothesis_specialists import (
    SpecialistAssessment,
    SpecialistNoteUpdate,
    _add_deduped,
    _remove_by_substring,
)
from agentic_fraud_servicing.providers.retry import run_with_retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

_ALL_CATEGORIES = [
    "THIRD_PARTY_FRAUD",
    "FIRST_PARTY_FRAUD",
    "SCAM",
    "DISPUTE",
    "UNABLE_TO_DETERMINE",
]


def _coerce_reasoning_values(v: dict) -> dict[str, str]:
    """Coerce non-string values in reasoning dict to strings.

    LLMs sometimes nest contradictions or specialist_assessments inside
    the reasoning dict as list/dict values instead of keeping them as
    separate top-level fields. Convert to strings so validation passes.
    """
    if not isinstance(v, dict):
        return v
    return {k: str(val) if not isinstance(val, str) else val for k, val in v.items()}


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

    scores: dict[str, float] = {cat: 0.20 for cat in _ALL_CATEGORIES}
    reasoning: dict[str, str] = {cat: "" for cat in _ALL_CATEGORIES}
    contradictions: list[str] = []
    assessment_summary: str = ""
    specialist_assessments: dict[str, SpecialistAssessment] = Field(
        default_factory=dict, exclude=True
    )

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning_values(cls, v: dict) -> dict[str, str]:
        return _coerce_reasoning_values(v)


class ReasoningNoteUpdate(BaseModel):
    """Incremental update to the hypothesis arbitrator's working notes.

    Scores, reasoning dict, and assessment_summary are regenerated each turn.
    Contradictions are incrementally updated to prevent flickering.
    """

    scores: dict[str, float] = {cat: 0.20 for cat in _ALL_CATEGORIES}
    reasoning: dict[str, str] = {cat: "" for cat in _ALL_CATEGORIES}
    assessment_summary: str = ""

    add_contradictions: list[str] = Field(default_factory=list)
    remove_contradictions: list[str] = Field(default_factory=list)

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning(cls, v: dict) -> dict[str, str]:
        return _coerce_reasoning_values(v)


def merge_reasoning_notes(
    previous: HypothesisAssessment,
    update: ReasoningNoteUpdate,
) -> HypothesisAssessment:
    """Merge an incremental update into previous assessment notes.

    Scores, reasoning dict, and assessment_summary are replaced wholesale.
    Contradictions are patched: removals first, then additions.
    """
    contradictions = _remove_by_substring(previous.contradictions, update.remove_contradictions)
    contradictions = _add_deduped(contradictions, update.add_contradictions)

    return HypothesisAssessment(
        scores=update.scores,
        reasoning=update.reasoning,
        contradictions=contradictions,
        assessment_summary=update.assessment_summary,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

HYPOTHESIS_INSTRUCTIONS = """\
You are a hypothesis scoring arbitrator for AMEX card dispute investigation.
You synthesize assessments from three category specialists (Dispute, Scam,
Third-Party Fraud) into a final 5-category probability distribution.

## Your Input

You receive the following context each turn:

1. **Specialist Assessments** — Three independent evaluations, each with
   policy-grounded reasoning, supporting/contradicting evidence, evidence gaps,
   and policy citations. Specialists evaluate their own category in isolation
   and cite specific policy documents.
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

   This applies equally to contradictions: a contradiction that was already
   reflected in previous scores has been priced in. Only **new** contradictions
   (not seen in the previous reasoning trace) should shift scores further.
   Re-citing the same contradictions across turns must not compound their
   effect.

7. **Contradictions are one signal among many.** Contradictions inform
   investigation direction but do not dominate it. Specialist likelihoods,
   evidence gaps, auth assessment, and the CM's narrative all carry weight.
   Even when contradictions are present, if specialist evidence or other
   signals point in a different direction, weigh them proportionally.
   No single signal type should consume more than ~0.40 of the total
   probability mass on its own.

8. **Score UNABLE_TO_DETERMINE based on evidence sufficiency.** This category
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
   - **High-impact evidence gaps exist.** If a specialist flags an evidence
     gap that could *reverse* the current leading category once obtained,
     lean heavily toward UNABLE_TO_DETERMINE. The current score is built on
     an unstable foundation. Examples of high-impact gaps:
     - Merchant delivery records missing when CM claims goods not received —
       confirmed delivery would collapse a dispute score and shift toward
       first-party fraud.
     - Device forensics pending when CM claims unauthorized — a device match
       would collapse third-party fraud.
     - Scammer communication trail unavailable when CM describes coercion —
       without it, scam vs. first-party fraud is indistinguishable.
     Low-impact gaps (evidence that would add confidence but not change
     direction) do not warrant shifting mass to UNABLE_TO_DETERMINE.

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
detected by you through cross-specialist analysis.

### Prerequisite: The CM Must Have Made a Claim

FIRST_PARTY_FRAUD means the cardmember is **misrepresenting** a transaction.
This requires TWO elements:
1. The CM has made an **active claim** — they allege fraud, dispute a charge,
   or describe being scammed.
2. Evidence **contradicts** that claim — system data shows the claim is false.

If the CM has NOT made a complaint or actively acknowledges/accepts the
transactions, FIRST_PARTY_FRAUD does not apply — there is no claim to
contradict. Route to UNABLE_TO_DETERMINE (insufficient basis for
any category) or to the category that best fits the conversation context.
Do NOT infer first-party fraud from the absence of a complaint.

### Scoring Signals (only when the CM has made an active claim)

- **All specialists report low likelihood — check WHY before routing.**
  Low likelihoods across all three specialists can mean two very different
  things. Look at the specialists' evidence_gaps and contradicting_evidence
  to distinguish:
  - If specialists cite **contradicting evidence** (evidence that actively
    doesn't fit their category), the remaining mass flows to
    FIRST_PARTY_FRAUD — something happened, and no external party explains it.
  - If specialists cite **evidence gaps** (insufficient data, key items only
    available offline), the remaining mass flows to UNABLE_TO_DETERMINE —
    there is simply not enough information yet to distinguish categories.

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

### Distinguish from UNABLE_TO_DETERMINE

| Condition | Route to |
|---|---|
| CM made a claim + evidence contradicts it | FIRST_PARTY_FRAUD |
| CM made a claim + evidence is missing/insufficient | UNABLE_TO_DETERMINE |
| CM has not made a clear claim | UNABLE_TO_DETERMINE |
| Specialists low because of gaps, not contradictions | UNABLE_TO_DETERMINE |

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


# ---------------------------------------------------------------------------
# Agent instance
# ---------------------------------------------------------------------------

hypothesis_agent = Agent(
    name="hypothesis",
    instructions=HYPOTHESIS_INSTRUCTIONS,
    output_type=AgentOutputSchema(HypothesisAssessment, strict_json_schema=False),
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_specialist_for_arbitrator(
    assessments: dict[str, SpecialistAssessment],
    deltas: dict[str, SpecialistNoteUpdate] | None = None,
) -> str:
    """Format specialist outputs into the arbitrator's user message section.

    Shows merged state for each specialist. When deltas are available,
    appends a "Changes this turn" section so the arbitrator can see what
    shifted since the last assessment.
    """
    if deltas is None:
        deltas = {}

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
            "\n".join(f"  - {c}" for c in a.policy_citations) if a.policy_citations else "  none"
        )
        gaps = ", ".join(a.evidence_gaps) if a.evidence_gaps else "none"
        section = (
            f"### {_LABELS[category]}\n"
            f"Eligibility: {a.eligibility}\n"
            f"Reasoning: {a.reasoning}\n"
            f"Supporting evidence: {supporting}\n"
            f"Contradicting evidence: {contradicting}\n"
            f"Evidence gaps: {gaps}\n"
            f"Policy citations:\n{citations}"
        )

        # Append diff summary if available for this specialist
        delta = deltas.get(category)
        if delta is not None:
            changes = _format_delta_summary(delta)
            if changes:
                section += f"\n\nChanges this turn:\n{changes}"

        parts.append(section)
    return "\n\n".join(parts)


def _format_delta_summary(delta: SpecialistNoteUpdate) -> str:
    """Format a specialist delta into a concise changes summary."""
    lines: list[str] = []
    for item in delta.add_supporting_evidence:
        lines.append(f"  + Supporting: {item}")
    for item in delta.remove_supporting_evidence:
        lines.append(f"  - Supporting (removed): {item}")
    for item in delta.add_contradicting_evidence:
        lines.append(f"  + Contradicting: {item}")
    for item in delta.remove_contradicting_evidence:
        lines.append(f"  - Contradicting (removed): {item}")
    for item in delta.add_evidence_gaps:
        lines.append(f"  + Gap: {item}")
    for item in delta.remove_evidence_gaps:
        lines.append(f"  - Gap (resolved): {item}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner wrapper
# ---------------------------------------------------------------------------


async def run_arbitrator(
    specialist_assessments: dict[str, SpecialistAssessment],
    allegations_summary: str,
    auth_summary: str,
    current_scores: dict[str, float],
    model_provider: ModelProvider,
    previous_reasoning: HypothesisAssessment | None = None,
    specialist_deltas: dict[str, SpecialistNoteUpdate] | None = None,
) -> HypothesisAssessment:
    """Run the arbitrator to score investigation categories.

    Takes pre-computed specialist assessments (run externally by the
    orchestrator) and synthesizes them into the final 5-category probability
    distribution with per-category reasoning.

    On subsequent turns, the arbitrator outputs a ReasoningNoteUpdate
    (incremental delta) which is merged into the previous HypothesisAssessment.
    Specialist deltas are included in the user message so the arbitrator can
    see what changed this turn.

    Args:
        specialist_assessments: Specialist outputs keyed by category
            (DISPUTE, SCAM, THIRD_PARTY_FRAUD).
        allegations_summary: Formatted allegations with types and entities.
        auth_summary: Auth assessment text (impersonation risk, risk factors).
        current_scores: Previous hypothesis scores (5-key dict).
        model_provider: LLM model provider for inference.
        previous_reasoning: Full HypothesisAssessment from the last successful
            run. Provides the reasoning trace for Bayesian updating.
        specialist_deltas: Raw SpecialistNoteUpdate objects from this turn,
            keyed by category. Empty dict or None on the first turn.

    Returns:
        HypothesisAssessment with updated scores, reasoning, and
        contradictions.

    Raises:
        RuntimeError: If the arbitrator agent SDK call fails.
    """
    if specialist_deltas is None:
        specialist_deltas = {}

    # 1. Format arbitrator user message
    scores_text = ", ".join(f"{k}: {v:.2f}" for k, v in current_scores.items())

    prev_reasoning_text = "First assessment — no prior reasoning."
    prev_contradictions_text = "None detected."
    is_update = previous_reasoning is not None
    if is_update:
        reasoning_lines = [f"- {k}: {v}" for k, v in previous_reasoning.reasoning.items() if v]
        parts = []
        if reasoning_lines:
            parts.append("Per-category reasoning:\n" + "\n".join(reasoning_lines))
        if previous_reasoning.assessment_summary:
            parts.append(f"Summary: {previous_reasoning.assessment_summary}")
        if parts:
            prev_reasoning_text = "\n".join(parts)
        if previous_reasoning.contradictions:
            numbered = [f"{i}. {c}" for i, c in enumerate(previous_reasoning.contradictions, 1)]
            prev_contradictions_text = "\n".join(numbered)

    user_msg = (
        f"## Specialist Assessments\n\n"
        f"{_format_specialist_for_arbitrator(specialist_assessments, specialist_deltas)}\n\n"
        f"## Auth Assessment\n{auth_summary}\n\n"
        f"## Accumulated Allegations\n{allegations_summary}\n\n"
        f"## Current Hypothesis Scores\n{scores_text}\n\n"
        f"## Previous Reasoning Trace\n{prev_reasoning_text}\n\n"
        f"## Previously Detected Contradictions\n{prev_contradictions_text}"
    )

    # 2. Run arbitrator — dual output type for diff/patch memory
    output_type = ReasoningNoteUpdate if is_update else HypothesisAssessment
    agent = Agent(
        name="hypothesis",
        instructions=HYPOTHESIS_INSTRUCTIONS,
        output_type=AgentOutputSchema(output_type, strict_json_schema=False),
    )

    try:
        result = await run_with_retry(
            agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        output = result.final_output
        if is_update:
            return merge_reasoning_notes(previous_reasoning, output)
        return output
    except Exception as exc:
        raise RuntimeError(f"Hypothesis agent failed: {exc}") from exc
