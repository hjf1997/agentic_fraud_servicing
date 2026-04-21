"""Hypothesis scoring arbitrator for 5-category investigation assessment.

Produces a 5-category probability distribution by combining two parallel calls:
1. Logprob-based scorer — forced-choice classification via raw OpenAI API,
   with UNABLE_TO_DETERMINE derived from Shannon entropy.
2. Reasoning agent — qualitative analysis via Agents SDK structured output,
   covering per-category reasoning, contradiction detection, and first-party
   fraud identification.

First-party fraud is detected cross-cuttingly by the reasoning agent — it has
no dedicated specialist. Specialists are run externally by the orchestrator.
"""

from __future__ import annotations

import asyncio
import logging

from agents import Agent, AgentOutputSchema, ModelProvider
from agents.run_config import RunConfig
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

from agentic_fraud_servicing.copilot.hypothesis_specialists import (
    SpecialistAssessment,
    SpecialistNoteUpdate,
    _add_deduped,
    _remove_by_substring,
)
from agentic_fraud_servicing.copilot.logit_scorer import compute_logprob_scores
from agentic_fraud_servicing.providers.retry import run_with_retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


_REASONING_CATEGORIES = ["THIRD_PARTY_FRAUD", "FIRST_PARTY_FRAUD", "SCAM", "DISPUTE"]


def _coerce_reasoning_values(v: dict) -> dict[str, str]:
    """Coerce non-string values in reasoning dict to strings.

    LLMs sometimes nest contradictions or specialist_assessments inside
    the reasoning dict as list/dict values instead of keeping them as
    separate top-level fields. Convert to strings so validation passes.
    """
    if not isinstance(v, dict):
        return v
    return {k: str(val) if not isinstance(val, str) else val for k, val in v.items()}


class HypothesisReasoning(BaseModel):
    """Reasoning-only output from the hypothesis reasoning agent.

    Does NOT produce scores — those come from the logprob-based scorer.
    UNABLE_TO_DETERMINE is entropy-derived and has no LLM reasoning.

    Attributes:
        reasoning: Per-category explanation grounded in specialist findings.
            Keys: THIRD_PARTY_FRAUD, FIRST_PARTY_FRAUD, SCAM, DISPUTE.
        contradictions: Detected contradictions between CM allegations and
            evidence, consolidated from specialist findings + cross-cutting
            analysis.
        assessment_summary: Overall assessment of the current situation.
    """

    reasoning: dict[str, str] = {cat: "" for cat in _REASONING_CATEGORIES}
    contradictions: list[str] = []
    assessment_summary: str = ""

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning(cls, v: dict) -> dict[str, str]:
        return _coerce_reasoning_values(v)


class ReasoningNoteUpdate(BaseModel):
    """Incremental update to the hypothesis reasoning agent's working notes.

    Reasoning dict and assessment_summary are regenerated each turn.
    Contradictions are incrementally updated to prevent flickering.
    """

    reasoning: dict[str, str] = {cat: "" for cat in _REASONING_CATEGORIES}
    assessment_summary: str = ""

    add_contradictions: list[str] = Field(default_factory=list)
    remove_contradictions: list[str] = Field(default_factory=list)

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning(cls, v: dict) -> dict[str, str]:
        return _coerce_reasoning_values(v)


class HypothesisAssessment(BaseModel):
    """Combined output from logprob scorer + reasoning agent.

    Attributes:
        scores: Probability distribution across 5 investigation categories.
            Keys: THIRD_PARTY_FRAUD, FIRST_PARTY_FRAUD, SCAM, DISPUTE,
            UNABLE_TO_DETERMINE. Values between 0.0 and 1.0, summing to
            approximately 1.0. Produced by the logprob-based scorer.
        reasoning: Per-category explanation of the assessment (4 keys).
            UNABLE_TO_DETERMINE has no reasoning — its score is entropy-derived.
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
    reasoning: dict[str, str] = {cat: "" for cat in _REASONING_CATEGORIES}
    contradictions: list[str] = []
    assessment_summary: str = ""
    specialist_assessments: dict[str, SpecialistAssessment] = Field(
        default_factory=dict, exclude=True
    )


def merge_reasoning_notes(
    previous: HypothesisReasoning,
    update: ReasoningNoteUpdate,
) -> HypothesisReasoning:
    """Merge an incremental update into previous reasoning notes.

    Reasoning dict and assessment_summary are replaced wholesale.
    Contradictions are patched: removals first, then additions.
    """
    contradictions = _remove_by_substring(previous.contradictions, update.remove_contradictions)
    contradictions = _add_deduped(contradictions, update.add_contradictions)

    return HypothesisReasoning(
        reasoning=update.reasoning,
        contradictions=contradictions,
        assessment_summary=update.assessment_summary,
    )


# ---------------------------------------------------------------------------
# Reasoning agent prompt
# ---------------------------------------------------------------------------

HYPOTHESIS_REASONING_INSTRUCTIONS = """\
You are a hypothesis reasoning analyst for AMEX card dispute investigation.
You analyze assessments from three category specialists (Dispute, Scam,
Third-Party Fraud) and provide qualitative reasoning for each of the 5
investigation categories. You do NOT produce scores — scoring is handled
separately.

## Your Input

You receive the following context each turn:

1. **Specialist Assessments** — Three independent evidence analyses, each with
   policy-grounded reasoning, supporting/contradicting evidence, evidence gaps,
   and eligibility determinations. They evaluate their own category in isolation
   and cite specific policy documents.
2. **Auth Assessment** — Impersonation risk score, risk factors, and step-up
   auth recommendations from the authentication specialist.
3. **Accumulated Allegations** — What the cardmember claims, with detail types
   and extracted entities. Needed for cross-cutting first-party fraud detection.
4. **Current Hypothesis Scores** — The current probability distribution across
   the 5 categories. Use these as context for your reasoning.
5. **Previous Reasoning Trace** — Your own per-category reasoning from the
   last assessment turn. Use this to ground your analysis: identify what changed
   and explain how it shifts the assessment for each category.

## Reasoning Discipline

1. **Use previous reasoning as context.** Compare the current specialist outputs
   against the previous reasoning trace. Explain what changed since then and
   how new evidence shifts the assessment for each category.

2. **Weigh specialist evidence critically.** Specialist assessments reflect how
   well currently available evidence fits their category — they do not account
   for evidence that could be collected offline after case opening. Your
   reasoning should also be grounded in available evidence only. Consider
   whether another specialist's contradicting evidence undermines a category.
   Look at the full picture across all three assessments.

3. **Allegations are not evidence.** CM claims establish which hypotheses to
   investigate, but cannot by themselves support a category. Only system
   evidence, specialist-cited policy findings, or contradictions should
   inform your reasoning.

4. **Repetition is not new evidence.** If the previous reasoning trace already
   accounted for an allegation, restating it is not grounds for changed
   assessments.

   This applies equally to contradictions: a contradiction that was already
   reflected in previous reasoning has been accounted for. Only **new**
   contradictions (not seen in the previous reasoning trace) should shift
   assessments further. Re-citing the same contradictions across turns must
   not compound their effect.

5. **Contradictions are one signal among many.** Contradictions inform
   investigation direction but do not dominate it. Specialist evidence,
   evidence gaps, auth assessment, and the CM's narrative all carry weight.
   Even when contradictions are present, if specialist evidence or other
   signals point in a different direction, weigh them proportionally.

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
contradict. Do NOT infer first-party fraud from the absence of a complaint.

### Detection Signals (only when the CM has made an active claim)

- **All specialists find weak support — check WHY.** Look at the specialists'
  evidence_gaps and contradicting_evidence to distinguish:
  - If specialists cite **contradicting evidence** (evidence that actively
    doesn't fit their category), this points toward FIRST_PARTY_FRAUD —
    something happened, and no external party explains it.
  - If specialists cite **evidence gaps** (insufficient data, key items only
    available offline), this points toward uncertainty, not first-party fraud.

- **Contradicting evidence without external manipulator**: If specialists
  flag contradicting evidence (e.g., CM claims unauthorized but chip+PIN
  from enrolled device) AND the scam specialist finds no evidence of an
  external deceiver, this strongly indicates first-party fraud.

- **Auth assessment shows CM's own credentials**: If the auth assessment
  confirms the CM's device/credentials were used for the disputed
  transactions, note this as a first-party fraud indicator.

- **Inconsistencies across specialist findings**: If the dispute specialist
  finds the CM did receive goods, the fraud specialist finds the CM's
  device was used, and the scam specialist finds no external manipulator —
  the convergence of contradictions points to first-party fraud.

- **ANY allegation type can be first-party fraud**: A CM claiming fraud,
  dispute, or scam can all turn out to be first-party fraud. Never rule it
  out based on what the CM alleges.

### Distinguish from UNABLE_TO_DETERMINE

| Condition | Points toward |
|---|---|
| CM made a claim + evidence contradicts it | FIRST_PARTY_FRAUD |
| CM made a claim + evidence is missing/insufficient | UNABLE_TO_DETERMINE |
| CM has not made a clear claim | UNABLE_TO_DETERMINE |
| Specialists unsupported because of gaps, not contradictions | UNABLE_TO_DETERMINE |

## Output Format

Provide your assessment as structured output with:
- reasoning: dict with exactly 4 keys (THIRD_PARTY_FRAUD, FIRST_PARTY_FRAUD,
  SCAM, DISPUTE), each a brief explanation (1-3 sentences) grounded in
  specialist findings and evidence. UNABLE_TO_DETERMINE has no reasoning key
  — its score is derived from entropy, not LLM output.
- contradictions: list of detected contradictions between allegations and
  evidence (consolidate from specialist findings + your own analysis)
- assessment_summary: 2-4 sentence overall assessment
"""


# ---------------------------------------------------------------------------
# Agent instance
# ---------------------------------------------------------------------------

hypothesis_reasoning_agent = Agent(
    name="hypothesis_reasoning",
    instructions=HYPOTHESIS_REASONING_INSTRUCTIONS,
    output_type=AgentOutputSchema(HypothesisReasoning, strict_json_schema=False),
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
    openai_client: AsyncOpenAI,
    model: str = "gpt-4.1",
    previous_reasoning: HypothesisAssessment | None = None,
    specialist_deltas: dict[str, SpecialistNoteUpdate] | None = None,
) -> HypothesisAssessment:
    """Run the arbitrator to produce hypothesis scores and reasoning.

    Makes two parallel calls:
    1. Logprob scorer (raw OpenAI API) — forced-choice classification to get
       a grounded probability distribution over the 4 real categories, with
       UNABLE_TO_DETERMINE derived from entropy.
    2. Reasoning agent (Agents SDK) — qualitative analysis producing
       per-category reasoning, contradictions, and assessment summary.

    On subsequent turns, the reasoning agent outputs a ReasoningNoteUpdate
    (incremental delta) which is merged into the previous HypothesisReasoning.
    Specialist deltas are included in both the reasoning and logit prompts
    so both scorers can see what changed this turn.

    Args:
        specialist_assessments: Specialist outputs keyed by category
            (DISPUTE, SCAM, THIRD_PARTY_FRAUD).
        allegations_summary: Formatted allegations with types and entities.
        auth_summary: Auth assessment text (impersonation risk, risk factors).
        current_scores: Previous hypothesis scores (5-key dict).
        model_provider: LLM model provider for the reasoning agent.
        openai_client: AsyncOpenAI client for the logprob scorer.
        model: OpenAI model identifier for the logprob scorer.
        previous_reasoning: Full HypothesisAssessment from the last successful
            run. Provides the reasoning trace for incremental analysis.
        specialist_deltas: Raw SpecialistNoteUpdate objects from this turn,
            keyed by category. Empty dict or None on the first turn.

    Returns:
        HypothesisAssessment with logprob-derived scores and LLM-generated
        reasoning.

    Raises:
        RuntimeError: If the reasoning agent SDK call fails.
    """
    if specialist_deltas is None:
        specialist_deltas = {}

    # 1. Format reasoning agent user message
    scores_text = ", ".join(f"{k}: {v:.2f}" for k, v in current_scores.items())

    prev_reasoning_text = "First assessment — no prior reasoning."
    is_update = previous_reasoning is not None
    if is_update:
        reasoning_lines = [f"- {k}: {v}" for k, v in previous_reasoning.reasoning.items() if v]
        parts = []
        if reasoning_lines:
            parts.append("Per-category reasoning:\n" + "\n".join(reasoning_lines))
        if previous_reasoning.contradictions:
            numbered = [f"  {i}. {c}" for i, c in enumerate(previous_reasoning.contradictions, 1)]
            parts.append("Contradictions:\n" + "\n".join(numbered))
        if previous_reasoning.assessment_summary:
            parts.append(f"Summary: {previous_reasoning.assessment_summary}")
        if parts:
            prev_reasoning_text = "\n".join(parts)

    user_msg = (
        f"## Specialist Assessments\n\n"
        f"{_format_specialist_for_arbitrator(specialist_assessments, specialist_deltas)}\n\n"
        f"## Auth Assessment\n{auth_summary}\n\n"
        f"## Accumulated Allegations\n{allegations_summary}\n\n"
        f"## Current Hypothesis Scores\n{scores_text}\n\n"
        f"## Previous Reasoning Trace\n{prev_reasoning_text}"
    )

    # 2. Run logprob scorer and reasoning agent in parallel
    async def _run_reasoning() -> HypothesisReasoning:
        output_type = ReasoningNoteUpdate if is_update else HypothesisReasoning
        agent = Agent(
            name="hypothesis_reasoning",
            instructions=HYPOTHESIS_REASONING_INSTRUCTIONS,
            output_type=AgentOutputSchema(output_type, strict_json_schema=False),
        )
        result = await run_with_retry(
            agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        output = result.final_output
        if is_update:
            return merge_reasoning_notes(previous_reasoning, output)
        return output

    async def _run_logit() -> dict[str, float]:
        return await compute_logprob_scores(
            client=openai_client,
            model=model,
            specialist_assessments=specialist_assessments,
            allegations_summary=allegations_summary,
            auth_summary=auth_summary,
            specialist_deltas=specialist_deltas,
        )

    try:
        logit_scores, reasoning_result = await asyncio.gather(
            _run_logit(),
            _run_reasoning(),
        )
    except Exception as exc:
        raise RuntimeError(f"Hypothesis arbitrator failed: {exc}") from exc

    # 3. Combine into final assessment
    return HypothesisAssessment(
        scores=logit_scores,
        reasoning=reasoning_result.reasoning,
        contradictions=reasoning_result.contradictions,
        assessment_summary=reasoning_result.assessment_summary,
    )
