"""Note alignment evaluator — LLM-powered copilot output vs CCP notes comparison.

Compares the copilot's accumulated state (hypothesis scores, extracted allegations,
evidence collected, case eligibility) against the CCP's handwritten notes from
ground truth. Scores three sub-dimensions: key facts coverage, allegation alignment,
and category/action agreement.
"""

from __future__ import annotations

import json

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel

from agentic_fraud_servicing.evaluation.models import (
    EvaluationRun,
    NoteAlignmentResult,
)
from agentic_fraud_servicing.ingestion.redaction import redact_all_gsgt

# --- Output model for LLM scoring ---


class NoteAlignmentScore(BaseModel):
    """Structured output from the note alignment scoring agent.

    Attributes:
        facts_coverage: 0.0-1.0 — did copilot capture the same key facts?
        allegation_alignment: 0.0-1.0 — does copilot's understanding match CCP's?
        category_action: 0.0-1.0 — do category and recommended actions agree?
        overall: 0.0-1.0 — average of the three sub-scores.
        explanation: Brief rationale for each sub-score.
    """

    facts_coverage: float = 0.0
    allegation_alignment: float = 0.0
    category_action: float = 0.0
    overall: float = 0.0
    explanation: str = ""


# --- Agent instance ---

_ALIGNMENT_INSTRUCTIONS = """\
You are an evaluation specialist comparing a fraud copilot's accumulated output
against the Contact Center Professional's (CCP) handwritten case notes.

You will receive:
1. **Copilot Output**: The copilot's final state after processing the full call,
   including hypothesis scores, extracted allegations, evidence collected, and
   case eligibility assessments.
2. **CCP Notes**: The verbatim notes written by the human CCP during/after the call.

## Score three sub-dimensions (each 0.0 to 1.0):

### 1. Facts Coverage
Did the copilot capture the same key facts the CCP noted?
- **1.0**: All key facts (amounts, dates, merchants, card details, transaction
  specifics) mentioned in CCP notes are present in copilot output.
- **0.5**: Most key facts are captured but some significant details are missing.
- **0.0**: Major facts from CCP notes are absent from copilot output.

### 2. Allegation Alignment
Does the copilot's understanding of the customer's claims match the CCP's
characterization?
- **1.0**: Copilot's extracted allegations fully align with what the CCP
  documented about the customer's complaint.
- **0.5**: Partial alignment — copilot captured the main complaint but missed
  nuances or secondary claims noted by the CCP.
- **0.0**: Copilot's allegation understanding contradicts or misses the CCP's
  characterization entirely.

### 3. Category & Action Agreement
Does the copilot's investigation category (highest hypothesis) and case
eligibility match the CCP's conclusion and recommended actions?

This dimension requires **reasoning trace comparison**, not just outcome matching:

**When copilot and CCP agree on the outcome:**
- Verify the copilot's reasoning is coherent with the CCP's reasoning. A correct
  outcome reached through wrong reasoning is unreliable and should score lower.
- **1.0**: Category and actions agree AND the copilot's reasoning trace (hypothesis
  scores, evidence cited, risk flags) is coherent with the CCP's rationale.
- **0.7**: Category and actions agree but the copilot's reasoning diverges from
  the CCP's — the right answer was reached for partially wrong reasons (e.g.,
  copilot flagged correct category but based on different evidence than CCP cited).
- **0.5**: Category matches but actions differ, or the copilot's reasoning is
  incoherent even though the outcome happens to be correct.

**When copilot and CCP disagree on the outcome:**
- Compare both reasoning traces to identify WHERE they diverge and WHY.
- **0.3**: Category or actions partially overlap — copilot's reasoning shows
  some valid signals but missed key evidence that led CCP to a different conclusion.
- **0.0**: Complete disagreement — copilot's reasoning contradicts the CCP's
  assessment with no valid supporting evidence.

## Rules
1. Be lenient on wording — focus on semantic equivalence, not exact phrasing.
2. The CCP notes may use informal language or abbreviations.
3. The copilot output uses structured fields — compare substance, not format.
4. Set `overall` to the average of the three sub-scores.
5. Provide a comprehensive explanation (4-8 sentences) that covers:
   - Facts coverage: which key facts were captured or missed.
   - Allegation alignment: how well the copilot understood the customer's claims.
   - Reasoning coherence: whether the copilot's reasoning trace (hypothesis
     scores, evidence, risk flags) is consistent with the CCP's reasoning —
     especially flag cases where the correct outcome was reached through
     incorrect or incomplete reasoning.
   - When outcomes disagree: explain specifically what evidence or reasoning
     the copilot missed or misinterpreted that led to the wrong conclusion.
"""

_alignment_agent = Agent(
    name="note_alignment_scorer",
    instructions=_ALIGNMENT_INSTRUCTIONS,
    output_type=AgentOutputSchema(NoteAlignmentScore, strict_json_schema=False),
)


# --- Public function ---


async def evaluate_note_alignment(
    run: EvaluationRun,
    model_provider: ModelProvider,
) -> NoteAlignmentResult:
    """Evaluate copilot output alignment with CCP handwritten notes.

    Args:
        run: A completed EvaluationRun with copilot_final_state and ground_truth.
        model_provider: LLM model provider for inference.

    Returns:
        NoteAlignmentResult with per-dimension scores and overall score.
    """
    ccp_notes, _ = redact_all_gsgt(run.ground_truth.get("ccp_notes", ""))
    if not ccp_notes:
        return NoteAlignmentResult(
            facts_coverage_score=0.0,
            allegation_alignment_score=0.0,
            category_action_score=0.0,
            overall_score=0.0,
            explanation="No CCP notes in ground truth — skipped.",
        )

    copilot_summary = _build_copilot_summary(run)
    score = await _score_alignment(copilot_summary, ccp_notes, model_provider)

    return NoteAlignmentResult(
        facts_coverage_score=score.facts_coverage,
        allegation_alignment_score=score.allegation_alignment,
        category_action_score=score.category_action,
        overall_score=score.overall,
        explanation=score.explanation,
    )


def _build_copilot_summary(run: EvaluationRun) -> str:
    """Build a text summary of the copilot's accumulated output for LLM comparison."""
    parts: list[str] = []

    # Hypothesis scores
    scores = run.copilot_final_state.get("hypothesis_scores", {})
    if scores:
        scores_text = ", ".join(f"{k}: {v:.2f}" for k, v in scores.items())
        parts.append(f"## Hypothesis Scores\n{scores_text}")

    # Accumulated allegations
    all_allegations: list[str] = []
    for turn in run.turn_metrics:
        for allegation in turn.allegations_extracted:
            detail = allegation.get("detail_type", "")
            desc = allegation.get("description", "")
            if detail:
                all_allegations.append(f"- {detail}: {desc}" if desc else f"- {detail}")
    if all_allegations:
        parts.append("## Extracted Allegations\n" + "\n".join(all_allegations))

    # Evidence collected
    evidence = run.copilot_final_state.get("evidence_collected", [])
    if evidence:
        parts.append("## Evidence Collected\n" + "\n".join(f"- {e}" for e in evidence))

    # Last copilot suggestion (case eligibility, risk flags, running summary)
    last_suggestion = None
    for turn in reversed(run.turn_metrics):
        if turn.copilot_suggestion is not None:
            last_suggestion = turn.copilot_suggestion
            break

    if last_suggestion:
        if last_suggestion.get("case_eligibility"):
            parts.append(
                "## Case Eligibility\n"
                + json.dumps(last_suggestion["case_eligibility"], indent=2)
            )
        if last_suggestion.get("risk_flags"):
            parts.append(
                "## Risk Flags\n"
                + "\n".join(f"- {f}" for f in last_suggestion["risk_flags"])
            )
        if last_suggestion.get("running_summary"):
            parts.append(f"## Running Summary\n{last_suggestion['running_summary']}")

    # Impersonation risk
    imp_risk = run.copilot_final_state.get("impersonation_risk")
    if imp_risk is not None:
        parts.append(f"## Impersonation Risk\n{imp_risk:.2f}")

    return "\n\n".join(parts) if parts else "(No copilot output captured)"


async def _score_alignment(
    copilot_summary: str,
    ccp_notes: str,
    model_provider: ModelProvider,
) -> NoteAlignmentScore:
    """Use LLM agent to score alignment between copilot output and CCP notes.

    Returns:
        NoteAlignmentScore. Falls back to zero scores on failure.
    """
    user_msg = (
        f"## Copilot Output\n{copilot_summary}\n\n"
        f"## CCP Notes\n{ccp_notes}"
    )

    try:
        result = await Runner.run(
            _alignment_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        from agentic_fraud_servicing.copilot.langfuse_tracing import extract_http_error

        status_code, error_body = extract_http_error(exc)
        detail = f"HTTP {status_code}: {error_body[:200]}" if status_code else str(exc)
        return NoteAlignmentScore(
            explanation=f"LLM scoring failed ({detail})",
        )
