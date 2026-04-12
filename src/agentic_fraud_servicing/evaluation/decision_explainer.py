"""Decision explanation evaluator — LLM-powered reasoning chain generation.

Uses an LLM agent (acting as a senior fraud analyst) to review the copilot's
performance on a case, produce a reasoning chain explaining how evidence drove
hypothesis scores, identify the most influential evidence, and suggest concrete
improvements.
"""

from __future__ import annotations

from agents import Agent, AgentOutputSchema, ModelProvider
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

from agentic_fraud_servicing.evaluation.models import (
    DecisionExplanation,
    EvaluationRun,
    NoteAlignmentResult,
)
from agentic_fraud_servicing.providers.retry import run_with_retry

# --- Structured output model for the LLM agent ---


class DecisionExplanationOutput(BaseModel):
    """Structured output from the decision explanation agent.

    Attributes:
        reasoning_chain: Narrative explanation of how evidence drove scores.
        influential_evidence: Top 3 evidence-to-decision links.
        improvement_suggestions: 2-4 actionable copilot improvement ideas.
        overall_quality_notes: Brief quality assessment.
    """

    reasoning_chain: str = ""
    influential_evidence: list[dict] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    overall_quality_notes: str = ""


# --- Agent instance ---

_DECISION_EXPLAINER_INSTRUCTIONS = """\
You are a senior fraud analyst reviewing an AI copilot's performance on a
real fraud servicing call. Your job is to explain HOW the copilot reached its
final hypothesis scores, identify the MOST INFLUENTIAL evidence, and suggest
concrete IMPROVEMENTS.

You will receive a comprehensive case summary including:
- Ground truth category and resolution
- Final hypothesis scores
- All allegations extracted during the call
- Risk flags raised
- Turn-by-turn hypothesis score evolution
- Key evidence retrieved
- CCP Note Alignment scores (how well copilot output matched the CCP's notes)

## Output Requirements

### reasoning_chain
Write a narrative (3-8 sentences) explaining the copilot's reasoning path:
- Reference specific turn numbers (e.g., "At turn 3, the chip+PIN auth event...")
- Note when hypothesis scores shifted and why
- Identify turning points where evidence changed the assessment
- Compare the copilot's final conclusion against the ground truth

### influential_evidence
List the top 3 most impactful evidence-to-decision links. Each entry must have:
- `evidence`: The evidence item (e.g., "chip+PIN auth event", "signed delivery proof")
- `influence`: How it influenced scoring (e.g., "Increased FIRST_PARTY_FRAUD by 0.25")
- `description`: Brief explanation of why this evidence mattered

### improvement_suggestions
Provide 2-4 concrete, actionable suggestions. Consider:
- Evidence timing: Were key pieces of evidence used promptly?
- CCP note alignment gaps: If provided, identify where the copilot's output
  diverged from the CCP's notes — missing facts, misaligned allegations, or
  disagreements on category/action. Suggest how the copilot could better
  capture what the CCP documented.
- Hypothesis accuracy: Did scores converge to the correct category?

Examples:
- "The chip+PIN auth event was retrieved at turn 5 but did not increase
  FIRST_PARTY_FRAUD until turn 8 — consider weighting auth contradictions
  more heavily in early turns."
- "The copilot missed the merchant familiarity signal in the cardmember's
  phrasing at turn 4 — add merchant familiarity detection to triage."
- "CCP notes mention a specific transaction amount ($499.99) that the copilot
  did not capture in its allegations — improve fact extraction to include
  transaction amounts from cardmember statements."

### overall_quality_notes
A brief (1-3 sentences) assessment of the copilot's reasoning quality.
Was the final decision correct? Was evidence used effectively?
"""

_decision_explainer_agent = Agent(
    name="decision_explainer",
    instructions=_DECISION_EXPLAINER_INSTRUCTIONS,
    output_type=AgentOutputSchema(DecisionExplanationOutput, strict_json_schema=False),
)


# --- Public function ---


async def evaluate_decision_explanation(
    run: EvaluationRun,
    model_provider: ModelProvider,
    note_alignment: NoteAlignmentResult | None = None,
) -> DecisionExplanation:
    """Generate a reasoning chain explaining how evidence drove hypothesis scores.

    Builds a comprehensive context from the EvaluationRun and uses an LLM agent
    to produce a narrative explanation, identify influential evidence, and suggest
    improvements.

    Args:
        run: A completed EvaluationRun with turn_metrics and ground_truth.
        model_provider: LLM model provider for inference.
        note_alignment: Optional CCP note alignment results. When provided,
            the LLM considers alignment gaps in its improvement suggestions.

    Returns:
        DecisionExplanation with reasoning chain, evidence links, and suggestions.
    """
    # Handle empty or minimal runs gracefully
    if not run.turn_metrics:
        return DecisionExplanation(
            reasoning_chain="No turn data available for analysis.",
            influential_evidence=[],
            improvement_suggestions=[],
            overall_quality_notes="Empty evaluation run — no copilot output to assess.",
        )

    context = _build_context(run, note_alignment)

    try:
        result = await run_with_retry(
            _decision_explainer_agent,
            input=context,
            run_config=RunConfig(model_provider=model_provider),
        )
        output: DecisionExplanationOutput = result.final_output

        # Cap influential_evidence at 3 entries
        evidence = output.influential_evidence[:3]

        return DecisionExplanation(
            reasoning_chain=output.reasoning_chain,
            influential_evidence=evidence,
            improvement_suggestions=output.improvement_suggestions,
            overall_quality_notes=output.overall_quality_notes,
        )
    except Exception as exc:
        # Graceful degradation on LLM failure
        from agentic_fraud_servicing.copilot.langfuse_tracing import extract_http_error

        status_code, error_body = extract_http_error(exc)
        detail = f"HTTP {status_code}: {error_body[:200]}" if status_code else str(exc)
        print(f"[decision_explainer] LLM failed ({detail})", file=__import__("sys").stderr)
        return DecisionExplanation(
            reasoning_chain="",
            influential_evidence=[],
            improvement_suggestions=[],
            overall_quality_notes=f"Decision explanation failed ({detail}).",
        )


def _build_context(run: EvaluationRun, note_alignment: NoteAlignmentResult | None = None) -> str:
    """Build comprehensive context string from the EvaluationRun for the LLM."""
    sections: list[str] = []

    # Ground truth
    gt = run.ground_truth
    gt_category = gt.get("investigation_category", "unknown")
    gt_resolution = gt.get("resolution", "unknown")
    sections.append(f"## Ground Truth\n- Category: {gt_category}\n- Resolution: {gt_resolution}")

    # Final hypothesis scores
    final_scores = run.copilot_final_state.get("hypothesis_scores", {})
    if not final_scores and run.turn_metrics:
        final_scores = run.turn_metrics[-1].hypothesis_scores
    if final_scores:
        scores_text = ", ".join(f"{k}: {v:.2f}" for k, v in final_scores.items())
        sections.append(f"## Final Hypothesis Scores\n{scores_text}")

    # Allegations extracted across all turns
    allegations: list[str] = []
    for turn in run.turn_metrics:
        for a in turn.allegations_extracted:
            detail = a.get("detail_type", a.get("claim_type", "unknown"))
            desc = a.get("claim_description", a.get("description", ""))
            allegations.append(f"- Turn {turn.turn_number}: [{detail}] {desc}")
    if allegations:
        sections.append("## Allegations Extracted\n" + "\n".join(allegations))

    # Risk flags (deduplicated, with first turn)
    flag_turns: dict[str, int] = {}
    for turn in run.turn_metrics:
        suggestion = turn.copilot_suggestion
        if suggestion is None:
            continue
        for flag in suggestion.get("risk_flags", []):
            if isinstance(flag, str) and flag not in flag_turns:
                flag_turns[flag] = turn.turn_number
    if flag_turns:
        flags_text = "\n".join(f"- Turn {t}: {f}" for f, t in flag_turns.items())
        sections.append(f"## Risk Flags Raised\n{flags_text}")

    # Hypothesis score evolution
    score_lines: list[str] = []
    for turn in run.turn_metrics:
        if turn.hypothesis_scores:
            scores = ", ".join(f"{k}: {v:.2f}" for k, v in turn.hypothesis_scores.items())
            score_lines.append(f"- Turn {turn.turn_number}: {scores}")
    if score_lines:
        sections.append("## Hypothesis Score Evolution\n" + "\n".join(score_lines))

    # Key evidence from retrieved_facts
    evidence_items: list[str] = []
    seen: set[str] = set()
    for turn in run.turn_metrics:
        suggestion = turn.copilot_suggestion
        if suggestion is None:
            continue
        for fact in suggestion.get("retrieved_facts", []):
            fact_str = str(fact)
            if fact_str not in seen:
                seen.add(fact_str)
                evidence_items.append(f"- Turn {turn.turn_number}: {fact_str[:200]}")
    if evidence_items:
        sections.append("## Key Evidence Retrieved\n" + "\n".join(evidence_items))

    # CCP Note Alignment (when available)
    if note_alignment and note_alignment.explanation:
        na_lines = [
            f"- Facts Coverage: {note_alignment.facts_coverage}",
            f"- Allegation Alignment: {note_alignment.allegation_alignment}",
            f"- Category & Action: {note_alignment.category_action}",
            f"- Overall: {note_alignment.overall}",
            f"- Explanation: {note_alignment.explanation}",
        ]
        sections.append("## CCP Note Alignment\n" + "\n".join(na_lines))

    return "\n\n".join(sections)
