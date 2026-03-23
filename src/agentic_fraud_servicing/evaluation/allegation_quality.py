"""Allegation quality evaluator — LLM-powered precision/recall/F1 assessment.

Computes precision, recall, and F1 for triage allegation extraction against
ground truth expected allegations. Uses an LLM agent for semantic matching
since AllegationDetailType values may not match exactly across ground truth
and extracted sets.
"""

from __future__ import annotations

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

from agentic_fraud_servicing.evaluation.models import (
    AllegationQualityResult,
    EvaluationRun,
)

# --- Output model for LLM matching ---


class AllegationMatchResult(BaseModel):
    """Structured output from the allegation matching agent.

    Attributes:
        matched: Ground truth allegations that have a matching extraction.
        missed: Ground truth allegations with no matching extraction.
        false_positives: Extracted allegations with no ground truth match.
        reasoning: Brief explanation of matching decisions.
    """

    matched: list[str] = Field(default_factory=list)
    missed: list[str] = Field(default_factory=list)
    false_positives: list[str] = Field(default_factory=list)
    reasoning: str = ""


# --- Agent instance ---

_MATCHING_INSTRUCTIONS = """\
You are an evaluation specialist assessing allegation extraction quality.

You will receive:
1. A list of **ground truth** allegation types expected for this case.
2. A list of **extracted** allegation types that the copilot triage agent produced.

Your task: determine which ground truth allegations were correctly extracted
(matched), which were missed, and which extractions are false positives.

## Matching Rules

- **Exact match**: If a ground truth value appears identically in the extracted
  list, it is a match (e.g., 'TRANSACTION_DISPUTE' matches 'TRANSACTION_DISPUTE').
- **Semantic equivalence**: If an extracted type clearly covers the same concept
  as a ground truth type, count it as a match. For example,
  'UNRECOGNIZED_TRANSACTION' matches 'TRANSACTION_DISPUTE' if the descriptions
  clearly overlap in meaning.
- **No loose matching**: Do NOT match types that are merely related but cover
  different aspects. For example, 'CARD_POSSESSION' should NOT match
  'LOST_STOLEN_CARD' unless the descriptions clearly overlap.
- Each ground truth allegation can match at most one extraction (no double counting).
- Each extraction can match at most one ground truth allegation.

## Output

- `matched`: List of ground truth allegation types that were correctly extracted.
- `missed`: List of ground truth allegation types NOT found in extractions.
- `false_positives`: List of extracted types NOT matching any ground truth.
- `reasoning`: Brief 2-3 sentence explanation of your matching decisions.
"""

_matching_agent = Agent(
    name="allegation_quality_matcher",
    instructions=_MATCHING_INSTRUCTIONS,
    output_type=AgentOutputSchema(AllegationMatchResult, strict_json_schema=False),
)


# --- Public function ---


async def evaluate_allegation_quality(
    run: EvaluationRun,
    model_provider: ModelProvider,
) -> AllegationQualityResult:
    """Evaluate allegation extraction quality against ground truth.

    Collects unique extracted allegation detail_types across all turns and
    uses an LLM to match them against the ground truth expected_allegations.

    Args:
        run: A completed EvaluationRun with turn_metrics and ground_truth.
        model_provider: LLM model provider for inference.

    Returns:
        AllegationQualityResult with precision, recall, F1, and detail lists.
    """
    ground_truth = run.ground_truth.get("expected_allegations", [])

    # Early return if no ground truth
    if not ground_truth:
        return AllegationQualityResult(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
        )

    # Collect unique extracted allegation detail_types across all turns
    extracted_set: set[str] = set()
    for turn in run.turn_metrics:
        for allegation in turn.allegations_extracted:
            detail_type = allegation.get("detail_type", "")
            if detail_type:
                extracted_set.add(detail_type)
    extracted = sorted(extracted_set)

    # No extractions — recall=0, precision=0
    if not extracted:
        return AllegationQualityResult(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            ground_truth_allegations=list(ground_truth),
            extracted_allegations=[],
            missed=list(ground_truth),
        )

    # Use LLM to match
    matched, missed, false_positives = await _match_allegations(
        ground_truth, extracted, model_provider
    )

    # Compute metrics
    precision = len(matched) / len(extracted) if extracted else 0.0
    recall = len(matched) / len(ground_truth) if ground_truth else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return AllegationQualityResult(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        ground_truth_allegations=list(ground_truth),
        extracted_allegations=extracted,
        matched=matched,
        missed=missed,
        false_positives=false_positives,
    )


async def _match_allegations(
    ground_truth: list[str],
    extracted: list[str],
    model_provider: ModelProvider,
) -> tuple[list[str], list[str], list[str]]:
    """Use LLM agent to match extracted allegations against ground truth.

    Returns:
        Tuple of (matched, missed, false_positives). Falls back to empty
        match on LLM failure (all ground truth missed, all extracted are FPs).
    """
    gt_text = "\n".join(f"- {a}" for a in ground_truth)
    ex_text = "\n".join(f"- {a}" for a in extracted)
    user_msg = f"## Ground Truth Allegations\n{gt_text}\n\n## Extracted Allegations\n{ex_text}"

    try:
        result = await Runner.run(
            _matching_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        output: AllegationMatchResult = result.final_output
        return output.matched, output.missed, output.false_positives
    except Exception:
        # Graceful degradation: treat all as missed / false positive
        return [], list(ground_truth), list(extracted)
