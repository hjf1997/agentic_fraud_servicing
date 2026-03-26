"""Prediction accuracy evaluator — LLM-powered outcome mapping and comparison.

Maps freeform ground truth outcome tags to InvestigationCategory values via an
LLM agent, then compares against the copilot's highest hypothesis score. Skips
the LLM call when the outcome tag is already a valid InvestigationCategory value.
"""

from __future__ import annotations

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel

from agentic_fraud_servicing.evaluation.models import EvaluationRun, PredictionResult
from agentic_fraud_servicing.models.enums import (
    INVESTIGATION_CATEGORIES_REFERENCE,
    InvestigationCategory,
)

# Valid category values for fast-path check
_VALID_CATEGORIES = {c.value for c in InvestigationCategory}


# --- Output model for LLM mapping ---


class OutcomeMapping(BaseModel):
    """Structured output from the outcome mapping agent.

    Attributes:
        mapped_category: One of the 4 InvestigationCategory values.
        reasoning: Brief explanation of why this category was chosen.
    """

    mapped_category: str = ""
    reasoning: str = ""


# --- Agent instance ---

_MAPPING_INSTRUCTIONS = f"""\
You are a classification specialist. Your task is to map a freeform outcome
description to exactly one of the 4 investigation categories defined below.

{INVESTIGATION_CATEGORIES_REFERENCE}

## Rules

1. Read the outcome text carefully.
2. Determine which InvestigationCategory best matches the described outcome.
3. Output the category value as one of: THIRD_PARTY_FRAUD, FIRST_PARTY_FRAUD, SCAM, DISPUTE.
4. Provide a brief reasoning (1-2 sentences) for your choice.

## Examples

- "Invalid fraud claim" → FIRST_PARTY_FRAUD (claim was determined to be false)
- "Friendly fraud" → FIRST_PARTY_FRAUD (customer misrepresented the transaction)
- "Confirmed unauthorized" → THIRD_PARTY_FRAUD (transaction was genuinely unauthorized)
- "Scam victim" → SCAM (customer was deceived by external party)
- "Merchant issue" → DISPUTE (problem with merchant, not fraud)
- "Denied - customer made purchase" → FIRST_PARTY_FRAUD (customer authorized transaction)
"""

_mapping_agent = Agent(
    name="outcome_mapper",
    instructions=_MAPPING_INSTRUCTIONS,
    output_type=AgentOutputSchema(OutcomeMapping, strict_json_schema=False),
)


# --- Public functions ---


async def map_outcome_to_category(
    outcome_tag: str,
    model_provider: ModelProvider,
) -> tuple[str, str]:
    """Map a freeform outcome tag to an InvestigationCategory value.

    If the tag is already a valid InvestigationCategory value (e.g.,
    'FIRST_PARTY_FRAUD'), returns it directly without an LLM call.

    Args:
        outcome_tag: Freeform text describing the case outcome.
        model_provider: LLM model provider for inference.

    Returns:
        Tuple of (mapped_category, reasoning).

    Raises:
        RuntimeError: If the LLM agent call fails.
    """
    # Fast path: already a valid category value
    if outcome_tag in _VALID_CATEGORIES:
        return outcome_tag, ""

    try:
        result = await Runner.run(
            _mapping_agent,
            input=f"Map this outcome to an InvestigationCategory:\n\n{outcome_tag}",
            run_config=RunConfig(model_provider=model_provider),
        )
        mapping: OutcomeMapping = result.final_output
        return mapping.mapped_category, mapping.reasoning
    except Exception as exc:
        raise RuntimeError(f"Outcome mapping agent failed: {exc}") from exc


def _get_top_two(scores: dict[str, float]) -> tuple[str, float, float]:
    """Extract the top category and top-2 scores from a hypothesis scores dict.

    Returns:
        Tuple of (top_category, top_score, second_score).
    """
    if not scores:
        return "", 0.0, 0.0

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_category = sorted_items[0][0]
    top_score = sorted_items[0][1]
    second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
    return top_category, top_score, second_score


async def evaluate_prediction(
    run: EvaluationRun,
    model_provider: ModelProvider,
) -> PredictionResult:
    """Evaluate copilot prediction accuracy against ground truth.

    Extracts the copilot's highest hypothesis score and compares it against the
    ground truth investigation category (mapped from freeform tag if needed).

    Args:
        run: A completed EvaluationRun with turn_metrics and ground_truth.
        model_provider: LLM model provider for outcome tag mapping.

    Returns:
        PredictionResult with match status, confidence delta, and reasoning.
    """
    # Extract ground truth tag
    ground_truth_tag = run.ground_truth.get("investigation_category", "")
    if not ground_truth_tag:
        return PredictionResult(
            predicted_category="",
            ground_truth_category="",
            match=False,
            confidence_delta=0.0,
            reasoning="No ground truth investigation_category provided.",
        )

    # Map ground truth tag to InvestigationCategory
    try:
        mapped_category, mapping_reasoning = await map_outcome_to_category(
            ground_truth_tag, model_provider
        )
    except RuntimeError as exc:
        return PredictionResult(
            predicted_category="",
            ground_truth_category=ground_truth_tag,
            match=False,
            confidence_delta=0.0,
            reasoning=f"Failed to map ground truth: {exc}",
        )

    # Extract predicted category from the last assessed turn's hypothesis scores
    scores: dict[str, float] = {}
    assessed = [tm for tm in run.turn_metrics if tm.copilot_suggestion is not None]
    if assessed:
        scores = assessed[-1].hypothesis_scores
    if not scores and run.copilot_final_state:
        scores = run.copilot_final_state.get("hypothesis_scores", {})

    predicted_category, top_score, second_score = _get_top_two(scores)
    confidence_delta = top_score - second_score

    # Compare
    match = predicted_category == mapped_category

    # Build reasoning
    parts = []
    if mapping_reasoning:
        parts.append(f"Ground truth mapping: {mapping_reasoning}")
    if match:
        parts.append(
            f"Prediction '{predicted_category}' matches ground truth "
            f"'{mapped_category}' (delta={confidence_delta:.2f})."
        )
    else:
        parts.append(
            f"Prediction '{predicted_category}' does NOT match ground truth "
            f"'{mapped_category}' (delta={confidence_delta:.2f})."
        )
    reasoning = " ".join(parts)

    return PredictionResult(
        predicted_category=predicted_category,
        ground_truth_category=mapped_category,
        match=match,
        confidence_delta=confidence_delta,
        reasoning=reasoning,
    )
