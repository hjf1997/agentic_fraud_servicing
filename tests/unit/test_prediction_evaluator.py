"""Tests for the prediction accuracy evaluator — LLM-powered outcome mapping."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.evaluation.models import EvaluationRun, PredictionResult, TurnMetric
from agentic_fraud_servicing.evaluation.prediction_evaluator import (
    OutcomeMapping,
    _get_top_two,
    evaluate_prediction,
    map_outcome_to_category,
)


_DUMMY_SUGGESTION = {"call_id": "test", "timestamp_ms": 0}


def _make_run(
    ground_truth_category: str = "FIRST_PARTY_FRAUD",
    hypothesis_scores: dict[str, float] | None = None,
    copilot_final_state: dict | None = None,
    num_turns: int = 3,
) -> EvaluationRun:
    """Helper to build an EvaluationRun with configurable hypothesis scores."""
    default_scores = {
        "THIRD_PARTY_FRAUD": 0.10,
        "FIRST_PARTY_FRAUD": 0.60,
        "SCAM": 0.15,
        "DISPUTE": 0.15,
    }
    scores = hypothesis_scores if hypothesis_scores is not None else default_scores

    metrics = [
        TurnMetric(
            turn_number=i + 1,
            speaker="CARDMEMBER",
            text=f"Turn {i + 1}",
            latency_ms=500.0,
            hypothesis_scores=scores if i == num_turns - 1 else {},
            copilot_suggestion=_DUMMY_SUGGESTION,
        )
        for i in range(num_turns)
    ]

    gt = {"investigation_category": ground_truth_category} if ground_truth_category else {}

    return EvaluationRun(
        scenario_name="test",
        ground_truth=gt,
        turn_metrics=metrics if num_turns > 0 else [],
        total_turns=num_turns,
        total_latency_ms=500.0 * num_turns,
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:01:00Z",
        copilot_final_state=copilot_final_state or {},
    )


# Patch target for Runner.run in the prediction_evaluator module
_RUNNER_PATCH = "agentic_fraud_servicing.evaluation.prediction_evaluator.run_with_retry"


def _mock_runner_result(mapped_category: str, reasoning: str = "Test reasoning"):
    """Create a mock Runner.run result with an OutcomeMapping output."""
    result = MagicMock()
    result.final_output = OutcomeMapping(mapped_category=mapped_category, reasoning=reasoning)
    return AsyncMock(return_value=result)


class TestOutcomeMapping:
    """OutcomeMapping Pydantic model."""

    def test_defaults(self):
        m = OutcomeMapping()
        assert m.mapped_category == ""
        assert m.reasoning == ""

    def test_all_fields(self):
        m = OutcomeMapping(mapped_category="FIRST_PARTY_FRAUD", reasoning="Friendly fraud case")
        assert m.mapped_category == "FIRST_PARTY_FRAUD"
        assert m.reasoning == "Friendly fraud case"


class TestMapOutcomeToCategory:
    """map_outcome_to_category function."""

    @pytest.mark.asyncio
    async def test_direct_category_skips_llm(self):
        """Valid InvestigationCategory value should skip LLM call."""
        provider = MagicMock()
        category, reasoning = await map_outcome_to_category("FIRST_PARTY_FRAUD", provider)
        assert category == "FIRST_PARTY_FRAUD"
        assert reasoning == ""

    @pytest.mark.asyncio
    async def test_all_valid_categories_skip_llm(self):
        """All 4 valid categories should skip LLM."""
        provider = MagicMock()
        for cat in ["THIRD_PARTY_FRAUD", "FIRST_PARTY_FRAUD", "SCAM", "DISPUTE"]:
            category, _ = await map_outcome_to_category(cat, provider)
            assert category == cat

    @pytest.mark.asyncio
    async def test_freeform_tag_uses_llm(self):
        """Freeform text should invoke LLM agent for mapping."""
        with patch(_RUNNER_PATCH, _mock_runner_result("FIRST_PARTY_FRAUD", "Friendly fraud")):
            category, reasoning = await map_outcome_to_category("Invalid fraud claim", MagicMock())
        assert category == "FIRST_PARTY_FRAUD"
        assert reasoning == "Friendly fraud"

    @pytest.mark.asyncio
    async def test_wraps_llm_exceptions(self):
        """LLM failures should be wrapped in RuntimeError."""
        with patch(_RUNNER_PATCH, side_effect=Exception("LLM down")):
            with pytest.raises(RuntimeError, match="Outcome mapping agent failed"):
                await map_outcome_to_category("some freeform tag", MagicMock())


class TestGetTopTwo:
    """_get_top_two helper function."""

    def test_normal_scores(self):
        scores = {"A": 0.6, "B": 0.2, "C": 0.1, "D": 0.1}
        top, s1, s2 = _get_top_two(scores)
        assert top == "A"
        assert s1 == 0.6
        assert s2 == 0.2

    def test_empty_scores(self):
        top, s1, s2 = _get_top_two({})
        assert top == ""
        assert s1 == 0.0
        assert s2 == 0.0

    def test_single_entry(self):
        top, s1, s2 = _get_top_two({"A": 0.8})
        assert top == "A"
        assert s1 == 0.8
        assert s2 == 0.0


class TestEvaluatePrediction:
    """evaluate_prediction async function."""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Predicted matches ground truth -> match=True."""
        run = _make_run(ground_truth_category="FIRST_PARTY_FRAUD")
        result = await evaluate_prediction(run, MagicMock())
        assert isinstance(result, PredictionResult)
        assert result.match is True
        assert result.predicted_category == "FIRST_PARTY_FRAUD"
        assert result.ground_truth_category == "FIRST_PARTY_FRAUD"

    @pytest.mark.asyncio
    async def test_mismatch(self):
        """Predicted differs from ground truth -> match=False."""
        run = _make_run(
            ground_truth_category="THIRD_PARTY_FRAUD",
            hypothesis_scores={
                "THIRD_PARTY_FRAUD": 0.10,
                "FIRST_PARTY_FRAUD": 0.60,
                "SCAM": 0.15,
                "DISPUTE": 0.15,
            },
        )
        result = await evaluate_prediction(run, MagicMock())
        assert result.match is False
        assert result.predicted_category == "FIRST_PARTY_FRAUD"
        assert result.ground_truth_category == "THIRD_PARTY_FRAUD"

    @pytest.mark.asyncio
    async def test_confidence_delta(self):
        """confidence_delta = top-1 minus top-2 score."""
        run = _make_run(
            hypothesis_scores={
                "THIRD_PARTY_FRAUD": 0.10,
                "FIRST_PARTY_FRAUD": 0.70,
                "SCAM": 0.10,
                "DISPUTE": 0.10,
            },
        )
        result = await evaluate_prediction(run, MagicMock())
        # delta = 0.70 - 0.10 = 0.60
        assert abs(result.confidence_delta - 0.60) < 0.01

    @pytest.mark.asyncio
    async def test_no_ground_truth(self):
        """Missing ground truth returns match=False with explanation."""
        run = _make_run(ground_truth_category="")
        result = await evaluate_prediction(run, MagicMock())
        assert result.match is False
        assert "No ground truth" in result.reasoning

    @pytest.mark.asyncio
    async def test_empty_turns_uses_copilot_final_state(self):
        """When no turn_metrics, falls back to copilot_final_state scores."""
        run = _make_run(
            ground_truth_category="FIRST_PARTY_FRAUD",
            num_turns=0,
            copilot_final_state={
                "hypothesis_scores": {
                    "THIRD_PARTY_FRAUD": 0.05,
                    "FIRST_PARTY_FRAUD": 0.80,
                    "SCAM": 0.10,
                    "DISPUTE": 0.05,
                }
            },
        )
        result = await evaluate_prediction(run, MagicMock())
        assert result.match is True
        assert result.predicted_category == "FIRST_PARTY_FRAUD"
        assert abs(result.confidence_delta - 0.70) < 0.01

    @pytest.mark.asyncio
    async def test_freeform_ground_truth_uses_llm(self):
        """Freeform ground truth tag invokes LLM for mapping."""
        run = _make_run(ground_truth_category="Friendly fraud - denied")
        with patch(_RUNNER_PATCH, _mock_runner_result("FIRST_PARTY_FRAUD", "Friendly fraud")):
            result = await evaluate_prediction(run, MagicMock())
        assert result.match is True
        assert result.ground_truth_category == "FIRST_PARTY_FRAUD"
        assert "Friendly fraud" in result.reasoning

    @pytest.mark.asyncio
    async def test_llm_failure_graceful(self):
        """LLM mapping failure returns match=False with error explanation."""
        run = _make_run(ground_truth_category="some freeform tag")
        with patch(_RUNNER_PATCH, side_effect=Exception("LLM timeout")):
            result = await evaluate_prediction(run, MagicMock())
        assert result.match is False
        assert "Failed to map ground truth" in result.reasoning
