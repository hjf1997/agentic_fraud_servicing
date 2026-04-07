"""Tests for the decision explanation evaluator — LLM-powered reasoning chain."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.evaluation.decision_explainer import (
    DecisionExplanationOutput,
    evaluate_decision_explanation,
)
from agentic_fraud_servicing.evaluation.models import (
    DecisionExplanation,
    EvaluationRun,
    TurnMetric,
)


def _make_turn(
    turn_number: int,
    speaker: str = "CARDMEMBER",
    hypothesis_scores: dict | None = None,
    risk_flags: list[str] | None = None,
    retrieved_facts: list[str] | None = None,
    running_summary: str = "",
    allegations: list[dict] | None = None,
) -> TurnMetric:
    """Build a TurnMetric with optional copilot suggestion data."""
    suggestion: dict | None = None
    if risk_flags is not None or retrieved_facts is not None or running_summary:
        suggestion = {
            "risk_flags": risk_flags or [],
            "retrieved_facts": retrieved_facts or [],
            "running_summary": running_summary,
            "suggested_questions": [],
        }
    return TurnMetric(
        turn_number=turn_number,
        speaker=speaker,
        text=f"Turn {turn_number}",
        latency_ms=500.0,
        copilot_suggestion=suggestion,
        hypothesis_scores=hypothesis_scores or {},
        allegations_extracted=allegations or [],
    )


def _make_run(
    turns: list[TurnMetric],
    ground_truth: dict | None = None,
    copilot_final_state: dict | None = None,
) -> EvaluationRun:
    """Build an EvaluationRun with optional ground truth and final state."""
    return EvaluationRun(
        scenario_name="test_scenario",
        ground_truth=ground_truth or {},
        turn_metrics=turns,
        total_turns=len(turns),
        total_latency_ms=500.0 * len(turns),
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:01:00Z",
        copilot_final_state=copilot_final_state or {},
    )


_RUNNER_PATCH = "agentic_fraud_servicing.evaluation.decision_explainer.run_with_retry"


def _mock_explanation(
    reasoning_chain: str = "The copilot correctly identified FIRST_PARTY_FRAUD.",
    influential_evidence: list[dict] | None = None,
    improvement_suggestions: list[str] | None = None,
    overall_quality_notes: str = "Good reasoning quality.",
) -> AsyncMock:
    """Create a mock Runner.run result with a DecisionExplanationOutput."""
    if influential_evidence is None:
        influential_evidence = [
            {
                "evidence": "chip+PIN auth event",
                "influence": "Increased FIRST_PARTY_FRAUD by 0.25",
                "description": "Contradicts card-not-present claim",
            },
            {
                "evidence": "signed delivery proof",
                "influence": "Increased FIRST_PARTY_FRAUD by 0.15",
                "description": "Contradicts goods-not-received claim",
            },
        ]
    if improvement_suggestions is None:
        improvement_suggestions = [
            "Weight auth contradictions more heavily in early turns.",
            "Add merchant familiarity detection to triage.",
        ]

    result = MagicMock()
    result.final_output = DecisionExplanationOutput(
        reasoning_chain=reasoning_chain,
        influential_evidence=influential_evidence,
        improvement_suggestions=improvement_suggestions,
        overall_quality_notes=overall_quality_notes,
    )
    return AsyncMock(return_value=result)


class TestDecisionExplanationOutput:
    """DecisionExplanationOutput Pydantic model."""

    def test_defaults(self):
        o = DecisionExplanationOutput()
        assert o.reasoning_chain == ""
        assert o.influential_evidence == []
        assert o.improvement_suggestions == []
        assert o.overall_quality_notes == ""

    def test_all_fields(self):
        o = DecisionExplanationOutput(
            reasoning_chain="chain",
            influential_evidence=[{"evidence": "x", "influence": "y", "description": "z"}],
            improvement_suggestions=["suggestion"],
            overall_quality_notes="notes",
        )
        assert o.reasoning_chain == "chain"
        assert len(o.influential_evidence) == 1
        assert o.improvement_suggestions == ["suggestion"]


class TestProducesDecisionExplanation:
    """evaluate_decision_explanation returns a populated DecisionExplanation."""

    @pytest.mark.asyncio
    async def test_all_fields_populated(self):
        turns = [
            _make_turn(
                1,
                hypothesis_scores={"THIRD_PARTY_FRAUD": 0.5, "FIRST_PARTY_FRAUD": 0.2},
                risk_flags=["chip+PIN contradiction"],
                retrieved_facts=["chip_pin auth event"],
                allegations=[
                    {"detail_type": "CARD_POSSESSION", "claim_description": "I had my card"}
                ],
            ),
            _make_turn(
                2,
                hypothesis_scores={"THIRD_PARTY_FRAUD": 0.3, "FIRST_PARTY_FRAUD": 0.5},
                risk_flags=["delivery contradiction"],
                retrieved_facts=["signed delivery proof"],
            ),
        ]
        run = _make_run(
            turns,
            ground_truth={"investigation_category": "FIRST_PARTY_FRAUD", "resolution": "denied"},
        )

        with patch(_RUNNER_PATCH, _mock_explanation()):
            result = await evaluate_decision_explanation(run, MagicMock())

        assert isinstance(result, DecisionExplanation)
        assert result.reasoning_chain != ""
        assert len(result.influential_evidence) > 0
        assert len(result.improvement_suggestions) > 0
        assert result.overall_quality_notes != ""


class TestReasoningChainReferencesGroundTruth:
    """The reasoning chain should reference the ground truth category."""

    @pytest.mark.asyncio
    async def test_reasoning_references_ground_truth(self):
        turns = [
            _make_turn(1, hypothesis_scores={"FIRST_PARTY_FRAUD": 0.6}),
        ]
        run = _make_run(
            turns,
            ground_truth={"investigation_category": "FIRST_PARTY_FRAUD"},
        )

        mock = _mock_explanation(
            reasoning_chain="The copilot correctly identified FIRST_PARTY_FRAUD at turn 2."
        )
        with patch(_RUNNER_PATCH, mock):
            result = await evaluate_decision_explanation(run, MagicMock())

        assert "FIRST_PARTY_FRAUD" in result.reasoning_chain


class TestInfluentialEvidenceCapped:
    """influential_evidence is capped at 3 entries even if LLM returns more."""

    @pytest.mark.asyncio
    async def test_max_three_entries(self):
        turns = [_make_turn(1, hypothesis_scores={"FIRST_PARTY_FRAUD": 0.5})]
        run = _make_run(turns, ground_truth={"investigation_category": "FIRST_PARTY_FRAUD"})

        # LLM returns 5 entries — should be capped to 3
        evidence = [
            {"evidence": f"ev{i}", "influence": f"inf{i}", "description": f"desc{i}"}
            for i in range(5)
        ]
        mock = _mock_explanation(influential_evidence=evidence)
        with patch(_RUNNER_PATCH, mock):
            result = await evaluate_decision_explanation(run, MagicMock())

        assert len(result.influential_evidence) == 3


class TestImprovementSuggestionsNonEmpty:
    """improvement_suggestions should be non-empty when LLM provides them."""

    @pytest.mark.asyncio
    async def test_suggestions_present(self):
        turns = [_make_turn(1, hypothesis_scores={"THIRD_PARTY_FRAUD": 0.8})]
        run = _make_run(turns, ground_truth={"investigation_category": "THIRD_PARTY_FRAUD"})

        with patch(_RUNNER_PATCH, _mock_explanation()):
            result = await evaluate_decision_explanation(run, MagicMock())

        assert len(result.improvement_suggestions) >= 1


class TestEmptyRunGraceful:
    """An empty EvaluationRun should return a minimal DecisionExplanation."""

    @pytest.mark.asyncio
    async def test_empty_turns(self):
        run = _make_run([], ground_truth={"investigation_category": "DISPUTE"})

        # No LLM call needed
        result = await evaluate_decision_explanation(run, MagicMock())

        assert isinstance(result, DecisionExplanation)
        assert "No turn data" in result.reasoning_chain
        assert result.influential_evidence == []
        assert result.improvement_suggestions == []
        assert result.overall_quality_notes != ""


class TestLlmFailureGraceful:
    """LLM failure should degrade gracefully — returns empty explanation."""

    @pytest.mark.asyncio
    async def test_wraps_exceptions(self):
        turns = [_make_turn(1, hypothesis_scores={"FIRST_PARTY_FRAUD": 0.5})]
        run = _make_run(turns, ground_truth={"investigation_category": "FIRST_PARTY_FRAUD"})

        with patch(_RUNNER_PATCH, side_effect=Exception("LLM timeout")):
            result = await evaluate_decision_explanation(run, MagicMock())

        assert isinstance(result, DecisionExplanation)
        assert result.reasoning_chain == ""
        assert result.influential_evidence == []
        assert result.improvement_suggestions == []
        assert "LLM error" in result.overall_quality_notes
