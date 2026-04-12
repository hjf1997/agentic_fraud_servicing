"""Tests for the allegation quality evaluator — LLM-powered precision/recall/F1."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.evaluation.allegation_quality import (
    AllegationMatchResult,
    evaluate_allegation_quality,
)
from agentic_fraud_servicing.evaluation.models import (
    AllegationQualityResult,
    EvaluationRun,
    TurnMetric,
)

_DUMMY_SUGGESTION = {"call_id": "test", "timestamp_ms": 0}


def _make_turn(
    turn_number: int,
    speaker: str = "CARDMEMBER",
    allegations: list[dict] | None = None,
) -> TurnMetric:
    """Build a TurnMetric with optional extracted allegations."""
    return TurnMetric(
        turn_number=turn_number,
        speaker=speaker,
        text=f"Turn {turn_number}",
        latency_ms=500.0,
        allegations_extracted=allegations or [],
        copilot_suggestion=_DUMMY_SUGGESTION,
    )


def _make_run(
    turns: list[TurnMetric],
    expected_allegations: list[str] | None = None,
) -> EvaluationRun:
    """Build an EvaluationRun with optional ground truth allegations."""
    gt = {}
    if expected_allegations is not None:
        gt["expected_allegations"] = expected_allegations
    return EvaluationRun(
        scenario_name="test",
        ground_truth=gt,
        turn_metrics=turns,
        total_turns=len(turns),
        total_latency_ms=500.0 * len(turns),
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:01:00Z",
    )


_RUNNER_PATCH = "agentic_fraud_servicing.evaluation.allegation_quality.run_with_retry"


def _mock_match(
    matched: list[str],
    missed: list[str],
    false_positives: list[str],
    reasoning: str = "Test matching",
) -> AsyncMock:
    """Create a mock Runner.run result with an AllegationMatchResult output."""
    result = MagicMock()
    result.final_output = AllegationMatchResult(
        matched=matched,
        missed=missed,
        false_positives=false_positives,
        reasoning=reasoning,
    )
    return AsyncMock(return_value=result)


class TestAllegationMatchResult:
    """AllegationMatchResult Pydantic model."""

    def test_defaults(self):
        r = AllegationMatchResult()
        assert r.matched == []
        assert r.missed == []
        assert r.false_positives == []
        assert r.reasoning == ""

    def test_all_fields(self):
        r = AllegationMatchResult(
            matched=["A"],
            missed=["B"],
            false_positives=["C"],
            reasoning="Matched A, missed B, FP C",
        )
        assert r.matched == ["A"]
        assert r.missed == ["B"]
        assert r.false_positives == ["C"]
        assert "Matched A" in r.reasoning


class TestPerfectMatch:
    """All ground truth found, no false positives."""

    @pytest.mark.asyncio
    async def test_precision_recall_f1_all_one(self):
        turns = [
            _make_turn(1, allegations=[{"detail_type": "TRANSACTION_DISPUTE"}]),
            _make_turn(2, allegations=[{"detail_type": "CARD_NOT_PRESENT_FRAUD"}]),
        ]
        gt = ["TRANSACTION_DISPUTE", "CARD_NOT_PRESENT_FRAUD"]
        run = _make_run(turns, expected_allegations=gt)

        mock = _mock_match(
            matched=["TRANSACTION_DISPUTE", "CARD_NOT_PRESENT_FRAUD"],
            missed=[],
            false_positives=[],
        )
        with patch(_RUNNER_PATCH, mock):
            result = await evaluate_allegation_quality(run, MagicMock())

        assert isinstance(result, AllegationQualityResult)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0
        assert result.matched == ["TRANSACTION_DISPUTE", "CARD_NOT_PRESENT_FRAUD"]
        assert result.missed == []
        assert result.false_positives == []


class TestPartialMatch:
    """Some missed, some false positives."""

    @pytest.mark.asyncio
    async def test_partial_metrics(self):
        turns = [
            _make_turn(1, allegations=[{"detail_type": "TRANSACTION_DISPUTE"}]),
            _make_turn(2, allegations=[{"detail_type": "CARD_POSSESSION"}]),
        ]
        gt = ["TRANSACTION_DISPUTE", "LOST_STOLEN_CARD"]
        run = _make_run(turns, expected_allegations=gt)

        # 1 matched, 1 missed, 1 false positive
        mock = _mock_match(
            matched=["TRANSACTION_DISPUTE"],
            missed=["LOST_STOLEN_CARD"],
            false_positives=["CARD_POSSESSION"],
        )
        with patch(_RUNNER_PATCH, mock):
            result = await evaluate_allegation_quality(run, MagicMock())

        # precision = 1/2, recall = 1/2
        assert result.precision == 0.5
        assert result.recall == 0.5
        # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        assert abs(result.f1_score - 0.5) < 0.001


class TestNoGroundTruth:
    """No ground truth allegations — returns zeros."""

    @pytest.mark.asyncio
    async def test_returns_zeros(self):
        turns = [
            _make_turn(1, allegations=[{"detail_type": "TRANSACTION_DISPUTE"}]),
        ]
        run = _make_run(turns, expected_allegations=[])

        # No LLM call needed
        result = await evaluate_allegation_quality(run, MagicMock())

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0
        assert result.ground_truth_allegations == []

    @pytest.mark.asyncio
    async def test_missing_key_returns_zeros(self):
        """ground_truth dict has no expected_allegations key at all."""
        turns = [_make_turn(1, allegations=[{"detail_type": "X"}])]
        run = _make_run(turns)  # No expected_allegations in ground_truth

        result = await evaluate_allegation_quality(run, MagicMock())

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0


class TestNoExtractions:
    """Ground truth exists but nothing was extracted."""

    @pytest.mark.asyncio
    async def test_recall_zero(self):
        turns = [_make_turn(1)]  # No allegations extracted
        gt = ["TRANSACTION_DISPUTE", "LOST_STOLEN_CARD"]
        run = _make_run(turns, expected_allegations=gt)

        result = await evaluate_allegation_quality(run, MagicMock())

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0
        assert result.missed == gt
        assert result.extracted_allegations == []


class TestF1Computation:
    """Verify F1 = 2 * P * R / (P + R) with specific numbers."""

    @pytest.mark.asyncio
    async def test_f1_with_asymmetric_pr(self):
        # 3 ground truth, 2 extracted, 1 matched
        turns = [
            _make_turn(1, allegations=[{"detail_type": "A"}]),
            _make_turn(2, allegations=[{"detail_type": "B"}]),
        ]
        gt = ["A", "C", "D"]
        run = _make_run(turns, expected_allegations=gt)

        mock = _mock_match(
            matched=["A"],
            missed=["C", "D"],
            false_positives=["B"],
        )
        with patch(_RUNNER_PATCH, mock):
            result = await evaluate_allegation_quality(run, MagicMock())

        # precision = 1/2 = 0.5, recall = 1/3 ≈ 0.333
        assert abs(result.precision - 0.5) < 0.001
        assert abs(result.recall - 1 / 3) < 0.001
        # F1 = 2 * 0.5 * 0.333 / (0.5 + 0.333) = 0.4
        expected_f1 = 2 * 0.5 * (1 / 3) / (0.5 + 1 / 3)
        assert abs(result.f1_score - expected_f1) < 0.001


class TestLlmFailureGraceful:
    """LLM failure should degrade gracefully — all missed, all FPs."""

    @pytest.mark.asyncio
    async def test_wraps_exceptions(self):
        turns = [
            _make_turn(1, allegations=[{"detail_type": "A"}]),
        ]
        gt = ["B"]
        run = _make_run(turns, expected_allegations=gt)

        with patch(_RUNNER_PATCH, side_effect=Exception("LLM timeout")):
            result = await evaluate_allegation_quality(run, MagicMock())

        # Graceful degradation: 0 matched, all missed, all FPs
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0
        assert result.missed == ["B"]
        assert result.false_positives == ["A"]
