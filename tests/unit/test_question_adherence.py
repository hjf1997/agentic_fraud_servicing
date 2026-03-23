"""Tests for the question adherence evaluator — LLM-powered CCP incorporation check."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.evaluation.models import (
    EvaluationRun,
    QuestionAdherenceResult,
    TurnMetric,
)
from agentic_fraud_servicing.evaluation.question_adherence import (
    AdherenceScore,
    evaluate_question_adherence,
)


def _make_turn(
    turn_number: int,
    speaker: str,
    text: str = "",
    suggested_questions: list[str] | None = None,
) -> TurnMetric:
    """Build a TurnMetric with optional copilot suggestion."""
    suggestion = None
    if suggested_questions is not None:
        suggestion = {"suggested_questions": suggested_questions}
    return TurnMetric(
        turn_number=turn_number,
        speaker=speaker,
        text=text or f"Turn {turn_number}",
        latency_ms=500.0,
        copilot_suggestion=suggestion,
    )


def _make_run(turns: list[TurnMetric]) -> EvaluationRun:
    """Build an EvaluationRun from a list of TurnMetrics."""
    return EvaluationRun(
        scenario_name="test",
        ground_truth={},
        turn_metrics=turns,
        total_turns=len(turns),
        total_latency_ms=500.0 * len(turns),
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:01:00Z",
    )


# Patch target for Runner.run in the question_adherence module
_RUNNER_PATCH = "agentic_fraud_servicing.evaluation.question_adherence.Runner.run"


def _mock_adherence(score: float, explanation: str = "Test") -> AsyncMock:
    """Create a mock Runner.run result with an AdherenceScore output."""
    result = MagicMock()
    result.final_output = AdherenceScore(score=score, explanation=explanation)
    return AsyncMock(return_value=result)


class TestAdherenceScore:
    """AdherenceScore Pydantic model."""

    def test_defaults(self):
        s = AdherenceScore()
        assert s.score == 0.0
        assert s.explanation == ""

    def test_all_fields(self):
        s = AdherenceScore(score=0.75, explanation="Partially rephrased")
        assert s.score == 0.75
        assert s.explanation == "Partially rephrased"


class TestFullAdherence:
    """All CCP turns fully incorporate suggestions."""

    @pytest.mark.asyncio
    async def test_all_turns_score_one(self):
        turns = [
            _make_turn(1, "CARDMEMBER", suggested_questions=["Ask about date"]),
            _make_turn(2, "CCP", text="When did the transaction happen?"),
            _make_turn(3, "CARDMEMBER", suggested_questions=["Ask about merchant"]),
            _make_turn(4, "CCP", text="Which merchant was this at?"),
        ]
        run = _make_run(turns)

        with patch(_RUNNER_PATCH, _mock_adherence(1.0, "Fully incorporated")):
            result = await evaluate_question_adherence(run, MagicMock())

        assert isinstance(result, QuestionAdherenceResult)
        assert result.turns_with_suggestions == 2
        assert result.turns_with_adherence == 2
        assert result.overall_adherence_rate == 1.0
        assert len(result.per_turn_scores) == 2


class TestPartialAdherence:
    """Mix of adherence scores across turns."""

    @pytest.mark.asyncio
    async def test_mixed_scores(self):
        turns = [
            _make_turn(1, "CARDMEMBER", suggested_questions=["Q1"]),
            _make_turn(2, "CCP", text="Response 1"),
            _make_turn(3, "CARDMEMBER", suggested_questions=["Q2"]),
            _make_turn(4, "CCP", text="Response 2"),
            _make_turn(5, "CARDMEMBER", suggested_questions=["Q3"]),
            _make_turn(6, "CCP", text="Response 3"),
        ]
        run = _make_run(turns)

        # Return different scores for each call: 1.0, 0.0, 0.5
        call_count = 0
        scores = [1.0, 0.0, 0.5]

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            score = scores[call_count]
            call_count += 1
            result = MagicMock()
            result.final_output = AdherenceScore(score=score, explanation="Test")
            return result

        with patch(_RUNNER_PATCH, side_effect=mock_run):
            result = await evaluate_question_adherence(run, MagicMock())

        assert result.turns_with_suggestions == 3
        # 1.0 >= 0.5 (yes), 0.0 < 0.5 (no), 0.5 >= 0.5 (yes)
        assert result.turns_with_adherence == 2
        assert abs(result.overall_adherence_rate - 2 / 3) < 0.01


class TestNoSuggestions:
    """No turns have suggested questions."""

    @pytest.mark.asyncio
    async def test_returns_zero_rate(self):
        turns = [
            _make_turn(1, "CARDMEMBER"),
            _make_turn(2, "CCP", text="Hello"),
            _make_turn(3, "CARDMEMBER"),
        ]
        run = _make_run(turns)

        result = await evaluate_question_adherence(run, MagicMock())

        assert result.turns_with_suggestions == 0
        assert result.turns_with_adherence == 0
        assert result.overall_adherence_rate == 0.0
        assert result.per_turn_scores == []


class TestNoCcpFollowUp:
    """Suggestion exists but no CCP turn follows — should be skipped."""

    @pytest.mark.asyncio
    async def test_skipped_without_penalty(self):
        turns = [
            _make_turn(1, "CARDMEMBER", suggested_questions=["Ask about amount"]),
            # No CCP turn after this
        ]
        run = _make_run(turns)

        result = await evaluate_question_adherence(run, MagicMock())

        # Counted as having suggestions, but no per-turn score recorded
        assert result.turns_with_suggestions == 1
        assert result.turns_with_adherence == 0
        assert result.per_turn_scores == []
        assert result.overall_adherence_rate == 0.0


class TestPerTurnScoreContent:
    """Verify per_turn_scores dict structure."""

    @pytest.mark.asyncio
    async def test_contains_expected_keys(self):
        turns = [
            _make_turn(1, "CARDMEMBER", suggested_questions=["What merchant?"]),
            _make_turn(2, "CCP", text="Can you tell me the merchant name?"),
        ]
        run = _make_run(turns)

        with patch(_RUNNER_PATCH, _mock_adherence(0.8, "Rephrased well")):
            result = await evaluate_question_adherence(run, MagicMock())

        assert len(result.per_turn_scores) == 1
        entry = result.per_turn_scores[0]
        assert entry["turn_number"] == 1
        assert entry["suggested_questions"] == ["What merchant?"]
        assert entry["ccp_response_turn"] == 2
        assert entry["ccp_text"] == "Can you tell me the merchant name?"
        assert entry["adherence_score"] == 0.8
        assert entry["explanation"] == "Rephrased well"


class TestLlmFailureGraceful:
    """LLM failure on a turn should return 0.0 for that turn, not crash."""

    @pytest.mark.asyncio
    async def test_returns_zero_on_failure(self):
        turns = [
            _make_turn(1, "CARDMEMBER", suggested_questions=["Q1"]),
            _make_turn(2, "CCP", text="Response"),
        ]
        run = _make_run(turns)

        with patch(_RUNNER_PATCH, side_effect=Exception("LLM timeout")):
            result = await evaluate_question_adherence(run, MagicMock())

        assert result.turns_with_suggestions == 1
        assert result.turns_with_adherence == 0
        assert len(result.per_turn_scores) == 1
        assert result.per_turn_scores[0]["adherence_score"] == 0.0
        assert "LLM scoring failed" in result.per_turn_scores[0]["explanation"]
