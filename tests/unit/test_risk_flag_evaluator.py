"""Tests for the risk flag timeliness evaluator — hybrid Python + LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.evaluation.models import (
    EvaluationRun,
    RiskFlagTimelinessResult,
    TurnMetric,
)
from agentic_fraud_servicing.evaluation.risk_flag_evaluator import (
    FlagMatchResult,
    evaluate_risk_flag_timeliness,
)


def _make_turn(
    turn_number: int,
    speaker: str = "CARDMEMBER",
    risk_flags: list[str] | None = None,
    retrieved_facts: list[str] | None = None,
    running_summary: str = "",
) -> TurnMetric:
    """Build a TurnMetric with optional risk flags and evidence data."""
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
    )


def _make_run(
    turns: list[TurnMetric],
    expected_risk_flags: list[str] | None = None,
) -> EvaluationRun:
    """Build an EvaluationRun with optional ground truth risk flags."""
    gt: dict = {}
    if expected_risk_flags is not None:
        gt["expected_risk_flags"] = expected_risk_flags
    return EvaluationRun(
        scenario_name="test",
        ground_truth=gt,
        turn_metrics=turns,
        total_turns=len(turns),
        total_latency_ms=500.0 * len(turns),
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:01:00Z",
    )


_RUNNER_PATCH = "agentic_fraud_servicing.evaluation.risk_flag_evaluator.run_with_retry"


def _mock_match(
    matches: list[dict],
    unmatched: list[str],
) -> AsyncMock:
    """Create a mock Runner.run result with a FlagMatchResult output."""
    result = MagicMock()
    result.final_output = FlagMatchResult(matches=matches, unmatched=unmatched)
    return AsyncMock(return_value=result)


class TestFlagMatchResult:
    """FlagMatchResult Pydantic model."""

    def test_defaults(self):
        r = FlagMatchResult()
        assert r.matches == []
        assert r.unmatched == []

    def test_all_fields(self):
        r = FlagMatchResult(
            matches=[{"expected_flag": "A", "raised_flag": "B", "raised_turn": 3}],
            unmatched=["C"],
        )
        assert len(r.matches) == 1
        assert r.matches[0]["raised_turn"] == 3
        assert r.unmatched == ["C"]


class TestAllFlagsPrompt:
    """All expected flags raised at the same turn as evidence — delay=0."""

    @pytest.mark.asyncio
    async def test_delay_zero(self):
        turns = [
            _make_turn(
                1,
                risk_flags=["chip+PIN contradiction detected"],
                retrieved_facts=["chip_pin auth event"],
            ),
            _make_turn(
                2,
                risk_flags=["high impersonation risk"],
                retrieved_facts=["device mismatch found"],
            ),
        ]
        run = _make_run(
            turns,
            expected_risk_flags=[
                "chip+PIN contradiction",
                "impersonation risk",
            ],
        )

        mock = _mock_match(
            matches=[
                {
                    "expected_flag": "chip+PIN contradiction",
                    "raised_flag": "chip+PIN contradiction detected",
                    "raised_turn": 1,
                },
                {
                    "expected_flag": "impersonation risk",
                    "raised_flag": "high impersonation risk",
                    "raised_turn": 2,
                },
            ],
            unmatched=[],
        )
        with patch(_RUNNER_PATCH, mock):
            result = await evaluate_risk_flag_timeliness(run, MagicMock())

        assert isinstance(result, RiskFlagTimelinessResult)
        assert result.flags_raised_count == 2
        assert result.flags_expected_count == 2
        assert result.average_delay_turns == 0.0
        assert len(result.per_flag_timing) == 2
        for timing in result.per_flag_timing:
            assert timing["delay_turns"] == 0


class TestLateFlags:
    """Some flags raised later than evidence became available."""

    @pytest.mark.asyncio
    async def test_positive_delay(self):
        turns = [
            # Turn 1: evidence appears
            _make_turn(1, risk_flags=[], retrieved_facts=["delivery proof signed"]),
            # Turn 2: still no flag
            _make_turn(2, risk_flags=[]),
            # Turn 3: flag finally raised
            _make_turn(3, risk_flags=["delivery contradiction"]),
        ]
        run = _make_run(turns, expected_risk_flags=["delivery contradiction"])

        mock = _mock_match(
            matches=[
                {
                    "expected_flag": "delivery contradiction",
                    "raised_flag": "delivery contradiction",
                    "raised_turn": 3,
                },
            ],
            unmatched=[],
        )
        with patch(_RUNNER_PATCH, mock):
            result = await evaluate_risk_flag_timeliness(run, MagicMock())

        assert result.flags_raised_count == 1
        assert result.flags_expected_count == 1
        # Evidence at turn 1, flag at turn 3 => delay = 2
        assert result.per_flag_timing[0]["delay_turns"] == 2
        assert result.per_flag_timing[0]["evidence_available_turn"] == 1
        assert result.average_delay_turns == 2.0


class TestNoExpectedFlags:
    """No expected risk flags in ground truth — returns zeros."""

    @pytest.mark.asyncio
    async def test_returns_zeros(self):
        turns = [_make_turn(1, risk_flags=["some flag"])]
        run = _make_run(turns, expected_risk_flags=[])

        # No LLM call needed
        result = await evaluate_risk_flag_timeliness(run, MagicMock())

        assert result.flags_expected_count == 0
        assert result.flags_raised_count == 0
        assert result.average_delay_turns == 0.0
        assert result.per_flag_timing == []

    @pytest.mark.asyncio
    async def test_missing_key_returns_zeros(self):
        """ground_truth dict has no expected_risk_flags key."""
        turns = [_make_turn(1)]
        run = _make_run(turns)

        result = await evaluate_risk_flag_timeliness(run, MagicMock())

        assert result.flags_expected_count == 0
        assert result.flags_raised_count == 0


class TestNoFlagsRaised:
    """Expected flags exist but copilot never raised any."""

    @pytest.mark.asyncio
    async def test_all_unmatched(self):
        turns = [_make_turn(1, risk_flags=[])]
        run = _make_run(turns, expected_risk_flags=["missed flag A", "missed flag B"])

        # No LLM call — early return path
        result = await evaluate_risk_flag_timeliness(run, MagicMock())

        assert result.flags_expected_count == 2
        assert result.flags_raised_count == 0
        assert result.average_delay_turns == 0.0
        assert result.per_flag_timing == []


class TestLlmFailureGraceful:
    """LLM failure should degrade gracefully — all flags unmatched."""

    @pytest.mark.asyncio
    async def test_wraps_exceptions(self):
        turns = [
            _make_turn(1, risk_flags=["some raised flag"]),
        ]
        run = _make_run(turns, expected_risk_flags=["expected flag"])

        with patch(_RUNNER_PATCH, side_effect=Exception("LLM timeout")):
            result = await evaluate_risk_flag_timeliness(run, MagicMock())

        # Graceful degradation: 0 matched
        assert result.flags_raised_count == 0
        assert result.flags_expected_count == 1
        assert result.average_delay_turns == 0.0
        assert result.per_flag_timing == []
