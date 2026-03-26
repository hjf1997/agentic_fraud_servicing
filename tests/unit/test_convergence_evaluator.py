"""Tests for evaluation/convergence_evaluator.py."""

from agentic_fraud_servicing.evaluation.convergence_evaluator import evaluate_convergence
from agentic_fraud_servicing.evaluation.models import EvaluationRun, TurnMetric


_DUMMY_SUGGESTION = {"call_id": "test", "timestamp_ms": 0}


def _make_run(
    ground_truth: dict,
    score_sequence: list[dict[str, float]],
) -> EvaluationRun:
    """Build a minimal EvaluationRun from a sequence of hypothesis score dicts."""
    metrics = [
        TurnMetric(
            turn_number=i + 1,
            speaker="CARDMEMBER",
            text=f"Turn {i + 1}",
            latency_ms=100.0,
            hypothesis_scores=scores,
            copilot_suggestion=_DUMMY_SUGGESTION,
        )
        for i, scores in enumerate(score_sequence)
    ]
    return EvaluationRun(
        scenario_name="test",
        ground_truth=ground_truth,
        turn_metrics=metrics,
        total_turns=len(metrics),
        total_latency_ms=100.0 * len(metrics),
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:01:00Z",
    )


class TestConvergesEarly:
    """Correct category is highest from turn 1."""

    def test_convergence_turn_is_first(self):
        run = _make_run(
            {"investigation_category": "THIRD_PARTY_FRAUD"},
            [
                {"THIRD_PARTY_FRAUD": 0.6, "FIRST_PARTY_FRAUD": 0.2, "SCAM": 0.1, "DISPUTE": 0.1},
                {"THIRD_PARTY_FRAUD": 0.7, "FIRST_PARTY_FRAUD": 0.1, "SCAM": 0.1, "DISPUTE": 0.1},
                {
                    "THIRD_PARTY_FRAUD": 0.8,
                    "FIRST_PARTY_FRAUD": 0.1,
                    "SCAM": 0.05,
                    "DISPUTE": 0.05,
                },
            ],
        )
        result = evaluate_convergence(run)
        assert result.convergence_turn == 1
        assert result.convergence_ratio is not None
        assert abs(result.convergence_ratio - 1 / 3) < 0.01

    def test_turn_scores_recorded(self):
        run = _make_run(
            {"investigation_category": "SCAM"},
            [{"THIRD_PARTY_FRAUD": 0.1, "FIRST_PARTY_FRAUD": 0.1, "SCAM": 0.7, "DISPUTE": 0.1}],
        )
        result = evaluate_convergence(run)
        assert len(result.turn_scores) == 1
        assert result.turn_scores[0]["SCAM"] == 0.7


class TestConvergesLate:
    """Correct category becomes highest partway through."""

    def test_convergence_at_turn_3(self):
        run = _make_run(
            {"investigation_category": "FIRST_PARTY_FRAUD"},
            [
                {"THIRD_PARTY_FRAUD": 0.5, "FIRST_PARTY_FRAUD": 0.2, "SCAM": 0.2, "DISPUTE": 0.1},
                {"THIRD_PARTY_FRAUD": 0.4, "FIRST_PARTY_FRAUD": 0.3, "SCAM": 0.2, "DISPUTE": 0.1},
                {"THIRD_PARTY_FRAUD": 0.2, "FIRST_PARTY_FRAUD": 0.5, "SCAM": 0.2, "DISPUTE": 0.1},
                {"THIRD_PARTY_FRAUD": 0.1, "FIRST_PARTY_FRAUD": 0.6, "SCAM": 0.2, "DISPUTE": 0.1},
            ],
        )
        result = evaluate_convergence(run)
        assert result.convergence_turn == 3
        assert result.total_turns == 4


class TestNeverConverges:
    """Correct category never becomes/stays highest."""

    def test_convergence_turn_is_none(self):
        run = _make_run(
            {"investigation_category": "DISPUTE"},
            [
                {"THIRD_PARTY_FRAUD": 0.5, "FIRST_PARTY_FRAUD": 0.2, "SCAM": 0.2, "DISPUTE": 0.1},
                {"THIRD_PARTY_FRAUD": 0.4, "FIRST_PARTY_FRAUD": 0.3, "SCAM": 0.2, "DISPUTE": 0.1},
                {"THIRD_PARTY_FRAUD": 0.6, "FIRST_PARTY_FRAUD": 0.2, "SCAM": 0.1, "DISPUTE": 0.1},
            ],
        )
        result = evaluate_convergence(run)
        assert result.convergence_turn is None
        assert result.convergence_ratio is None

    def test_oscillating_scores(self):
        """Correct category highest on some turns but not all subsequent."""
        run = _make_run(
            {"investigation_category": "SCAM"},
            [
                {"THIRD_PARTY_FRAUD": 0.1, "FIRST_PARTY_FRAUD": 0.1, "SCAM": 0.7, "DISPUTE": 0.1},
                {"THIRD_PARTY_FRAUD": 0.5, "FIRST_PARTY_FRAUD": 0.1, "SCAM": 0.3, "DISPUTE": 0.1},
                {"THIRD_PARTY_FRAUD": 0.1, "FIRST_PARTY_FRAUD": 0.1, "SCAM": 0.7, "DISPUTE": 0.1},
            ],
        )
        result = evaluate_convergence(run)
        # SCAM is highest at turn 3 and stays highest (last turn) → converges at 3
        assert result.convergence_turn == 3


class TestTies:
    """Tied scores resolved alphabetically."""

    def test_tie_resolved_alphabetically(self):
        run = _make_run(
            {"investigation_category": "DISPUTE"},
            [
                {"THIRD_PARTY_FRAUD": 0.5, "FIRST_PARTY_FRAUD": 0.5, "SCAM": 0.0, "DISPUTE": 0.0},
                {"THIRD_PARTY_FRAUD": 0.0, "FIRST_PARTY_FRAUD": 0.0, "SCAM": 0.0, "DISPUTE": 0.5},
            ],
        )
        result = evaluate_convergence(run)
        # Turn 1: tie FIRST_PARTY_FRAUD vs THIRD_PARTY_FRAUD → FIRST_PARTY_FRAUD (alpha)
        # Turn 2: DISPUTE is highest → converges at turn 2
        assert result.convergence_turn == 2


class TestEmptyTurns:
    """No turn metrics."""

    def test_empty_returns_none_convergence(self):
        run = _make_run({"investigation_category": "SCAM"}, [])
        result = evaluate_convergence(run)
        assert result.convergence_turn is None
        assert result.total_turns == 0
        assert result.turn_scores == []


class TestNoGroundTruth:
    """Missing investigation_category in ground truth."""

    def test_no_category_returns_none(self):
        run = _make_run(
            {},
            [{"THIRD_PARTY_FRAUD": 0.5, "FIRST_PARTY_FRAUD": 0.2, "SCAM": 0.2, "DISPUTE": 0.1}],
        )
        result = evaluate_convergence(run)
        assert result.convergence_turn is None
        assert result.correct_category == ""
        assert result.turn_scores == []


class TestSingleTurn:
    """Only one turn in the run."""

    def test_single_turn_converges_if_highest(self):
        run = _make_run(
            {"investigation_category": "THIRD_PARTY_FRAUD"},
            [{"THIRD_PARTY_FRAUD": 0.8, "FIRST_PARTY_FRAUD": 0.1, "SCAM": 0.05, "DISPUTE": 0.05}],
        )
        result = evaluate_convergence(run)
        assert result.convergence_turn == 1
        assert result.total_turns == 1
        assert result.convergence_ratio == 1.0

    def test_single_turn_no_convergence_if_not_highest(self):
        run = _make_run(
            {"investigation_category": "DISPUTE"},
            [{"THIRD_PARTY_FRAUD": 0.8, "FIRST_PARTY_FRAUD": 0.1, "SCAM": 0.05, "DISPUTE": 0.05}],
        )
        result = evaluate_convergence(run)
        assert result.convergence_turn is None
