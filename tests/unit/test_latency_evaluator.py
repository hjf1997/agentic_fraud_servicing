"""Tests for the latency evaluator — pure Python percentile and compliance analysis."""

from __future__ import annotations

from agentic_fraud_servicing.evaluation.latency_evaluator import evaluate_latency
from agentic_fraud_servicing.evaluation.models import EvaluationRun, TurnMetric


def _make_run(latencies: list[float]) -> EvaluationRun:
    """Helper to build an EvaluationRun from a list of latency values."""
    # Include a dummy copilot_suggestion so turns are treated as assessed
    _dummy_suggestion = {"call_id": "test", "timestamp_ms": 0}
    metrics = [
        TurnMetric(
            turn_number=i + 1,
            speaker="CARDMEMBER",
            text=f"Turn {i + 1}",
            latency_ms=lat,
            copilot_suggestion=_dummy_suggestion,
        )
        for i, lat in enumerate(latencies)
    ]
    return EvaluationRun(
        scenario_name="test",
        ground_truth={},
        turn_metrics=metrics,
        total_turns=len(metrics),
        total_latency_ms=sum(latencies),
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:01:00Z",
    )


class TestAllBelowTarget:
    """All turns within the 1500ms compliance target."""

    def test_compliance_rate_is_one(self):
        report = evaluate_latency(_make_run([100.0, 200.0, 300.0, 400.0, 500.0]))
        assert report.compliance_rate == 1.0

    def test_no_flagged_turns(self):
        report = evaluate_latency(_make_run([100.0, 200.0, 300.0]))
        assert report.flagged_turns == []


class TestMixedValues:
    """Some turns above, some below the 1500ms target."""

    def test_compliance_rate(self):
        # 3 below, 2 above -> 60% compliance
        report = evaluate_latency(_make_run([100.0, 200.0, 300.0, 2000.0, 3000.0]))
        assert report.compliance_rate == 0.6

    def test_flagged_turns_correct(self):
        report = evaluate_latency(_make_run([100.0, 200.0, 300.0, 2000.0, 3000.0]))
        assert report.flagged_turns == [4, 5]

    def test_per_turn_latency_preserved(self):
        latencies = [100.0, 2000.0, 300.0]
        report = evaluate_latency(_make_run(latencies))
        assert report.per_turn_latency_ms == latencies


class TestAllAboveTarget:
    """All turns exceed the 1500ms target."""

    def test_compliance_rate_is_zero(self):
        report = evaluate_latency(_make_run([2000.0, 3000.0, 4000.0]))
        assert report.compliance_rate == 0.0

    def test_all_turns_flagged(self):
        report = evaluate_latency(_make_run([2000.0, 3000.0, 4000.0]))
        assert report.flagged_turns == [1, 2, 3]


class TestPercentiles:
    """Percentile computation with known sorted data."""

    def test_known_percentiles(self):
        # 10 values: 100, 200, ..., 1000
        latencies = [float(i * 100) for i in range(1, 11)]
        report = evaluate_latency(_make_run(latencies))
        assert report.p50_ms == 500.0
        assert report.max_ms == 1000.0

    def test_p95_p99_with_100_points(self):
        # 100 values: 1.0, 2.0, ..., 100.0
        latencies = [float(i) for i in range(1, 101)]
        report = evaluate_latency(_make_run(latencies))
        assert report.p50_ms == 50.0
        assert report.p95_ms == 95.0
        assert report.p99_ms == 99.0
        assert report.max_ms == 100.0


class TestEmptyTurnMetrics:
    """Edge case: no turn metrics at all."""

    def test_returns_zeros(self):
        report = evaluate_latency(_make_run([]))
        assert report.p50_ms == 0.0
        assert report.p95_ms == 0.0
        assert report.p99_ms == 0.0
        assert report.max_ms == 0.0
        assert report.compliance_rate == 0.0
        assert report.flagged_turns == []
        assert report.per_turn_latency_ms == []


class TestSingleTurn:
    """Edge case: exactly one turn metric."""

    def test_single_below_target(self):
        report = evaluate_latency(_make_run([500.0]))
        assert report.p50_ms == 500.0
        assert report.p95_ms == 500.0
        assert report.p99_ms == 500.0
        assert report.max_ms == 500.0
        assert report.compliance_rate == 1.0
        assert report.flagged_turns == []

    def test_single_above_target(self):
        report = evaluate_latency(_make_run([2000.0]))
        assert report.compliance_rate == 0.0
        assert report.flagged_turns == [1]


class TestNonAssessedTurnsFiltered:
    """Non-assessed turns (copilot_suggestion=None) should be excluded."""

    def test_non_assessed_turns_excluded(self):
        """Mix of assessed CM turns and non-assessed CCP turns."""
        metrics = [
            TurnMetric(
                turn_number=1,
                speaker="CCP",
                text="Hello",
                latency_ms=5.0,
                copilot_suggestion=None,
            ),
            TurnMetric(
                turn_number=2,
                speaker="CARDMEMBER",
                text="I have a charge",
                latency_ms=800.0,
                copilot_suggestion={"call_id": "test", "timestamp_ms": 0},
            ),
            TurnMetric(
                turn_number=3,
                speaker="CCP",
                text="Let me check",
                latency_ms=3.0,
                copilot_suggestion=None,
            ),
            TurnMetric(
                turn_number=4,
                speaker="CARDMEMBER",
                text="It was $500",
                latency_ms=1200.0,
                copilot_suggestion={"call_id": "test", "timestamp_ms": 0},
            ),
        ]
        run = EvaluationRun(
            scenario_name="test",
            ground_truth={},
            turn_metrics=metrics,
            total_turns=4,
            total_latency_ms=2008.0,
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-01T00:01:00Z",
        )
        report = evaluate_latency(run)
        # Only 2 assessed turns (turns 2 and 4)
        assert len(report.per_turn_latency_ms) == 2
        assert report.per_turn_latency_ms == [800.0, 1200.0]
        assert report.assessed_turns == [2, 4]
        assert report.compliance_rate == 1.0  # both under 1500ms


class TestComplianceTarget:
    """Boundary: latency exactly at the 1500ms target."""

    def test_exact_target_is_compliant(self):
        report = evaluate_latency(_make_run([1500.0]))
        assert report.compliance_rate == 1.0
        assert report.flagged_turns == []
