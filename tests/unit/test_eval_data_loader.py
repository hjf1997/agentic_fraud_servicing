"""Tests for evaluation.eval_data_loader — evaluation data loader functions."""

import os

import pytest

from agentic_fraud_servicing.evaluation.eval_data_loader import (
    discover_eval_scenarios,
    extract_dimension_scores,
    load_evaluation_report,
    load_evaluation_run,
    load_transcript_for_eval,
)
from agentic_fraud_servicing.evaluation.models import (
    ConvergenceResult,
    EvaluationReport,
    EvaluationRun,
    EvidenceUtilizationResult,
    LatencyReport,
    PredictionResult,
    TurnMetric,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evaluation_run() -> EvaluationRun:
    """Create a minimal EvaluationRun for testing."""
    return EvaluationRun(
        scenario_name="test_scenario",
        ground_truth={"investigation_category": "FIRST_PARTY_FRAUD"},
        turn_metrics=[
            TurnMetric(
                turn_number=1,
                speaker="CCP",
                text="Hello, how can I help?",
                latency_ms=120.5,
                hypothesis_scores={"THIRD_PARTY_FRAUD": 0.25, "FIRST_PARTY_FRAUD": 0.25},
            ),
            TurnMetric(
                turn_number=2,
                speaker="CARDMEMBER",
                text="I didn't make this purchase.",
                latency_ms=980.3,
                hypothesis_scores={"THIRD_PARTY_FRAUD": 0.60, "FIRST_PARTY_FRAUD": 0.10},
            ),
        ],
        total_turns=2,
        total_latency_ms=1100.8,
        start_time="2026-03-23T10:00:00Z",
        end_time="2026-03-23T10:01:00Z",
        copilot_final_state={"hypothesis_scores": {"THIRD_PARTY_FRAUD": 0.60}},
    )


def _make_evaluation_report() -> EvaluationReport:
    """Create a minimal EvaluationReport for testing."""
    return EvaluationReport(
        scenario_name="test_scenario",
        overall_score=0.75,
        latency=LatencyReport(
            per_turn_latency_ms=[120.5, 980.3],
            p50_ms=550.4,
            p95_ms=980.3,
            p99_ms=980.3,
            max_ms=980.3,
            compliance_rate=1.0,
        ),
        prediction=PredictionResult(
            predicted_category="FIRST_PARTY_FRAUD",
            ground_truth_category="FIRST_PARTY_FRAUD",
            match=True,
            confidence_delta=0.3,
        ),
        generated_at="2026-03-23T10:02:00Z",
    )


def _seed_scenario(base_dir: str, name: str, *, run: bool = True, report: bool = True) -> str:
    """Create a scenario directory with optional JSON files."""
    scenario_dir = os.path.join(base_dir, name)
    os.makedirs(scenario_dir, exist_ok=True)
    if run:
        run_path = os.path.join(scenario_dir, "evaluation_run.json")
        with open(run_path, "w") as f:
            f.write(_make_evaluation_run().model_dump_json(indent=2))
    if report:
        report_path = os.path.join(scenario_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            f.write(_make_evaluation_report().model_dump_json(indent=2))
    return scenario_dir


# ---------------------------------------------------------------------------
# TestDiscoverEvalScenarios
# ---------------------------------------------------------------------------


class TestDiscoverEvalScenarios:
    """Tests for discover_eval_scenarios."""

    def test_finds_scenarios_with_report_json(self, tmp_path):
        _seed_scenario(str(tmp_path), "alpha")
        _seed_scenario(str(tmp_path), "beta")
        result = discover_eval_scenarios(str(tmp_path))
        assert result == ["alpha", "beta"]

    def test_returns_empty_for_nonexistent_dir(self):
        result = discover_eval_scenarios("/nonexistent/path/xyzzy")
        assert result == []

    def test_returns_empty_for_empty_dir(self, tmp_path):
        result = discover_eval_scenarios(str(tmp_path))
        assert result == []

    def test_returns_sorted(self, tmp_path):
        _seed_scenario(str(tmp_path), "zebra")
        _seed_scenario(str(tmp_path), "apple")
        result = discover_eval_scenarios(str(tmp_path))
        assert result == ["apple", "zebra"]

    def test_ignores_dirs_without_report_json(self, tmp_path):
        """Directories with only evaluation_run.json (no report) are excluded."""
        _seed_scenario(str(tmp_path), "has_report")
        _seed_scenario(str(tmp_path), "no_report", report=False)
        result = discover_eval_scenarios(str(tmp_path))
        assert result == ["has_report"]


# ---------------------------------------------------------------------------
# TestLoadEvaluationRun
# ---------------------------------------------------------------------------


class TestLoadEvaluationRun:
    """Tests for load_evaluation_run."""

    def test_loads_valid_json(self, tmp_path):
        scenario_dir = _seed_scenario(str(tmp_path), "s1")
        result = load_evaluation_run(scenario_dir)
        assert isinstance(result, EvaluationRun)
        assert result.scenario_name == "test_scenario"
        assert result.total_turns == 2

    def test_missing_file_returns_none(self, tmp_path):
        result = load_evaluation_run(str(tmp_path))
        assert result is None

    def test_invalid_json_returns_none(self, tmp_path):
        scenario_dir = os.path.join(str(tmp_path), "bad")
        os.makedirs(scenario_dir)
        with open(os.path.join(scenario_dir, "evaluation_run.json"), "w") as f:
            f.write("{invalid json!!")
        result = load_evaluation_run(scenario_dir)
        assert result is None


# ---------------------------------------------------------------------------
# TestLoadEvaluationReport
# ---------------------------------------------------------------------------


class TestLoadEvaluationReport:
    """Tests for load_evaluation_report."""

    def test_loads_valid_json(self, tmp_path):
        scenario_dir = _seed_scenario(str(tmp_path), "s1")
        result = load_evaluation_report(scenario_dir)
        assert isinstance(result, EvaluationReport)
        assert result.scenario_name == "test_scenario"
        assert result.overall_score == 0.75

    def test_missing_file_returns_none(self, tmp_path):
        result = load_evaluation_report(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# TestLoadTranscriptForEval
# ---------------------------------------------------------------------------


class TestLoadTranscriptForEval:
    """Tests for load_transcript_for_eval."""

    def test_extracts_turns_from_run(self, tmp_path):
        scenario_dir = _seed_scenario(str(tmp_path), "s1")
        turns = load_transcript_for_eval(scenario_dir)
        assert len(turns) == 2
        assert turns[0]["turn_number"] == 1
        assert turns[0]["speaker"] == "CCP"
        assert turns[0]["text"] == "Hello, how can I help?"
        assert turns[0]["latency_ms"] == 120.5
        assert "THIRD_PARTY_FRAUD" in turns[0]["hypothesis_scores"]

    def test_empty_when_no_run(self, tmp_path):
        result = load_transcript_for_eval(str(tmp_path))
        assert result == []

    def test_second_turn_data(self, tmp_path):
        scenario_dir = _seed_scenario(str(tmp_path), "s1")
        turns = load_transcript_for_eval(scenario_dir)
        assert turns[1]["turn_number"] == 2
        assert turns[1]["speaker"] == "CARDMEMBER"
        assert turns[1]["latency_ms"] == 980.3


# ---------------------------------------------------------------------------
# TestExtractDimensionScores
# ---------------------------------------------------------------------------


class TestExtractDimensionScores:
    """Tests for extract_dimension_scores."""

    def test_all_dimensions_present(self):
        """All 8 dimension keys are returned even when sub-reports are None."""
        report = EvaluationReport(scenario_name="test", overall_score=0.0)
        scores = extract_dimension_scores(report)
        assert len(scores) == 8
        expected_keys = {
            "latency",
            "prediction",
            "question_adherence",
            "allegation_quality",
            "evidence_utilization",
            "convergence",
            "risk_flag_timeliness",
            "decision_explanation",
        }
        assert set(scores.keys()) == expected_keys

    def test_all_none_when_no_sub_reports(self):
        report = EvaluationReport(scenario_name="test", overall_score=0.0)
        scores = extract_dimension_scores(report)
        for v in scores.values():
            assert v is None

    def test_partial_dimensions(self):
        """Only populated dimensions return non-None scores."""
        report = _make_evaluation_report()
        scores = extract_dimension_scores(report)
        # latency compliance_rate = 1.0
        assert scores["latency"] == 1.0
        # prediction match = True -> 1.0
        assert scores["prediction"] == 1.0
        # Others not set -> None
        assert scores["question_adherence"] is None
        assert scores["convergence"] is None

    def test_convergence_with_ratio(self):
        report = EvaluationReport(
            scenario_name="test",
            overall_score=0.5,
            convergence=ConvergenceResult(
                convergence_turn=3,
                total_turns=10,
                convergence_ratio=0.3,
                correct_category="FIRST_PARTY_FRAUD",
            ),
        )
        scores = extract_dimension_scores(report)
        assert scores["convergence"] == pytest.approx(0.7)

    def test_evidence_utilization_average(self):
        report = EvaluationReport(
            scenario_name="test",
            overall_score=0.5,
            evidence_utilization=EvidenceUtilizationResult(
                total_evidence_nodes=10,
                retrieved_nodes=8,
                referenced_in_reasoning=6,
                retrieval_coverage=0.8,
                reasoning_coverage=0.6,
            ),
        )
        scores = extract_dimension_scores(report)
        assert scores["evidence_utilization"] == pytest.approx(0.7)
