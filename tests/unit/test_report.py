"""Tests for the evaluation report aggregator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.evaluation.models import (
    AllegationQualityResult,
    ConvergenceResult,
    DecisionExplanation,
    EvaluationReport,
    EvaluationRun,
    EvidenceUtilizationResult,
    LatencyReport,
    PredictionResult,
    QuestionAdherenceResult,
    RiskFlagTimelinessResult,
    TurnMetric,
)
from agentic_fraud_servicing.evaluation.report import (
    _compute_overall_score,
    _extract_dimension_score,
    generate_report,
    save_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_LATENCY = LatencyReport(
    per_turn_latency_ms=[800.0, 1200.0],
    p50_ms=1000.0,
    p95_ms=1200.0,
    p99_ms=1200.0,
    max_ms=1200.0,
    compliance_rate=1.0,
)

_CONVERGENCE = ConvergenceResult(
    convergence_turn=2,
    total_turns=5,
    convergence_ratio=0.4,
    correct_category="THIRD_PARTY_FRAUD",
)

_EVIDENCE = EvidenceUtilizationResult(
    total_evidence_nodes=4,
    retrieved_nodes=3,
    referenced_in_reasoning=2,
    retrieval_coverage=0.75,
    reasoning_coverage=0.50,
)

_PREDICTION = PredictionResult(
    predicted_category="THIRD_PARTY_FRAUD",
    ground_truth_category="THIRD_PARTY_FRAUD",
    match=True,
    confidence_delta=0.3,
)

_QUESTION_ADHERENCE = QuestionAdherenceResult(
    per_turn_scores=[],
    overall_adherence_rate=0.8,
    turns_with_suggestions=5,
    turns_with_adherence=4,
)

_ALLEGATION_QUALITY = AllegationQualityResult(
    precision=0.9,
    recall=0.8,
    f1_score=0.85,
)

_RISK_FLAG = RiskFlagTimelinessResult(
    per_flag_timing=[],
    average_delay_turns=1.0,
    flags_raised_count=3,
    flags_expected_count=4,
)

_DECISION_EXPLANATION = DecisionExplanation(
    reasoning_chain="The evidence clearly shows...",
    influential_evidence=[{"node_id": "n1"}],
    improvement_suggestions=["Probe earlier"],
)


def _make_run() -> EvaluationRun:
    """Create a minimal EvaluationRun for testing."""
    return EvaluationRun(
        scenario_name="test_scenario",
        ground_truth={"investigation_category": "THIRD_PARTY_FRAUD"},
        turn_metrics=[
            TurnMetric(turn_number=1, speaker="CCP", text="Hello", latency_ms=800.0),
            TurnMetric(turn_number=2, speaker="CARDMEMBER", text="I dispute", latency_ms=1200.0),
        ],
        total_turns=2,
        total_latency_ms=2000.0,
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:01:00Z",
    )


# Patch targets for all evaluators
_PATCH_BASE = "agentic_fraud_servicing.evaluation.report"


# ---------------------------------------------------------------------------
# TestGenerateReport — pure-Python-only mode
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for generate_report with model_provider=None (pure-Python only)."""

    @pytest.fixture(autouse=True)
    def _patch_pure_evaluators(self):
        with (
            patch(f"{_PATCH_BASE}.evaluate_latency", return_value=_LATENCY) as self.mock_lat,
            patch(
                f"{_PATCH_BASE}.evaluate_convergence", return_value=_CONVERGENCE
            ) as self.mock_conv,
            patch(
                f"{_PATCH_BASE}.evaluate_evidence_utilization", return_value=_EVIDENCE
            ) as self.mock_ev,
        ):
            yield

    async def test_returns_evaluation_report(self):
        report = await generate_report(_make_run())
        assert isinstance(report, EvaluationReport)

    async def test_three_dimensions_populated(self):
        report = await generate_report(_make_run())
        assert report.latency is not None
        assert report.convergence is not None
        assert report.evidence_utilization is not None

    async def test_llm_dimensions_are_none(self):
        report = await generate_report(_make_run())
        assert report.prediction is None
        assert report.question_adherence is None
        assert report.allegation_quality is None
        assert report.risk_flag_timeliness is None
        assert report.decision_explanation is None

    async def test_scenario_name_propagated(self):
        report = await generate_report(_make_run())
        assert report.scenario_name == "test_scenario"

    async def test_generated_at_set(self):
        report = await generate_report(_make_run())
        assert report.generated_at != ""
        assert "T" in report.generated_at  # ISO format

    async def test_overall_score_uses_available_dimensions(self):
        """Overall score should be weighted average of 3 available dimensions only."""
        report = await generate_report(_make_run())
        # latency: compliance_rate=1.0, weight=0.10
        # convergence: 1.0 - 0.4 = 0.6, weight=0.15
        # evidence: (0.75+0.50)/2 = 0.625, weight=0.10
        # total_weight = 0.10 + 0.15 + 0.10 = 0.35
        # weighted_sum = 0.10*1.0 + 0.15*0.6 + 0.10*0.625 = 0.10 + 0.09 + 0.0625 = 0.2525
        # overall = 0.2525 / 0.35 ≈ 0.7214
        assert 0.70 < report.overall_score < 0.75


# ---------------------------------------------------------------------------
# TestGenerateReportWithLlm — all 8 dimensions
# ---------------------------------------------------------------------------


class TestGenerateReportWithLlm:
    """Tests for generate_report with a mock model_provider (all 8 evaluators)."""

    @pytest.fixture(autouse=True)
    def _patch_all_evaluators(self):
        with (
            patch(f"{_PATCH_BASE}.evaluate_latency", return_value=_LATENCY),
            patch(f"{_PATCH_BASE}.evaluate_convergence", return_value=_CONVERGENCE),
            patch(f"{_PATCH_BASE}.evaluate_evidence_utilization", return_value=_EVIDENCE),
            patch(
                f"{_PATCH_BASE}.evaluate_prediction",
                new_callable=AsyncMock,
                return_value=_PREDICTION,
            ),
            patch(
                f"{_PATCH_BASE}.evaluate_question_adherence",
                new_callable=AsyncMock,
                return_value=_QUESTION_ADHERENCE,
            ),
            patch(
                f"{_PATCH_BASE}.evaluate_allegation_quality",
                new_callable=AsyncMock,
                return_value=_ALLEGATION_QUALITY,
            ),
            patch(
                f"{_PATCH_BASE}.evaluate_risk_flag_timeliness",
                new_callable=AsyncMock,
                return_value=_RISK_FLAG,
            ),
            patch(
                f"{_PATCH_BASE}.evaluate_decision_explanation",
                new_callable=AsyncMock,
                return_value=_DECISION_EXPLANATION,
            ),
        ):
            yield

    async def test_all_eight_dimensions_populated(self):
        provider = MagicMock()
        report = await generate_report(_make_run(), model_provider=provider)
        assert report.latency is not None
        assert report.convergence is not None
        assert report.evidence_utilization is not None
        assert report.prediction is not None
        assert report.question_adherence is not None
        assert report.allegation_quality is not None
        assert report.risk_flag_timeliness is not None
        assert report.decision_explanation is not None

    async def test_overall_score_uses_all_weights(self):
        provider = MagicMock()
        report = await generate_report(_make_run(), model_provider=provider)
        # All 8 weights sum to 1.0, so no re-normalization needed
        # Manually compute expected score:
        # latency: 1.0 * 0.10 = 0.10
        # prediction: 1.0 * 0.20 = 0.20
        # question_adherence: 0.8 * 0.10 = 0.08
        # allegation_quality: 0.85 * 0.15 = 0.1275
        # evidence: 0.625 * 0.10 = 0.0625
        # convergence: 0.6 * 0.15 = 0.09
        # risk_flag: 0.75 * 0.10 = 0.075
        # decision: 1.0 * 0.10 = 0.10
        # total = 0.835
        assert 0.83 < report.overall_score < 0.84


# ---------------------------------------------------------------------------
# TestGracefulDegradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Evaluator failures produce None without crashing the report."""

    async def test_latency_failure_continues(self):
        with (
            patch(f"{_PATCH_BASE}.evaluate_latency", side_effect=RuntimeError("boom")),
            patch(f"{_PATCH_BASE}.evaluate_convergence", return_value=_CONVERGENCE),
            patch(f"{_PATCH_BASE}.evaluate_evidence_utilization", return_value=_EVIDENCE),
        ):
            report = await generate_report(_make_run())
            assert report.latency is None
            assert report.convergence is not None
            assert report.evidence_utilization is not None

    async def test_llm_evaluator_failure_continues(self):
        with (
            patch(f"{_PATCH_BASE}.evaluate_latency", return_value=_LATENCY),
            patch(f"{_PATCH_BASE}.evaluate_convergence", return_value=_CONVERGENCE),
            patch(f"{_PATCH_BASE}.evaluate_evidence_utilization", return_value=_EVIDENCE),
            patch(
                f"{_PATCH_BASE}.evaluate_prediction",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LLM timeout"),
            ),
            patch(
                f"{_PATCH_BASE}.evaluate_question_adherence",
                new_callable=AsyncMock,
                return_value=_QUESTION_ADHERENCE,
            ),
            patch(
                f"{_PATCH_BASE}.evaluate_allegation_quality",
                new_callable=AsyncMock,
                return_value=_ALLEGATION_QUALITY,
            ),
            patch(
                f"{_PATCH_BASE}.evaluate_risk_flag_timeliness",
                new_callable=AsyncMock,
                return_value=_RISK_FLAG,
            ),
            patch(
                f"{_PATCH_BASE}.evaluate_decision_explanation",
                new_callable=AsyncMock,
                return_value=_DECISION_EXPLANATION,
            ),
        ):
            provider = MagicMock()
            report = await generate_report(_make_run(), model_provider=provider)
            assert report.prediction is None
            assert report.question_adherence is not None


# ---------------------------------------------------------------------------
# TestSaveReport
# ---------------------------------------------------------------------------


class TestSaveReport:
    """Tests for save_report file persistence."""

    def test_writes_json_file(self, tmp_path):
        report = EvaluationReport(scenario_name="test", overall_score=0.5)
        path = save_report(report, str(tmp_path))
        assert path.endswith("evaluation_report.json")
        assert (tmp_path / "evaluation_report.json").exists()

    def test_file_is_valid_json(self, tmp_path):
        report = EvaluationReport(scenario_name="test", overall_score=0.5)
        save_report(report, str(tmp_path))
        data = json.loads((tmp_path / "evaluation_report.json").read_text())
        assert data["scenario_name"] == "test"
        assert data["overall_score"] == 0.5

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        report = EvaluationReport(scenario_name="test", overall_score=0.0)
        save_report(report, str(nested))
        assert nested.exists()
        assert (nested / "evaluation_report.json").exists()


# ---------------------------------------------------------------------------
# TestWeightNormalization
# ---------------------------------------------------------------------------


class TestWeightNormalization:
    """Tests for weight re-normalization when some dimensions are None."""

    def test_three_of_eight_dimensions(self):
        """With 3/8 dimensions, weights should re-normalize to sum to 1.0."""
        scores = {
            "latency": 1.0,
            "prediction": None,
            "question_adherence": None,
            "allegation_quality": None,
            "evidence_utilization": 0.5,
            "convergence": 0.6,
            "risk_flag_timeliness": None,
            "decision_explanation": None,
        }
        overall = _compute_overall_score(scores)
        # Weights: latency=0.10, evidence=0.10, convergence=0.15 → total=0.35
        # Weighted: 0.10*1.0 + 0.10*0.5 + 0.15*0.6 = 0.10 + 0.05 + 0.09 = 0.24
        # Normalized: 0.24 / 0.35 ≈ 0.6857
        assert abs(overall - 0.24 / 0.35) < 0.001

    def test_all_dimensions_none_returns_zero(self):
        scores = {dim: None for dim in _compute_overall_score.__code__.co_varnames}
        # Use actual weight keys
        scores = {
            dim: None
            for dim in [
                "latency",
                "prediction",
                "question_adherence",
                "allegation_quality",
                "evidence_utilization",
                "convergence",
                "risk_flag_timeliness",
                "decision_explanation",
            ]
        }
        assert _compute_overall_score(scores) == 0.0


# ---------------------------------------------------------------------------
# TestExtractDimensionScore
# ---------------------------------------------------------------------------


class TestExtractDimensionScore:
    """Tests for _extract_dimension_score helper."""

    def test_latency_returns_compliance_rate(self):
        assert _extract_dimension_score("latency", _LATENCY) == 1.0

    def test_prediction_match_returns_one(self):
        assert _extract_dimension_score("prediction", _PREDICTION) == 1.0

    def test_prediction_mismatch_returns_zero(self):
        result = PredictionResult(
            predicted_category="SCAM",
            ground_truth_category="DISPUTE",
            match=False,
            confidence_delta=0.1,
        )
        assert _extract_dimension_score("prediction", result) == 0.0

    def test_convergence_with_ratio(self):
        assert _extract_dimension_score("convergence", _CONVERGENCE) == 0.6

    def test_convergence_never_converged(self):
        result = ConvergenceResult(
            convergence_turn=None,
            total_turns=10,
            convergence_ratio=None,
            correct_category="SCAM",
        )
        assert _extract_dimension_score("convergence", result) == 0.0

    def test_none_result_returns_none(self):
        assert _extract_dimension_score("latency", None) is None
