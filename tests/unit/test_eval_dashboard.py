"""Unit tests for the evaluation results dashboard."""

from __future__ import annotations

from unittest.mock import patch

import gradio as gr
import matplotlib.pyplot as plt

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
from agentic_fraud_servicing.ui.eval_dashboard import (
    _build_evidence_table_html,
    _build_hypothesis_chart,
    _build_latency_chart,
    _build_radar_chart,
    _build_summary_html,
    create_eval_dashboard_app,
)


def _make_report(**overrides) -> EvaluationReport:
    """Build a minimal EvaluationReport with optional overrides."""
    defaults = {
        "scenario_name": "test_scenario",
        "overall_score": 0.75,
        "latency": LatencyReport(
            per_turn_latency_ms=[500.0, 800.0, 1200.0],
            p50_ms=800.0,
            p95_ms=1200.0,
            p99_ms=1200.0,
            max_ms=1200.0,
            compliance_rate=1.0,
            flagged_turns=[],
        ),
        "prediction": PredictionResult(
            predicted_category="FIRST_PARTY_FRAUD",
            ground_truth_category="FIRST_PARTY_FRAUD",
            match=True,
            confidence_delta=0.3,
        ),
        "convergence": ConvergenceResult(
            convergence_turn=3,
            total_turns=5,
            convergence_ratio=0.6,
            correct_category="FIRST_PARTY_FRAUD",
        ),
        "question_adherence": QuestionAdherenceResult(
            per_turn_scores=[],
            overall_adherence_rate=0.8,
            turns_with_suggestions=3,
            turns_with_adherence=2,
        ),
        "allegation_quality": AllegationQualityResult(
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
        ),
        "evidence_utilization": EvidenceUtilizationResult(
            total_evidence_nodes=5,
            retrieved_nodes=4,
            referenced_in_reasoning=3,
            retrieval_coverage=0.8,
            reasoning_coverage=0.6,
            missed_evidence=[{"node_id": "n1", "node_type": "Transaction", "source_type": "FACT"}],
        ),
        "risk_flag_timeliness": RiskFlagTimelinessResult(
            per_flag_timing=[],
            average_delay_turns=1.0,
            flags_raised_count=2,
            flags_expected_count=3,
        ),
        "decision_explanation": DecisionExplanation(
            reasoning_chain="The copilot identified contradictions.",
            influential_evidence=[{"node_id": "txn1", "node_type": "Transaction"}],
            improvement_suggestions=["Faster convergence"],
            overall_quality_notes="Good performance.",
        ),
        "generated_at": "2026-03-23T12:00:00",
    }
    defaults.update(overrides)
    return EvaluationReport(**defaults)


def _make_run() -> EvaluationRun:
    """Build a minimal EvaluationRun with 3 turns."""
    metrics = [
        TurnMetric(
            turn_number=i + 1,
            speaker=["CCP", "CARDMEMBER", "SYSTEM"][i],
            text=f"Turn {i + 1} text",
            latency_ms=500.0 + i * 300,
            hypothesis_scores={
                "THIRD_PARTY_FRAUD": 0.3 - i * 0.05,
                "FIRST_PARTY_FRAUD": 0.2 + i * 0.15,
                "SCAM": 0.3 - i * 0.05,
                "DISPUTE": 0.2 - i * 0.05,
            },
            allegations_extracted=[{"detail_type": "CARD_NOT_PRESENT_FRAUD"}] if i == 1 else [],
        )
        for i in range(3)
    ]
    return EvaluationRun(
        scenario_name="test_scenario",
        ground_truth={"investigation_category": "FIRST_PARTY_FRAUD"},
        turn_metrics=metrics,
        total_turns=3,
        total_latency_ms=2100.0,
        start_time="2026-03-23T12:00:00",
        end_time="2026-03-23T12:01:00",
    )


class TestCreateEvalDashboardApp:
    """Verify create_eval_dashboard_app returns a Blocks instance."""

    @patch(
        "agentic_fraud_servicing.ui.eval_dashboard.discover_eval_scenarios",
        return_value=["test_scenario"],
    )
    def test_returns_blocks(self, _mock_discover):
        app = create_eval_dashboard_app()
        assert isinstance(app, gr.Blocks)


class TestBuildRadarChart:
    """Verify radar chart builds correctly."""

    def test_returns_figure(self):
        report = _make_report()
        fig = _build_radar_chart(report)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_none_without_report(self):
        assert _build_radar_chart(None) is None


class TestBuildLatencyChart:
    """Verify latency chart builds correctly."""

    def test_returns_figure(self):
        report = _make_report()
        fig = _build_latency_chart(report)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_none_without_report(self):
        assert _build_latency_chart(None) is None


class TestBuildHypothesisChart:
    """Verify hypothesis evolution chart builds correctly."""

    def test_returns_figure(self):
        run = _make_run()
        report = _make_report()
        fig = _build_hypothesis_chart(run, report)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_none_without_run(self):
        assert _build_hypothesis_chart(None, None) is None


class TestBuildSummaryHtml:
    """Verify summary HTML contains expected elements."""

    def test_contains_overall_quality(self):
        report = _make_report()
        run = _make_run()
        html = _build_summary_html(report, run)
        assert "Overall Quality" in html
        assert "75%" in html

    def test_returns_placeholder_without_report(self):
        html = _build_summary_html(None, None)
        assert "No evaluation report" in html


class TestBuildEvidenceTableHtml:
    """Verify evidence utilization HTML contains expected elements."""

    def test_contains_coverage_metrics(self):
        report = _make_report()
        html = _build_evidence_table_html(report)
        assert "Retrieval Coverage" in html
        assert "Reasoning Coverage" in html
        assert "Missed Evidence" in html

    def test_returns_placeholder_without_report(self):
        html = _build_evidence_table_html(None)
        assert "No evidence utilization" in html
