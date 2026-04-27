"""Integration tests for the full evaluation pipeline.

Exercises the end-to-end flow: CopilotOrchestrator transcript replay with mocked
specialist agents -> EvaluationRun construction -> generate_report (pure-Python
and LLM-powered) -> save_report JSON persistence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from agentic_fraud_servicing.copilot.question_planner import QuestionPlan

from agentic_fraud_servicing.copilot.auth_agent import AuthAssessment
from agentic_fraud_servicing.copilot.case_advisor import CaseAdvisory, CaseTypeAssessment
from agentic_fraud_servicing.copilot.hypothesis_agent import HypothesisAssessment
from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator
from agentic_fraud_servicing.copilot.retrieval_agent import RetrievalResult
from agentic_fraud_servicing.evaluation.models import (
    EvaluationReport,
    EvaluationRun,
    TurnMetric,
)
from agentic_fraud_servicing.evaluation.report import generate_report, save_report
from agentic_fraud_servicing.models.allegations import (
    AllegationDetailType,
    AllegationExtraction,
    AllegationExtractionResult,
)

# ---------------------------------------------------------------------------
# Canned specialist results (same pattern as test_copilot_flow.py)
# ---------------------------------------------------------------------------

_TRIAGE_RESULT = AllegationExtractionResult(
    allegations=[
        AllegationExtraction(
            detail_type=AllegationDetailType.UNRECOGNIZED_TRANSACTION,
            description="CM disputes charge at Electronics Store",
            entities={"merchant_name": "Electronics Store"},
            confidence=0.85,
        ),
    ]
)

_AUTH_ASSESSMENT = AuthAssessment(
    impersonation_risk=0.1,
    risk_factors=["Device recognized"],
    step_up_recommended=False,
    step_up_method="NONE",
    assessment_summary="Low risk.",
)

_QUESTION_PLAN = QuestionPlan(
    questions=["When did you first notice this charge?"],
    rationale=["Establish timeline"],
    priority_field="transaction_date",
    confidence=0.8,
)

_RETRIEVAL_RESULT = RetrievalResult(
    transactions=[{"node_id": "txn-001", "node_type": "TRANSACTION", "amount": 499.99}],
    auth_events=[{"node_type": "AUTH_EVENT", "result": "passed"}],
    customer_profile={"name": "Test User"},
    retrieval_summary="Found 1 transaction.",
    data_gaps=[],
)

_HYPOTHESIS_RESULT = HypothesisAssessment(
    scores={
        "THIRD_PARTY_FRAUD": 0.55,
        "FIRST_PARTY_FRAUD": 0.15,
        "SCAM": 0.10,
        "DISPUTE": 0.20,
    },
    reasoning={
        "THIRD_PARTY_FRAUD": "CM claims unauthorized charge.",
        "FIRST_PARTY_FRAUD": "No contradiction signals.",
        "SCAM": "No social engineering detected.",
        "DISPUTE": "Possible merchant issue.",
    },
    contradictions=[],
    assessment_summary="Likely third-party fraud.",
)

_CASE_ADVISORY = CaseAdvisory(
    assessments=[
        CaseTypeAssessment(
            case_type="fraud",
            eligibility="incomplete",
            met_criteria=["Transaction identified"],
            unmet_criteria=["Identity verification pending"],
        ),
    ],
    general_warnings=[],
    next_info_needed=["Identity verification"],
    summary="Fraud case incomplete.",
)

# Patch targets
_PATCH_TRIAGE = "agentic_fraud_servicing.copilot.orchestrator.run_triage"
_PATCH_AUTH = "agentic_fraud_servicing.copilot.orchestrator.run_auth_assessment"
_PATCH_QUESTION = "agentic_fraud_servicing.copilot.orchestrator.run_question_planner"
_PATCH_RETRIEVAL = "agentic_fraud_servicing.copilot.orchestrator.run_retrieval"
_PATCH_HYPOTHESIS = "agentic_fraud_servicing.copilot.orchestrator.run_hypothesis"
_PATCH_CASE_ADVISOR = "agentic_fraud_servicing.copilot.orchestrator.run_case_advisor"

# Ground truth for evaluation
_GROUND_TRUTH = {
    "investigation_category": "THIRD_PARTY_FRAUD",
    "resolution": "approved",
    "expected_allegations": ["UNRECOGNIZED_TRANSACTION"],
    "expected_risk_flags": ["step-up auth recommended"],
    "key_evidence_nodes": ["txn-001"],
}


@pytest.fixture()
def _mock_specialists():
    """Patch all 6 specialist run_* functions with canned results."""
    with (
        patch(_PATCH_TRIAGE, new_callable=AsyncMock, return_value=_TRIAGE_RESULT),
        patch(_PATCH_AUTH, new_callable=AsyncMock, return_value=_AUTH_ASSESSMENT),
        patch(_PATCH_QUESTION, new_callable=AsyncMock, return_value=_QUESTION_PLAN),
        patch(_PATCH_RETRIEVAL, new_callable=AsyncMock, return_value=_RETRIEVAL_RESULT),
        patch(_PATCH_HYPOTHESIS, new_callable=AsyncMock, return_value=_HYPOTHESIS_RESULT),
        patch(_PATCH_CASE_ADVISOR, new_callable=AsyncMock, return_value=_CASE_ADVISORY),
    ):
        yield


async def _build_evaluation_run(
    sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
) -> EvaluationRun:
    """Run the copilot on sample events and build an EvaluationRun."""
    gateway = gateway_factory(tmp_path)
    orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)

    turn_metrics: list[TurnMetric] = []
    start_time = time.perf_counter()

    for i, event in enumerate(sample_transcript_events):
        t0 = time.perf_counter()
        suggestion = await orch.process_event(event)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Extract allegations from this turn's triage result
        allegations_extracted = []
        if suggestion and hasattr(orch, "accumulated_allegations"):
            for a in orch.accumulated_allegations[len(turn_metrics) :]:
                allegations_extracted.append(
                    {"detail_type": a.detail_type.value, "description": a.description}
                )

        turn_metrics.append(
            TurnMetric(
                turn_number=i + 1,
                speaker=event.speaker.value,
                text=event.text,
                latency_ms=latency_ms,
                copilot_suggestion=suggestion.model_dump(mode="json") if suggestion else None,
                hypothesis_scores=dict(suggestion.hypothesis_scores) if suggestion else {},
                allegations_extracted=allegations_extracted,
            )
        )

    total_latency = (time.perf_counter() - start_time) * 1000.0

    return EvaluationRun(
        scenario_name="test_eval_scenario",
        ground_truth=_GROUND_TRUTH,
        turn_metrics=turn_metrics,
        total_turns=len(turn_metrics),
        total_latency_ms=total_latency,
        start_time="2026-03-23T00:00:00Z",
        end_time="2026-03-23T00:01:00Z",
        copilot_final_state={
            "hypothesis_scores": dict(orch.hypothesis_scores),
            "impersonation_risk": orch.impersonation_risk,
        },
    )


# ---------------------------------------------------------------------------
# Tests: EvaluationRun construction
# ---------------------------------------------------------------------------


class TestEvaluationRunConstruction:
    """Verify that the copilot orchestrator produces a valid EvaluationRun."""

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_evaluation_run_has_correct_turn_count(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """EvaluationRun should have one TurnMetric per transcript event."""
        run = await _build_evaluation_run(
            sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
        )
        assert run.total_turns == len(sample_transcript_events)
        assert len(run.turn_metrics) == 4

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_turn_metrics_have_latency(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Every TurnMetric should have a positive latency measurement."""
        run = await _build_evaluation_run(
            sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
        )
        for tm in run.turn_metrics:
            assert tm.latency_ms > 0.0

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_turn_metrics_have_copilot_suggestion(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """CARDMEMBER TurnMetrics should have a copilot_suggestion dict."""
        run = await _build_evaluation_run(
            sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
        )
        cm_metrics = [tm for tm in run.turn_metrics if tm.speaker == "CARDMEMBER"]
        assert len(cm_metrics) > 0
        for tm in cm_metrics:
            assert tm.copilot_suggestion is not None
            assert isinstance(tm.copilot_suggestion, dict)

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_ground_truth_propagated(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Ground truth should be stored in the EvaluationRun."""
        run = await _build_evaluation_run(
            sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
        )
        assert run.ground_truth["investigation_category"] == "THIRD_PARTY_FRAUD"


# ---------------------------------------------------------------------------
# Tests: Pure-Python report generation (no LLM)
# ---------------------------------------------------------------------------


class TestPurePythonReport:
    """Verify generate_report with model_provider=None (pure-Python evaluators only)."""

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_report_has_three_pure_python_dimensions(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """generate_report(run, None) populates latency, convergence, evidence_utilization."""
        run = await _build_evaluation_run(
            sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
        )
        report = await generate_report(run, model_provider=None)

        assert isinstance(report, EvaluationReport)
        assert report.latency is not None
        assert report.convergence is not None
        assert report.evidence_utilization is not None

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_llm_dimensions_are_none_without_provider(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """LLM-powered dimensions should be None without a model_provider."""
        run = await _build_evaluation_run(
            sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
        )
        report = await generate_report(run, model_provider=None)

        assert report.prediction is None
        assert report.question_adherence is not None  # pure-Python evaluator
        assert report.allegation_quality is None
        assert report.risk_flag_timeliness is None
        assert report.decision_explanation is None

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_overall_score_computed(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Overall score should be > 0 with at least some pure-Python dimensions."""
        run = await _build_evaluation_run(
            sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
        )
        report = await generate_report(run, model_provider=None)

        assert report.overall_score >= 0.0
        assert report.scenario_name == "test_eval_scenario"


# ---------------------------------------------------------------------------
# Tests: Full report with mocked LLM evaluators
# ---------------------------------------------------------------------------


class TestFullReportWithMockedLLM:
    """Verify generate_report with mocked LLM evaluators produces all 8 dimensions."""

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_all_eight_dimensions_populated(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """All 8 evaluation dimensions should be populated with mocked LLM evaluators."""
        from agentic_fraud_servicing.evaluation.models import (
            AllegationQualityResult,
            DecisionExplanation,
            PredictionResult,
            RiskFlagTimelinessResult,
        )

        run = await _build_evaluation_run(
            sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
        )

        # Mock LLM-powered evaluator functions (question_adherence is now pure-Python)
        mock_prediction = PredictionResult(
            predicted_category="THIRD_PARTY_FRAUD",
            ground_truth_category="THIRD_PARTY_FRAUD",
            match=True,
            confidence_delta=0.3,
        )
        mock_allegation = AllegationQualityResult(
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
        )
        mock_risk_flag = RiskFlagTimelinessResult(
            per_flag_timing=[],
            average_delay_turns=1.0,
            flags_raised_count=2,
            flags_expected_count=3,
        )
        mock_decision = DecisionExplanation(
            reasoning_chain="Clear evidence of unauthorized access.",
            influential_evidence=[{"node_id": "txn-001"}],
            improvement_suggestions=["Ask about card possession earlier"],
        )

        with (
            patch(
                "agentic_fraud_servicing.evaluation.report.evaluate_prediction",
                new_callable=AsyncMock,
                return_value=mock_prediction,
            ),
            patch(
                "agentic_fraud_servicing.evaluation.report.evaluate_allegation_quality",
                new_callable=AsyncMock,
                return_value=mock_allegation,
            ),
            patch(
                "agentic_fraud_servicing.evaluation.report.evaluate_risk_flag_timeliness",
                new_callable=AsyncMock,
                return_value=mock_risk_flag,
            ),
            patch(
                "agentic_fraud_servicing.evaluation.report.evaluate_decision_explanation",
                new_callable=AsyncMock,
                return_value=mock_decision,
            ),
        ):
            report = await generate_report(run, model_provider=mock_model_provider)

        assert isinstance(report, EvaluationReport)
        # Pure-Python dimensions
        assert report.latency is not None
        assert report.convergence is not None
        assert report.evidence_utilization is not None
        # LLM-powered dimensions
        assert report.prediction is not None
        assert report.prediction.match is True
        assert report.question_adherence is not None
        assert report.allegation_quality is not None
        assert report.allegation_quality.f1_score == pytest.approx(0.85)
        assert report.risk_flag_timeliness is not None
        assert report.decision_explanation is not None
        assert "unauthorized" in report.decision_explanation.reasoning_chain
        # Overall score should be higher with all 8 dimensions
        assert report.overall_score > 0.0


# ---------------------------------------------------------------------------
# Tests: save_report persistence
# ---------------------------------------------------------------------------


class TestSaveReport:
    """Verify save_report writes valid JSON."""

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_save_report_writes_valid_json(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """save_report should write a parseable JSON file."""
        run = await _build_evaluation_run(
            sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
        )
        report = await generate_report(run, model_provider=None)

        output_dir = tmp_path / "eval_output"
        file_path = save_report(report, str(output_dir))

        assert Path(file_path).exists()
        with open(file_path) as f:
            data = json.load(f)
        assert data["scenario_name"] == "test_eval_scenario"
        assert "overall_score" in data
        assert "latency" in data
        assert data["latency"] is not None

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_save_report_creates_directory(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """save_report should create the output directory if it doesn't exist."""
        run = await _build_evaluation_run(
            sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
        )
        report = await generate_report(run, model_provider=None)

        nested_dir = tmp_path / "deep" / "nested" / "dir"
        file_path = save_report(report, str(nested_dir))
        assert Path(file_path).exists()


# ---------------------------------------------------------------------------
# Live test skeleton
# ---------------------------------------------------------------------------


class TestLiveEvaluation:
    """Live evaluation test requiring real LLM credentials."""

    @pytest.mark.live
    async def test_evaluation_pipeline_live(self, gateway_factory, tmp_path):
        """End-to-end evaluation pipeline test with real LLM provider.

        Requires ConnectChain credentials configured in .env
        (LLM_PROVIDER=connectchain, CONNECTCHAIN_MODEL_INDEX).
        Skipped by default — run with: pytest -m live
        """
        from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator
        from agentic_fraud_servicing.models.enums import SpeakerType
        from agentic_fraud_servicing.models.transcript import TranscriptEvent
        from agentic_fraud_servicing.ui.helpers import create_provider

        try:
            provider = create_provider()
        except Exception:
            pytest.skip("LLM provider not configured — skipping live test")

        events = [
            TranscriptEvent(
                call_id="call-live-eval",
                event_id="evt-live-001",
                timestamp_ms=1000,
                speaker=SpeakerType.CARDMEMBER,
                text="I see a $500 charge at Best Buy that I did not make.",
            ),
        ]

        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, provider)

        suggestion = await orch.process_event(events[0])

        run = EvaluationRun(
            scenario_name="live_test",
            ground_truth={"investigation_category": "THIRD_PARTY_FRAUD"},
            turn_metrics=[
                TurnMetric(
                    turn_number=1,
                    speaker="CARDMEMBER",
                    text=events[0].text,
                    latency_ms=500.0,
                    copilot_suggestion=suggestion.model_dump(mode="json"),
                    hypothesis_scores=dict(suggestion.hypothesis_scores),
                )
            ],
            total_turns=1,
            total_latency_ms=500.0,
            start_time="2026-03-23T00:00:00Z",
            end_time="2026-03-23T00:00:01Z",
        )

        report = await generate_report(run, model_provider=provider)
        assert isinstance(report, EvaluationReport)
        assert report.latency is not None
