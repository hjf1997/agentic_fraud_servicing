"""Tests for evaluation data models."""

from agentic_fraud_servicing.evaluation.models import (
    AllegationQualityResult,
    ConvergenceResult,
    DecisionExplanation,
    EvaluationConfig,
    EvaluationReport,
    EvaluationRun,
    EvidenceUtilizationResult,
    LatencyReport,
    PredictionResult,
    QuestionAdherenceResult,
    RiskFlagTimelinessResult,
    TurnMetric,
)

# ---------------------------------------------------------------------------
# EvaluationConfig
# ---------------------------------------------------------------------------


class TestEvaluationConfig:
    def test_required_fields(self):
        cfg = EvaluationConfig(
            scenario_name="test_scenario",
            ground_truth={"investigation_category": "FIRST_PARTY_FRAUD"},
            transcript_path="scripts/transcripts/test.json",
        )
        assert cfg.scenario_name == "test_scenario"
        assert cfg.ground_truth["investigation_category"] == "FIRST_PARTY_FRAUD"
        assert cfg.transcript_path == "scripts/transcripts/test.json"
        assert cfg.evaluator_flags == {}

    def test_all_fields(self):
        cfg = EvaluationConfig(
            scenario_name="scam_techvault",
            ground_truth={"investigation_category": "SCAM", "resolution": "denied"},
            transcript_path="/tmp/transcript.json",
            evaluator_flags={"latency": True, "prediction": False},
        )
        assert cfg.evaluator_flags["latency"] is True
        assert cfg.evaluator_flags["prediction"] is False

    def test_json_round_trip(self):
        cfg = EvaluationConfig(
            scenario_name="round_trip",
            ground_truth={"category": "DISPUTE", "amount": 99.99},
            transcript_path="t.json",
            evaluator_flags={"latency": True},
        )
        json_str = cfg.model_dump_json()
        restored = EvaluationConfig.model_validate_json(json_str)
        assert restored == cfg


# ---------------------------------------------------------------------------
# TurnMetric
# ---------------------------------------------------------------------------


class TestTurnMetric:
    def test_defaults(self):
        tm = TurnMetric(turn_number=1, speaker="CARDMEMBER", text="Hello", latency_ms=120.5)
        assert tm.turn_number == 1
        assert tm.speaker == "CARDMEMBER"
        assert tm.text == "Hello"
        assert tm.latency_ms == 120.5
        assert tm.copilot_suggestion is None
        assert tm.hypothesis_scores == {}
        assert tm.allegations_extracted == []

    def test_all_fields(self):
        tm = TurnMetric(
            turn_number=3,
            speaker="CARDMEMBER",
            text="I didn't make that purchase",
            latency_ms=850.2,
            copilot_suggestion={"suggested_questions": ["When did you notice?"]},
            hypothesis_scores={
                "THIRD_PARTY_FRAUD": 0.6,
                "FIRST_PARTY_FRAUD": 0.1,
                "SCAM": 0.2,
                "DISPUTE": 0.1,
            },
            allegations_extracted=[
                {"allegation_type": "TRANSACTION_DISPUTE", "description": "unauthorized purchase"}
            ],
        )
        assert tm.copilot_suggestion is not None
        assert len(tm.hypothesis_scores) == 4
        assert len(tm.allegations_extracted) == 1

    def test_none_copilot_suggestion(self):
        """CCP and SYSTEM turns have no copilot suggestion."""
        tm = TurnMetric(
            turn_number=2,
            speaker="CCP",
            text="How can I help?",
            latency_ms=0.0,
            copilot_suggestion=None,
        )
        assert tm.copilot_suggestion is None
        assert tm.latency_ms == 0.0

    def test_json_round_trip(self):
        tm = TurnMetric(
            turn_number=5,
            speaker="SYSTEM",
            text="Auth verified",
            latency_ms=45.0,
            hypothesis_scores={"THIRD_PARTY_FRAUD": 0.5, "DISPUTE": 0.5},
            allegations_extracted=[{"type": "IDENTITY_VERIFICATION"}],
        )
        json_str = tm.model_dump_json()
        restored = TurnMetric.model_validate_json(json_str)
        assert restored == tm


# ---------------------------------------------------------------------------
# EvaluationRun
# ---------------------------------------------------------------------------


class TestEvaluationRun:
    def _make_turn_metrics(self, count: int = 3) -> list[TurnMetric]:
        return [
            TurnMetric(
                turn_number=i + 1,
                speaker="CARDMEMBER" if i % 2 == 0 else "CCP",
                text=f"Turn {i + 1}",
                latency_ms=100.0 * (i + 1),
            )
            for i in range(count)
        ]

    def test_required_fields(self):
        metrics = self._make_turn_metrics(2)
        run = EvaluationRun(
            scenario_name="test",
            ground_truth={"investigation_category": "DISPUTE"},
            turn_metrics=metrics,
            total_turns=2,
            total_latency_ms=300.0,
            start_time="2026-03-23T10:00:00Z",
            end_time="2026-03-23T10:00:05Z",
        )
        assert run.scenario_name == "test"
        assert len(run.turn_metrics) == 2
        assert run.total_turns == 2
        assert run.copilot_final_state == {}

    def test_with_turn_metrics(self):
        metrics = self._make_turn_metrics(4)
        run = EvaluationRun(
            scenario_name="multi_turn",
            ground_truth={"investigation_category": "THIRD_PARTY_FRAUD"},
            turn_metrics=metrics,
            total_turns=4,
            total_latency_ms=1000.0,
            start_time="2026-03-23T10:00:00Z",
            end_time="2026-03-23T10:00:10Z",
            copilot_final_state={
                "hypothesis_scores": {"THIRD_PARTY_FRAUD": 0.7},
                "impersonation_risk": 0.3,
            },
        )
        assert len(run.turn_metrics) == run.total_turns
        assert run.copilot_final_state["impersonation_risk"] == 0.3

    def test_json_round_trip(self):
        metrics = self._make_turn_metrics(3)
        run = EvaluationRun(
            scenario_name="round_trip",
            ground_truth={"investigation_category": "SCAM", "resolution": "denied"},
            turn_metrics=metrics,
            total_turns=3,
            total_latency_ms=600.0,
            start_time="2026-03-23T10:00:00Z",
            end_time="2026-03-23T10:00:03Z",
            copilot_final_state={"accumulated_allegations": 5},
        )
        json_str = run.model_dump_json()
        restored = EvaluationRun.model_validate_json(json_str)
        assert restored == run

    def test_total_turns_matches_metrics_length(self):
        """Verify total_turns can be set independently but should match len(turn_metrics)."""
        metrics = self._make_turn_metrics(3)
        run = EvaluationRun(
            scenario_name="count_check",
            ground_truth={},
            turn_metrics=metrics,
            total_turns=3,
            total_latency_ms=600.0,
            start_time="2026-03-23T10:00:00Z",
            end_time="2026-03-23T10:00:03Z",
        )
        assert run.total_turns == len(run.turn_metrics)


# ---------------------------------------------------------------------------
# LatencyReport
# ---------------------------------------------------------------------------


class TestLatencyReport:
    def test_defaults(self):
        lr = LatencyReport(
            per_turn_latency_ms=[100.0, 200.0],
            p50_ms=150.0,
            p95_ms=195.0,
            p99_ms=199.0,
            max_ms=200.0,
            compliance_rate=1.0,
        )
        assert lr.compliance_target_ms == 1500.0
        assert lr.flagged_turns == []
        assert lr.compliance_rate == 1.0

    def test_all_fields(self):
        lr = LatencyReport(
            per_turn_latency_ms=[500.0, 1200.0, 1800.0, 900.0],
            p50_ms=1050.0,
            p95_ms=1740.0,
            p99_ms=1788.0,
            max_ms=1800.0,
            compliance_target_ms=1500.0,
            compliance_rate=0.75,
            flagged_turns=[3],
        )
        assert len(lr.per_turn_latency_ms) == 4
        assert lr.flagged_turns == [3]
        assert lr.compliance_rate == 0.75

    def test_json_round_trip(self):
        lr = LatencyReport(
            per_turn_latency_ms=[100.0, 2000.0],
            p50_ms=1050.0,
            p95_ms=1910.0,
            p99_ms=1982.0,
            max_ms=2000.0,
            compliance_rate=0.5,
            flagged_turns=[2],
        )
        restored = LatencyReport.model_validate_json(lr.model_dump_json())
        assert restored == lr


# ---------------------------------------------------------------------------
# PredictionResult
# ---------------------------------------------------------------------------


class TestPredictionResult:
    def test_defaults(self):
        pr = PredictionResult(
            predicted_category="THIRD_PARTY_FRAUD",
            ground_truth_category="THIRD_PARTY_FRAUD",
            match=True,
            confidence_delta=0.35,
        )
        assert pr.reasoning == ""
        assert pr.match is True

    def test_all_fields(self):
        pr = PredictionResult(
            predicted_category="SCAM",
            ground_truth_category="FIRST_PARTY_FRAUD",
            match=False,
            confidence_delta=0.05,
            reasoning="Close scores between SCAM and FIRST_PARTY_FRAUD",
        )
        assert pr.match is False
        assert pr.confidence_delta == 0.05
        assert "Close scores" in pr.reasoning

    def test_json_round_trip(self):
        pr = PredictionResult(
            predicted_category="DISPUTE",
            ground_truth_category="DISPUTE",
            match=True,
            confidence_delta=0.45,
            reasoning="Clear dispute pattern",
        )
        restored = PredictionResult.model_validate_json(pr.model_dump_json())
        assert restored == pr


# ---------------------------------------------------------------------------
# QuestionAdherenceResult
# ---------------------------------------------------------------------------


class TestQuestionAdherenceResult:
    def test_defaults(self):
        qa = QuestionAdherenceResult()
        assert qa.probing_questions == []
        assert qa.total_questions == 0
        assert qa.answered == 0
        assert qa.overall_adherence_rate == 0.0
        assert qa.information_sufficient is False

    def test_all_fields(self):
        qa = QuestionAdherenceResult(
            probing_questions=[
                {"text": "Q1", "status": "answered", "target_category": "SCAM"},
                {"text": "Q2", "status": "skipped", "target_category": "DISPUTE"},
            ],
            total_questions=2,
            answered=1,
            invalidated=0,
            skipped=1,
            pending=0,
            information_sufficient=True,
            overall_adherence_rate=0.5,
        )
        assert len(qa.probing_questions) == 2
        assert qa.answered == 1
        assert qa.skipped == 1

    def test_json_round_trip(self):
        qa = QuestionAdherenceResult(
            probing_questions=[{"text": "Q1", "status": "answered"}],
            total_questions=1,
            answered=1,
            overall_adherence_rate=1.0,
            information_sufficient=True,
        )
        restored = QuestionAdherenceResult.model_validate_json(qa.model_dump_json())
        assert restored == qa


# ---------------------------------------------------------------------------
# AllegationQualityResult
# ---------------------------------------------------------------------------


class TestAllegationQualityResult:
    def test_defaults(self):
        aq = AllegationQualityResult(precision=0.8, recall=0.9, f1_score=0.85)
        assert aq.ground_truth_allegations == []
        assert aq.extracted_allegations == []
        assert aq.matched == []
        assert aq.missed == []
        assert aq.false_positives == []

    def test_all_fields(self):
        aq = AllegationQualityResult(
            precision=0.75,
            recall=0.67,
            f1_score=0.71,
            ground_truth_allegations=["TRANSACTION_DISPUTE", "CARD_NOT_PRESENT_FRAUD"],
            extracted_allegations=["TRANSACTION_DISPUTE", "LOST_STOLEN_CARD"],
            matched=["TRANSACTION_DISPUTE"],
            missed=["CARD_NOT_PRESENT_FRAUD"],
            false_positives=["LOST_STOLEN_CARD"],
        )
        assert len(aq.matched) == 1
        assert len(aq.missed) == 1
        assert len(aq.false_positives) == 1

    def test_json_round_trip(self):
        aq = AllegationQualityResult(
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            ground_truth_allegations=["TRANSACTION_DISPUTE"],
            extracted_allegations=["TRANSACTION_DISPUTE"],
            matched=["TRANSACTION_DISPUTE"],
        )
        restored = AllegationQualityResult.model_validate_json(aq.model_dump_json())
        assert restored == aq


# ---------------------------------------------------------------------------
# EvidenceUtilizationResult
# ---------------------------------------------------------------------------


class TestEvidenceUtilizationResult:
    def test_defaults(self):
        eu = EvidenceUtilizationResult(
            total_evidence_nodes=8,
            retrieved_nodes=6,
            referenced_in_reasoning=4,
            retrieval_coverage=0.75,
            reasoning_coverage=0.5,
        )
        assert eu.missed_evidence == []
        assert eu.retrieval_coverage == 0.75

    def test_all_fields(self):
        eu = EvidenceUtilizationResult(
            total_evidence_nodes=10,
            retrieved_nodes=7,
            referenced_in_reasoning=5,
            retrieval_coverage=0.7,
            reasoning_coverage=0.5,
            missed_evidence=[
                {"node_id": "node-001", "node_type": "DELIVERY_PROOF"},
                {"node_id": "node-002", "node_type": "REFUND_RECORD"},
            ],
        )
        assert len(eu.missed_evidence) == 2
        assert eu.missed_evidence[0]["node_type"] == "DELIVERY_PROOF"

    def test_json_round_trip(self):
        eu = EvidenceUtilizationResult(
            total_evidence_nodes=5,
            retrieved_nodes=5,
            referenced_in_reasoning=3,
            retrieval_coverage=1.0,
            reasoning_coverage=0.6,
            missed_evidence=[{"node_id": "n1", "node_type": "AUTH_EVENT"}],
        )
        restored = EvidenceUtilizationResult.model_validate_json(eu.model_dump_json())
        assert restored == eu


# ---------------------------------------------------------------------------
# ConvergenceResult
# ---------------------------------------------------------------------------


class TestConvergenceResult:
    def test_defaults(self):
        cr = ConvergenceResult(
            convergence_turn=5,
            total_turns=10,
            convergence_ratio=0.5,
            correct_category="FIRST_PARTY_FRAUD",
        )
        assert cr.turn_scores == []
        assert cr.convergence_ratio == 0.5

    def test_never_converged(self):
        cr = ConvergenceResult(
            convergence_turn=None,
            total_turns=12,
            convergence_ratio=None,
            correct_category="SCAM",
        )
        assert cr.convergence_turn is None
        assert cr.convergence_ratio is None

    def test_all_fields(self):
        cr = ConvergenceResult(
            convergence_turn=3,
            total_turns=8,
            convergence_ratio=0.375,
            correct_category="THIRD_PARTY_FRAUD",
            turn_scores=[
                {
                    "turn_number": 1,
                    "THIRD_PARTY_FRAUD": 0.4,
                    "FIRST_PARTY_FRAUD": 0.2,
                    "SCAM": 0.2,
                    "DISPUTE": 0.2,
                },
                {
                    "turn_number": 2,
                    "THIRD_PARTY_FRAUD": 0.6,
                    "FIRST_PARTY_FRAUD": 0.15,
                    "SCAM": 0.15,
                    "DISPUTE": 0.1,
                },
            ],
        )
        assert len(cr.turn_scores) == 2

    def test_json_round_trip(self):
        cr = ConvergenceResult(
            convergence_turn=None,
            total_turns=5,
            convergence_ratio=None,
            correct_category="DISPUTE",
            turn_scores=[{"turn_number": 1, "DISPUTE": 0.3}],
        )
        restored = ConvergenceResult.model_validate_json(cr.model_dump_json())
        assert restored == cr


# ---------------------------------------------------------------------------
# RiskFlagTimelinessResult
# ---------------------------------------------------------------------------


class TestRiskFlagTimelinessResult:
    def test_defaults(self):
        rf = RiskFlagTimelinessResult(
            average_delay_turns=1.5,
            flags_raised_count=3,
            flags_expected_count=4,
        )
        assert rf.per_flag_timing == []

    def test_all_fields(self):
        rf = RiskFlagTimelinessResult(
            per_flag_timing=[
                {
                    "flag_text": "Impersonation risk elevated",
                    "raised_turn": 5,
                    "evidence_available_turn": 3,
                    "delay_turns": 2,
                }
            ],
            average_delay_turns=2.0,
            flags_raised_count=1,
            flags_expected_count=2,
        )
        assert len(rf.per_flag_timing) == 1
        assert rf.per_flag_timing[0]["delay_turns"] == 2

    def test_json_round_trip(self):
        rf = RiskFlagTimelinessResult(
            per_flag_timing=[{"flag_text": "step-up", "raised_turn": 2, "delay_turns": 0}],
            average_delay_turns=0.0,
            flags_raised_count=1,
            flags_expected_count=1,
        )
        restored = RiskFlagTimelinessResult.model_validate_json(rf.model_dump_json())
        assert restored == rf


# ---------------------------------------------------------------------------
# DecisionExplanation
# ---------------------------------------------------------------------------


class TestDecisionExplanation:
    def test_defaults(self):
        de = DecisionExplanation()
        assert de.reasoning_chain == ""
        assert de.influential_evidence == []
        assert de.improvement_suggestions == []
        assert de.overall_quality_notes == ""

    def test_all_fields(self):
        de = DecisionExplanation(
            reasoning_chain="Chip+PIN auth contradicts CM claim of unauthorized use.",
            influential_evidence=[
                {"node_id": "auth-001", "influence": "high", "description": "chip+PIN verified"}
            ],
            improvement_suggestions=["Probe for external manipulator earlier"],
            overall_quality_notes="Strong evidence utilization, timely convergence.",
        )
        assert "Chip+PIN" in de.reasoning_chain
        assert len(de.influential_evidence) == 1
        assert len(de.improvement_suggestions) == 1

    def test_json_round_trip(self):
        de = DecisionExplanation(
            reasoning_chain="Merchant dispute with delivery proof.",
            influential_evidence=[{"node_id": "dp-001", "influence": "medium"}],
            improvement_suggestions=["Ask about return policy earlier"],
            overall_quality_notes="Good.",
        )
        restored = DecisionExplanation.model_validate_json(de.model_dump_json())
        assert restored == de


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------


class TestEvaluationReport:
    def test_defaults_none_sub_reports(self):
        report = EvaluationReport(scenario_name="test", overall_score=0.82)
        assert report.latency is None
        assert report.prediction is None
        assert report.question_adherence is None
        assert report.allegation_quality is None
        assert report.evidence_utilization is None
        assert report.convergence is None
        assert report.risk_flag_timeliness is None
        assert report.decision_explanation is None
        assert report.generated_at == ""

    def test_with_populated_sub_reports(self):
        latency = LatencyReport(
            per_turn_latency_ms=[100.0, 200.0],
            p50_ms=150.0,
            p95_ms=195.0,
            p99_ms=199.0,
            max_ms=200.0,
            compliance_rate=1.0,
        )
        prediction = PredictionResult(
            predicted_category="THIRD_PARTY_FRAUD",
            ground_truth_category="THIRD_PARTY_FRAUD",
            match=True,
            confidence_delta=0.4,
        )
        report = EvaluationReport(
            scenario_name="full_report",
            overall_score=0.91,
            latency=latency,
            prediction=prediction,
            generated_at="2026-03-23T12:00:00Z",
        )
        assert report.latency is not None
        assert report.prediction is not None
        assert report.prediction.match is True
        assert report.convergence is None

    def test_json_round_trip(self):
        report = EvaluationReport(
            scenario_name="round_trip",
            overall_score=0.75,
            latency=LatencyReport(
                per_turn_latency_ms=[500.0],
                p50_ms=500.0,
                p95_ms=500.0,
                p99_ms=500.0,
                max_ms=500.0,
                compliance_rate=1.0,
            ),
            prediction=PredictionResult(
                predicted_category="DISPUTE",
                ground_truth_category="DISPUTE",
                match=True,
                confidence_delta=0.5,
            ),
            convergence=ConvergenceResult(
                convergence_turn=4,
                total_turns=10,
                convergence_ratio=0.4,
                correct_category="DISPUTE",
            ),
            decision_explanation=DecisionExplanation(
                reasoning_chain="Merchant failed to deliver.",
            ),
            generated_at="2026-03-23T12:00:00Z",
        )
        json_str = report.model_dump_json()
        restored = EvaluationReport.model_validate_json(json_str)
        assert restored == report
        assert restored.latency is not None
        assert restored.convergence is not None
        assert restored.question_adherence is None
