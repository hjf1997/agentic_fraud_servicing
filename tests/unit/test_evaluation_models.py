"""Tests for evaluation data models."""

from agentic_fraud_servicing.evaluation.models import (
    EvaluationConfig,
    EvaluationRun,
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
