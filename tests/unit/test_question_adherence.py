"""Tests for the probing question lifecycle evaluator."""

from __future__ import annotations

from agentic_fraud_servicing.evaluation.models import (
    EvaluationRun,
    QuestionAdherenceResult,
    TurnMetric,
)
from agentic_fraud_servicing.evaluation.question_adherence import (
    evaluate_question_adherence,
)


def _make_turn(
    turn_number: int,
    speaker: str,
    text: str = "",
    copilot_suggestion: dict | None = None,
) -> TurnMetric:
    """Build a TurnMetric with optional copilot suggestion."""
    return TurnMetric(
        turn_number=turn_number,
        speaker=speaker,
        text=text or f"Turn {turn_number}",
        latency_ms=500.0,
        copilot_suggestion=copilot_suggestion,
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


def _pq(text: str, status: str, target: str = "", reason: str = "") -> dict:
    """Shorthand to build a probing question dict."""
    return {
        "text": text,
        "status": status,
        "target_category": target,
        "reason": reason,
        "turn_suggested": 1,
        "assessment_suggested": 1,
    }


class TestLifecycleExtraction:
    """Extracts probing question stats from the last copilot suggestion."""

    def test_all_answered(self):
        suggestion = {
            "probing_questions": [
                _pq("Q1", "answered", "SCAM"),
                _pq("Q2", "answered", "THIRD_PARTY_FRAUD"),
            ],
            "information_sufficient": True,
        }
        turns = [
            _make_turn(1, "CARDMEMBER"),
            _make_turn(10, "CARDMEMBER", copilot_suggestion=suggestion),
        ]
        result = evaluate_question_adherence(_make_run(turns))

        assert isinstance(result, QuestionAdherenceResult)
        assert result.total_questions == 2
        assert result.answered == 2
        assert result.invalidated == 0
        assert result.skipped == 0
        assert result.pending == 0
        assert result.information_sufficient is True
        assert result.overall_adherence_rate == 1.0

    def test_mixed_statuses(self):
        suggestion = {
            "probing_questions": [
                _pq("Q1", "answered", "SCAM"),
                _pq("Q2", "invalidated", "DISPUTE", "hypothesis collapsed"),
                _pq("Q3", "skipped", "THIRD_PARTY_FRAUD", "CCP did not ask"),
                _pq("Q4", "pending", "SCAM"),
            ],
            "information_sufficient": False,
        }
        turns = [_make_turn(5, "CCP", copilot_suggestion=suggestion)]
        result = evaluate_question_adherence(_make_run(turns))

        assert result.total_questions == 4
        assert result.answered == 1
        assert result.invalidated == 1
        assert result.skipped == 1
        assert result.pending == 1
        assert result.information_sufficient is False
        assert result.overall_adherence_rate == 0.25

    def test_uses_last_suggestion(self):
        """When multiple suggestions exist, uses the last one with probing_questions."""
        early = {
            "probing_questions": [_pq("Old Q", "pending", "SCAM")],
            "information_sufficient": False,
        }
        late = {
            "probing_questions": [
                _pq("Old Q", "answered", "SCAM"),
                _pq("New Q", "pending", "DISPUTE"),
            ],
            "information_sufficient": False,
        }
        turns = [
            _make_turn(1, "CARDMEMBER", copilot_suggestion=early),
            _make_turn(5, "CARDMEMBER"),
            _make_turn(10, "CCP", copilot_suggestion=late),
        ]
        result = evaluate_question_adherence(_make_run(turns))

        assert result.total_questions == 2
        assert result.answered == 1
        assert result.pending == 1


class TestNoQuestions:
    """No probing questions in the run."""

    def test_no_suggestions_at_all(self):
        turns = [_make_turn(1, "CARDMEMBER"), _make_turn(2, "CCP")]
        result = evaluate_question_adherence(_make_run(turns))

        assert result.total_questions == 0
        assert result.overall_adherence_rate == 0.0
        assert result.probing_questions == []

    def test_suggestions_without_probing_questions(self):
        suggestion = {"suggested_questions": ["Q1"], "probing_questions": []}
        turns = [_make_turn(1, "CCP", copilot_suggestion=suggestion)]
        result = evaluate_question_adherence(_make_run(turns))

        assert result.total_questions == 0
        assert result.probing_questions == []


class TestQuestionDetails:
    """Verify the probing_questions list is preserved in the result."""

    def test_contains_full_question_data(self):
        pqs = [
            _pq("Ask about phishing texts", "answered", "SCAM", "CM described the text"),
            _pq("Confirm card chip type", "skipped", "THIRD_PARTY_FRAUD", "staleness window"),
        ]
        suggestion = {"probing_questions": pqs, "information_sufficient": True}
        turns = [_make_turn(1, "CCP", copilot_suggestion=suggestion)]
        result = evaluate_question_adherence(_make_run(turns))

        assert len(result.probing_questions) == 2
        assert result.probing_questions[0]["text"] == "Ask about phishing texts"
        assert result.probing_questions[0]["status"] == "answered"
        assert result.probing_questions[0]["reason"] == "CM described the text"
        assert result.probing_questions[1]["status"] == "skipped"
