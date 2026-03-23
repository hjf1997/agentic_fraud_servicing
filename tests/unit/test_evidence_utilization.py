"""Tests for evaluation/evidence_utilization.py."""

from agentic_fraud_servicing.evaluation.evidence_utilization import (
    evaluate_evidence_utilization,
)
from agentic_fraud_servicing.evaluation.models import EvaluationRun, TurnMetric


def _make_run(
    ground_truth: dict,
    suggestions: list[dict | None],
) -> EvaluationRun:
    """Build a minimal EvaluationRun from a list of copilot_suggestion dicts."""
    metrics = [
        TurnMetric(
            turn_number=i + 1,
            speaker="CARDMEMBER",
            text=f"Turn {i + 1}",
            latency_ms=100.0,
            copilot_suggestion=s,
        )
        for i, s in enumerate(suggestions)
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


class TestAllEvidenceUsed:
    """All key evidence nodes retrieved and referenced."""

    def test_full_coverage(self):
        run = _make_run(
            {"key_evidence_nodes": ["node-1", "node-2"]},
            [
                {
                    "retrieved_facts": [{"node_id": "node-1"}, {"node_id": "node-2"}],
                    "running_summary": "Analysis of node-1 and node-2 shows fraud.",
                    "risk_flags": [],
                },
            ],
        )
        result = evaluate_evidence_utilization(run)
        assert result.total_evidence_nodes == 2
        assert result.retrieved_nodes == 2
        assert result.referenced_in_reasoning == 2
        assert result.retrieval_coverage == 1.0
        assert result.reasoning_coverage == 1.0
        assert result.missed_evidence == []


class TestPartialCoverage:
    """Only some evidence nodes retrieved or referenced."""

    def test_half_retrieved(self):
        run = _make_run(
            {"key_evidence_nodes": ["node-1", "node-2", "node-3", "node-4"]},
            [
                {
                    "retrieved_facts": [{"node_id": "node-1"}, {"node_id": "node-2"}],
                    "running_summary": "Found node-1 relevant.",
                    "risk_flags": [],
                },
            ],
        )
        result = evaluate_evidence_utilization(run)
        assert result.retrieved_nodes == 2
        assert result.retrieval_coverage == 0.5
        assert result.referenced_in_reasoning == 1
        assert result.reasoning_coverage == 0.25

    def test_reference_in_risk_flags(self):
        """Evidence referenced in risk_flags counts toward reasoning coverage."""
        run = _make_run(
            {"key_evidence_nodes": ["node-x"]},
            [
                {
                    "retrieved_facts": [],
                    "running_summary": "",
                    "risk_flags": ["High risk: node-x contradicts claim"],
                },
            ],
        )
        result = evaluate_evidence_utilization(run)
        assert result.referenced_in_reasoning == 1
        assert result.reasoning_coverage == 1.0


class TestNoEvidenceInGroundTruth:
    """Ground truth has no key_evidence_nodes."""

    def test_returns_zeros(self):
        run = _make_run(
            {"investigation_category": "SCAM"},
            [{"retrieved_facts": [{"node_id": "a"}], "running_summary": "stuff"}],
        )
        result = evaluate_evidence_utilization(run)
        assert result.total_evidence_nodes == 0
        assert result.retrieval_coverage == 0.0
        assert result.reasoning_coverage == 0.0
        assert result.missed_evidence == []


class TestNoRetrievedFacts:
    """Copilot suggestions have no retrieved_facts."""

    def test_zero_retrieval_coverage(self):
        run = _make_run(
            {"key_evidence_nodes": ["node-1", "node-2"]},
            [
                {"retrieved_facts": [], "running_summary": "", "risk_flags": []},
                {"retrieved_facts": [], "running_summary": "", "risk_flags": []},
            ],
        )
        result = evaluate_evidence_utilization(run)
        assert result.retrieved_nodes == 0
        assert result.retrieval_coverage == 0.0
        assert len(result.missed_evidence) == 2


class TestMissedEvidenceList:
    """Missed evidence list is correct."""

    def test_missed_ids(self):
        run = _make_run(
            {"key_evidence_nodes": ["node-a", "node-b", "node-c"]},
            [
                {
                    "retrieved_facts": [{"node_id": "node-a"}],
                    "running_summary": "Analyzed node-a.",
                    "risk_flags": [],
                },
            ],
        )
        result = evaluate_evidence_utilization(run)
        missed_ids = [m["node_id"] for m in result.missed_evidence]
        assert "node-b" in missed_ids
        assert "node-c" in missed_ids
        assert "node-a" not in missed_ids


class TestEmptyRun:
    """No turn metrics at all."""

    def test_empty_with_key_nodes(self):
        run = _make_run({"key_evidence_nodes": ["node-1"]}, [])
        result = evaluate_evidence_utilization(run)
        assert result.retrieved_nodes == 0
        assert result.retrieval_coverage == 0.0
        assert len(result.missed_evidence) == 1

    def test_empty_without_key_nodes(self):
        run = _make_run({}, [])
        result = evaluate_evidence_utilization(run)
        assert result.total_evidence_nodes == 0
        assert result.missed_evidence == []


class TestStringFacts:
    """Retrieved facts as plain strings instead of dicts."""

    def test_string_facts_matched(self):
        run = _make_run(
            {"key_evidence_nodes": ["node-1"]},
            [{"retrieved_facts": ["node-1"], "running_summary": "", "risk_flags": []}],
        )
        result = evaluate_evidence_utilization(run)
        assert result.retrieved_nodes == 1
        assert result.retrieval_coverage == 1.0
