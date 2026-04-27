"""Tests for ui.dashboard_data — dashboard data loader functions."""

import json
import os
from datetime import datetime, timezone

import pytest

from agentic_fraud_servicing.models.case import Case
from agentic_fraud_servicing.models.enums import (
    AllegationType,
    AuthMethod,
    CaseStatus,
    EvidenceEdgeType,
    EvidenceSourceType,
    TransactionChannel,
)
from agentic_fraud_servicing.models.evidence import EvidenceEdge, Transaction
from agentic_fraud_servicing.storage.case_store import CaseStore
from agentic_fraud_servicing.storage.evidence_store import EvidenceStore
from agentic_fraud_servicing.storage.trace_store import TraceStore
from agentic_fraud_servicing.ui.dashboard_data import (
    discover_scenarios,
    load_audit_trail,
    load_case,
    load_copilot_final_state,
    load_copilot_suggestions,
    load_evidence,
    load_transcript_turns,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CASE_ID = "case-test-dash-001"
CALL_ID = "call-test-dash-001"


def _seed_case(db_dir: str) -> None:
    """Seed a Case into the case store."""
    store = CaseStore(os.path.join(db_dir, "cases.db"))
    case = Case(
        case_id=CASE_ID,
        call_id=CALL_ID,
        customer_id="cust-001",
        account_id="acct-001",
        allegation_type=AllegationType.FRAUD,
        status=CaseStatus.OPEN,
        created_at=datetime(2026, 3, 9, tzinfo=timezone.utc),
    )
    store.create_case(case)
    store.close()


def _seed_evidence(db_dir: str) -> None:
    """Seed evidence nodes and edges."""
    store = EvidenceStore(os.path.join(db_dir, "evidence.db"))
    node = Transaction(
        node_id="txn-001",
        case_id=CASE_ID,
        source_type=EvidenceSourceType.FACT,
        created_at=datetime(2026, 3, 2, tzinfo=timezone.utc),
        amount=2847.99,
        merchant_name="TechVault Electronics",
        transaction_date=datetime(2026, 3, 2, tzinfo=timezone.utc),
        auth_method=AuthMethod.CHIP,
        channel=TransactionChannel.POS,
    )
    store.add_node(node)
    edge = EvidenceEdge(
        edge_id="edge-001",
        case_id=CASE_ID,
        source_node_id="txn-001",
        target_node_id="auth-001",
        edge_type=EvidenceEdgeType.SUPPORTS,
        created_at=datetime(2026, 3, 2, tzinfo=timezone.utc),
    )
    store.add_edge(edge)
    store.close()


def _seed_traces(db_dir: str) -> None:
    """Seed trace records matching simulation runner conventions."""
    store = TraceStore(os.path.join(db_dir, "traces.db"))

    # Transcript turns
    for turn_num in range(1, 4):
        speaker = "CCP" if turn_num % 2 == 1 else "CARDMEMBER"
        store.log_invocation(
            trace_id=f"tr-turn-{turn_num}",
            case_id=CASE_ID,
            agent_id="transcript",
            action="conversation_turn",
            input_data=json.dumps({"turn": turn_num, "speaker": speaker}),
            output_data=json.dumps(
                {
                    "turn": turn_num,
                    "speaker": speaker,
                    "text": f"Turn {turn_num} text by {speaker}",
                }
            ),
            duration_ms=0.0,
            timestamp=datetime(2026, 3, 9, 12, 0, turn_num, tzinfo=timezone.utc),
            status="success",
        )

    # Copilot suggestions
    for turn_num in range(1, 4):
        store.log_invocation(
            trace_id=f"tr-copilot-{turn_num}",
            case_id=CASE_ID,
            agent_id="copilot_suggestion",
            action="suggestion",
            input_data=json.dumps({"turn": turn_num}),
            output_data=json.dumps(
                {
                    "hypothesis_scores": {"THIRD_PARTY_FRAUD": 0.5, "FIRST_PARTY_FRAUD": 0.1},
                    "specialist_likelihoods": {
                        "THIRD_PARTY_FRAUD": 0.4,
                        "SCAM": 0.2,
                        "DISPUTE": 0.3,
                    },
                    "suggested_questions": [f"Question for turn {turn_num}"],
                    "risk_flags": [],
                }
            ),
            duration_ms=0.0,
            timestamp=datetime(2026, 3, 9, 12, 1, turn_num, tzinfo=timezone.utc),
            status="success",
        )

    # Copilot final state
    store.log_invocation(
        trace_id="tr-copilot-final",
        case_id=CASE_ID,
        agent_id="copilot_final",
        action="final_state",
        input_data="{}",
        output_data=json.dumps(
            {
                "hypothesis_scores": {"THIRD_PARTY_FRAUD": 0.3, "FIRST_PARTY_FRAUD": 0.6},
                "impersonation_risk": 0.1,
                "evidence_collected": ["txn-001"],
            }
        ),
        duration_ms=0.0,
        timestamp=datetime(2026, 3, 9, 12, 2, 0, tzinfo=timezone.utc),
        status="success",
    )

    store.close()


@pytest.fixture()
def seeded_dir(tmp_path):
    """Create a fully seeded scenario directory."""
    db_dir = str(tmp_path / "scenario_test")
    os.makedirs(db_dir, exist_ok=True)
    _seed_case(db_dir)
    _seed_evidence(db_dir)
    _seed_traces(db_dir)
    return db_dir


# ---------------------------------------------------------------------------
# Tests: discover_scenarios
# ---------------------------------------------------------------------------


class TestDiscoverScenarios:
    def test_finds_scenarios_with_cases_db(self, tmp_path):
        """Discovers directories containing cases.db."""
        s1 = tmp_path / "alpha"
        s1.mkdir()
        (s1 / "cases.db").touch()

        s2 = tmp_path / "beta"
        s2.mkdir()
        (s2 / "cases.db").touch()

        # Directory without cases.db should be excluded
        s3 = tmp_path / "gamma"
        s3.mkdir()

        result = discover_scenarios(str(tmp_path))
        assert result == ["alpha", "beta"]

    def test_returns_empty_for_nonexistent_dir(self):
        """Returns empty list when base_dir doesn't exist."""
        assert discover_scenarios("/nonexistent/path") == []

    def test_returns_empty_for_empty_dir(self, tmp_path):
        """Returns empty list when no scenarios found."""
        assert discover_scenarios(str(tmp_path)) == []

    def test_returns_sorted(self, tmp_path):
        """Results are sorted alphabetically."""
        for name in ["zebra", "apple", "mango"]:
            d = tmp_path / name
            d.mkdir()
            (d / "cases.db").touch()

        result = discover_scenarios(str(tmp_path))
        assert result == ["apple", "mango", "zebra"]


# ---------------------------------------------------------------------------
# Tests: load_case
# ---------------------------------------------------------------------------


class TestLoadCase:
    def test_loads_case_dict(self, seeded_dir):
        """Loads case and returns dict with expected fields."""
        result = load_case(seeded_dir)
        assert result is not None
        assert result["case_id"] == CASE_ID
        assert result["allegation_type"] == "FRAUD"
        assert result["status"] == "OPEN"

    def test_returns_none_for_missing_db(self, tmp_path):
        """Returns None when cases.db doesn't exist."""
        assert load_case(str(tmp_path)) is None

    def test_returns_none_for_empty_db(self, tmp_path):
        """Returns None when cases table has no rows."""
        store = CaseStore(os.path.join(str(tmp_path), "cases.db"))
        store.close()
        assert load_case(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Tests: load_transcript_turns
# ---------------------------------------------------------------------------


class TestLoadTranscriptTurns:
    def test_loads_sorted_turns(self, seeded_dir):
        """Loads transcript turns sorted by turn number."""
        turns = load_transcript_turns(seeded_dir, CASE_ID)
        assert len(turns) == 3
        assert turns[0]["turn"] == 1
        assert turns[0]["speaker"] == "CCP"
        assert "Turn 1" in turns[0]["text"]
        assert turns[1]["turn"] == 2
        assert turns[2]["turn"] == 3

    def test_returns_empty_for_missing_db(self, tmp_path):
        """Returns empty list when traces.db doesn't exist."""
        assert load_transcript_turns(str(tmp_path), CASE_ID) == []

    def test_returns_empty_for_wrong_case(self, seeded_dir):
        """Returns empty list for a case_id with no transcript traces."""
        assert load_transcript_turns(seeded_dir, "nonexistent-case") == []


# ---------------------------------------------------------------------------
# Tests: load_copilot_suggestions
# ---------------------------------------------------------------------------


class TestLoadCopilotSuggestions:
    def test_loads_sorted_suggestions(self, seeded_dir):
        """Loads copilot suggestions sorted by turn number."""
        suggestions = load_copilot_suggestions(seeded_dir, CASE_ID)
        assert len(suggestions) == 3
        assert suggestions[0]["turn"] == 1
        assert "hypothesis_scores" in suggestions[0]["suggestion"]
        assert suggestions[2]["turn"] == 3

    def test_returns_empty_for_missing_db(self, tmp_path):
        """Returns empty list when traces.db doesn't exist."""
        assert load_copilot_suggestions(str(tmp_path), CASE_ID) == []


# ---------------------------------------------------------------------------
# Tests: load_copilot_final_state
# ---------------------------------------------------------------------------


class TestLoadCopilotFinalState:
    def test_loads_final_state(self, seeded_dir):
        """Loads final copilot state with hypothesis scores."""
        state = load_copilot_final_state(seeded_dir, CASE_ID)
        assert state is not None
        assert state["hypothesis_scores"]["FIRST_PARTY_FRAUD"] == 0.6
        assert state["impersonation_risk"] == 0.1

    def test_returns_none_for_missing_db(self, tmp_path):
        """Returns None when traces.db doesn't exist."""
        assert load_copilot_final_state(str(tmp_path), CASE_ID) is None

    def test_returns_none_for_wrong_case(self, seeded_dir):
        """Returns None for a case_id with no copilot_final trace."""
        assert load_copilot_final_state(seeded_dir, "nonexistent-case") is None


# ---------------------------------------------------------------------------
# Tests: load_evidence
# ---------------------------------------------------------------------------


class TestLoadEvidence:
    def test_loads_nodes_and_edges(self, seeded_dir):
        """Loads evidence nodes and edges for the case."""
        nodes, edges = load_evidence(seeded_dir, CASE_ID)
        assert len(nodes) == 1
        assert nodes[0]["node_id"] == "txn-001"
        assert nodes[0]["amount"] == 2847.99
        assert len(edges) == 1
        assert edges[0]["edge_id"] == "edge-001"

    def test_returns_empty_for_missing_db(self, tmp_path):
        """Returns empty tuples when evidence.db doesn't exist."""
        nodes, edges = load_evidence(str(tmp_path), CASE_ID)
        assert nodes == []
        assert edges == []


# ---------------------------------------------------------------------------
# Tests: load_audit_trail
# ---------------------------------------------------------------------------


class TestLoadAuditTrail:
    def test_loads_all_traces(self, seeded_dir):
        """Loads all trace records for the case."""
        trail = load_audit_trail(seeded_dir, CASE_ID)
        # 3 transcript + 3 copilot_suggestion + 1 copilot_final = 7
        assert len(trail) == 7

    def test_ordered_by_timestamp(self, seeded_dir):
        """Traces are ordered by timestamp ascending."""
        trail = load_audit_trail(seeded_dir, CASE_ID)
        timestamps = [t["timestamp"] for t in trail]
        assert timestamps == sorted(timestamps)

    def test_returns_empty_for_missing_db(self, tmp_path):
        """Returns empty list when traces.db doesn't exist."""
        assert load_audit_trail(str(tmp_path), CASE_ID) == []
