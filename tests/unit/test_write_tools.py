"""Tests for gateway/tools/write_tools.py: write data access tools."""

from datetime import datetime, timezone

import pytest

from agentic_fraud_servicing.gateway.tool_gateway import (
    AuthContext,
    GatewayAuthError,
    ToolGateway,
)
from agentic_fraud_servicing.gateway.tools.write_tools import (
    append_evidence_edge,
    append_evidence_node,
    create_case,
    mark_transactions_disputed,
    update_case_status,
)
from agentic_fraud_servicing.models.case import Case
from agentic_fraud_servicing.models.enums import (
    CaseStatus,
    EvidenceEdgeType,
    EvidenceSourceType,
)
from agentic_fraud_servicing.models.evidence import EvidenceEdge, Transaction
from agentic_fraud_servicing.storage.case_store import CaseStore
from agentic_fraud_servicing.storage.evidence_store import EvidenceStore
from agentic_fraud_servicing.storage.trace_store import TraceStore

NOW = datetime.now(timezone.utc)


@pytest.fixture()
def stores(tmp_path):
    """Create real stores backed by tmp_path SQLite databases."""
    cs = CaseStore(str(tmp_path / "cases.db"))
    es = EvidenceStore(str(tmp_path / "evidence.db"))
    ts = TraceStore(str(tmp_path / "traces.db"))
    yield cs, es, ts
    cs.close()
    es.close()
    ts.close()


@pytest.fixture()
def gateway(stores):
    """Create a ToolGateway with real stores."""
    cs, es, ts = stores
    return ToolGateway(case_store=cs, evidence_store=es, trace_store=ts)


@pytest.fixture()
def write_ctx():
    """Auth context with write permission."""
    return AuthContext(agent_id="test-agent", case_id="case-1", permissions={"write"})


@pytest.fixture()
def no_write_ctx():
    """Auth context without write permission."""
    return AuthContext(agent_id="test-agent", case_id="case-1", permissions={"read"})


def _make_case(case_id: str = "case-1") -> Case:
    """Helper to create a minimal Case model."""
    return Case(
        case_id=case_id,
        call_id="call-1",
        customer_id="cust-1",
        account_id="acct-1",
        created_at=NOW,
    )


def _make_transaction(node_id: str = "txn-1", case_id: str = "case-1") -> Transaction:
    """Helper to create a minimal Transaction node."""
    return Transaction(
        node_id=node_id,
        case_id=case_id,
        source_type=EvidenceSourceType.FACT,
        created_at=NOW,
        amount=99.99,
        merchant_name="Test Merchant",
        transaction_date=NOW,
    )


def _make_edge(
    edge_id: str = "edge-1",
    case_id: str = "case-1",
    source_node_id: str = "txn-1",
    target_node_id: str = "txn-2",
) -> EvidenceEdge:
    """Helper to create a minimal EvidenceEdge."""
    return EvidenceEdge(
        edge_id=edge_id,
        case_id=case_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=EvidenceEdgeType.SUPPORTS,
        created_at=NOW,
    )


# --- create_case tests ---


class TestCreateCase:
    def test_persists_and_retrievable(self, gateway, stores, write_ctx):
        """Case is persisted and retrievable via case_store.get_case."""
        cs, _, _ = stores
        case = _make_case()
        create_case(gateway, write_ctx, case)

        retrieved = cs.get_case("case-1")
        assert retrieved is not None
        assert retrieved.case_id == "case-1"
        assert retrieved.customer_id == "cust-1"

    def test_auth_failure_raises(self, gateway, no_write_ctx):
        """Raises GatewayAuthError when ctx lacks 'write' permission."""
        with pytest.raises(GatewayAuthError, match="Permission 'write' not granted"):
            create_case(gateway, no_write_ctx, _make_case())

    def test_duplicate_raises_runtime_error(self, gateway, write_ctx):
        """Duplicate case_id raises RuntimeError."""
        create_case(gateway, write_ctx, _make_case())
        with pytest.raises(RuntimeError, match="already exists"):
            create_case(gateway, write_ctx, _make_case())

    def test_logs_call_to_trace_store(self, gateway, stores, write_ctx):
        """The call is logged to the trace store."""
        _, _, ts = stores
        create_case(gateway, write_ctx, _make_case())

        traces = ts.get_traces_by_case("case-1")
        assert len(traces) == 1
        assert traces[0]["action"] == "create_case"
        assert traces[0]["agent_id"] == "test-agent"
        assert traces[0]["status"] == "success"


# --- update_case_status tests ---


class TestUpdateCaseStatus:
    def test_status_changes_in_store(self, gateway, stores, write_ctx):
        """Status updates are reflected in the case store."""
        cs, _, _ = stores
        case = _make_case()
        cs.create_case(case)

        update_case_status(gateway, write_ctx, "case-1", CaseStatus.INVESTIGATING)

        retrieved = cs.get_case("case-1")
        assert retrieved is not None
        assert retrieved.status == CaseStatus.INVESTIGATING

    def test_nonexistent_case_raises(self, gateway, write_ctx):
        """Raises RuntimeError when case_id doesn't exist."""
        with pytest.raises(RuntimeError, match="not found"):
            update_case_status(gateway, write_ctx, "nonexistent", CaseStatus.CLOSED)

    def test_logs_call_to_trace_store(self, gateway, stores, write_ctx):
        """The call is logged to the trace store."""
        cs, _, ts = stores
        cs.create_case(_make_case())

        update_case_status(gateway, write_ctx, "case-1", CaseStatus.INVESTIGATING)

        traces = ts.get_traces_by_case("case-1")
        assert len(traces) == 1
        assert traces[0]["action"] == "update_case_status"


# --- append_evidence_node tests ---


class TestAppendEvidenceNode:
    def test_persists_and_retrievable(self, gateway, stores, write_ctx):
        """Node is persisted and retrievable via evidence_store."""
        _, es, _ = stores
        node = _make_transaction()
        append_evidence_node(gateway, write_ctx, node)

        nodes = es.get_nodes_by_case("case-1")
        assert len(nodes) == 1
        assert nodes[0]["node_id"] == "txn-1"
        assert nodes[0]["node_type"] == "TRANSACTION"

    def test_auth_failure_raises(self, gateway, no_write_ctx):
        """Raises GatewayAuthError when ctx lacks 'write' permission."""
        with pytest.raises(GatewayAuthError, match="Permission 'write' not granted"):
            append_evidence_node(gateway, no_write_ctx, _make_transaction())

    def test_logs_call_to_trace_store(self, gateway, stores, write_ctx):
        """The call is logged to the trace store."""
        _, _, ts = stores
        append_evidence_node(gateway, write_ctx, _make_transaction())

        traces = ts.get_traces_by_case("case-1")
        assert len(traces) == 1
        assert traces[0]["action"] == "append_evidence_node"
        assert traces[0]["status"] == "success"


# --- append_evidence_edge tests ---


class TestAppendEvidenceEdge:
    def test_persists_and_retrievable(self, gateway, stores, write_ctx):
        """Edge is persisted and retrievable via evidence_store."""
        _, es, _ = stores
        edge = _make_edge()
        append_evidence_edge(gateway, write_ctx, edge)

        edges = es.get_edges_by_case("case-1")
        assert len(edges) == 1
        assert edges[0]["edge_id"] == "edge-1"
        assert edges[0]["edge_type"] == "SUPPORTS"

    def test_auth_failure_raises(self, gateway, no_write_ctx):
        """Raises GatewayAuthError when ctx lacks 'write' permission."""
        with pytest.raises(GatewayAuthError, match="Permission 'write' not granted"):
            append_evidence_edge(gateway, no_write_ctx, _make_edge())

    def test_logs_call_to_trace_store(self, gateway, stores, write_ctx):
        """The call is logged to the trace store."""
        _, _, ts = stores
        append_evidence_edge(gateway, write_ctx, _make_edge())

        traces = ts.get_traces_by_case("case-1")
        assert len(traces) == 1
        assert traces[0]["action"] == "append_evidence_edge"
        assert traces[0]["status"] == "success"


# --- mark_transactions_disputed tests ---


class TestMarkTransactionsDisputed:
    def test_marks_specified_transactions(self, gateway, stores, write_ctx):
        """Only the specified transaction should be marked as disputed."""
        _, es, _ = stores
        append_evidence_node(gateway, write_ctx, _make_transaction("txn-1"))
        append_evidence_node(gateway, write_ctx, _make_transaction("txn-2"))
        append_evidence_node(gateway, write_ctx, _make_transaction("txn-3"))

        updated = mark_transactions_disputed(gateway, write_ctx, "case-1", ["txn-2"])
        assert updated == 1

        nodes = es.get_nodes_by_case("case-1")
        for n in nodes:
            if n["node_id"] == "txn-2":
                assert n["is_disputed"] is True
            else:
                assert n["is_disputed"] is False

    def test_auth_failure_raises(self, gateway, no_write_ctx):
        """Raises GatewayAuthError when ctx lacks 'write' permission."""
        with pytest.raises(GatewayAuthError, match="Permission 'write' not granted"):
            mark_transactions_disputed(gateway, no_write_ctx, "case-1", ["txn-1"])

    def test_logs_call_to_trace_store(self, gateway, stores, write_ctx):
        """The call is logged to the trace store."""
        _, es, ts = stores
        append_evidence_node(gateway, write_ctx, _make_transaction("txn-1"))

        mark_transactions_disputed(gateway, write_ctx, "case-1", ["txn-1"])

        traces = ts.get_traces_by_case("case-1")
        # 2 traces: one for append_evidence_node, one for mark_transactions_disputed
        mark_trace = next(t for t in traces if t["action"] == "mark_transactions_disputed")
        assert mark_trace is not None
