"""Tests for gateway/tools/read_tools.py: read-only data access tools."""

from datetime import datetime, timezone

import pytest

from agentic_fraud_servicing.gateway.tool_gateway import (
    AuthContext,
    GatewayAuthError,
    ToolGateway,
)
from agentic_fraud_servicing.gateway.tools.read_tools import (
    fetch_customer_profile,
    lookup_transactions,
    query_auth_logs,
)
from agentic_fraud_servicing.models.enums import EvidenceSourceType
from agentic_fraud_servicing.models.evidence import AuthEvent, Customer, Transaction
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
def read_ctx():
    """Auth context with read permission."""
    return AuthContext(agent_id="test-agent", case_id="case-1", permissions={"read"})


@pytest.fixture()
def no_read_ctx():
    """Auth context without read permission."""
    return AuthContext(agent_id="test-agent", case_id="case-1", permissions={"write"})


def _add_transaction(es: EvidenceStore, case_id: str, node_id: str, pan: str = "") -> None:
    """Helper to add a Transaction node to the evidence store."""
    txn = Transaction(
        node_id=node_id,
        case_id=case_id,
        source_type=EvidenceSourceType.FACT,
        created_at=NOW,
        amount=99.99,
        merchant_name=f"Merchant {pan}" if pan else "Test Merchant",
        transaction_date=NOW,
    )
    es.add_node(txn)


def _add_auth_event(es: EvidenceStore, case_id: str, node_id: str) -> None:
    """Helper to add an AuthEvent node to the evidence store."""
    event = AuthEvent(
        node_id=node_id,
        case_id=case_id,
        source_type=EvidenceSourceType.FACT,
        created_at=NOW,
        auth_type="OTP",
        result="success",
        timestamp=NOW,
    )
    es.add_node(event)


def _add_customer(es: EvidenceStore, case_id: str, node_id: str) -> None:
    """Helper to add a Customer node to the evidence store."""
    customer = Customer(
        node_id=node_id,
        case_id=case_id,
        source_type=EvidenceSourceType.FACT,
        created_at=NOW,
        profile_hash="hash-abc",
    )
    es.add_node(customer)


# --- lookup_transactions tests ---


class TestLookupTransactions:
    def test_returns_only_transaction_nodes(self, gateway, stores, read_ctx):
        """Filters to TRANSACTION nodes, excludes other types."""
        _, es, _ = stores
        _add_transaction(es, "case-1", "txn-1")
        _add_auth_event(es, "case-1", "auth-1")
        _add_customer(es, "case-1", "cust-1")

        result = lookup_transactions(gateway, read_ctx, "case-1")
        assert len(result) == 1
        assert result[0]["node_type"] == "TRANSACTION"

    def test_masks_pan_in_output(self, gateway, stores, read_ctx):
        """PAN patterns in transaction dicts are masked."""
        _, es, _ = stores
        # Create a transaction with a PAN in the merchant_name field
        txn = Transaction(
            node_id="txn-pan",
            case_id="case-1",
            source_type=EvidenceSourceType.FACT,
            created_at=NOW,
            amount=50.0,
            merchant_name="Card 4111111111111111 used",
            transaction_date=NOW,
        )
        es.add_node(txn)

        result = lookup_transactions(gateway, read_ctx, "case-1")
        assert len(result) == 1
        assert "4111111111111111" not in result[0]["merchant_name"]
        assert "[PAN_REDACTED]" in result[0]["merchant_name"]

    def test_auth_failure_raises(self, gateway, no_read_ctx):
        """Raises GatewayAuthError when ctx lacks 'read' permission."""
        with pytest.raises(GatewayAuthError, match="Permission 'read' not granted"):
            lookup_transactions(gateway, no_read_ctx, "case-1")

    def test_empty_case_returns_empty_list(self, gateway, read_ctx):
        """Returns empty list when no evidence nodes exist for the case."""
        result = lookup_transactions(gateway, read_ctx, "nonexistent-case")
        assert result == []

    def test_logs_call_to_trace_store(self, gateway, stores, read_ctx):
        """The call is logged to the trace store."""
        _, es, ts = stores
        _add_transaction(es, "case-1", "txn-1")

        lookup_transactions(gateway, read_ctx, "case-1")

        traces = ts.get_traces_by_case("case-1")
        assert len(traces) == 1
        assert traces[0]["action"] == "lookup_transactions"
        assert traces[0]["agent_id"] == "test-agent"

    def test_disputed_flag_in_returned_transactions(self, gateway, stores, read_ctx):
        """Disputed transactions should have is_disputed=True in returned dicts."""
        _, es, _ = stores
        txn = Transaction(
            node_id="txn-disp",
            case_id="case-1",
            source_type=EvidenceSourceType.FACT,
            created_at=NOW,
            amount=100.0,
            merchant_name="Test",
            transaction_date=NOW,
            is_disputed=True,
        )
        es.add_node(txn)

        result = lookup_transactions(gateway, read_ctx, "case-1")
        assert len(result) == 1
        assert result[0]["is_disputed"] is True

    def test_undisputed_default(self, gateway, stores, read_ctx):
        """Default transactions should have is_disputed=False."""
        _, es, _ = stores
        _add_transaction(es, "case-1", "txn-1")

        result = lookup_transactions(gateway, read_ctx, "case-1")
        assert len(result) == 1
        assert result[0]["is_disputed"] is False


# --- query_auth_logs tests ---


class TestQueryAuthLogs:
    def test_returns_only_auth_event_nodes(self, gateway, stores, read_ctx):
        """Filters to AUTH_EVENT nodes, excludes other types."""
        _, es, _ = stores
        _add_auth_event(es, "case-1", "auth-1")
        _add_auth_event(es, "case-1", "auth-2")
        _add_transaction(es, "case-1", "txn-1")

        result = query_auth_logs(gateway, read_ctx, "case-1")
        assert len(result) == 2
        assert all(n["node_type"] == "AUTH_EVENT" for n in result)

    def test_auth_failure_raises(self, gateway, no_read_ctx):
        """Raises GatewayAuthError when ctx lacks 'read' permission."""
        with pytest.raises(GatewayAuthError, match="Permission 'read' not granted"):
            query_auth_logs(gateway, no_read_ctx, "case-1")

    def test_empty_case_returns_empty_list(self, gateway, read_ctx):
        """Returns empty list when no auth events exist."""
        result = query_auth_logs(gateway, read_ctx, "nonexistent-case")
        assert result == []

    def test_logs_call_to_trace_store(self, gateway, stores, read_ctx):
        """The call is logged to the trace store."""
        _, _, ts = stores

        query_auth_logs(gateway, read_ctx, "case-1")

        traces = ts.get_traces_by_case("case-1")
        assert len(traces) == 1
        assert traces[0]["action"] == "query_auth_logs"


# --- fetch_customer_profile tests ---


class TestFetchCustomerProfile:
    def test_returns_first_customer_node(self, gateway, stores, read_ctx):
        """Returns the first CUSTOMER node for the case."""
        _, es, _ = stores
        _add_customer(es, "case-1", "cust-1")
        _add_transaction(es, "case-1", "txn-1")

        result = fetch_customer_profile(gateway, read_ctx, "case-1")
        assert result is not None
        assert result["node_type"] == "CUSTOMER"
        assert result["profile_hash"] == "hash-abc"

    def test_returns_none_when_no_customer(self, gateway, read_ctx):
        """Returns None when no CUSTOMER nodes exist for the case."""
        result = fetch_customer_profile(gateway, read_ctx, "nonexistent-case")
        assert result is None

    def test_auth_failure_raises(self, gateway, no_read_ctx):
        """Raises GatewayAuthError when ctx lacks 'read' permission."""
        with pytest.raises(GatewayAuthError, match="Permission 'read' not granted"):
            fetch_customer_profile(gateway, no_read_ctx, "case-1")

    def test_logs_call_to_trace_store(self, gateway, stores, read_ctx):
        """The call is logged to the trace store."""
        _, es, ts = stores
        _add_customer(es, "case-1", "cust-1")

        fetch_customer_profile(gateway, read_ctx, "case-1")

        traces = ts.get_traces_by_case("case-1")
        assert len(traces) == 1
        assert traces[0]["action"] == "fetch_customer_profile"

    def test_masks_pan_in_customer_profile(self, gateway, stores, read_ctx):
        """PAN patterns in customer node dicts are masked."""
        _, es, _ = stores
        customer = Customer(
            node_id="cust-pan",
            case_id="case-1",
            source_type=EvidenceSourceType.FACT,
            created_at=NOW,
            profile_hash="hash-xyz",
            risk_indicators=["card 371449635398431 compromised"],
        )
        es.add_node(customer)

        result = fetch_customer_profile(gateway, read_ctx, "case-1")
        assert result is not None
        assert "371449635398431" not in str(result)
        assert "[PAN_REDACTED]" in str(result)
