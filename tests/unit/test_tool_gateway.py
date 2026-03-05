"""Tests for gateway/tool_gateway.py: GatewayAuthError, AuthContext, ToolGateway."""

import pytest

from agentic_fraud_servicing.gateway.tool_gateway import (
    AuthContext,
    GatewayAuthError,
    ToolGateway,
)
from agentic_fraud_servicing.storage.case_store import CaseStore
from agentic_fraud_servicing.storage.evidence_store import EvidenceStore
from agentic_fraud_servicing.storage.trace_store import TraceStore


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


# --- GatewayAuthError tests ---


class TestGatewayAuthError:
    def test_is_runtime_error_subclass(self):
        err = GatewayAuthError("test")
        assert isinstance(err, RuntimeError)

    def test_str_includes_context(self):
        err = GatewayAuthError("denied", agent_id="agent-1", action="write")
        result = str(err)
        assert "denied" in result
        assert "agent_id=agent-1" in result
        assert "action=write" in result

    def test_str_without_context(self):
        err = GatewayAuthError("plain error")
        assert str(err) == "plain error"


# --- AuthContext tests ---


class TestAuthContext:
    def test_creation_with_defaults(self):
        ctx = AuthContext(agent_id="triage-agent")
        assert ctx.agent_id == "triage-agent"
        assert ctx.case_id is None
        assert ctx.permissions == set()

    def test_creation_with_all_fields(self):
        ctx = AuthContext(
            agent_id="retrieval-agent",
            case_id="case-123",
            permissions={"read", "compliance"},
        )
        assert ctx.agent_id == "retrieval-agent"
        assert ctx.case_id == "case-123"
        assert ctx.permissions == {"read", "compliance"}


# --- ToolGateway tests ---


class TestToolGatewayInit:
    def test_stores_accessible_via_properties(self, gateway, stores):
        cs, es, ts = stores
        assert gateway.case_store is cs
        assert gateway.evidence_store is es
        assert gateway.trace_store is ts


class TestCheckAuth:
    def test_passes_with_valid_permission(self, gateway):
        ctx = AuthContext(agent_id="agent-1", permissions={"read", "write"})
        # Should not raise
        gateway.check_auth(ctx, "read")
        gateway.check_auth(ctx, "write")

    def test_raises_on_missing_permission(self, gateway):
        ctx = AuthContext(agent_id="agent-1", permissions={"read"})
        with pytest.raises(GatewayAuthError, match="Permission 'write' not granted"):
            gateway.check_auth(ctx, "write")

    def test_raises_on_empty_agent_id(self, gateway):
        ctx = AuthContext(agent_id="", permissions={"read"})
        with pytest.raises(GatewayAuthError, match="agent_id must not be empty"):
            gateway.check_auth(ctx, "read")


class TestLogCall:
    def test_writes_trace_to_store(self, gateway, stores):
        _, _, ts = stores
        ctx = AuthContext(agent_id="triage-agent", case_id="case-42")
        gateway.log_call(
            ctx=ctx,
            action="lookup_transactions",
            input_summary='{"case_id": "case-42"}',
            output_summary='{"count": 3}',
            duration_ms=45.2,
        )

        traces = ts.get_traces_by_case("case-42")
        assert len(traces) == 1
        trace = traces[0]
        assert trace["agent_id"] == "triage-agent"
        assert trace["case_id"] == "case-42"
        assert trace["action"] == "lookup_transactions"
        assert trace["status"] == "success"

    def test_uses_unknown_when_no_case_id(self, gateway, stores):
        _, _, ts = stores
        ctx = AuthContext(agent_id="agent-1")
        gateway.log_call(
            ctx=ctx,
            action="test_action",
            input_summary="in",
            output_summary="out",
            duration_ms=10.0,
        )

        traces = ts.get_traces_by_case("unknown")
        assert len(traces) == 1


class TestMaskPanInDict:
    def test_masks_amex_pan(self, gateway):
        data = {"card": "371449635398431", "name": "John"}
        result = gateway.mask_pan_in_dict(data)
        assert result["card"] == "[PAN_REDACTED]"
        assert result["name"] == "John"
        # Original not mutated
        assert data["card"] == "371449635398431"

    def test_masks_visa_pan(self, gateway):
        data = {"card": "4111111111111111"}
        result = gateway.mask_pan_in_dict(data)
        assert result["card"] == "[PAN_REDACTED]"

    def test_masks_mc_pan(self, gateway):
        data = {"card": "5425233430109903"}
        result = gateway.mask_pan_in_dict(data)
        assert result["card"] == "[PAN_REDACTED]"

    def test_no_mutation_of_original(self, gateway):
        data = {"card": "4111111111111111", "nested": {"val": "4111111111111111"}}
        original_card = data["card"]
        gateway.mask_pan_in_dict(data)
        assert data["card"] == original_card
        assert data["nested"]["val"] == original_card

    def test_handles_nested_dicts(self, gateway):
        data = {"outer": {"inner": "card is 4111111111111111"}}
        result = gateway.mask_pan_in_dict(data)
        assert "[PAN_REDACTED]" in result["outer"]["inner"]

    def test_handles_lists(self, gateway):
        data = {"cards": ["4111111111111111", "safe-text"]}
        result = gateway.mask_pan_in_dict(data)
        assert result["cards"][0] == "[PAN_REDACTED]"
        assert result["cards"][1] == "safe-text"

    def test_handles_non_string_values(self, gateway):
        data = {"amount": 100.50, "count": 3, "active": True, "empty": None}
        result = gateway.mask_pan_in_dict(data)
        assert result == data
