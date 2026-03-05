"""Tests for gateway/tools/compliance.py: compliance tools for agents."""

from datetime import datetime, timedelta, timezone

import pytest

from agentic_fraud_servicing.gateway.tool_gateway import (
    AuthContext,
    GatewayAuthError,
    ToolGateway,
)
from agentic_fraud_servicing.gateway.tools.compliance import (
    RETENTION_DAYS,
    check_retention,
    redact_case_fields,
    verify_consent,
)
from agentic_fraud_servicing.models.case import AuditEntry, Case
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
def compliance_ctx():
    """Auth context with compliance permission."""
    return AuthContext(agent_id="test-agent", case_id="case-1", permissions={"compliance"})


@pytest.fixture()
def no_compliance_ctx():
    """Auth context without compliance permission."""
    return AuthContext(agent_id="test-agent", case_id="case-1", permissions={"read"})


def _make_case(
    case_id: str = "case-1",
    created_at: datetime = NOW,
    audit_trail: list | None = None,
) -> Case:
    """Helper to create a minimal Case model."""
    return Case(
        case_id=case_id,
        call_id="call-1",
        customer_id="cust-1",
        account_id="acct-1",
        created_at=created_at,
        audit_trail=audit_trail or [],
    )


# --- check_retention tests ---


class TestCheckRetention:
    def test_within_retention_returns_true(self, gateway, stores, compliance_ctx):
        """Recent case is within the retention window."""
        cs, _, _ = stores
        cs.create_case(_make_case())

        result = check_retention(gateway, compliance_ctx, "case-1")

        assert result["case_id"] == "case-1"
        assert result["within_retention"] is True
        assert result["age_days"] >= 0
        assert result["retention_limit_days"] == RETENTION_DAYS

    def test_old_case_returns_false(self, gateway, stores, compliance_ctx):
        """Case older than retention window returns False."""
        cs, _, _ = stores
        old_date = NOW - timedelta(days=RETENTION_DAYS + 100)
        cs.create_case(_make_case(created_at=old_date))

        result = check_retention(gateway, compliance_ctx, "case-1")

        assert result["within_retention"] is False
        assert result["age_days"] > RETENTION_DAYS

    def test_case_not_found_raises(self, gateway, compliance_ctx):
        """Raises RuntimeError when case doesn't exist."""
        with pytest.raises(RuntimeError, match="not found"):
            check_retention(gateway, compliance_ctx, "nonexistent")

    def test_auth_failure_raises(self, gateway, no_compliance_ctx):
        """Raises GatewayAuthError when ctx lacks 'compliance' permission."""
        with pytest.raises(GatewayAuthError, match="Permission 'compliance' not granted"):
            check_retention(gateway, no_compliance_ctx, "case-1")

    def test_logs_call_to_trace_store(self, gateway, stores, compliance_ctx):
        """The call is logged to the trace store."""
        cs, _, ts = stores
        cs.create_case(_make_case())

        check_retention(gateway, compliance_ctx, "case-1")

        traces = ts.get_traces_by_case("case-1")
        assert len(traces) == 1
        assert traces[0]["action"] == "check_retention"
        assert traces[0]["agent_id"] == "test-agent"
        assert traces[0]["status"] == "success"


# --- verify_consent tests ---


class TestVerifyConsent:
    def test_consent_found_returns_true(self, gateway, stores, compliance_ctx):
        """Case with a consent audit entry returns consent_recorded=True."""
        cs, _, _ = stores
        audit = [
            AuditEntry(
                timestamp=NOW,
                action="record_consent",
                details="Customer gave verbal consent",
            )
        ]
        cs.create_case(_make_case(audit_trail=audit))

        result = verify_consent(gateway, compliance_ctx, "case-1")

        assert result["case_id"] == "case-1"
        assert result["consent_recorded"] is True
        assert len(result["consent_entries"]) == 1
        assert "verbal consent" in result["consent_entries"][0]

    def test_no_consent_returns_false(self, gateway, stores, compliance_ctx):
        """Case without consent audit entries returns consent_recorded=False."""
        cs, _, _ = stores
        audit = [AuditEntry(timestamp=NOW, action="open_case", details="Case opened")]
        cs.create_case(_make_case(audit_trail=audit))

        result = verify_consent(gateway, compliance_ctx, "case-1")

        assert result["consent_recorded"] is False
        assert result["consent_entries"] == []

    def test_case_not_found_raises(self, gateway, compliance_ctx):
        """Raises RuntimeError when case doesn't exist."""
        with pytest.raises(RuntimeError, match="not found"):
            verify_consent(gateway, compliance_ctx, "nonexistent")


# --- redact_case_fields tests ---


class TestRedactCaseFields:
    def test_redacts_specified_fields(self, gateway, stores, compliance_ctx):
        """Specified fields are replaced with '[REDACTED]' in the returned dict."""
        cs, _, _ = stores
        cs.create_case(_make_case())

        result = redact_case_fields(
            gateway, compliance_ctx, "case-1", ["customer_id", "account_id"]
        )

        assert result["customer_id"] == "[REDACTED]"
        assert result["account_id"] == "[REDACTED]"
        # Non-redacted fields remain unchanged.
        assert result["case_id"] == "case-1"

    def test_invalid_field_raises_value_error(self, gateway, stores, compliance_ctx):
        """Raises ValueError when a field doesn't exist on the Case model."""
        cs, _, _ = stores
        cs.create_case(_make_case())

        with pytest.raises(ValueError, match="does not exist"):
            redact_case_fields(gateway, compliance_ctx, "case-1", ["nonexistent_field"])

    def test_auth_failure_raises(self, gateway, no_compliance_ctx):
        """Raises GatewayAuthError when ctx lacks 'compliance' permission."""
        with pytest.raises(GatewayAuthError, match="Permission 'compliance' not granted"):
            redact_case_fields(gateway, no_compliance_ctx, "case-1", ["customer_id"])

    def test_case_not_found_raises_value_error(self, gateway, compliance_ctx):
        """Raises ValueError when case doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            redact_case_fields(gateway, compliance_ctx, "nonexistent", ["customer_id"])

    def test_logs_call_to_trace_store(self, gateway, stores, compliance_ctx):
        """The call is logged to the trace store."""
        cs, _, ts = stores
        cs.create_case(_make_case())

        redact_case_fields(gateway, compliance_ctx, "case-1", ["customer_id"])

        traces = ts.get_traces_by_case("case-1")
        assert len(traces) == 1
        assert traces[0]["action"] == "redact_case_fields"
        assert traces[0]["status"] == "success"
