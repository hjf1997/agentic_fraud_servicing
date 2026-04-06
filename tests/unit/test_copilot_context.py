"""Unit tests for copilot context dataclass and tool wrappers."""

import json
from unittest.mock import MagicMock, patch

import pytest
from agents.tool import FunctionTool

from agentic_fraud_servicing.copilot.context import (
    CopilotContext,
    _make_auth_ctx,
    tool_fetch_customer_profile,
    tool_lookup_transactions,
    tool_query_auth_logs,
)
from agentic_fraud_servicing.models.transcript import TranscriptEvent

# ---------------------------------------------------------------------------
# CopilotContext dataclass tests
# ---------------------------------------------------------------------------


class TestCopilotContext:
    """Tests for CopilotContext dataclass fields and defaults."""

    def test_required_fields(self) -> None:
        """CopilotContext requires case_id, call_id, and gateway."""
        gw = MagicMock()
        ctx = CopilotContext(case_id="C-1", call_id="CALL-1", gateway=gw)
        assert ctx.case_id == "C-1"
        assert ctx.call_id == "CALL-1"
        assert ctx.gateway is gw

    def test_default_fields(self) -> None:
        """Optional fields have correct defaults."""
        gw = MagicMock()
        ctx = CopilotContext(case_id="C-1", call_id="CALL-1", gateway=gw)
        assert ctx.hypothesis_scores == {}
        assert ctx.impersonation_risk == 0.0
        assert ctx.evidence_collected == []
        assert ctx.transcript_history == []

    def test_all_fields(self) -> None:
        """CopilotContext accepts all 7 fields."""
        gw = MagicMock()
        event = TranscriptEvent(
            call_id="CALL-1",
            event_id="E-1",
            timestamp_ms=1000,
            speaker="CCP",
            text="hello",
        )
        ctx = CopilotContext(
            case_id="C-1",
            call_id="CALL-1",
            gateway=gw,
            hypothesis_scores={"fraud": 0.7},
            impersonation_risk=0.3,
            evidence_collected=["ref-1"],
            transcript_history=[event],
        )
        assert ctx.hypothesis_scores == {"fraud": 0.7}
        assert ctx.impersonation_risk == 0.3
        assert ctx.evidence_collected == ["ref-1"]
        assert len(ctx.transcript_history) == 1


# ---------------------------------------------------------------------------
# _make_auth_ctx helper tests
# ---------------------------------------------------------------------------


class TestMakeAuthCtx:
    """Tests for the internal _make_auth_ctx helper."""

    def test_creates_correct_auth_context(self) -> None:
        """AuthContext has agent_id='copilot', case_id from context, read permission."""
        gw = MagicMock()
        ctx = CopilotContext(case_id="C-42", call_id="CALL-1", gateway=gw)
        auth = _make_auth_ctx(ctx)
        assert auth.agent_id == "copilot"
        assert auth.case_id == "C-42"
        assert auth.permissions == {"read"}


# ---------------------------------------------------------------------------
# Tool wrapper tests
# ---------------------------------------------------------------------------


class TestToolLookupTransactions:
    """Tests for the tool_lookup_transactions function_tool wrapper."""

    def test_is_function_tool(self) -> None:
        """tool_lookup_transactions is a FunctionTool instance."""
        assert isinstance(tool_lookup_transactions, FunctionTool)

    @pytest.mark.asyncio
    async def test_returns_json_string(self) -> None:
        """Wrapper returns a JSON string of transaction dicts."""
        gw = MagicMock()
        copilot_ctx = CopilotContext(case_id="C-1", call_id="CALL-1", gateway=gw)
        wrapper_ctx = MagicMock()
        wrapper_ctx.context = copilot_ctx

        fake_txns = [
            {
                "amount": 100.00,
                "merchant_name": "SHOP",
                "is_disputed": True,
                "transaction_date": "2024-06-01",
            },
        ]
        with patch(
            "agentic_fraud_servicing.copilot.context.lookup_transactions",
            return_value=fake_txns,
        ) as mock_fn:
            result = await tool_lookup_transactions.on_invoke_tool(wrapper_ctx, "")
            mock_fn.assert_called_once()
            # Verify auth context
            auth_arg = mock_fn.call_args[0][1]
            assert auth_arg.agent_id == "copilot"
            assert auth_arg.permissions == {"read"}

        assert isinstance(result, str)
        parsed = json.loads(result)
        # New format: {"disputed_transactions": [...], "summary": "..."}
        assert "disputed_transactions" in parsed
        assert "summary" in parsed
        assert len(parsed["disputed_transactions"]) == 1
        assert "Disputed Transactions" in parsed["summary"]


class TestToolQueryAuthLogs:
    """Tests for the tool_query_auth_logs function_tool wrapper."""

    def test_is_function_tool(self) -> None:
        """tool_query_auth_logs is a FunctionTool instance."""
        assert isinstance(tool_query_auth_logs, FunctionTool)

    @pytest.mark.asyncio
    async def test_returns_json_string(self) -> None:
        """Wrapper returns a JSON string of auth event dicts."""
        gw = MagicMock()
        copilot_ctx = CopilotContext(case_id="C-2", call_id="CALL-2", gateway=gw)
        wrapper_ctx = MagicMock()
        wrapper_ctx.context = copilot_ctx

        fake_logs = [{"result": "pass"}]
        with patch(
            "agentic_fraud_servicing.copilot.context.query_auth_logs",
            return_value=fake_logs,
        ) as mock_fn:
            result = await tool_query_auth_logs.on_invoke_tool(wrapper_ctx, "")
            mock_fn.assert_called_once()
            auth_arg = mock_fn.call_args[0][1]
            assert auth_arg.agent_id == "copilot"

        assert json.loads(result) == fake_logs


class TestToolFetchCustomerProfile:
    """Tests for the tool_fetch_customer_profile function_tool wrapper."""

    def test_is_function_tool(self) -> None:
        """tool_fetch_customer_profile is a FunctionTool instance."""
        assert isinstance(tool_fetch_customer_profile, FunctionTool)

    @pytest.mark.asyncio
    async def test_returns_json_string(self) -> None:
        """Wrapper returns a JSON string of the customer profile dict."""
        gw = MagicMock()
        copilot_ctx = CopilotContext(case_id="C-3", call_id="CALL-3", gateway=gw)
        wrapper_ctx = MagicMock()
        wrapper_ctx.context = copilot_ctx

        fake_profile = {"name": "John"}
        with patch(
            "agentic_fraud_servicing.copilot.context.fetch_customer_profile",
            return_value=fake_profile,
        ) as mock_fn:
            result = await tool_fetch_customer_profile.on_invoke_tool(wrapper_ctx, "")
            mock_fn.assert_called_once()

        assert json.loads(result) == fake_profile

    @pytest.mark.asyncio
    async def test_returns_null_when_not_found(self) -> None:
        """Wrapper returns 'null' when no customer profile exists."""
        gw = MagicMock()
        copilot_ctx = CopilotContext(case_id="C-4", call_id="CALL-4", gateway=gw)
        wrapper_ctx = MagicMock()
        wrapper_ctx.context = copilot_ctx

        with patch(
            "agentic_fraud_servicing.copilot.context.fetch_customer_profile",
            return_value=None,
        ):
            result = await tool_fetch_customer_profile.on_invoke_tool(wrapper_ctx, "")

        assert result == "null"
