"""Tests for the fast data retrieval agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.copilot.retrieval_agent import (
    RETRIEVAL_INSTRUCTIONS,
    RetrievalResult,
    retrieval_agent,
    run_retrieval,
)


class TestRetrievalResult:
    """Tests for the RetrievalResult Pydantic model."""

    def test_defaults(self):
        """RetrievalResult with all defaults has correct empty values."""
        result = RetrievalResult()
        assert result.transactions == []
        assert result.auth_events == []
        assert result.customer_profile is None
        assert result.retrieval_summary == ""
        assert result.data_gaps == []

    def test_all_fields(self):
        """RetrievalResult accepts all fields with correct types."""
        result = RetrievalResult(
            transactions=[{"amount": 100.0, "merchant": "ACME"}],
            auth_events=[{"auth_type": "sms_otp", "result": "success"}],
            customer_profile={"name": "Jane Doe", "risk_level": "low"},
            retrieval_summary="Found 1 transaction, 1 auth event, and customer profile.",
            data_gaps=["No delivery proof found"],
        )
        assert len(result.transactions) == 1
        assert len(result.auth_events) == 1
        assert result.customer_profile is not None
        assert "1 transaction" in result.retrieval_summary
        assert len(result.data_gaps) == 1

    def test_round_trip_json(self):
        """RetrievalResult survives JSON round-trip serialization."""
        original = RetrievalResult(
            transactions=[{"amount": 50.0}],
            auth_events=[],
            customer_profile={"id": "cust-1"},
            retrieval_summary="Found 1 transaction.",
            data_gaps=["No auth events"],
        )
        json_str = original.model_dump_json()
        restored = RetrievalResult.model_validate_json(json_str)
        assert restored == original

    def test_customer_profile_none(self):
        """RetrievalResult allows None for customer_profile."""
        result = RetrievalResult(customer_profile=None)
        assert result.customer_profile is None
        dumped = result.model_dump()
        assert dumped["customer_profile"] is None


class TestRetrievalAgent:
    """Tests for the retrieval_agent Agent instance."""

    def test_agent_name(self):
        """Agent has the correct name."""
        assert retrieval_agent.name == "fast_retrieval"

    def test_agent_output_type(self):
        """Agent has RetrievalResult as output_type."""
        assert retrieval_agent.output_type is RetrievalResult

    def test_agent_has_three_tools(self):
        """Agent has exactly 3 tools registered."""
        assert len(retrieval_agent.tools) == 3

    def test_agent_instructions(self):
        """Agent instructions reference key retrieval concepts."""
        assert "tool_lookup_transactions" in RETRIEVAL_INSTRUCTIONS
        assert "tool_query_auth_logs" in RETRIEVAL_INSTRUCTIONS
        assert "tool_fetch_customer_profile" in RETRIEVAL_INSTRUCTIONS


class TestRunRetrieval:
    """Tests for the run_retrieval async function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock ModelProvider."""
        return MagicMock()

    @pytest.fixture
    def mock_gateway(self):
        """Create a mock ToolGateway."""
        return MagicMock()

    @pytest.fixture
    def sample_retrieval_result(self):
        """Create a sample RetrievalResult for mocking."""
        return RetrievalResult(
            transactions=[{"amount": 200.0, "merchant": "STORE"}],
            auth_events=[{"auth_type": "password", "result": "success"}],
            customer_profile={"name": "John Doe"},
            retrieval_summary="Retrieved 1 transaction, 1 auth event, profile.",
            data_gaps=[],
        )

    async def test_run_retrieval_returns_result(
        self, mock_provider, mock_gateway, sample_retrieval_result
    ):
        """run_retrieval returns RetrievalResult from Runner.run."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = sample_retrieval_result

        with patch(
            "agentic_fraud_servicing.copilot.retrieval_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_retrieval("case-1", "call-1", mock_gateway, mock_provider)

        assert isinstance(result, RetrievalResult)
        assert len(result.transactions) == 1
        assert result.customer_profile is not None

    async def test_run_retrieval_passes_model_provider(self, mock_provider, mock_gateway):
        """run_retrieval passes model_provider in RunConfig."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = RetrievalResult()

        with patch(
            "agentic_fraud_servicing.copilot.retrieval_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_retrieval("case-1", "call-1", mock_gateway, mock_provider)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["run_config"].model_provider is mock_provider

    async def test_run_retrieval_creates_copilot_context(self, mock_provider, mock_gateway):
        """run_retrieval creates CopilotContext with correct case_id and gateway."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = RetrievalResult()

        with patch(
            "agentic_fraud_servicing.copilot.retrieval_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_retrieval("case-42", "call-99", mock_gateway, mock_provider)

        call_kwargs = mock_run.call_args
        ctx = call_kwargs.kwargs["context"]
        assert ctx.case_id == "case-42"
        assert ctx.call_id == "call-99"
        assert ctx.gateway is mock_gateway

    async def test_run_retrieval_wraps_exceptions(self, mock_provider, mock_gateway):
        """run_retrieval wraps SDK exceptions in RuntimeError."""
        with patch(
            "agentic_fraud_servicing.copilot.retrieval_agent.Runner.run",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM call failed"),
        ):
            with pytest.raises(RuntimeError, match="Retrieval agent failed"):
                await run_retrieval("case-1", "call-1", mock_gateway, mock_provider)
