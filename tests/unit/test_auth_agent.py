"""Tests for the authentication and impersonation risk agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from agentic_fraud_servicing.copilot.auth_agent import (
    AUTH_INSTRUCTIONS,
    AuthAssessment,
    auth_agent,
    run_auth_assessment,
)


class TestAuthAssessment:
    """Tests for the AuthAssessment Pydantic model."""

    def test_defaults(self):
        """AuthAssessment with all defaults has correct values."""
        result = AuthAssessment()
        assert result.impersonation_risk == 0.0
        assert result.risk_factors == []
        assert result.step_up_recommended is False
        assert result.step_up_method == "NONE"
        assert result.assessment_summary == ""

    def test_all_fields(self):
        """AuthAssessment accepts all fields with correct types."""
        result = AuthAssessment(
            impersonation_risk=0.85,
            risk_factors=["failed auth attempt", "new device"],
            step_up_recommended=True,
            step_up_method="CALLBACK",
            assessment_summary="High risk due to multiple red flags.",
        )
        assert result.impersonation_risk == 0.85
        assert len(result.risk_factors) == 2
        assert result.step_up_recommended is True
        assert result.step_up_method == "CALLBACK"
        assert "High risk" in result.assessment_summary

    def test_risk_validation_too_high(self):
        """Impersonation risk above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            AuthAssessment(impersonation_risk=1.5)

    def test_risk_validation_too_low(self):
        """Impersonation risk below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            AuthAssessment(impersonation_risk=-0.1)

    def test_round_trip_json(self):
        """AuthAssessment survives JSON round-trip serialization."""
        original = AuthAssessment(
            impersonation_risk=0.6,
            risk_factors=["hesitation on account details"],
            step_up_recommended=True,
            step_up_method="SMS_OTP",
            assessment_summary="Medium-high risk.",
        )
        json_str = original.model_dump_json()
        restored = AuthAssessment.model_validate_json(json_str)
        assert restored == original


class TestAuthAgent:
    """Tests for the auth_agent Agent instance."""

    def test_agent_name(self):
        """Agent has the correct name."""
        assert auth_agent.name == "auth_assessor"

    def test_agent_output_type(self):
        """Agent has AuthAssessment as output_type."""
        assert auth_agent.output_type.output_type is AuthAssessment

    def test_agent_instructions(self):
        """Agent instructions reference key auth assessment concepts."""
        assert "impersonation" in AUTH_INSTRUCTIONS.lower()
        assert "step-up" in AUTH_INSTRUCTIONS.lower()
        assert "SMS_OTP" in AUTH_INSTRUCTIONS

    def test_agent_instructions_four_categories(self):
        """Agent instructions reference the 4 investigation categories."""
        assert "THIRD_PARTY_FRAUD" in AUTH_INSTRUCTIONS
        assert "FIRST_PARTY_FRAUD" in AUTH_INSTRUCTIONS
        assert "SCAM" in AUTH_INSTRUCTIONS


class TestRunAuthAssessment:
    """Tests for the run_auth_assessment async function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock ModelProvider."""
        return MagicMock()

    @pytest.fixture
    def sample_assessment(self):
        """Create a sample AuthAssessment for mocking."""
        return AuthAssessment(
            impersonation_risk=0.7,
            risk_factors=["failed OTP", "new device"],
            step_up_recommended=True,
            step_up_method="CALLBACK",
            assessment_summary="High risk indicators present.",
        )

    async def test_returns_assessment(self, mock_provider, sample_assessment):
        """run_auth_assessment returns AuthAssessment from Runner.run."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = sample_assessment

        with patch(
            "agentic_fraud_servicing.copilot.auth_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_auth_assessment(
                "I need to check my account",
                [{"type": "OTP", "result": "fail"}],
                None,
                mock_provider,
            )

        assert isinstance(result, AuthAssessment)
        assert result.impersonation_risk == 0.7
        assert result.step_up_recommended is True

    async def test_passes_model_provider(self, mock_provider):
        """run_auth_assessment passes model_provider in RunConfig."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = AuthAssessment()

        with patch(
            "agentic_fraud_servicing.copilot.auth_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_auth_assessment("text", [], None, mock_provider)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["run_config"].model_provider is mock_provider

    async def test_includes_auth_events_in_message(self, mock_provider):
        """run_auth_assessment includes auth events JSON in user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = AuthAssessment()

        with patch(
            "agentic_fraud_servicing.copilot.auth_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_auth_assessment(
                "text", [{"type": "OTP", "result": "fail"}], None, mock_provider
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "OTP" in user_input
        assert "fail" in user_input

    async def test_includes_customer_profile(self, mock_provider):
        """run_auth_assessment includes customer profile when provided."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = AuthAssessment()

        with patch(
            "agentic_fraud_servicing.copilot.auth_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            profile = {"name": "John Doe", "risk_tier": "low"}
            await run_auth_assessment("text", [], profile, mock_provider)

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "John Doe" in user_input
        assert "Not available" not in user_input

    async def test_handles_none_profile(self, mock_provider):
        """run_auth_assessment handles None customer_profile gracefully."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = AuthAssessment()

        with patch(
            "agentic_fraud_servicing.copilot.auth_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_auth_assessment("text", [], None, mock_provider)

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "Not available" in user_input

    async def test_wraps_exceptions(self, mock_provider):
        """run_auth_assessment wraps SDK exceptions in RuntimeError."""
        with patch(
            "agentic_fraud_servicing.copilot.auth_agent.Runner.run",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM call failed"),
        ):
            with pytest.raises(RuntimeError, match="Auth assessment agent failed"):
                await run_auth_assessment("bad input", [], None, mock_provider)
