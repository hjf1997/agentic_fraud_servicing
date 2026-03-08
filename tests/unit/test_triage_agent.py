"""Tests for the triage specialist agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from agentic_fraud_servicing.copilot.triage_agent import (
    TRIAGE_INSTRUCTIONS,
    TriageResult,
    run_triage,
    triage_agent,
)
from agentic_fraud_servicing.models.enums import AllegationType


class TestTriageResult:
    """Tests for the TriageResult Pydantic model."""

    def test_defaults(self):
        """TriageResult with all defaults has correct empty/zero values."""
        result = TriageResult()
        assert result.claims == []
        assert result.allegation_type is None
        assert result.confidence == 0.0
        assert result.category_shift_detected is False
        assert result.key_phrases == []

    def test_all_fields(self):
        """TriageResult accepts all fields with correct types."""
        result = TriageResult(
            claims=["I didn't make this purchase", "Someone used my card"],
            allegation_type=AllegationType.FRAUD,
            confidence=0.92,
            category_shift_detected=True,
            key_phrases=["didn't make", "someone used"],
        )
        assert len(result.claims) == 2
        assert result.allegation_type == AllegationType.FRAUD
        assert result.confidence == 0.92
        assert result.category_shift_detected is True
        assert len(result.key_phrases) == 2

    def test_confidence_validation_too_high(self):
        """Confidence above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            TriageResult(confidence=1.5)

    def test_confidence_validation_too_low(self):
        """Confidence below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            TriageResult(confidence=-0.1)

    def test_round_trip_json(self):
        """TriageResult survives JSON round-trip serialization."""
        original = TriageResult(
            claims=["charge was unauthorized"],
            allegation_type=AllegationType.SCAM,
            confidence=0.75,
            category_shift_detected=False,
            key_phrases=["unauthorized"],
        )
        json_str = original.model_dump_json()
        restored = TriageResult.model_validate_json(json_str)
        assert restored == original


class TestTriageAgent:
    """Tests for the triage_agent Agent instance."""

    def test_agent_name(self):
        """Agent has the correct name."""
        assert triage_agent.name == "triage"

    def test_agent_output_type(self):
        """Agent has TriageResult as output_type."""
        assert triage_agent.output_type.output_type is TriageResult

    def test_agent_instructions(self):
        """Agent instructions reference key triage concepts."""
        assert "FRAUD" in TRIAGE_INSTRUCTIONS
        assert "DISPUTE" in TRIAGE_INSTRUCTIONS
        assert "SCAM" in TRIAGE_INSTRUCTIONS

    def test_agent_instructions_four_categories(self):
        """Agent instructions reference the 4 investigation categories."""
        assert "THIRD_PARTY_FRAUD" in TRIAGE_INSTRUCTIONS
        assert "FIRST_PARTY_FRAUD" in TRIAGE_INSTRUCTIONS
        assert "InvestigationCategory" in TRIAGE_INSTRUCTIONS
        assert "AllegationType" in TRIAGE_INSTRUCTIONS


class TestRunTriage:
    """Tests for the run_triage async function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock ModelProvider."""
        return MagicMock()

    @pytest.fixture
    def sample_triage_result(self):
        """Create a sample TriageResult for mocking."""
        return TriageResult(
            claims=["I was charged twice for the same item"],
            allegation_type=AllegationType.DISPUTE,
            confidence=0.85,
            category_shift_detected=False,
            key_phrases=["charged twice"],
        )

    async def test_run_triage_returns_result(self, mock_provider, sample_triage_result):
        """run_triage returns TriageResult from Runner.run."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = sample_triage_result

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_triage("I was charged twice", None, mock_provider)

        assert isinstance(result, TriageResult)
        assert result.allegation_type == AllegationType.DISPUTE
        assert result.confidence == 0.85

    async def test_run_triage_passes_model_provider(self, mock_provider):
        """run_triage passes model_provider in RunConfig."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = TriageResult()

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_triage("test text", None, mock_provider)

        # Verify Runner.run was called with the correct agent and RunConfig
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["run_config"].model_provider is mock_provider

    async def test_run_triage_includes_previous_type(self, mock_provider):
        """run_triage includes previous classification in user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = TriageResult()

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_triage("some text", AllegationType.FRAUD, mock_provider)

        # The input should mention the previous classification
        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "FRAUD" in user_input

    async def test_run_triage_no_previous_type(self, mock_provider):
        """run_triage handles None previous_type with 'classify from scratch' message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = TriageResult()

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_triage("some text", None, mock_provider)

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "from scratch" in user_input

    async def test_run_triage_wraps_exceptions(self, mock_provider):
        """run_triage wraps SDK exceptions in RuntimeError."""
        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.Runner.run",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM call failed"),
        ):
            with pytest.raises(RuntimeError, match="Triage agent failed"):
                await run_triage("bad input", None, mock_provider)
