"""Tests for the triage specialist agent module (allegation extraction)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentic_fraud_servicing.models.allegations import (
    AllegationExtraction,
    AllegationExtractionResult,
)

from agentic_fraud_servicing.copilot.triage_agent import (
    TRIAGE_INSTRUCTIONS,
    run_triage,
    triage_agent,
)
from agentic_fraud_servicing.models.enums import AllegationDetailType


class TestTriageAgent:
    """Tests for the triage_agent Agent instance."""

    def test_agent_name(self):
        """Agent has the correct name."""
        assert triage_agent.name == "triage"

    def test_agent_output_type(self):
        """Agent has AllegationExtractionResult as output_type."""
        assert triage_agent.output_type.output_type is AllegationExtractionResult

    def test_instructions_contain_all_17_allegation_detail_types(self):
        """Instructions reference all 17 AllegationDetailType enum values."""
        for ct in AllegationDetailType:
            assert ct.value in TRIAGE_INSTRUCTIONS, f"Missing {ct.value}"

    def test_instructions_contain_entity_guidance(self):
        """Instructions include entity extraction guidance for claim types."""
        # Spot-check a few entity fields from different claim types
        assert "merchant_name" in TRIAGE_INSTRUCTIONS
        assert "amount" in TRIAGE_INSTRUCTIONS
        assert "tracking_number" in TRIAGE_INSTRUCTIONS
        assert "cancellation_date" in TRIAGE_INSTRUCTIONS

    def test_instructions_emphasize_claimed_not_conclusions(self):
        """Instructions tell the agent to describe what CM claimed."""
        lower = TRIAGE_INSTRUCTIONS.lower()
        assert "claimed" in lower or "what the cardmember claimed" in lower

    def test_instructions_do_not_contain_allegation_type_output(self):
        """Instructions do not reference AllegationType as an output field."""
        # The word AllegationType may appear in INVESTIGATION_CATEGORIES_REFERENCE
        # but should not appear as an output instruction
        assert "allegation_type:" not in TRIAGE_INSTRUCTIONS
        assert "category_shift_detected" not in TRIAGE_INSTRUCTIONS

    def test_instructions_contain_investigation_categories_reference(self):
        """Instructions include the 4-category reference for context."""
        assert "THIRD_PARTY_FRAUD" in TRIAGE_INSTRUCTIONS
        assert "FIRST_PARTY_FRAUD" in TRIAGE_INSTRUCTIONS

    def test_instructions_contain_example_phrases(self):
        """Instructions contain natural language example phrases."""
        assert "I didn't make this charge" in TRIAGE_INSTRUCTIONS
        assert "Package never arrived" in TRIAGE_INSTRUCTIONS
        assert "charged me twice" in TRIAGE_INSTRUCTIONS


class TestRunTriage:
    """Tests for the run_triage async function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock ModelProvider."""
        return MagicMock()

    @pytest.fixture
    def sample_result(self):
        """Create a sample AllegationExtractionResult for mocking."""
        return AllegationExtractionResult(
            allegations=[
                AllegationExtraction(
                    detail_type=AllegationDetailType.UNRECOGNIZED_TRANSACTION,
                    description="CM says they did not make a $499 charge",
                    entities={"amount": 499.0, "merchant_name": "Electronics Store"},
                    confidence=0.9,
                    context="I didn't make this purchase at Electronics Store",
                ),
            ]
        )

    async def test_returns_allegation_extraction_result(self, mock_provider, sample_result):
        """run_triage returns AllegationExtractionResult from Runner.run."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = sample_result

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_triage("I didn't make this purchase", mock_provider)

        assert isinstance(result, AllegationExtractionResult)
        assert len(result.allegations) == 1
        assert result.allegations[0].detail_type == AllegationDetailType.UNRECOGNIZED_TRANSACTION

    async def test_passes_model_provider(self, mock_provider):
        """run_triage passes model_provider in RunConfig."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = AllegationExtractionResult()

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_triage("test text", mock_provider)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["run_config"].model_provider is mock_provider

    async def test_conversation_history_included(self, mock_provider):
        """run_triage builds message from conversation_history when provided."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = AllegationExtractionResult()

        history = [
            ("CCP", "How can I help you today?"),
            ("CARDMEMBER", "I didn't make this charge"),
        ]

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_triage(
                "I didn't make this charge",
                mock_provider,
                conversation_history=history,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "Conversation history:" in user_input
        assert "[LATEST TURN]" in user_input
        assert "CARDMEMBER: I didn't make this charge" in user_input

    async def test_falls_back_to_transcript_text(self, mock_provider):
        """run_triage uses transcript_text when no conversation_history."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = AllegationExtractionResult()

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_triage("some transcript text", mock_provider)

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "Transcript segment:" in user_input
        assert "some transcript text" in user_input

    async def test_no_previous_type_parameter(self):
        """run_triage does not accept previous_type parameter."""
        import inspect

        sig = inspect.signature(run_triage)
        assert "previous_type" not in sig.parameters

    async def test_wraps_exceptions(self, mock_provider):
        """run_triage wraps SDK exceptions in RuntimeError."""
        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.Runner.run",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM call failed"),
        ):
            with pytest.raises(RuntimeError, match="Triage agent failed"):
                await run_triage("bad input", mock_provider)
