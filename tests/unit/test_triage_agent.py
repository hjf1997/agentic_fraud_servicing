"""Tests for the triage specialist agent module (allegation extraction)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.copilot.triage_agent import (
    TRIAGE_INSTRUCTIONS,
    run_triage,
    triage_agent,
)
from agentic_fraud_servicing.models.allegations import (
    AllegationExtraction,
    AllegationExtractionResult,
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

    def test_instructions_contain_allegation_detail_types(self):
        """Instructions reference all AllegationDetailType values except UNRECOGNIZED_TRANSACTION.

        UNRECOGNIZED_TRANSACTION is handled by the retrieval agent, not
        the triage agent. It is kept in the enum for reuse elsewhere.
        """
        excluded = {"UNRECOGNIZED_TRANSACTION"}
        for ct in AllegationDetailType:
            if ct.value in excluded:
                continue
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

        history = [("CARDMEMBER", "I didn't make this purchase")]

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_triage(history, mock_provider)

        assert isinstance(result, AllegationExtractionResult)
        assert len(result.allegations) == 1
        assert result.allegations[0].detail_type == AllegationDetailType.UNRECOGNIZED_TRANSACTION

    async def test_passes_model_provider(self, mock_provider):
        """run_triage passes model_provider in RunConfig."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = AllegationExtractionResult()

        history = [("CARDMEMBER", "test text")]

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_triage(history, mock_provider)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["run_config"].model_provider is mock_provider

    async def test_conversation_history_markers(self, mock_provider):
        """run_triage marks turns with [CONTEXT], [NEW], and [LATEST TURN]."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = AllegationExtractionResult()

        history = [
            ("CCP", "How can I help you today?"),
            ("CARDMEMBER", "I have a charge"),
            ("CCP", "Let me check"),
            ("CARDMEMBER", "I didn't make this charge"),
        ]

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_triage(history, mock_provider, new_turn_offset=2)

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "[CONTEXT] CCP: How can I help you today?" in user_input
        assert "[CONTEXT] CARDMEMBER: I have a charge" in user_input
        assert "[NEW] CCP: Let me check" in user_input
        assert "[LATEST TURN] CARDMEMBER: I didn't make this charge" in user_input

    async def test_all_new_when_offset_zero(self, mock_provider):
        """When new_turn_offset=0, all turns except last are [NEW]."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = AllegationExtractionResult()

        history = [
            ("CCP", "How can I help?"),
            ("CARDMEMBER", "I didn't make this charge"),
        ]

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_triage(history, mock_provider, new_turn_offset=0)

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "[NEW] CCP: How can I help?" in user_input
        assert "[LATEST TURN] CARDMEMBER: I didn't make this charge" in user_input
        assert "[CONTEXT]" not in user_input

    async def test_includes_allegation_summary_when_provided(self, mock_provider):
        """run_triage prepends allegation summary to user message when provided."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = AllegationExtractionResult()

        history = [
            ("CCP", "How can I help?"),
            ("CARDMEMBER", "I see a duplicate charge"),
        ]
        summary = "1. [UNRECOGNIZED_TRANSACTION] CM says unauthorized purchase"

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_triage(
                history,
                mock_provider,
                allegation_summary=summary,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "Previously extracted allegations" in user_input
        assert "UNRECOGNIZED_TRANSACTION" in user_input
        assert "do NOT re-extract" in user_input
        assert "[LATEST TURN]" in user_input

    async def test_no_allegation_summary_omits_section(self, mock_provider):
        """run_triage omits allegation summary section when not provided."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = AllegationExtractionResult()

        history = [
            ("CCP", "How can I help?"),
            ("CARDMEMBER", "I didn't make this charge"),
        ]

        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_triage(history, mock_provider)

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "Previously extracted allegations" not in user_input
        assert "[LATEST TURN]" in user_input

    async def test_no_previous_type_or_transcript_text_parameter(self):
        """run_triage does not accept previous_type or transcript_text parameters."""
        import inspect

        sig = inspect.signature(run_triage)
        assert "previous_type" not in sig.parameters
        assert "transcript_text" not in sig.parameters

    async def test_wraps_exceptions(self, mock_provider):
        """run_triage wraps SDK exceptions in RuntimeError."""
        history = [("CARDMEMBER", "bad input")]
        with patch(
            "agentic_fraud_servicing.copilot.triage_agent.run_with_retry",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM call failed"),
        ):
            with pytest.raises(RuntimeError, match="Triage agent failed"):
                await run_triage(history, mock_provider)
