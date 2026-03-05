"""Tests for the question planner specialist agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from agentic_fraud_servicing.copilot.question_planner import (
    QUESTION_INSTRUCTIONS,
    QuestionPlan,
    question_agent,
    run_question_planner,
)


class TestQuestionPlan:
    """Tests for the QuestionPlan Pydantic model."""

    def test_defaults(self):
        """QuestionPlan with all defaults has correct empty/zero values."""
        plan = QuestionPlan()
        assert plan.questions == []
        assert plan.rationale == []
        assert plan.priority_field == ""
        assert plan.confidence == 0.0

    def test_all_fields(self):
        """QuestionPlan accepts all fields with correct types."""
        plan = QuestionPlan(
            questions=[
                "Can you describe the transaction you're disputing?",
                "Did you authorize anyone else to use your card?",
            ],
            rationale=[
                "Clarifies which transaction is in scope",
                "Determines if card was shared with an authorized user",
            ],
            priority_field="transaction_details",
            confidence=0.88,
        )
        assert len(plan.questions) == 2
        assert len(plan.rationale) == 2
        assert plan.priority_field == "transaction_details"
        assert plan.confidence == 0.88

    def test_confidence_validation_too_high(self):
        """Confidence above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            QuestionPlan(confidence=1.5)

    def test_confidence_validation_too_low(self):
        """Confidence below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            QuestionPlan(confidence=-0.1)

    def test_round_trip_json(self):
        """QuestionPlan survives JSON round-trip serialization."""
        original = QuestionPlan(
            questions=["When did you first notice the charge?"],
            rationale=["Establishes timeline of awareness"],
            priority_field="discovery_date",
            confidence=0.75,
        )
        json_str = original.model_dump_json()
        restored = QuestionPlan.model_validate_json(json_str)
        assert restored == original


class TestQuestionAgent:
    """Tests for the question_agent Agent instance."""

    def test_agent_name(self):
        """Agent has the correct name."""
        assert question_agent.name == "question_planner"

    def test_agent_output_type(self):
        """Agent has QuestionPlan as output_type."""
        assert question_agent.output_type is QuestionPlan

    def test_agent_instructions_content(self):
        """Agent instructions reference key question planning concepts."""
        assert "PAN" in QUESTION_INSTRUCTIONS
        assert "CVV" in QUESTION_INSTRUCTIONS
        assert "open-ended" in QUESTION_INSTRUCTIONS
        assert "missing field" in QUESTION_INSTRUCTIONS


class TestRunQuestionPlanner:
    """Tests for the run_question_planner async function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock ModelProvider."""
        return MagicMock()

    @pytest.fixture
    def sample_plan(self):
        """Create a sample QuestionPlan for mocking."""
        return QuestionPlan(
            questions=["Can you tell me more about the disputed charge?"],
            rationale=["Gathers basic transaction context"],
            priority_field="transaction_details",
            confidence=0.80,
        )

    async def test_returns_result(self, mock_provider, sample_plan):
        """run_question_planner returns QuestionPlan from Runner.run."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = sample_plan

        with patch(
            "agentic_fraud_servicing.copilot.question_planner.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_question_planner(
                "Customer disputes a $500 charge",
                ["transaction_date", "merchant_name"],
                {"fraud": 0.6, "dispute": 0.3},
                mock_provider,
            )

        assert isinstance(result, QuestionPlan)
        assert result.priority_field == "transaction_details"
        assert result.confidence == 0.80

    async def test_passes_model_provider(self, mock_provider):
        """run_question_planner passes model_provider in RunConfig."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = QuestionPlan()

        with patch(
            "agentic_fraud_servicing.copilot.question_planner.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_question_planner("summary", [], {}, mock_provider)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["run_config"].model_provider is mock_provider

    async def test_includes_missing_fields(self, mock_provider):
        """run_question_planner includes missing fields in user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = QuestionPlan()

        with patch(
            "agentic_fraud_servicing.copilot.question_planner.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_question_planner(
                "some case",
                ["transaction_date", "merchant_name"],
                {},
                mock_provider,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "transaction_date" in user_input
        assert "merchant_name" in user_input

    async def test_includes_hypothesis_scores(self, mock_provider):
        """run_question_planner includes hypothesis scores in user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = QuestionPlan()

        with patch(
            "agentic_fraud_servicing.copilot.question_planner.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_question_planner(
                "some case",
                [],
                {"fraud": 0.7, "scam": 0.2},
                mock_provider,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "fraud" in user_input
        assert "0.70" in user_input

    async def test_handles_empty_missing_fields(self, mock_provider):
        """run_question_planner handles empty missing_fields list."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = QuestionPlan()

        with patch(
            "agentic_fraud_servicing.copilot.question_planner.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_question_planner("summary", [], {}, mock_provider)

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "No missing fields" in user_input

    async def test_wraps_exceptions(self, mock_provider):
        """run_question_planner wraps SDK exceptions in RuntimeError."""
        with patch(
            "agentic_fraud_servicing.copilot.question_planner.Runner.run",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM call failed"),
        ):
            with pytest.raises(RuntimeError, match="Question planner agent failed"):
                await run_question_planner("bad input", [], {}, mock_provider)
