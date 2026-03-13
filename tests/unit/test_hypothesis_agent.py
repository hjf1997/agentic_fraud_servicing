"""Tests for the hypothesis scoring specialist agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.copilot.hypothesis_agent import (
    HYPOTHESIS_INSTRUCTIONS,
    HypothesisAssessment,
    hypothesis_agent,
    run_hypothesis,
)


class TestHypothesisAssessment:
    """Tests for the HypothesisAssessment Pydantic model."""

    def test_defaults(self):
        """Default scores are uniform 0.25 across all 4 categories."""
        assessment = HypothesisAssessment()
        assert assessment.scores == {
            "THIRD_PARTY_FRAUD": 0.25,
            "FIRST_PARTY_FRAUD": 0.25,
            "SCAM": 0.25,
            "DISPUTE": 0.25,
        }
        assert assessment.contradictions == []
        assert assessment.assessment_summary == ""

    def test_all_fields(self):
        """All fields can be set explicitly."""
        assessment = HypothesisAssessment(
            scores={
                "THIRD_PARTY_FRAUD": 0.1,
                "FIRST_PARTY_FRAUD": 0.6,
                "SCAM": 0.2,
                "DISPUTE": 0.1,
            },
            reasoning={
                "THIRD_PARTY_FRAUD": "Low — auth from enrolled device",
                "FIRST_PARTY_FRAUD": "High — chip+PIN contradicts claim",
                "SCAM": "Moderate — some urgency in language",
                "DISPUTE": "Low — no merchant issue",
            },
            contradictions=[
                "CM claims unauthorized but chip+PIN auth from enrolled device",
                "CM knew merchant name before being told",
            ],
            assessment_summary="Strong indicators of first-party fraud.",
        )
        assert assessment.scores["FIRST_PARTY_FRAUD"] == 0.6
        assert len(assessment.contradictions) == 2
        assert "first-party fraud" in assessment.assessment_summary

    def test_scores_dict_has_four_keys(self):
        """Default scores dict contains exactly the 4 investigation categories."""
        assessment = HypothesisAssessment()
        expected_keys = {
            "THIRD_PARTY_FRAUD",
            "FIRST_PARTY_FRAUD",
            "SCAM",
            "DISPUTE",
        }
        assert set(assessment.scores.keys()) == expected_keys

    def test_reasoning_dict_has_four_keys(self):
        """Default reasoning dict contains exactly the 4 investigation categories."""
        assessment = HypothesisAssessment()
        expected_keys = {
            "THIRD_PARTY_FRAUD",
            "FIRST_PARTY_FRAUD",
            "SCAM",
            "DISPUTE",
        }
        assert set(assessment.reasoning.keys()) == expected_keys

    def test_contradictions_is_list(self):
        """Contradictions field is a list of strings."""
        assessment = HypothesisAssessment(contradictions=["Claim A contradicts Evidence B"])
        assert isinstance(assessment.contradictions, list)
        assert assessment.contradictions[0] == "Claim A contradicts Evidence B"

    def test_json_round_trip(self):
        """HypothesisAssessment survives JSON serialization round-trip."""
        original = HypothesisAssessment(
            scores={
                "THIRD_PARTY_FRAUD": 0.1,
                "FIRST_PARTY_FRAUD": 0.7,
                "SCAM": 0.1,
                "DISPUTE": 0.1,
            },
            reasoning={
                "THIRD_PARTY_FRAUD": "Low",
                "FIRST_PARTY_FRAUD": "High",
                "SCAM": "Low",
                "DISPUTE": "Low",
            },
            contradictions=["chip+PIN from enrolled device"],
            assessment_summary="Likely first-party fraud.",
        )
        json_str = original.model_dump_json()
        restored = HypothesisAssessment.model_validate_json(json_str)
        assert restored.scores == original.scores
        assert restored.reasoning == original.reasoning
        assert restored.contradictions == original.contradictions
        assert restored.assessment_summary == original.assessment_summary


class TestHypothesisAgent:
    """Tests for the hypothesis_agent Agent instance."""

    def test_agent_name(self):
        """Agent has the correct name."""
        assert hypothesis_agent.name == "hypothesis"

    def test_agent_output_type(self):
        """Agent has HypothesisAssessment as output_type."""
        assert hypothesis_agent.output_type.output_type is HypothesisAssessment

    def test_instructions_contain_investigation_categories_reference(self):
        """Instructions include the full INVESTIGATION_CATEGORIES_REFERENCE."""
        assert "THIRD_PARTY_FRAUD" in HYPOTHESIS_INSTRUCTIONS
        assert "FIRST_PARTY_FRAUD" in HYPOTHESIS_INSTRUCTIONS
        assert "SCAM" in HYPOTHESIS_INSTRUCTIONS
        assert "DISPUTE" in HYPOTHESIS_INSTRUCTIONS
        # Full definitions, not just names
        assert "Authorization: NO" in HYPOTHESIS_INSTRUCTIONS
        assert "Fraud actor: External criminal" in HYPOTHESIS_INSTRUCTIONS

    def test_instructions_contain_key_reasoning_patterns(self):
        """Instructions include the specific reasoning patterns."""
        lower = HYPOTHESIS_INSTRUCTIONS.lower()
        assert "chip+pin" in lower or "chip" in lower
        assert "enrolled device" in lower
        assert "external manipulator" in lower
        assert "cross-cutting" in lower
        assert "bayesian prior" in lower or "prior" in lower

    def test_instructions_contain_scoring_rules(self):
        """Instructions explain probability distribution scoring."""
        lower = HYPOTHESIS_INSTRUCTIONS.lower()
        assert "probability distribution" in lower
        assert "sum to" in lower

    def test_instructions_contain_first_party_fraud_signals(self):
        """Instructions describe first-party fraud detection signals."""
        assert "story shifts" in HYPOTHESIS_INSTRUCTIONS.lower() or (
            "story" in HYPOTHESIS_INSTRUCTIONS.lower()
            and "shift" in HYPOTHESIS_INSTRUCTIONS.lower()
        )
        assert "delivery proof" in HYPOTHESIS_INSTRUCTIONS.lower() or (
            "delivery" in HYPOTHESIS_INSTRUCTIONS.lower()
            and "proof" in HYPOTHESIS_INSTRUCTIONS.lower()
        )
        assert "merchant familiarity" in HYPOTHESIS_INSTRUCTIONS.lower() or (
            "merchant" in HYPOTHESIS_INSTRUCTIONS.lower()
            and "familiarity" in HYPOTHESIS_INSTRUCTIONS.lower()
        )


class TestRunHypothesis:
    """Tests for the run_hypothesis async function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock ModelProvider."""
        return MagicMock()

    @pytest.fixture
    def sample_assessment(self):
        """Create a sample HypothesisAssessment for mocking."""
        return HypothesisAssessment(
            scores={
                "THIRD_PARTY_FRAUD": 0.1,
                "FIRST_PARTY_FRAUD": 0.6,
                "SCAM": 0.2,
                "DISPUTE": 0.1,
            },
            reasoning={
                "THIRD_PARTY_FRAUD": "Low — enrolled device used",
                "FIRST_PARTY_FRAUD": "High — chip+PIN contradiction",
                "SCAM": "Moderate — some urgency",
                "DISPUTE": "Low — no merchant issue",
            },
            contradictions=["CM claims unauthorized but chip+PIN auth"],
            assessment_summary="First-party fraud indicators present.",
        )

    @pytest.fixture
    def default_scores(self):
        """Default hypothesis scores for testing."""
        return {
            "THIRD_PARTY_FRAUD": 0.25,
            "FIRST_PARTY_FRAUD": 0.25,
            "SCAM": 0.25,
            "DISPUTE": 0.25,
        }

    async def test_returns_hypothesis_assessment(
        self, mock_provider, sample_assessment, default_scores
    ):
        """run_hypothesis returns HypothesisAssessment from Runner.run."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = sample_assessment

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_hypothesis(
                allegations_summary="UNRECOGNIZED_TRANSACTION: CM says didn't make $499 charge",
                auth_summary="Impersonation risk: 0.2, no step-up needed",
                evidence_summary="Chip+PIN auth from enrolled device",
                current_scores=default_scores,
                conversation_summary="CM called about unauthorized charge",
                model_provider=mock_provider,
            )

        assert isinstance(result, HypothesisAssessment)
        assert result.scores["FIRST_PARTY_FRAUD"] == 0.6

    async def test_passes_model_provider(self, mock_provider, default_scores):
        """run_hypothesis passes model_provider in RunConfig."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = HypothesisAssessment()

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_hypothesis(
                allegations_summary="test claims",
                auth_summary="test auth",
                evidence_summary="test evidence",
                current_scores=default_scores,
                conversation_summary="test summary",
                model_provider=mock_provider,
            )

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["run_config"].model_provider is mock_provider

    async def test_includes_allegations_in_message(self, mock_provider, default_scores):
        """run_hypothesis includes allegations_summary in the user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = HypothesisAssessment()

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_hypothesis(
                allegations_summary="UNRECOGNIZED_TRANSACTION: unauthorized $2847 at TechVault",
                auth_summary="low risk",
                evidence_summary="chip+PIN",
                current_scores=default_scores,
                conversation_summary="summary",
                model_provider=mock_provider,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "UNRECOGNIZED_TRANSACTION: unauthorized $2847 at TechVault" in user_input
        assert "Accumulated Allegations" in user_input

    async def test_includes_evidence_in_message(self, mock_provider, default_scores):
        """run_hypothesis includes evidence_summary in the user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = HypothesisAssessment()

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_hypothesis(
                allegations_summary="claims",
                auth_summary="auth",
                evidence_summary="Chip+PIN auth from enrolled device ID dev-123",
                current_scores=default_scores,
                conversation_summary="summary",
                model_provider=mock_provider,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "Chip+PIN auth from enrolled device ID dev-123" in user_input
        assert "Retrieved Evidence" in user_input

    async def test_includes_current_scores_in_message(self, mock_provider):
        """run_hypothesis includes formatted current scores in the user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = HypothesisAssessment()

        scores = {
            "THIRD_PARTY_FRAUD": 0.10,
            "FIRST_PARTY_FRAUD": 0.60,
            "SCAM": 0.20,
            "DISPUTE": 0.10,
        }

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_agent.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_hypothesis(
                allegations_summary="claims",
                auth_summary="auth",
                evidence_summary="evidence",
                current_scores=scores,
                conversation_summary="summary",
                model_provider=mock_provider,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "FIRST_PARTY_FRAUD: 0.60" in user_input
        assert "Current Hypothesis Scores" in user_input

    async def test_wraps_exceptions(self, mock_provider, default_scores):
        """run_hypothesis wraps SDK exceptions in RuntimeError."""
        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_agent.Runner.run",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM call failed"),
        ):
            with pytest.raises(RuntimeError, match="Hypothesis agent failed"):
                await run_hypothesis(
                    allegations_summary="claims",
                    auth_summary="auth",
                    evidence_summary="evidence",
                    current_scores=default_scores,
                    conversation_summary="summary",
                    model_provider=mock_provider,
                )
