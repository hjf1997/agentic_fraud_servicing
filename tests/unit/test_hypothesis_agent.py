"""Tests for the hypothesis scoring arbitrator and specialist panel."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.copilot.hypothesis_agent import (
    HYPOTHESIS_REASONING_INSTRUCTIONS,
    HypothesisAssessment,
    HypothesisReasoning,
    hypothesis_reasoning_agent,
    run_arbitrator,
)
from agentic_fraud_servicing.copilot.hypothesis_specialists import (
    SpecialistAssessment,
    run_specialists,
)

# ---------------------------------------------------------------------------
# SpecialistAssessment model tests
# ---------------------------------------------------------------------------


class TestSpecialistAssessment:
    """Tests for the SpecialistAssessment Pydantic model."""

    def test_defaults(self):
        """SpecialistAssessment with required fields has correct defaults."""
        assessment = SpecialistAssessment(category="DISPUTE")
        assert assessment.category == "DISPUTE"
        assert assessment.reasoning == ""
        assert assessment.supporting_evidence == []
        assert assessment.contradicting_evidence == []
        assert assessment.policy_citations == []
        assert assessment.evidence_gaps == []
        assert assessment.eligibility == "eligible"

    def test_all_fields(self):
        """SpecialistAssessment accepts all fields."""
        assessment = SpecialistAssessment(
            category="SCAM",
            reasoning="Strong social engineering indicators.",
            supporting_evidence=["Coached language in transcript"],
            contradicting_evidence=["No unusual payment method"],
            policy_citations=["Per fraud_case_checklist.md: 'scam documentation'"],
            evidence_gaps=["Communication trail with scammer"],
            eligibility="blocked",
        )
        assert len(assessment.supporting_evidence) == 1
        assert len(assessment.policy_citations) == 1
        assert assessment.evidence_gaps == ["Communication trail with scammer"]
        assert assessment.eligibility == "blocked"

    def test_json_round_trip(self):
        """SpecialistAssessment survives JSON serialization."""
        original = SpecialistAssessment(
            category="THIRD_PARTY_FRAUD",
            reasoning="Some unfamiliar device activity.",
            evidence_gaps=["Device fingerprint data"],
            eligibility="eligible",
        )
        json_str = original.model_dump_json()
        restored = SpecialistAssessment.model_validate_json(json_str)
        assert restored == original


# ---------------------------------------------------------------------------
# HypothesisAssessment model tests
# ---------------------------------------------------------------------------


class TestHypothesisAssessment:
    """Tests for the HypothesisAssessment Pydantic model."""

    def test_defaults(self):
        """Default scores are uniform 0.20 across all 5 categories."""
        assessment = HypothesisAssessment()
        assert assessment.scores == {
            "THIRD_PARTY_FRAUD": 0.20,
            "FIRST_PARTY_FRAUD": 0.20,
            "SCAM": 0.20,
            "DISPUTE": 0.20,
            "UNABLE_TO_DETERMINE": 0.20,
        }
        assert assessment.contradictions == []
        assert assessment.assessment_summary == ""
        assert assessment.specialist_assessments == {}

    def test_all_fields(self):
        """All fields can be set explicitly."""
        assessment = HypothesisAssessment(
            scores={
                "THIRD_PARTY_FRAUD": 0.1,
                "FIRST_PARTY_FRAUD": 0.5,
                "SCAM": 0.15,
                "DISPUTE": 0.1,
                "UNABLE_TO_DETERMINE": 0.15,
            },
            reasoning={
                "THIRD_PARTY_FRAUD": "Low — auth from enrolled device",
                "FIRST_PARTY_FRAUD": "High — chip+PIN contradicts claim",
                "SCAM": "Moderate — some urgency in language",
                "DISPUTE": "Low — no merchant issue",
                "UNABLE_TO_DETERMINE": "Some evidence gaps remain",
            },
            contradictions=[
                "CM claims unauthorized but chip+PIN auth from enrolled device",
                "CM knew merchant name before being told",
            ],
            assessment_summary="Strong indicators of first-party fraud.",
        )
        assert assessment.scores["FIRST_PARTY_FRAUD"] == 0.5
        assert len(assessment.contradictions) == 2
        assert "first-party fraud" in assessment.assessment_summary

    def test_scores_dict_has_five_keys(self):
        """Default scores dict contains exactly the 5 investigation categories."""
        assessment = HypothesisAssessment()
        expected_keys = {
            "THIRD_PARTY_FRAUD",
            "FIRST_PARTY_FRAUD",
            "SCAM",
            "DISPUTE",
            "UNABLE_TO_DETERMINE",
        }
        assert set(assessment.scores.keys()) == expected_keys

    def test_reasoning_dict_has_five_keys(self):
        """Default reasoning dict contains exactly the 5 investigation categories."""
        assessment = HypothesisAssessment()
        expected_keys = {
            "THIRD_PARTY_FRAUD",
            "FIRST_PARTY_FRAUD",
            "SCAM",
            "DISPUTE",
            "UNABLE_TO_DETERMINE",
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
                "FIRST_PARTY_FRAUD": 0.6,
                "SCAM": 0.1,
                "DISPUTE": 0.1,
                "UNABLE_TO_DETERMINE": 0.1,
            },
            reasoning={
                "THIRD_PARTY_FRAUD": "Low",
                "FIRST_PARTY_FRAUD": "High",
                "SCAM": "Low",
                "DISPUTE": "Low",
                "UNABLE_TO_DETERMINE": "Low",
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

    def test_specialist_assessments_excluded_from_json(self):
        """specialist_assessments is excluded from JSON serialization."""
        assessment = HypothesisAssessment()
        assessment.specialist_assessments = {
            "DISPUTE": SpecialistAssessment(category="DISPUTE"),
        }
        dumped = assessment.model_dump()
        assert "specialist_assessments" not in dumped

    def test_specialist_assessments_stored(self):
        """specialist_assessments can be set and read programmatically."""
        assessment = HypothesisAssessment()
        sa = SpecialistAssessment(category="DISPUTE")
        assessment.specialist_assessments = {"DISPUTE": sa}
        assert assessment.specialist_assessments["DISPUTE"].category == "DISPUTE"


# ---------------------------------------------------------------------------
# Agent instance tests
# ---------------------------------------------------------------------------


class TestHypothesisReasoningAgent:
    """Tests for the hypothesis_reasoning_agent Agent instance."""

    def test_agent_name(self):
        """Agent has the correct name."""
        assert hypothesis_reasoning_agent.name == "hypothesis_reasoning"

    def test_agent_output_type(self):
        """Agent has HypothesisReasoning as output_type."""
        assert hypothesis_reasoning_agent.output_type.output_type is HypothesisReasoning

    def test_instructions_contain_reasoning_concepts(self):
        """Instructions reference reasoning-specific concepts."""
        lower = HYPOTHESIS_REASONING_INSTRUCTIONS.lower()
        assert "specialist" in lower
        assert "first_party_fraud" in lower or "first-party fraud" in lower
        assert "reasoning" in lower
        assert "contradictions" in lower

    def test_instructions_contain_first_party_fraud_detection(self):
        """Instructions describe first-party fraud cross-cutting detection."""
        lower = HYPOTHESIS_REASONING_INSTRUCTIONS.lower()
        assert "cross-cutting" in lower or "cross-specialist" in lower
        assert "external manipulator" in lower or "external deceiver" in lower
        assert "contradicting evidence" in lower


# ---------------------------------------------------------------------------
# run_specialists tests
# ---------------------------------------------------------------------------


class TestRunSpecialists:
    """Tests for the run_specialists parallel runner."""

    @pytest.fixture
    def mock_provider(self):
        return MagicMock()

    async def test_returns_three_assessments(self, mock_provider):
        """run_specialists returns dict with 3 category keys."""
        mock_result = MagicMock()
        mock_result.final_output = SpecialistAssessment(category="placeholder", reasoning="Test.")

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_specialists.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            results = await run_specialists(
                "allegations", "evidence", "conversation", mock_provider
            )

        assert set(results.keys()) == {"DISPUTE", "SCAM", "THIRD_PARTY_FRAUD"}
        for assessment in results.values():
            assert isinstance(assessment, SpecialistAssessment)

    async def test_handles_specialist_failure(self, mock_provider):
        """Failing specialist returns default assessment with eligibility 'eligible'."""
        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("LLM timeout")
            mock_result = MagicMock()
            mock_result.final_output = SpecialistAssessment(category="ok", reasoning="Fine.")
            return mock_result

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_specialists.run_with_retry",
            new_callable=AsyncMock,
            side_effect=_side_effect,
        ):
            results = await run_specialists(
                "allegations", "evidence", "conversation", mock_provider
            )

        # One failed (default: reasoning contains error), two succeeded
        reasonings = [a.reasoning for a in results.values()]
        assert any("unavailable" in r.lower() for r in reasonings)
        assert sum(1 for r in reasonings if r == "Fine.") == 2

    async def test_passes_previous_assessments(self, mock_provider):
        """Previous assessments are included in specialist user messages."""
        captured_inputs: list[str] = []

        async def _capture(*args, **kwargs):
            captured_inputs.append(kwargs.get("input", args[1] if len(args) > 1 else ""))
            mock_result = MagicMock()
            mock_result.final_output = SpecialistAssessment(category="test", reasoning="Ok.")
            return mock_result

        prev = {
            "DISPUTE": SpecialistAssessment(category="DISPUTE", reasoning="Looks like dispute."),
        }

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_specialists.run_with_retry",
            new_callable=AsyncMock,
            side_effect=_capture,
        ):
            await run_specialists(
                "allegations",
                "evidence",
                "conversation",
                mock_provider,
                previous_assessments=prev,
            )

        # The dispute specialist should see its previous assessment
        dispute_input = captured_inputs[0]  # DISPUTE is first in _SPECIALISTS
        assert "Your Previous Assessment" in dispute_input
        assert "eligible" in dispute_input


# ---------------------------------------------------------------------------
# run_arbitrator tests
# ---------------------------------------------------------------------------


class TestRunArbitrator:
    """Tests for the run_arbitrator async function."""

    @pytest.fixture
    def mock_provider(self):
        return MagicMock()

    @pytest.fixture
    def mock_openai_client(self):
        return AsyncMock()

    @pytest.fixture
    def sample_assessment(self):
        return HypothesisAssessment(
            scores={
                "THIRD_PARTY_FRAUD": 0.1,
                "FIRST_PARTY_FRAUD": 0.5,
                "SCAM": 0.15,
                "DISPUTE": 0.1,
                "UNABLE_TO_DETERMINE": 0.15,
            },
            reasoning={
                "THIRD_PARTY_FRAUD": "Low — enrolled device used",
                "FIRST_PARTY_FRAUD": "High — chip+PIN contradiction",
                "SCAM": "Moderate — some urgency",
                "DISPUTE": "Low — no merchant issue",
                "UNABLE_TO_DETERMINE": "Some evidence gaps remain",
            },
            contradictions=["CM claims unauthorized but chip+PIN auth"],
            assessment_summary="First-party fraud indicators present.",
        )

    @pytest.fixture
    def mock_specialist_outputs(self):
        return {
            "DISPUTE": SpecialistAssessment(category="DISPUTE", reasoning="No merchant issue."),
            "SCAM": SpecialistAssessment(category="SCAM", reasoning="Some urgency."),
            "THIRD_PARTY_FRAUD": SpecialistAssessment(
                category="THIRD_PARTY_FRAUD", reasoning="Enrolled device."
            ),
        }

    @pytest.fixture
    def default_scores(self):
        return {
            "THIRD_PARTY_FRAUD": 0.20,
            "FIRST_PARTY_FRAUD": 0.20,
            "SCAM": 0.20,
            "DISPUTE": 0.20,
            "UNABLE_TO_DETERMINE": 0.20,
        }

    @pytest.fixture
    def sample_reasoning(self):
        return HypothesisReasoning(
            reasoning={
                "THIRD_PARTY_FRAUD": "Low — enrolled device used",
                "FIRST_PARTY_FRAUD": "High — chip+PIN contradiction",
                "SCAM": "Moderate — some urgency",
                "DISPUTE": "Low — no merchant issue",
                "UNABLE_TO_DETERMINE": "Some evidence gaps remain",
            },
            contradictions=["CM claims unauthorized but chip+PIN auth"],
            assessment_summary="First-party fraud indicators present.",
        )

    @pytest.fixture
    def sample_logit_scores(self):
        return {
            "THIRD_PARTY_FRAUD": 0.1,
            "FIRST_PARTY_FRAUD": 0.5,
            "SCAM": 0.15,
            "DISPUTE": 0.1,
            "UNABLE_TO_DETERMINE": 0.15,
        }

    async def test_returns_hypothesis_assessment(
        self,
        mock_provider,
        mock_openai_client,
        sample_reasoning,
        sample_logit_scores,
        mock_specialist_outputs,
        default_scores,
    ):
        """run_arbitrator returns HypothesisAssessment."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = sample_reasoning

        with (
            patch(
                "agentic_fraud_servicing.copilot.hypothesis_agent.run_with_retry",
                new_callable=AsyncMock,
                return_value=mock_run_result,
            ),
            patch(
                "agentic_fraud_servicing.copilot.hypothesis_agent.compute_logprob_scores",
                new_callable=AsyncMock,
                return_value=sample_logit_scores,
            ),
        ):
            result = await run_arbitrator(
                specialist_assessments=mock_specialist_outputs,
                allegations_summary="UNRECOGNIZED_TRANSACTION: $499 charge",
                auth_summary="Impersonation risk: 0.2",
                current_scores=default_scores,
                model_provider=mock_provider,
                openai_client=mock_openai_client,
            )

        assert isinstance(result, HypothesisAssessment)
        assert result.scores["FIRST_PARTY_FRAUD"] == 0.5

    async def test_includes_specialist_outputs_in_reasoning_message(
        self,
        mock_provider,
        mock_openai_client,
        mock_specialist_outputs,
        default_scores,
        sample_logit_scores,
    ):
        """Reasoning agent user message contains formatted specialist assessments."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = HypothesisReasoning()

        with (
            patch(
                "agentic_fraud_servicing.copilot.hypothesis_agent.run_with_retry",
                new_callable=AsyncMock,
                return_value=mock_run_result,
            ) as mock_run,
            patch(
                "agentic_fraud_servicing.copilot.hypothesis_agent.compute_logprob_scores",
                new_callable=AsyncMock,
                return_value=sample_logit_scores,
            ),
        ):
            await run_arbitrator(
                specialist_assessments=mock_specialist_outputs,
                allegations_summary="claims",
                auth_summary="auth",
                current_scores=default_scores,
                model_provider=mock_provider,
                openai_client=mock_openai_client,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "Specialist Assessments" in user_input
        assert "Dispute Specialist" in user_input
        assert "Scam Specialist" in user_input
        assert "Fraud Specialist (Third-Party)" in user_input

    def test_does_not_import_run_specialists(self):
        """run_arbitrator module does not import run_specialists — decoupled."""
        import agentic_fraud_servicing.copilot.hypothesis_agent as mod

        assert not hasattr(mod, "run_specialists")

    async def test_wraps_exceptions(self, mock_provider, mock_openai_client, default_scores):
        """run_arbitrator wraps exceptions in RuntimeError."""
        mock_specialist_outputs = {
            "DISPUTE": SpecialistAssessment(category="DISPUTE"),
            "SCAM": SpecialistAssessment(category="SCAM"),
            "THIRD_PARTY_FRAUD": SpecialistAssessment(category="THIRD_PARTY_FRAUD"),
        }

        with (
            patch(
                "agentic_fraud_servicing.copilot.hypothesis_agent.run_with_retry",
                new_callable=AsyncMock,
                side_effect=ValueError("LLM call failed"),
            ),
            patch(
                "agentic_fraud_servicing.copilot.hypothesis_agent.compute_logprob_scores",
                new_callable=AsyncMock,
                side_effect=ValueError("API failed"),
            ),
        ):
            with pytest.raises(RuntimeError, match="Hypothesis arbitrator failed"):
                await run_arbitrator(
                    specialist_assessments=mock_specialist_outputs,
                    allegations_summary="claims",
                    auth_summary="auth",
                    current_scores=default_scores,
                    model_provider=mock_provider,
                    openai_client=mock_openai_client,
                )
